//! Tool arguments against the declared schema: coercion, buried-param hoist, the inferred filter.

const std = @import("std");
const log = @import("../../../sdk/util/log.zig");
const testing = std.testing;

const parseToolCalls = @import("parse.zig").parseToolCalls;
const ParsedToolCall = @import("parse.zig").ParsedToolCall;
const looseRepairContainer = @import("json_repair.zig").looseRepairContainer;

/// Locate a tool's `properties` map in an OpenAI-shaped tools array. Accepts
/// both the wrapped (`{"type":"function","function":{…}}`) and flat forms.
/// The tool's JSON-Schema `parameters` object (Anthropic `input_schema` too).
/// Callers that need only the property map use `toolPropertiesFor`; the hoist
/// also needs `required`, which lives beside `properties` here.
fn toolParametersFor(tools: std.json.Value, name: []const u8) ?std.json.ObjectMap {
    if (tools != .array) return null;
    for (tools.array.items) |tool_val| {
        if (tool_val != .object) continue;
        const func = blk: {
            if (tool_val.object.get("function")) |f| {
                if (f == .object) break :blk f.object;
            }
            break :blk tool_val.object;
        };
        const nv = func.get("name") orelse continue;
        if (nv != .string or !std.mem.eql(u8, nv.string, name)) continue;
        const params = func.get("parameters") orelse func.get("input_schema") orelse return null;
        if (params != .object) return null;
        return params.object;
    }
    return null;
}

fn toolPropertiesFor(tools: std.json.Value, name: []const u8) ?std.json.ObjectMap {
    const params = toolParametersFor(tools, name) orelse return null;
    const props = params.get("properties") orelse return null;
    if (props != .object) return null;
    return props.object;
}

/// The types a misplaced value may be hoisted on. A required OBJECT/ARRAY param
/// found inside another container is too speculative to move — only scalars.
fn isScalarJsonType(want: []const u8) bool {
    return std.mem.eql(u8, want, "string") or std.mem.eql(u8, want, "integer") or
        std.mem.eql(u8, want, "number") or std.mem.eql(u8, want, "boolean");
}

/// The JSON type a property declares. `"type"` may be a union array
/// (`["string","null"]`) — the first non-null entry wins. Absent/odd → null,
/// which means "leave the value alone". Without a `"type"`, a `oneOf`/`anyOf`
/// union declares one only when every non-null branch names the same.
fn declaredJsonType(prop: std.json.Value) ?[]const u8 {
    if (prop != .object) return null;
    const t = prop.object.get("type") orelse return unionJsonType(prop.object);
    switch (t) {
        .string => |s| return s,
        .array => |arr| {
            for (arr.items) |item| {
                if (item != .string) continue;
                if (std.mem.eql(u8, item.string, "null")) continue;
                return item.string;
            }
            return null;
        },
        else => return null,
    }
}

fn unionJsonType(prop: std.json.ObjectMap) ?[]const u8 {
    const branches = (prop.get("oneOf") orelse prop.get("anyOf") orelse return null);
    if (branches != .array) return null;
    var agreed: ?[]const u8 = null;
    for (branches.array.items) |b| {
        const bt = declaredJsonType(b) orelse return null;
        if (std.mem.eql(u8, bt, "null")) continue;
        if (agreed) |a| {
            if (!std.mem.eql(u8, a, bt)) return null;
        } else agreed = bt;
    }
    return agreed;
}

/// Tolerant boolean spelling — the union of what weak models actually emit:
/// Python's `False`, shell's `0`/`1`, prose `yes`/`no`, plus the trailing-comma
/// weld the app's `appendFlagIsTrue` already tolerates. Anything else → null
/// (never guess; an honest client validation error beats a wrong value).
fn looseBoolSpelling(raw: []const u8) ?bool {
    const s = std.mem.trim(u8, raw, " \t\r\n,");
    if (s.len == 0 or s.len > 5) return null;
    var buf: [5]u8 = undefined;
    const lower = std.ascii.lowerString(buf[0..s.len], s);
    if (std.mem.eql(u8, lower, "true") or std.mem.eql(u8, lower, "yes") or
        std.mem.eql(u8, lower, "on") or std.mem.eql(u8, lower, "1")) return true;
    if (std.mem.eql(u8, lower, "false") or std.mem.eql(u8, lower, "no") or
        std.mem.eql(u8, lower, "off") or std.mem.eql(u8, lower, "0")) return false;
    return null;
}

/// Strict parse of a container string that admits a key repeated with the SAME
/// value (issue #402: a model spelling `header` twice per item) and refuses a
/// key repeated with different values: the two are told apart by parsing
/// first-wins and last-wins and comparing the documents.
fn parseContainerAllowingRepeats(allocator: std.mem.Allocator, text: []const u8) ?std.json.Parsed(std.json.Value) {
    var first = std.json.parseFromSlice(std.json.Value, allocator, text, .{ .duplicate_field_behavior = .use_first }) catch return null;
    var last = std.json.parseFromSlice(std.json.Value, allocator, text, .{ .duplicate_field_behavior = .use_last }) catch {
        first.deinit();
        return null;
    };
    defer last.deinit();
    const a = std.json.Stringify.valueAlloc(allocator, first.value, .{}) catch {
        first.deinit();
        return null;
    };
    defer allocator.free(a);
    const b = std.json.Stringify.valueAlloc(allocator, last.value, .{}) catch {
        first.deinit();
        return null;
    };
    defer allocator.free(b);
    if (!std.mem.eql(u8, a, b)) {
        first.deinit();
        return null;
    }
    return first;
}

/// Coerce ONE value to the declared type. Returns null when the value already
/// matches or the spelling is undecidable. Allocated replacements are appended
/// to `owned`; parsed sub-documents to `docs` — both are freed by the caller
/// AFTER the object is re-serialized, because the returned Value borrows them.
fn coerceValueToType(
    allocator: std.mem.Allocator,
    v: std.json.Value,
    want: []const u8,
    owned: *std.ArrayList([]const u8),
    docs: *std.ArrayList(std.json.Parsed(std.json.Value)),
) !?std.json.Value {
    // A CONTAINER param carried through a tag format arrives as a STRING — no tag
    // format types its values. Live (15 hits in one session): pi's `edit.edits`,
    // an array of {oldText,newText}, shipped as `"[{\"oldText\":…}]"` on every
    // call. Strict parse only: a value that isn't well-formed JSON of the
    // DECLARED kind is left alone rather than guessed at.
    if (std.mem.eql(u8, want, "array") or std.mem.eql(u8, want, "object")) {
        if (v != .string) return null;
        const t = std.mem.trim(u8, v.string, " \t\r\n");
        const want_array = want[0] == 'a';
        if (t.len == 0 or t[0] != (if (want_array) @as(u8, '[') else @as(u8, '{'))) return null;

        // Strict first; a well-formed container is the common case.
        if (parseContainerAllowingRepeats(allocator, t)) |doc| {
            const kind_ok = if (want_array) doc.value == .array else doc.value == .object;
            if (kind_ok) {
                try docs.append(allocator, doc);
                return doc.value;
            }
            var d = doc;
            d.deinit();
            return null;
        }

        // The container string is itself MANGLED (live: pi's `edits` array with a
        // missing comma between two object values). Same big-file escaping class
        // as looseRepairToolCallJson — re-serialize tolerantly, then STRICT-parse
        // the result so a mis-repair that yields invalid JSON is discarded.
        const repaired = looseRepairContainer(allocator, t) orelse return null;
        defer allocator.free(repaired);
        var rdoc = parseContainerAllowingRepeats(allocator, repaired) orelse {
            log.debug("[tool-autocorrect] {s} param left as a string: not well-formed JSON of that kind\n", .{want});
            return null;
        };
        const rkind_ok = if (want_array) rdoc.value == .array else rdoc.value == .object;
        if (!rkind_ok) {
            rdoc.deinit();
            return null;
        }
        try docs.append(allocator, rdoc); // ownership transfers; never deinit here
        return rdoc.value;
    }
    if (std.mem.eql(u8, want, "boolean")) {
        return switch (v) {
            .bool => null,
            .string => |s| if (looseBoolSpelling(s)) |b| std.json.Value{ .bool = b } else null,
            .integer => |i| if (i == 0 or i == 1) std.json.Value{ .bool = i == 1 } else null,
            else => null,
        };
    }
    if (std.mem.eql(u8, want, "integer")) {
        return switch (v) {
            .string => |s| blk: {
                const n = std.fmt.parseInt(i64, std.mem.trim(u8, s, " \t\r\n"), 10) catch break :blk null;
                break :blk std.json.Value{ .integer = n };
            },
            else => null,
        };
    }
    if (std.mem.eql(u8, want, "number")) {
        return switch (v) {
            .string => |s| blk: {
                const t = std.mem.trim(u8, s, " \t\r\n");
                if (std.fmt.parseInt(i64, t, 10)) |n| break :blk std.json.Value{ .integer = n } else |_| {}
                const f = std.fmt.parseFloat(f64, t) catch break :blk null;
                if (std.math.isNan(f) or std.math.isInf(f)) break :blk null; // not representable in JSON
                break :blk std.json.Value{ .float = f };
            },
            else => null,
        };
    }
    if (std.mem.eql(u8, want, "string")) {
        // The inverse repair: the tag/custom formats infer type from the value's
        // SPELLING (isJsonLiteral), so a string param whose content happens to
        // read as `false`/`42` was shipped as a bool/number. The schema is the
        // only thing that can tell those apart.
        return switch (v) {
            .bool => |b| std.json.Value{ .string = if (b) "true" else "false" },
            .integer => |i| blk: {
                const s = try std.fmt.allocPrint(allocator, "{d}", .{i});
                try owned.append(allocator, s);
                break :blk std.json.Value{ .string = s };
            },
            .float => |f| blk: {
                const s = try std.fmt.allocPrint(allocator, "{d}", .{f});
                try owned.append(allocator, s);
                break :blk std.json.Value{ .string = s };
            },
            .number_string => |s| std.json.Value{ .string = s },
            else => null,
        };
    }
    return null; // a type we don't model — never guessed
}

fn jsonTypeMatches(v: std.json.Value, want: []const u8) bool {
    if (v == .null) return true; // an explicit null satisfies any optional field
    if (std.mem.eql(u8, want, "boolean")) return v == .bool;
    if (std.mem.eql(u8, want, "integer")) return v == .integer or v == .number_string;
    if (std.mem.eql(u8, want, "number")) return v == .integer or v == .float or v == .number_string;
    if (std.mem.eql(u8, want, "string")) return v == .string;
    if (std.mem.eql(u8, want, "array")) return v == .array;
    if (std.mem.eql(u8, want, "object")) return v == .object;
    return true; // a type we don't model constrains nothing
}

/// True when every argument whose type the tool DECLARES actually carries that
/// JSON type. This is the invariant strict clients (Claude Code, pi, opencode)
/// enforce and reject on; the format corpus asserts it universally so a new
/// entry in any family is covered without a bespoke assertion. Undeclared keys
/// and unmodelled types constrain nothing.
pub fn toolCallConformsToSchema(
    allocator: std.mem.Allocator,
    call: ParsedToolCall,
    tools_json: []const u8,
) bool {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return true;
    defer tools.deinit();
    const props = toolPropertiesFor(tools.value, call.name) orelse return true;

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch return true;
    defer parsed.deinit();
    if (parsed.value != .object) return true;

    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        const prop = props.get(entry.key_ptr.*) orelse continue;
        const want = declaredJsonType(prop) orelse continue;
        if (!jsonTypeMatches(entry.value_ptr.*, want)) return false;
    }
    return true;
}

/// True when a REQUIRED scalar param is MISSING at the top level while sitting
/// buried inside one of the tool's declared container args — exactly the shape
/// `hoistMisplacedRequiredParams` exists to repair, and exactly what a strict
/// client rejects with "must have required properties X". The format corpus
/// asserts this is FALSE for every entry that declares tools, so a future family
/// that produces the buried shape is covered without a bespoke assertion.
pub fn requiredParamIsBuried(
    allocator: std.mem.Allocator,
    call: ParsedToolCall,
    tools_json: []const u8,
) bool {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return false;
    defer tools.deinit();
    const params = toolParametersFor(tools.value, call.name) orelse return false;
    const props_v = params.get("properties") orelse return false;
    const required_v = params.get("required") orelse return false;
    if (props_v != .object or required_v != .array) return false;
    const props = props_v.object;

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;

    for (required_v.array.items) |req_v| {
        if (req_v != .string) continue;
        const name = req_v.string;
        if (parsed.value.object.get(name) != null) continue;
        const want = declaredJsonType(props.get(name) orelse continue) orelse continue;
        if (!isScalarJsonType(want)) continue;

        var it = parsed.value.object.iterator();
        while (it.next()) |e| {
            const prop = props.get(e.key_ptr.*) orelse continue;
            if (containerItemDeclares(prop, name)) continue;
            if (buriedParamIn(e.value_ptr, name, want, false) != null) return true;
        }
    }
    return false;
}

/// Coerce every parsed call's `arguments` to the types its tool DECLARES.
///
/// Class: **type inference from value spelling.** Neither the tag formats
/// (`<parameter=x>False</parameter>`, Gemma's `key:false`) nor a model writing
/// JSON by hand carries type information a parser can trust — `isJsonLiteral`
/// has to guess from the bytes, and it guesses wrong in both directions:
///   • `False` / `"false"` on a boolean param → shipped as the STRING "false";
///   • `false` / `42` as a string param's CONTENT → shipped as a bool/number.
/// Strict clients (Claude Code, pi, opencode) reject both and the model, which
/// cannot see its own serialized request, loops forever "fixing" a value that
/// was already correct. The tool schema is the only disambiguator, and it is
/// already threaded to every parse site as `tools_json`.
///
/// Contract: only where the schema declares a type, and only when the spelling
/// is unambiguous. An undecidable value is left alone so the client's
/// validation error stays honest. Correct calls from strong models are
/// byte-unchanged (nothing re-serializes unless a coercion actually fired).
///
/// Scalars AND containers: a tag format types nothing, so a declared
/// `array`/`object` arrives as a string of JSON and `coerceValueToType` parses
/// it back (strict, then one tolerant repair re-validated strictly). This
/// comment used to say "only SCALARS are touched" — it was two weeks stale by
/// the time it was read as a spec and filed as issue #94. A contract comment is
/// load-bearing documentation: pin it with a test (`issue #94: a container
/// param carried as literal TEXT…`), don't let it drift.
pub fn coerceToolArgsToSchema(
    allocator: std.mem.Allocator,
    calls: []ParsedToolCall,
    tools_json: []const u8,
) !void {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return;
    defer tools.deinit();
    if (tools.value != .array) return;

    for (calls) |*call| {
        const props = toolPropertiesFor(tools.value, call.name) orelse continue;

        var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch continue;
        defer parsed.deinit();
        if (parsed.value != .object) continue;

        var owned = std.ArrayList([]const u8).empty;
        defer {
            for (owned.items) |s| allocator.free(s);
            owned.deinit(allocator);
        }
        // Parsed container sub-documents: the rewritten object BORROWS from these,
        // so they must outlive the Stringify below.
        var docs = std.ArrayList(std.json.Parsed(std.json.Value)).empty;
        defer {
            for (docs.items) |*d| d.deinit();
            docs.deinit(allocator);
        }

        var changed = false;
        var it = parsed.value.object.iterator();
        while (it.next()) |entry| {
            const prop = props.get(entry.key_ptr.*) orelse continue;
            const want = declaredJsonType(prop) orelse continue;
            const new_val = try coerceValueToType(allocator, entry.value_ptr.*, want, &owned, &docs) orelse continue;
            entry.value_ptr.* = new_val;
            changed = true;
        }
        if (!changed) continue;

        const rewritten = std.json.Stringify.valueAlloc(allocator, parsed.value, .{}) catch continue;
        log.info("  [tool-parse] coerced {s} arguments to the declared schema types\n", .{call.name});
        allocator.free(call.arguments);
        call.arguments = rewritten;
    }
}

/// Does this container's ITEM schema declare `name` as one of its own
/// properties? If it does, a `name` sitting inside it BELONGS there and must
/// never be lifted out — a multi-file edit tool would lose its per-item paths.
/// This is the whole safety argument for the hoist below, so it is deliberately
/// conservative: no declared item `properties` map ⇒ nothing is proven ⇒ false.
fn containerItemDeclares(prop: std.json.Value, name: []const u8) bool {
    if (prop != .object) return false;
    // Array property → the item schema. Object property → the schema itself.
    const inner = if (prop.object.get("items")) |items| items else prop;
    if (inner != .object) return false;
    const props = inner.object.get("properties") orelse return false;
    if (props != .object) return false;
    return props.object.get(name) != null;
}

/// Scalar equality — the only shapes we ever hoist on. A container value is
/// never a candidate, so a structural compare is not needed.
fn jsonScalarEql(a: std.json.Value, b: std.json.Value) bool {
    return switch (a) {
        .string => |s| b == .string and std.mem.eql(u8, s, b.string),
        .integer => |i| b == .integer and i == b.integer,
        .bool => |x| b == .bool and x == b.bool,
        else => false,
    };
}

/// Find `name` buried one level down inside `container`, requiring UNANIMITY:
/// every object in the container carries `name`, all with the same scalar value,
/// and that value matches the type the schema declares for the top-level param.
/// Anything less is ambiguous → null. With `remove`, also deletes it from each
/// object (the returned value keeps pointing into the parse arena, which
/// outlives the removal).
fn buriedParamIn(
    container: *std.json.Value,
    name: []const u8,
    want: []const u8,
    remove: bool,
) ?std.json.Value {
    switch (container.*) {
        .array => |*arr| {
            if (arr.items.len == 0) return null;
            var found: ?std.json.Value = null;
            for (arr.items) |*item| {
                if (item.* != .object) return null;
                const v = item.object.get(name) orelse return null; // must be in EVERY item
                if (v == .null or !jsonTypeMatches(v, want)) return null;
                if (found) |f| {
                    if (!jsonScalarEql(f, v)) return null; // items disagree → ambiguous
                } else found = v;
            }
            if (remove) for (arr.items) |*item| {
                _ = item.object.orderedRemove(name);
            };
            return found;
        },
        .object => |*obj| {
            const v = obj.get(name) orelse return null;
            if (v == .null or !jsonTypeMatches(v, want)) return null;
            if (remove) _ = obj.orderedRemove(name);
            return v;
        },
        else => return null,
    }
}

/// Lift a REQUIRED top-level parameter the model BURIED inside a container arg.
///
/// Class: **misplaced required param.** A weak model that has internalized "the
/// edit object holds everything about the edit" writes the file path INSIDE each
/// `edits[]` item and never emits it at the top level:
///
///     {"edits":[{"newText":"…","oldText":"…","path":"us_presidents/generate_site.sh"}]}
///
/// The call is valid JSON and correctly escaped, so no repair path fires and no
/// type is wrong — it is purely in the wrong PLACE, which only the schema can
/// know. Strict clients reject it ("path: must have required properties path"),
/// and the model, which cannot see its own serialized request, re-emits the same
/// call until it gives up and falls back to rewriting the whole file with
/// `write` — three dead multi-thousand-token rounds in the live pi
/// session (gemma-4-26B-A4B). Same shape as the Claude Code replace_all loop.
///
/// Contract — every condition is read off the SCHEMA, never guessed:
///   • the param is declared REQUIRED at the top level and is ABSENT there;
///   • it is declared a SCALAR (hoisting a container is too speculative);
///   • it sits inside a DECLARED container arg whose item schema does NOT
///     declare it (`containerItemDeclares` — otherwise it belongs where it is);
///   • every object in that container carries it with the SAME value and the
///     declared type.
/// Ambiguity of any kind (items disagreeing, two containers offering different
/// values, a wrong-typed value) leaves the call untouched so the client's
/// validation error stays honest. A compliant call never re-serializes.
pub fn hoistMisplacedRequiredParams(
    allocator: std.mem.Allocator,
    calls: []ParsedToolCall,
    tools_json: []const u8,
) !void {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return;
    defer tools.deinit();
    if (tools.value != .array) return;

    for (calls) |*call| {
        const params = toolParametersFor(tools.value, call.name) orelse continue;
        const props_v = params.get("properties") orelse continue;
        const required_v = params.get("required") orelse continue;
        if (props_v != .object or required_v != .array) continue;
        const props = props_v.object;

        var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch continue;
        defer parsed.deinit();
        if (parsed.value != .object) continue;

        var changed = false;
        for (required_v.array.items) |req_v| {
            if (req_v != .string) continue;
            const name = req_v.string; // borrows from `tools`, which outlives the Stringify
            if (parsed.value.object.get(name) != null) continue; // present — nothing misplaced

            const want = declaredJsonType(props.get(name) orelse continue) orelse continue;
            if (!isScalarJsonType(want)) continue;

            // Probe every declared container WITHOUT mutating: a value found in
            // two containers that disagree is ambiguous, and we must not have
            // already gutted the first one by then.
            var found: ?std.json.Value = null;
            var ambiguous = false;
            var it = parsed.value.object.iterator();
            while (it.next()) |e| {
                const prop = props.get(e.key_ptr.*) orelse continue;
                if (containerItemDeclares(prop, name)) continue;
                const v = buriedParamIn(e.value_ptr, name, want, false) orelse continue;
                if (found) |f| {
                    if (!jsonScalarEql(f, v)) {
                        ambiguous = true;
                        break;
                    }
                } else found = v;
            }
            if (ambiguous or found == null) continue;

            // Commit: strip it from the containers, install it at the top level.
            it = parsed.value.object.iterator();
            while (it.next()) |e| {
                const prop = props.get(e.key_ptr.*) orelse continue;
                if (containerItemDeclares(prop, name)) continue;
                _ = buriedParamIn(e.value_ptr, name, want, true);
            }
            try parsed.value.object.put(parsed.arena.allocator(), name, found.?);
            log.info("  [tool-parse] hoisted misplaced required arg '{s}' to the top level for {s}\n", .{ name, call.name });
            changed = true;
        }
        if (!changed) continue;

        const rewritten = std.json.Stringify.valueAlloc(allocator, parsed.value, .{}) catch continue;
        allocator.free(call.arguments);
        call.arguments = rewritten;
    }
}

/// Does the OpenAI-shaped tools array declare a function with this name?
/// Accepts both the wrapped ({"type":"function","function":{…}}) and flat
/// forms, mirroring `toolPropertiesFor`. An unparseable/odd `tools_json`
/// returns TRUE — never drop a call because OUR schema parse failed.
pub fn toolNameIsDeclared(allocator: std.mem.Allocator, tools_json: []const u8, name: []const u8) bool {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return true;
    defer parsed.deinit();
    if (parsed.value != .array) return true;
    for (parsed.value.array.items) |tool_val| {
        if (tool_val != .object) continue;
        const func = blk: {
            if (tool_val.object.get("function")) |f| {
                if (f == .object) break :blk f.object;
            }
            break :blk tool_val.object;
        };
        const nv = func.get("name") orelse continue;
        if (nv == .string and std.mem.eql(u8, nv.string, name)) return true;
    }
    return false;
}

/// Drop HEURISTICALLY-inferred raw-JSON calls (`ParsedToolCall.inferred`)
/// whose name the request never declared. The George Washington class: a
/// generation truncated mid-data-script leaves `{"name": "George Washington",
/// "num": 1, …}` as the first balanced object, and the flat-shape synthesis
/// promoted it to a tool call the client can only answer with "tool not
/// found" — burning the turn. Explicit tag-format calls are NEVER dropped
/// here (undeclared name → honest client feedback the model corrects from).
///
/// Returns the retained slice (shrunk in place), or null when every call was
/// dropped — the caller treats that exactly like "no tool calls", so the text
/// stays visible content and finish_reason (e.g. "length") is preserved.
pub fn filterInferredBySchema(
    allocator: std.mem.Allocator,
    calls: []ParsedToolCall,
    tools_json: []const u8,
) !?[]ParsedToolCall {
    var kept: usize = 0;
    for (calls) |tc| {
        if (!tc.inferred or toolNameIsDeclared(allocator, tools_json, tc.name)) {
            calls[kept] = tc;
            kept += 1;
        } else {
            log.warn("  [tool-parse] dropped inferred raw-JSON call '{s}' — name not among the request's tools (data object, not a call)\n", .{tc.name});
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    if (kept == calls.len) return calls;
    if (kept == 0) {
        allocator.free(calls);
        return null;
    }
    return try allocator.realloc(calls, kept);
}

// ── tests ──

test "filterInferredBySchema drops undeclared inferred calls, keeps declared and explicit ones" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    ;
    // Inferred + undeclared → dropped; slice becomes null.
    {
        var calls = try allocator.alloc(ParsedToolCall, 1);
        calls[0] = .{
            .name = try allocator.dupe(u8, "George Washington"),
            .arguments = try allocator.dupe(u8, "{\"num\":1}"),
            .inferred = true,
        };
        try testing.expect((try filterInferredBySchema(allocator, calls, tools)) == null);
    }
    // Inferred + declared → kept.
    {
        var calls = try allocator.alloc(ParsedToolCall, 1);
        calls[0] = .{
            .name = try allocator.dupe(u8, "write"),
            .arguments = try allocator.dupe(u8, "{\"path\":\"a\"}"),
            .inferred = true,
        };
        const kept = (try filterInferredBySchema(allocator, calls, tools)).?;
        defer {
            for (kept) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(kept);
        }
        try testing.expectEqual(@as(usize, 1), kept.len);
    }
    // Explicit + undeclared → kept (client feedback, not our guess); the
    // inferred sibling with an undeclared name is dropped and the slice shrinks.
    {
        var calls = try allocator.alloc(ParsedToolCall, 2);
        calls[0] = .{
            .name = try allocator.dupe(u8, "searchWeb"),
            .arguments = try allocator.dupe(u8, "{\"q\":\"zig\"}"),
            .inferred = false,
        };
        calls[1] = .{
            .name = try allocator.dupe(u8, "John Adams"),
            .arguments = try allocator.dupe(u8, "{\"num\":2}"),
            .inferred = true,
        };
        const kept = (try filterInferredBySchema(allocator, calls, tools)).?;
        defer {
            for (kept) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(kept);
        }
        try testing.expectEqual(@as(usize, 1), kept.len);
        try testing.expectEqualStrings("searchWeb", kept[0].name);
    }
}

test "toolNameIsDeclared: wrapped + flat forms; unparseable schema never drops" {
    const allocator = testing.allocator;
    const wrapped =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{}}}}]
    ;
    const flat =
        \\[{"name":"write","parameters":{"type":"object","properties":{}}}]
    ;
    try testing.expect(toolNameIsDeclared(allocator, wrapped, "write"));
    try testing.expect(toolNameIsDeclared(allocator, flat, "write"));
    try testing.expect(!toolNameIsDeclared(allocator, wrapped, "George Washington"));
    // Broken schema JSON → safe default TRUE (never drop on OUR parse failure).
    try testing.expect(toolNameIsDeclared(allocator, "not json", "anything"));
}

test "issue #94: a container param carried as literal TEXT coerces to a real array/object" {
    // #94 reports this as an open gap on the strength of the doc comment
    // ("only SCALARS are touched"), which was stale — `coerceValueToType` has
    // had its array/object arm since 60ba5ec, two weeks before
    // the issue was filed. Pinned as a TEST rather than left as a code read,
    // so the claim is settled by evidence and cannot silently regress.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"stats","parameters":{"type":"object","properties":{"nums":{"type":"array","items":{"type":"integer"}},"opts":{"type":"object"},"label":{"type":"string"}}}}}]
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "stats"),
        .arguments = try allocator.dupe(u8, "{\"nums\":\"[1, 2, 3]\",\"opts\":\"{\\\"grid\\\": true}\",\"label\":\"[not an array]\"}"),
    }};
    defer {
        allocator.free(calls[0].name);
        allocator.free(calls[0].arguments);
    }

    try coerceToolArgsToSchema(allocator, &calls, tools);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const o = parsed.value.object;

    const nums = o.get("nums").?.array;
    try testing.expectEqual(@as(usize, 3), nums.items.len);
    try testing.expectEqual(@as(i64, 2), nums.items[1].integer);
    try testing.expectEqual(true, o.get("opts").?.object.get("grid").?.bool);
    // A STRING param whose content merely looks bracket-y is untouched — the
    // schema decides, never the spelling.
    try testing.expectEqualStrings("[not an array]", o.get("label").?.string);
}

/// The Edit tool as Claude Code declares it (OpenAI shape, post
/// `buildOpenAIToolsJson`): `replace_all` is a BOOLEAN, the rest are strings.
const edit_tools_json_test =
    \\[{"type":"function","function":{"name":"Edit","description":"Edit a file","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},"replace_all":{"type":"boolean","default":false}},"required":["file_path","old_string","new_string"]}}}]
;

fn parseArgsObj(allocator: std.mem.Allocator, args: []const u8) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, allocator, args, .{});
}

test "coerceToolArgsToSchema: Python-style False on a boolean param becomes JSON false" {
    // VERBATIM capture, Qwen3.6-35B-A3B via Claude Code on
    // ~/.mlx-serve/logs/mlx-serve-11234.log:109471. The model emits the Hermes
    // XML parameter form with Python's `False`; parseHermesToolCall's
    // isJsonLiteral only knows lowercase, so it shipped the STRING "False" and
    // Claude Code rejected every Edit with
    //   "The parameter `replace_all` type is expected as `boolean` but provided as `string`"
    // — six times in a row, until the model gave up and rewrote the whole file.
    const allocator = testing.allocator;
    const raw =
        "\n<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n" ++
        "<parameter=file_path>\n/Users/david/doom/index.html\n</parameter>\n" ++
        "<parameter=old_string>\n  <script src=\"game.js\"></script>\n</parameter>\n" ++
        "<parameter=new_string>\n  <script src=\"game.js\" type=\"module\"></script>\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const v = parsed.value.object.get("replace_all").?;
    try testing.expect(v == .bool);
    try testing.expectEqual(false, v.bool);
    // Same capture, the whitespace half of the class: `old_string` must match
    // the file "exactly, including indentation", so the two leading spaces the
    // model sent are payload. Trimming them either failed the edit or matched a
    // DIFFERENT, un-indented occurrence mid-line and rewrote it at the wrong
    // nesting; coercion leaves a declared string alone, so the parser is the
    // only layer that can preserve them.
    try testing.expectEqualStrings(
        "  <script src=\"game.js\"></script>",
        parsed.value.object.get("old_string").?.string,
    );
    try testing.expectEqualStrings(
        "  <script src=\"game.js\" type=\"module\"></script>",
        parsed.value.object.get("new_string").?.string,
    );
}

test "coerceToolArgsToSchema: a STRING param spelled `false` stays a string" {
    // The inverse half of the same class: isJsonLiteral guesses the type from
    // the value's SPELLING, so a code edit whose old_string is literally the
    // token `false` was shipped as JSON `false` (boolean) — "expected string,
    // provided boolean". The schema is the only thing that can disambiguate.
    const allocator = testing.allocator;
    const raw = "<tool_call>\n<function=Edit>\n<parameter=file_path>\na.js\n</parameter>\n" ++
        "<parameter=old_string>\nfalse\n</parameter>\n<parameter=new_string>\n42\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const old = parsed.value.object.get("old_string").?;
    try testing.expect(old == .string);
    try testing.expectEqualStrings("false", old.string);
    const new = parsed.value.object.get("new_string").?;
    try testing.expect(new == .string);
    try testing.expectEqualStrings("42", new.string);
}

test "coerceToolArgsToSchema: quoted boolean in a JSON body is coerced too" {
    // Same class, different producer: the model emits well-formed JSON but
    // quotes the boolean. Strict parse succeeds, so no repair path ever runs.
    const allocator = testing.allocator;
    const raw = "<tool_call>{\"name\":\"Edit\",\"arguments\":{\"file_path\":\"a.js\"," ++
        "\"old_string\":\"a\",\"new_string\":\"b\",\"replace_all\":\"true\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const v = parsed.value.object.get("replace_all").?;
    try testing.expect(v == .bool);
    try testing.expectEqual(true, v.bool);
}

test "coerceToolArgsToSchema: unknown tool / unknown key / undecidable value pass through" {
    const allocator = testing.allocator;
    // Unknown tool name → args untouched.
    {
        const raw = "<tool_call>{\"name\":\"Nope\",\"arguments\":{\"replace_all\":\"False\"}}</tool_call>";
        const calls = (try parseToolCalls(allocator, raw)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
        const parsed = try parseArgsObj(allocator, calls[0].arguments);
        defer parsed.deinit();
        try testing.expect(parsed.value.object.get("replace_all").? == .string);
    }
    // Known tool, boolean param, value that is not a recognizable boolean →
    // left alone so the client's validation error stays honest.
    {
        const raw = "<tool_call>{\"name\":\"Edit\",\"arguments\":{\"replace_all\":\"maybe\"}}</tool_call>";
        const calls = (try parseToolCalls(allocator, raw)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
        const parsed = try parseArgsObj(allocator, calls[0].arguments);
        defer parsed.deinit();
        const v = parsed.value.object.get("replace_all").?;
        try testing.expect(v == .string);
        try testing.expectEqualStrings("maybe", v.string);
    }
}

/// pi's `edit` tool, verbatim from its own schema
/// (@earendil-works/pi-coding-agent dist/core/tools/edit.js — typebox
/// `editSchema`). Two facts the tests below lean on: `path` is REQUIRED at the
/// TOP level, and the array's item schema declares ONLY oldText/newText — it
/// never declares `path`. That asymmetry is what makes a buried `path`
/// provably misplaced rather than a judgment call.
const pi_edit_tools_json_test =
    \\[{"type":"function","function":{"name":"edit","description":"Edit a file","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path to the file to edit (relative or absolute)"},"edits":{"type":"array","items":{"type":"object","properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},"required":["oldText","newText"]},"description":"One or more targeted replacements."}},"required":["path","edits"]}}}]
;

const edit_array_tools_json_test =
    \\[{"type":"function","function":{"name":"edit","description":"Edit a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array","items":{"type":"object"}},"dry_run":{"type":"boolean"}},"required":["path","edits"]}}}]
;

test "coerceToolArgsToSchema: an ARRAY param passed through Hermes XML is not left a string" {
    // Live capture class (15 occurrences in one session): `<parameter=edits>` holds
    // a JSON array, but parseHermesToolCall's isJsonLiteral only recognizes
    // scalars, so the whole array shipped as a STRING. Strict clients reject it
    // ("edits: want array, got str"); lenient ones silently mis-execute.
    const allocator = testing.allocator;
    const raw = "<tool_call>\n<function=edit>\n<parameter=path>\n/tmp/a.js\n</parameter>\n" ++
        "<parameter=edits>\n[{\"oldText\": \"const a = 1;\", \"newText\": \"const a = 2;\"}]\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_array_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits == .array);
    try testing.expectEqual(@as(usize, 1), edits.array.items.len);
    try testing.expectEqualStrings("const a = 2;", edits.array.items[0].object.get("newText").?.string);
    // sibling scalars still coerce/pass through
    try testing.expectEqualStrings("/tmp/a.js", parsed.value.object.get("path").?.string);
}

test "coerceToolArgsToSchema: identical duplicate keys inside a container param still coerce; conflicting ones do not" {
    // Issue #402: Qwen Code's ask_user_question array arrived as a string whose
    // items repeated `header` with the same value; strict parse rejects the
    // duplicate, the tolerant re-emit keeps it, and the string reached the
    // client unchanged. Identical duplicates carry no ambiguity; conflicting
    // ones do, and guessing a value is worse than leaving the string alone.
    const allocator = testing.allocator;
    const cases = [_]struct { inner: []const u8, want_array: bool }{
        .{ .inner = "[{\"header\":\"Scope\",\"question\":\"Which scope?\",\"header\":\"Scope\",\"options\":[{\"label\":\"Small\"}]}]", .want_array = true },
        .{ .inner = "[{\"header\":\"Scope\",\"header\":\"Size\"}]", .want_array = false },
    };
    for (cases) |c| {
        const raw = try std.fmt.allocPrint(allocator, "<tool_call>\n<function=edit>\n<parameter=edits>\n{s}\n</parameter>\n</function>\n</tool_call>", .{c.inner});
        defer allocator.free(raw);
        const calls = (try parseToolCalls(allocator, raw)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try coerceToolArgsToSchema(allocator, calls, edit_array_tools_json_test);
        const parsed = try parseArgsObj(allocator, calls[0].arguments);
        defer parsed.deinit();
        const edits = parsed.value.object.get("edits").?;
        if (c.want_array) {
            try testing.expect(edits == .array);
            try testing.expectEqualStrings("Scope", edits.array.items[0].object.get("header").?.string);
            try testing.expectEqual(@as(usize, 3), edits.array.items[0].object.count());
        } else {
            try testing.expect(edits == .string);
        }
    }
}

test "coerceToolArgsToSchema: a STRING param whose content is JSON stays a string" {
    // The inverse guard for arrays/objects: `old_string` may legitimately BE the
    // text `[1,2]`. Only the schema decides.
    const allocator = testing.allocator;
    const raw = "<tool_call>\n<function=Edit>\n<parameter=file_path>\na.js\n</parameter>\n" ++
        "<parameter=old_string>\n[1,2]\n</parameter>\n<parameter=new_string>\n{\"k\":1}\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    try testing.expect(parsed.value.object.get("old_string").? == .string);
    try testing.expectEqualStrings("[1,2]", parsed.value.object.get("old_string").?.string);
    try testing.expect(parsed.value.object.get("new_string").? == .string);
}

test "coerceToolArgsToSchema: already-conforming args are byte-identical (no-op guard)" {
    // The auto-correct layer must never touch a good call. Property: when every
    // declared type already matches, `arguments` comes out byte-for-byte the same.
    const allocator = testing.allocator;
    const raw = "<tool_call>{\"name\":\"Edit\",\"arguments\":{\"file_path\":\"a.js\",\"old_string\":\"a\"," ++
        "\"new_string\":\"b\",\"replace_all\":true}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
    try testing.expectEqualStrings(before, calls[0].arguments);
}

// ── Auto-correct safety properties (deterministic fuzz) ─────────────────────
// The repair layer is only ever allowed to touch input that is actually broken.
// These generate well-formed tool calls whose values deliberately SPELL other
// JSON types ("false", "42", "[1,2]", "null") — exactly the strings the
// value-spelling inference used to mis-type — and assert round-trip identity.

fn jsonValueEql(a: std.json.Value, b: std.json.Value) bool {
    if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
    return switch (a) {
        .null => true,
        .bool => a.bool == b.bool,
        .integer => a.integer == b.integer,
        .float => a.float == b.float,
        .number_string => std.mem.eql(u8, a.number_string, b.number_string),
        .string => std.mem.eql(u8, a.string, b.string),
        .array => blk: {
            if (a.array.items.len != b.array.items.len) break :blk false;
            for (a.array.items, b.array.items) |x, y| if (!jsonValueEql(x, y)) break :blk false;
            break :blk true;
        },
        .object => blk: {
            if (a.object.count() != b.object.count()) break :blk false;
            var it = a.object.iterator();
            while (it.next()) |e| {
                const other = b.object.get(e.key_ptr.*) orelse break :blk false;
                if (!jsonValueEql(e.value_ptr.*, other)) break :blk false;
            }
            break :blk true;
        },
    };
}

/// Value strings chosen to collide with every JSON literal spelling the
/// tag-format parsers used to guess from.
const adversarial_strings = [_][]const u8{
    "false",      "true",         "False",   "True",    "null",   "None",
    "42",         "-7",           "3.14",    "0",       "1",      "",
    "[1,2]",      "{\"k\":1}",    "nan",     "inf",     "  ",     "yes",
    "a\"b",  "line\nline", "tab\there", "back\\slash",
    "café ☕",
    "0x1f",  "{not json",  "[unclosed", "1e400",       "00",   "+5",
    "-",
};

test "fuzz: a conforming tool call round-trips byte-identical through parse+coerce" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE_1234);
    const rand = prng.random();

    const types = [_][]const u8{ "string", "integer", "number", "boolean", "array", "object" };

    var iter: usize = 0;
    while (iter < 400) : (iter += 1) {
        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        // Build a random schema + a CONFORMING args object for it.
        var props: std.json.ObjectMap = .empty;
        var args: std.json.ObjectMap = .empty;
        const nprops = 1 + rand.uintLessThan(usize, 5);
        var p: usize = 0;
        while (p < nprops) : (p += 1) {
            const key = try std.fmt.allocPrint(arena, "p{d}", .{p});
            const want = types[rand.uintLessThan(usize, types.len)];

            var prop: std.json.ObjectMap = .empty;
            try prop.put(arena, "type", .{ .string = want });
            try props.put(arena, key, .{ .object = prop });

            const v: std.json.Value = if (std.mem.eql(u8, want, "string"))
                .{ .string = adversarial_strings[rand.uintLessThan(usize, adversarial_strings.len)] }
            else if (std.mem.eql(u8, want, "integer"))
                .{ .integer = @as(i64, @intCast(rand.uintLessThan(u32, 1000))) - 500 }
            else if (std.mem.eql(u8, want, "number"))
                .{ .float = 1.5 }
            else if (std.mem.eql(u8, want, "boolean"))
                .{ .bool = rand.boolean() }
            else if (std.mem.eql(u8, want, "array")) blk: {
                var arr = std.json.Array.init(arena);
                try arr.append(.{ .string = "x" });
                try arr.append(.{ .integer = 1 });
                break :blk .{ .array = arr };
            } else blk: {
                var o: std.json.ObjectMap = .empty;
                try o.put(arena, "k", .{ .string = "v" });
                break :blk .{ .object = o };
            };
            try args.put(arena, key, v);
        }

        var func: std.json.ObjectMap = .empty;
        try func.put(arena, "name", .{ .string = "t" });
        var params: std.json.ObjectMap = .empty;
        try params.put(arena, "type", .{ .string = "object" });
        try params.put(arena, "properties", .{ .object = props });
        try func.put(arena, "parameters", .{ .object = params });
        var tool: std.json.ObjectMap = .empty;
        try tool.put(arena, "type", .{ .string = "function" });
        try tool.put(arena, "function", .{ .object = func });
        var tools_arr = std.json.Array.init(arena);
        try tools_arr.append(.{ .object = tool });
        const tools_json = try std.json.Stringify.valueAlloc(arena, std.json.Value{ .array = tools_arr }, .{});

        // Serialize as a canonical Hermes JSON call — always VALID input.
        const args_json = try std.json.Stringify.valueAlloc(arena, std.json.Value{ .object = args }, .{});
        const raw = try std.fmt.allocPrint(arena, "<tool_call>{{\"name\":\"t\",\"arguments\":{s}}}</tool_call>", .{args_json});

        const calls = (try parseToolCalls(allocator, raw)) orelse {
            std.debug.print("\n[fuzz iter {d}] valid tool call did not parse\n  {s}\n", .{ iter, raw });
            return error.FuzzFailed;
        };
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }

        const before = try arena.dupe(u8, calls[0].arguments);
        try coerceToolArgsToSchema(allocator, calls, tools_json);

        // Property 1: a conforming call is left BYTE-identical.
        if (!std.mem.eql(u8, before, calls[0].arguments)) {
            std.debug.print("\n[fuzz iter {d}] auto-correct mutated a conforming call\n  before: {s}\n  after : {s}\n", .{ iter, before, calls[0].arguments });
            return error.FuzzFailed;
        }

        // Property 2: the values survive verbatim (no spelling-based retyping).
        var got = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer got.deinit();
        if (!jsonValueEql(std.json.Value{ .object = args }, got.value)) {
            std.debug.print("\n[fuzz iter {d}] value mismatch\n  want: {s}\n  got : {s}\n", .{ iter, args_json, calls[0].arguments });
            return error.FuzzFailed;
        }

        // Property 3: conformance holds, and coercion is idempotent.
        try testing.expect(toolCallConformsToSchema(allocator, calls[0], tools_json));
        try coerceToolArgsToSchema(allocator, calls, tools_json);
        try testing.expectEqualStrings(before, calls[0].arguments);
    }
}

test "coerceToolArgsToSchema: a MANGLED array-typed param is tolerantly repaired" {
    // Live soak capture (record 268): pi's `edits` array with a MISSING COMMA
    // between two object values — `"newText": "..." "path": "..."`. Strict parse
    // rejects it, so pre-fix `edits` stayed a STRING (schema contradiction). The
    // tolerant container repair recovers a real array.
    const allocator = testing.allocator;
    const raw = "<tool_call>\n<function=edit>\n<parameter=path>\n./game.js\n</parameter>\n" ++
        "<parameter=edits>\n[{\"oldText\": \"  speed: 8,\", \"newText\": \"  vy: 0,\" \"extra\": \"x\"}]\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, pi_edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits == .array);
    try testing.expectEqual(@as(usize, 1), edits.array.items.len);
    try testing.expectEqualStrings("  speed: 8,", edits.array.items[0].object.get("oldText").?.string);
}

// ── Misplaced required param (buried-`path` class) ──────────────────────────
//
// VERBATIM captured arguments pi session (gemma-4-26B-A4B-it-qat-4bit,
// ~/.mlx-serve/logs/mlx-serve-11234.log around the us_presidents run). The model
// put `path` INSIDE each edits[] item and emitted no top-level `path`, so pi
// answered
//     Validation failed for tool "edit":
//       - path: must have required properties path
// three times in a row. Each rejection was a full multi-thousand-token
// generation; the model then abandoned `edit` and rewrote the whole file with
// `write` — the same "give up on the cheap tool, re-emit the expensive one"
// degradation as the Claude Code replace_all loop.
//
// The parse layer is NOT at fault (verified: convertGemma4Object/Array are a
// structural walk that preserves the model's own nesting and key order, and
// coerceToolArgsToSchema only rewrites values in place) — the model genuinely
// emitted it this way. The schema is what makes it provably fixable: `path` is
// required at the TOP level and the item schema declares only oldText/newText.
//
// The script bodies are the real ones, excerpted for readability — their LENGTH
// is not load-bearing (the JSON was already valid and correctly escaped; that is
// exactly why no repair path fired). The nesting and the escaping are verbatim.
const pi_buried_path_args_captured =
    \\{"edits":[{"newText":"#!/bin/bash\n\n# Get the directory where the script is located\nSCRIPT_DIR=\"$( cd \"$( dirname \"${BASH_SOURCE[0]}\" )\" && pwd )\"\ncd \"$SCRIPT_DIR\"\n","oldText":"#!/bin/bash\n\n# Data for the presidents\n# Format: \"Name|Term|Bio\"\n","path":"us_presidents/generate_site.sh"}]}
;

test "hoistMisplacedRequiredParams: pi's buried `path` is lifted out of the edits items" {
    const allocator = testing.allocator;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, pi_buried_path_args_captured),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }

    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();

    // The required param now satisfies the schema at the top level…
    const path = parsed.value.object.get("path") orelse return error.PathNotHoisted;
    try testing.expect(path == .string);
    try testing.expectEqualStrings("us_presidents/generate_site.sh", path.string);

    // …and is gone from the item, whose schema never declared it.
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits == .array);
    try testing.expectEqual(@as(usize, 1), edits.array.items.len);
    const item = edits.array.items[0].object;
    try testing.expect(item.get("path") == null);

    // The payload the model worked for survives byte-exact, escaping intact.
    try testing.expectEqualStrings(
        "#!/bin/bash\n\n# Data for the presidents\n# Format: \"Name|Term|Bio\"\n",
        item.get("oldText").?.string,
    );
    try testing.expect(std.mem.startsWith(u8, item.get("newText").?.string, "#!/bin/bash\n"));
    try testing.expect(std.mem.indexOf(u8, item.get("newText").?.string, "${BASH_SOURCE[0]}") != null);

    // The whole call now conforms — which is the point: pi accepts it.
    try testing.expect(toolCallConformsToSchema(allocator, calls[0], pi_edit_tools_json_test));
}

test "hoistMisplacedRequiredParams: a buried `path` survives the Gemma 4 parse path end to end" {
    // The same class through the REAL parse chain (Gemma's call:name{...} form —
    // this session's model). The raw pre-parse bytes were not dumped (the server
    // was not at --log-level debug), so the tag wrapper here is reconstructed;
    // the ARGUMENT SHAPE it produces is the verbatim captured one asserted above,
    // which is what the hoist actually operates on.
    const allocator = testing.allocator;
    const raw = "<|tool_call>call:edit{edits:[{oldText:<|\"|>old line<|\"|>,newText:<|\"|>new line<|\"|>," ++
        "path:<|\"|>us_presidents/generate_site.sh<|\"|>}]}<tool_call|>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try hoistMisplacedRequiredParams(allocator, calls, pi_edit_tools_json_test);
    try coerceToolArgsToSchema(allocator, calls, pi_edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const path = parsed.value.object.get("path") orelse return error.PathNotHoisted;
    try testing.expectEqualStrings("us_presidents/generate_site.sh", path.string);
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits.array.items[0].object.get("path") == null);
    try testing.expectEqualStrings("old line", edits.array.items[0].object.get("oldText").?.string);
}

test "hoistMisplacedRequiredParams: a compliant call is byte-identical" {
    // The strong-model path: nothing may re-serialize when nothing was misplaced.
    const allocator = testing.allocator;
    const args =
        \\{"path":"a.js","edits":[{"oldText":"a","newText":"b"}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "hoistMisplacedRequiredParams: never touches a key the item schema DECLARES" {
    // Safety gate. A tool whose items legitimately carry their own `path` (a
    // multi-file edit) must never have it stripped out from under them — hoisting
    // there would DESTROY data, not repair it. The item schema declaring the key
    // is the proof that it belongs where it is.
    const allocator = testing.allocator;
    const multifile_schema =
        \\[{"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array","items":{"type":"object","properties":{"path":{"type":"string"},"oldText":{"type":"string"}},"required":["path","oldText"]}}},"required":["path","edits"]}}}]
    ;
    const args =
        \\{"edits":[{"path":"a.js","oldText":"a"},{"path":"b.js","oldText":"b"}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, multifile_schema);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "hoistMisplacedRequiredParams: items that DISAGREE on the value stay honest" {
    // Two items, two different paths, and the tool edits ONE file. There is no
    // correct hoist here — picking either would silently write the wrong file.
    // Leave it broken so the client's validation error reaches the model.
    const allocator = testing.allocator;
    const args =
        \\{"edits":[{"oldText":"a","newText":"b","path":"a.js"},{"oldText":"c","newText":"d","path":"b.js"}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "hoistMisplacedRequiredParams: a wrong-typed buried value is never hoisted" {
    // `path` is declared a string; an object spelled `path` inside the item is
    // not the missing param. Hoisting it would just move the schema violation.
    const allocator = testing.allocator;
    const args =
        \\{"edits":[{"oldText":"a","newText":"b","path":{"nested":"x"}}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "coerceToolArgsToSchema: quoted integer/number strings coerce to numeric" {
    // A model emitting `"limit":"5"` for an integer param → strict clients reject
    // "expected integer, got string". Coerce string→numeric per the schema.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"},"scale":{"type":"number"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"Read\",\"arguments\":{\"path\":\"a.txt\",\"limit\":\"5\",\"scale\":\"1.5\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, tools);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    try testing.expectEqual(@as(i64, 5), parsed.value.object.get("limit").?.integer);
    const scale = parsed.value.object.get("scale").?;
    try testing.expect(scale == .float or scale == .integer);
    // A string param is untouched.
    try testing.expectEqualStrings("a.txt", parsed.value.object.get("path").?.string);
}

test "coerceToolArgsToSchema: nullable union type [\"string\",\"null\"] coerces to the non-null member" {
    // Real schemas declare optional params as a type UNION. declaredJsonType
    // picks the first non-null member, so a boolean-spelled value under
    // ["boolean","null"] still coerces.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"E","parameters":{"type":"object","properties":{"flag":{"type":["boolean","null"]},"note":{"type":["string","null"]}}}}}]
    ;
    const raw = "<tool_call>\n<function=E>\n<parameter=flag>\nFalse\n</parameter>\n<parameter=note>\nhi\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, tools);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    try testing.expect(parsed.value.object.get("flag").? == .bool);
    try testing.expectEqual(false, parsed.value.object.get("flag").?.bool);
    try testing.expectEqualStrings("hi", parsed.value.object.get("note").?.string);
}

test "coerceToolArgsToSchema: an object param declared as a oneOf union coerces from its JSON text" {
    // fx's shell tool: `request` is a discriminated union with no top-level "type".
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"shell","parameters":{"type":"object","required":["request"],"properties":{"request":{"oneOf":[{"type":"object","required":["action","command"],"properties":{"action":{"const":"run"},"command":{"type":"string"}}},{"type":"object","required":["action"],"properties":{"action":{"const":"cancel"}}}]}}}}}]
    ;
    const raw = "<tool_call>\n<function=shell>\n<parameter=request>\n{\"action\":\"run\",\"command\":\"ls\"}\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, tools);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const req = parsed.value.object.get("request").?;
    try testing.expect(req == .object);
    try testing.expectEqualStrings("ls", req.object.get("command").?.string);
}

test "declaredJsonType: a oneOf/anyOf union declares a type only when its branches agree" {
    const allocator = testing.allocator;
    const cases = [_]struct { schema: []const u8, want: ?[]const u8 }{
        .{ .schema = "{\"oneOf\":[{\"type\":\"object\"},{\"type\":\"object\"}]}", .want = "object" },
        .{ .schema = "{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"null\"}]}", .want = "string" },
        .{ .schema = "{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"object\"}]}", .want = null },
        .{ .schema = "{\"oneOf\":[{\"type\":\"object\"},{\"const\":1}]}", .want = null },
    };
    for (cases) |c| {
        var doc = try std.json.parseFromSlice(std.json.Value, allocator, c.schema, .{});
        defer doc.deinit();
        const got = declaredJsonType(doc.value);
        if (c.want) |w| {
            try testing.expectEqualStrings(w, got.?);
        } else {
            try testing.expect(got == null);
        }
    }
}

test "coerceToolArgsToSchema: an explicit JSON null satisfies any typed param (not coerced away)" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"E","parameters":{"type":"object","properties":{"limit":{"type":"integer"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"E\",\"arguments\":{\"limit\":null}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, tools);
    // null is a valid absent-value marker for any type — left byte-identical.
    try testing.expectEqualStrings(before, calls[0].arguments);
}

test "coerceToolArgsToSchema: coercion is SHALLOW — values nested inside a param are untouched" {
    // Documented boundary: the layer coerces TOP-LEVEL params against the tool's
    // declared property types. It does NOT recurse into array/object element
    // properties (the tool schema's `items`/nested `properties` are the client's
    // to validate). A nested `"port":"8080"` stays a string — byte-identical.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"W","parameters":{"type":"object","properties":{"config":{"type":"object"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"W\",\"arguments\":{\"config\":{\"port\":\"8080\",\"tls\":\"false\"}}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, tools);
    try testing.expectEqualStrings(before, calls[0].arguments);
}

test "coerceToolArgsToSchema: an unmodelled declared type leaves the value alone" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"E","parameters":{"type":"object","properties":{"x":{"type":"weird"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"E\",\"arguments\":{\"x\":\"False\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, tools);
    try testing.expectEqualStrings(before, calls[0].arguments);
}

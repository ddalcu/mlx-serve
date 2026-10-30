//! The DFlash config contract: a drafter declares `block_size`, `mask_token_id` and `target_layer_ids`, at the root
//! or nested under `dflash_config`. Discovery and the engine's loader read it the same way.

const std = @import("std");

pub const Contract = struct {
    nested: ?std.json.ObjectMap,
    root: std.json.ObjectMap,

    pub fn get(self: Contract, key: []const u8) ?std.json.Value {
        if (self.nested) |n| {
            if (n.get(key)) |v| return v;
        }
        return self.root.get(key);
    }
};

pub fn object(root: std.json.ObjectMap) ?Contract {
    const nested: ?std.json.ObjectMap = blk: {
        if (root.get("dflash_config")) |dc| {
            if (dc == .object) break :blk dc.object;
        }
        break :blk null;
    };
    const c = Contract{ .nested = nested, .root = root };
    if (c.get("block_size") == null) return null;
    if (c.get("mask_token_id") == null) return null;
    if (c.get("target_layer_ids") == null) return null;
    return c;
}

pub fn isDflashConfigJson(root: std.json.ObjectMap) bool {
    return object(root) != null;
}

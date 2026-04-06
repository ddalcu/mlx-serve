#include "lexer.h"
#include "parser.h"
#include "runtime.h"
#include "value.h"
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <cstring>
#include <string>

using json = nlohmann::ordered_json;

static thread_local std::string g_last_error;

extern "C" {

char* jinja_render_chat(
    const char* template_str,
    const char* messages_json,
    const char* tools_json,
    const char* extra_json,
    int add_generation_prompt
) {
    g_last_error.clear();
    try {
        // Parse the template: lexer → parser → AST
        jinja::lexer lex;
        auto lex_result = lex.tokenize(template_str);
        auto prog = jinja::parse_from_tokens(lex_result);

        // Build context with all variables the template needs
        json context_data = json::object();
        context_data["messages"] = json::parse(messages_json);
        context_data["add_generation_prompt"] = (add_generation_prompt != 0);

        if (tools_json && tools_json[0] != '\0') {
            context_data["tools"] = json::parse(tools_json);
        }

        // Merge extra context (bos_token, eos_token, enable_thinking, etc.)
        if (extra_json && extra_json[0] != '\0') {
            auto extra = json::parse(extra_json);
            for (auto it = extra.begin(); it != extra.end(); ++it) {
                context_data[it.key()] = it.value();
            }
        }

        // Create jinja context and populate with data
        jinja::context ctx(template_str);
        jinja::global_from_json(ctx, context_data, /* mark_input = */ true);

        // Execute template
        jinja::runtime rt(ctx);
        auto results = rt.execute(prog);
        auto parts = jinja::runtime::gather_string_parts(results);
        std::string result = parts->as_string().str();

        char* out = (char*)std::malloc(result.size() + 1);
        if (!out) return nullptr;
        std::memcpy(out, result.c_str(), result.size() + 1);
        return out;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    } catch (...) {
        g_last_error = "unknown jinja render error";
        return nullptr;
    }
}

void jinja_str_free(char* s) {
    std::free(s);
}

const char* jinja_last_error(void) {
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

} // extern "C"

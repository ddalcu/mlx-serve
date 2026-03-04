#include "jinja.hpp"
#include <cstdlib>
#include <cstring>
#include <string>

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
        jinja::Template tpl(template_str);
        auto messages = jinja::json::parse(messages_json);
        auto tools = tools_json ? jinja::json::parse(tools_json) : jinja::json::array();
        auto extra = extra_json ? jinja::json::parse(extra_json) : jinja::json::object();

        std::string result = tpl.apply_chat_template(
            messages,
            add_generation_prompt != 0,
            tools,
            extra
        );

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

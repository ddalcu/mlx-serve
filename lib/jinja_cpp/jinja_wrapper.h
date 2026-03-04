#ifndef JINJA_WRAPPER_H
#define JINJA_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Render a Jinja2 chat template with the given messages and tools.
 *
 * template_str:  the Jinja2 template source
 * messages_json: JSON array of message objects [{"role":"...","content":"..."},...]
 * tools_json:    JSON array of tool objects, or NULL if no tools
 * extra_json:    JSON object with extra context (bos_token, eos_token, etc.), or NULL
 * add_generation_prompt: 1 to add, 0 to skip
 *
 * Returns a malloc'd string on success, NULL on failure.
 * The caller must free the result with jinja_str_free().
 * On failure, jinja_last_error() returns the error message.
 */
char* jinja_render_chat(
    const char* template_str,
    const char* messages_json,
    const char* tools_json,
    const char* extra_json,
    int add_generation_prompt
);

/* Free a string returned by jinja_render_chat. */
void jinja_str_free(char* s);

/* Returns the last error message, or NULL if no error. Thread-local. */
const char* jinja_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* JINJA_WRAPPER_H */

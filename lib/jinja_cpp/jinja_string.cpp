#include "jinja_string.h"
#include "value.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace jinja {

bool is_py_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

static constexpr uint32_t not_a_code_point = 0x110000;

// The code point starting at s[i] and its byte length; a malformed byte is one byte of not_a_code_point.
static uint32_t utf8_decode(std::string_view s, size_t i, size_t & len) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
    if (len == 0 || i + len > s.size()) { len = 1; return not_a_code_point; }
    if (len == 1) return c;
    uint32_t cp = c & (0x7F >> len);
    for (size_t k = 1; k < len; ++k) {
        const unsigned char cc = static_cast<unsigned char>(s[i + k]);
        if ((cc & 0xC0) != 0x80) { len = 1; return not_a_code_point; }
        cp = (cp << 6) | (cc & 0x3F);
    }
    return cp;
}

template <typename Pred>
static size_t utf8_prefix(std::string_view s, Pred pred) {
    size_t i = 0, len = 0;
    while (i < s.size() && pred(utf8_decode(s, i, len))) i += len;
    return i;
}

template <typename Pred>
static size_t utf8_suffix(std::string_view s, Pred pred) {
    size_t end = s.size();
    while (end > 0) {
        size_t start = end - 1;
        while (start > 0 && end - start < 4 && (static_cast<unsigned char>(s[start]) & 0xC0) == 0x80) --start;
        size_t len = 0;
        uint32_t cp = utf8_decode(s, start, len);
        if (start + len != end) { start = end - 1; cp = not_a_code_point; }
        if (!pred(cp)) break;
        end = start;
    }
    return s.size() - end;
}

size_t py_space_prefix(std::string_view s) { return utf8_prefix(s, is_py_space); }
size_t py_space_suffix(std::string_view s) { return utf8_suffix(s, is_py_space); }

//
// string_part
//

bool string_part::is_uppercase() const {
    for (char c : val) {
        if (std::islower(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

bool string_part::is_lowercase() const {
    for (char c : val) {
        if (std::isupper(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

//
// string
//

void string::mark_input() {
    for (auto & part : parts) {
        part.is_input = true;
    }
}

std::string string::str() const {
    if (parts.size() == 1) {
        return parts[0].val;
    }
    std::ostringstream oss;
    for (const auto & part : parts) {
        oss << part.val;
    }
    return oss.str();
}

size_t string::length() const {
    size_t len = 0;
    for (const auto & part : parts) {
        len += part.val.length();
    }
    return len;
}

void string::hash_update(hasher & hash) const noexcept {
    for (const auto & part : parts) {
        hash.update(part.val.data(), part.val.length());
    }
}

bool string::all_parts_are_input() const {
    for (const auto & part : parts) {
        if (!part.is_input) {
            return false;
        }
    }
    return true;
}

bool string::is_uppercase() const {
    for (const auto & part : parts) {
        if (!part.is_uppercase()) {
            return false;
        }
    }
    return true;
}

bool string::is_lowercase() const {
    for (const auto & part : parts) {
        if (!part.is_lowercase()) {
            return false;
        }
    }
    return true;
}

// mark this string as input if other has ALL parts as input
void string::mark_input_based_on(const string & other) {
    if (other.all_parts_are_input()) {
        for (auto & part : parts) {
            part.is_input = true;
        }
    }
}

string string::append(const string & other) {
    for (const auto & part : other.parts) {
        parts.push_back(part);
    }
    return *this;
}

// in-place transformation

using transform_fn = std::function<std::string(const std::string&)>;
static string apply_transform(string & self, const transform_fn & fn) {
    for (auto & part : self.parts) {
        part.val = fn(part.val);
    }
    return self;
}

string string::uppercase() {
    return apply_transform(*this, [](const std::string & s) {
        std::string res = s;
        std::transform(res.begin(), res.end(), res.begin(), ::toupper);
        return res;
    });
}
string string::lowercase() {
    return apply_transform(*this, [](const std::string & s) {
        std::string res = s;
        std::transform(res.begin(), res.end(), res.begin(), ::tolower);
        return res;
    });
}
string string::capitalize() {
    return apply_transform(*this, [](const std::string & s) {
        if (s.empty()) return s;
        std::string res = s;
        res[0] = ::toupper(static_cast<unsigned char>(res[0]));
        std::transform(res.begin() + 1, res.end(), res.begin() + 1, ::tolower);
        return res;
    });
}
string string::titlecase() {
    return apply_transform(*this, [](const std::string & s) {
        std::string res = s;
        bool capitalize_next = true;
        for (char &c : res) {
            if (isspace(static_cast<unsigned char>(c))) {
                capitalize_next = true;
            } else if (capitalize_next) {
                c = ::toupper(static_cast<unsigned char>(c));
                capitalize_next = false;
            } else {
                c = ::tolower(static_cast<unsigned char>(c));
            }
        }
        return res;
    });
}
string string::strip(bool left, bool right, std::optional<const std::string_view> chars) {
    // Python's str.strip: whitespace by str.isspace, `chars` as a set of code points.
    static auto strip_part = [](const std::string & s, bool left, bool right, std::optional<const std::string_view> chars) -> std::string {
        std::vector<uint32_t> set;
        if (chars) {
            for (size_t i = 0, len = 0; i < chars->size(); i += len) set.push_back(utf8_decode(*chars, i, len));
        }
        auto match = [&](uint32_t cp) -> bool {
            if (!chars) return is_py_space(cp);
            return cp != not_a_code_point && std::find(set.begin(), set.end(), cp) != set.end();
        };
        const size_t start = left ? utf8_prefix(s, match) : 0;
        const size_t end = start == s.size() ? start : s.size() - (right ? utf8_suffix(s, match) : 0);
        return s.substr(start, end - start);
    };
    if (parts.empty()) {
        return *this;
    }
    if (left) {
        for (size_t i = 0; i < parts.size(); ++i) {
            parts[i].val = strip_part(parts[i].val, true, false, chars);
            if (parts[i].val.empty()) {
                // remove empty part
                parts.erase(parts.begin() + i);
                --i;
                continue;
            } else {
                break;
            }
        }
    }
    if (right) {
        for (size_t i = parts.size(); i-- > 0;) {
            parts[i].val = strip_part(parts[i].val, false, true, chars);
            if (parts[i].val.empty()) {
                // remove empty part
                parts.erase(parts.begin() + i);
                continue;
            } else {
                break;
            }
        }
    }
    return *this;
}

} // namespace jinja

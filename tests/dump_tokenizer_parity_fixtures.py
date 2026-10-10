"""HF `tokenizers` reference ids for the env-gated "tokenizer parity" test in src/tokenizer.zig.

Each case is {name, text, ids}, ids from the Python `tokenizers` library (no special tokens) on the model's own
tokenizer.json: whitespace runs (blank lines holding spaces, CRLF, tabs), Unicode White_Space and zero-width
non-spaces, CJK and kana with full-width spaces, whitespace before digit and CJK runs, plus a seeded fuzz over the
same alphabet.

  python tests/dump_tokenizer_parity_fixtures.py <model_dir> tests/fixtures/tokenizer_parity/qwen3_8.json
  zig build test-build -Dtest-filter="tokenizer parity" && TOKENIZER_PARITY_MODEL=<model_dir> \\
    TOKENIZER_PARITY_CASES=$PWD/tests/fixtures/tokenizer_parity/qwen3_8.json ./zig-out/tests/test

qwen3_8.json: Qwen3.8's tokenizer.json (the 27B and Flash-Next packs ship the same file). deepseek_v4.json:
DeepSeek-V4's (Hy-MT2 ships the same pre_tokenizer), with --no-zero-width. tokenizers 0.23.2.
"""
import argparse
import json
import os
import random

from tokenizers import Tokenizer

# Unicode White_Space past ASCII, minus U+2000/U+2001 (NFC rewrites them and the Zig encoder has no NFC
# step), then zero-width non-spaces.
UNICODE_WS = ["\u0085", "\xa0", "\u1680", "\u2002", "\u2003", "\u2009", "\u200a", "\u2028", "\u2029", "\u202f",
              "\u205f", "\u3000"]
NOT_WS = ["\u200b", "\u180e", "\ufeff"]

HANDWRITTEN = [
    ("blank_line_with_spaces", "a\n    \nb"),
    ("blank_line_then_indent", "a\n  \n  b"),
    ("spaces_newlines_spaces", "a  \n\n  \n b"),
    ("tab_in_newline_run", "x\n\t\n\ny"),
    ("crlf_blank_line", "a\r\n   \r\nb"),
    ("python_indented_blank_line", "def f():\n    x = 1\n    \n    return x\n"),
    ("markdown_blank_lines", "- one\n  \n- two\n   \n\n## Title\n"),
    ("json_blank_line", '{\n  "a": 1,\n  \n  "b": [1, 2]\n}\n'),
    ("trailing_spaces_eof", "end   \n   "),
    ("only_whitespace", " \n \t \r\n  "),
    ("gutenberg_header", "The Project Gutenberg eBook of Pride and Prejudice\n    \nThis eBook is for the use"),
    ("ideographic_indent", "\u3000\u3000\u957f\u6c5f\u53d1\u6e90\u4e8e\u9752\u85cf\u9ad8\u539f\u3002"
                           "\u3000\u3000\u5b83\u6d41\u7ecf\u5341\u4e00\u4e2a\u7701\u5e02\u3002"),
    ("ideographic_between_words", "\u4f60\u597d\u3000\u4e16\u754c\u3000\u3000\uff01"),
    ("ideographic_in_brackets", "\u300c\u3000\u3000\u300d\u3000\u2014\u2014\n\u3000\u3000\uff08\u6ce8\uff09"),
    ("nbsp_between_words", "Price:\xa0100\xa0USD and\xa0\xa0more"),
    ("nbsp_before_newline", "line\xa0\xa0\nnext"),
    ("thin_spaces", "10\u2009000\u202fkm\u205fx"),
    ("line_paragraph_separators", "first\u2028second\u2029third"),
    ("next_line_char", "a\u0085b\u0085\u0085c"),
    ("ogham_space", "a\u1680b"),
    ("mixed_run_before_newline", "x\u3000 \xa0\ny"),
    ("mixed_run_before_letter", "x \u3000\xa0y"),
    ("zero_width_not_space", "a\u200bb\ufeffc\u180ed"),
    ("indented_numbers", "[\n    1,\n    22,\n  333\n]"),
    ("unicode_spaces_before_digits", "x \u20031 \u3000\u300012 \xa0\xa03"),
    ("latin_next_to_cjk_and_kana", "abc\u4e2d\u6587def \u304b\u306a\u30ab\u30ca ok"),
    ("kana_with_ideographic_spaces", "\u3053\u3093\u306b\u3061\u306f\u3000\u4e16\u754c\u3002"
                                     "\u3000\u3000\u3088\u308d\u3057\u304f\u3002"),
]

ALPHABET = ([" ", " ", "\t", "\n", "\n", "\r\n", "\r", "  ", "\n\n"] + UNICODE_WS +
            list("aZ\xe9\u4e2d\u6587\u300c\u300d\uff0c\u3002?.,;:'\"-_()[]{}0123") + ["don't"])


def fuzz_cases(n, seed, zero_width):
    rng = random.Random(seed)
    alphabet = ALPHABET + NOT_WS if zero_width else ALPHABET
    return [(f"fuzz_{i:03d}", "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 16)))) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--fuzz", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-zero-width", action="store_true",
                    help="fuzz without zero-width characters: DeepSeek's main regex reads them as neither punctuation "
                         "nor space, which the Zig grammar does not emulate")
    a = ap.parse_args()
    tok = Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    cases = [{"name": name, "text": text, "ids": tok.encode(text, add_special_tokens=False).ids}
             for name, text in HANDWRITTEN + fuzz_cases(a.fuzz, a.seed, not a.no_zero_width)]
    with open(a.out, "w", encoding="ascii") as f:
        f.write("[\n" + ",\n".join(json.dumps(c) for c in cases) + "\n]\n")
    print(f"{len(cases)} cases -> {a.out}")


if __name__ == "__main__":
    main()

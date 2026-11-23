// src/poly/grammar_test.cpp - CPU-only tests for strata-poly's grammar, on a small hand-made vocabulary.
//
//   clang++ -std=c++20 -Iinclude -Isrc/poly src/poly/grammar_test.cpp src/poly/grammar.cpp -o grammar_test && ./grammar_test
// and ./grammar_test --dump model.gguf for tools/test_poly_vocab_texts.py, ./grammar_test --parse a.gbnf ... for
// serve/test_grammar.py
#include "grammar.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "vocab_text.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

using strata::poly::Grammar;

enum : int32_t { A, B, AB, ABC, LBRACE, RBRACE, X, EURO_HEAD, EURO_TAIL, CONTROL, EOS, N };

llama_vocab make_vocab() {
    llama_vocab v;
    v.pieces = {"a", "b", "ab", "abc", "{", "}", "x", "\xE2\x82", "\xAC", "", "<eos>"};
    v.eog = {EOS};
    return v;
}

std::unique_ptr<Grammar> parse(const llama_vocab& v, const char* text, std::string* why = nullptr) {
    std::string err;
    auto g = Grammar::parse(text, v, err);
    if (why) *why = err;
    return g;
}

// what the grammar lets through, in token ids
std::vector<int> allowed(const Grammar& g) {
    std::vector<float> logits(N, 0.f);
    g.mask(logits.data(), N);
    std::vector<int> out;
    for (int i = 0; i < N; ++i)
        if (logits[(size_t) i] != -INFINITY) out.push_back(i);
    return out;
}

void test_parse_errors() {
    const llama_vocab v = make_vocab();
    std::string why;
    check(!parse(v, "root ::= (\"a\"", &why) && !why.empty(), "an unclosed group is an error with a message");
    check(!parse(v, "root ::= foo", &why) && !why.empty(), "an undefined rule is an error");
    check(!parse(v, "root ::= root \"a\"", &why) && !why.empty(), "left recursion is an error");
    check(!parse(v, "root ::= \"a", &why) && !why.empty(), "an unclosed string is an error");
    check(!parse(v, "main ::= \"a\"", &why) && !why.empty(), "a grammar without root is an error");
    check(why.find('\n') == std::string::npos, "the message is one line");
    check(parse(v, "root ::= \"a\"") != nullptr, "a good grammar parses after bad ones");
}

void test_literal() {
    const llama_vocab v = make_vocab();
    auto g = parse(v, "root ::= \"ab\"");
    check(g != nullptr, "literal parses");
    check(g->accepts(A) && g->accepts(AB), "a and ab start the literal");
    check(!g->accepts(B) && !g->accepts(ABC) && !g->accepts(X), "b, abc and x do not");
    check(!g->accepts(CONTROL), "a token without text matches nothing");
    check(!g->accepts(EOS) && !g->complete(), "no end before the literal is done");
    check((allowed(*g) == std::vector<int>{A, AB}), "the mask leaves a and ab");
    g->accept(A);
    check(g->accepts(B) && !g->accepts(A) && !g->accepts(AB), "after a, only b");
    check(!g->accepts(EOS), "still no end");
    g->accept(B);
    check(g->complete() && g->accepts(EOS), "after ab the end is allowed");
    check(!g->accepts(A) && !g->accepts(X), "and nothing else");
    check((allowed(*g) == std::vector<int>{EOS}), "only the end token is left");
    float all_gone[N];
    for (float& f : all_gone) f = 1.f;
    check(g->mask(all_gone, N) == 1, "mask counts what is left");
    g->accept(EOS);
    check(g->complete(), "an end token changes nothing");

    auto h = parse(v, "root ::= \"ab\"");
    h->accept(AB);
    check(h->complete(), "a token can be the whole literal");
}

void test_repetition() {
    const llama_vocab v = make_vocab();
    std::string why_not;
    auto g = parse(v, "root ::= \"a\"{2,3}");
    check(g != nullptr, "bounds parse");
    check(!g->accepts(EOS), "no end with none");
    g->accept(A);
    check(!g->accepts(EOS) && g->accepts(A), "one is too few");
    g->accept(A);
    check(g->complete() && g->accepts(EOS) && g->accepts(A), "two is enough and a third is allowed");
    g->accept(A);
    check(g->accepts(EOS) && !g->accepts(A), "three is the most");
    check(!parse(v, "root ::= \"a\"{3,2}", &why_not) && !why_not.empty(), "an inverted bound is an error, not a hang");
}

void test_class() {
    const llama_vocab v = make_vocab();
    auto g = parse(v, "root ::= [a-b]+ \"x\"");
    check(g->accepts(A) && g->accepts(B) && g->accepts(AB), "the class takes a, b and ab");
    check(!g->accepts(ABC) && !g->accepts(X) && !g->accepts(LBRACE), "not abc (c is outside), not x first");
    g->accept(AB);
    check(g->accepts(X) && g->accepts(A) && !g->accepts(EOS), "after ab: x or more of the class");
    g->accept(X);
    check(g->complete() && allowed(*g) == std::vector<int>{EOS}, "x ends it");
}

void test_utf8_split() {
    const llama_vocab v = make_vocab();
    auto g = parse(v, "root ::= \"\xE2\x82\xAC\"");   // the euro sign
    check(g != nullptr, "a multi-byte literal parses");
    check(g->accepts(EURO_HEAD), "the first two bytes of a character are a prefix");
    check(!g->accepts(EURO_TAIL), "the last byte alone is not");
    g->accept(EURO_HEAD);
    check(!g->complete(), "half a character is not complete");
    check(g->accepts(EURO_TAIL) && !g->accepts(A) && !g->accepts(EOS), "only the rest follows");
    g->accept(EURO_TAIL);
    check(g->complete() && g->accepts(EOS), "the character is whole");
}

void test_accepts_changes_nothing() {
    const llama_vocab v = make_vocab();
    auto g = parse(v, "root ::= \"ab\" | \"x\"");
    for (int i = 0; i < 3; ++i) check(g->accepts(A) && g->accepts(X) && !g->accepts(B), "asking twice answers the same");
    g->accept(X);
    check(g->complete(), "x alone is a sentence");
}

void test_gpt2_texts() {
    // "Ġ" is the space, "Ċ" the newline, "Ã©" the two bytes of an é, "âĤ¬" the three of a euro sign
    const std::vector<std::string> pieces = {"a", "\xC4\xA0the", "\xC4\x8A", "\xC3\x83\xC2\xA9", "\xC3\xA2\xC4\xA4\xC2\xAC",
                                             "<|im_end|>", "\xE2\x82\xAC"};
    const std::vector<int32_t> types = {1, 1, 1, 1, 1, 3, 1};
    const auto t = strata::poly::gpt2_token_texts("gpt2", pieces, types, 9);
    check(t.size() == 9, "one text for each id of the vocabulary");
    check(t[0] == "a" && t[1] == " the" && t[2] == "\n", "ASCII, the space and the newline");
    check(t[3] == "\xC3\xA9", "an é is its two UTF-8 bytes");
    check(t[4] == "\xE2\x82\xAC", "a euro sign is its three");
    check(t[5].empty(), "a control token has no text");
    check(t[6].empty(), "a piece outside the alphabet has no text");
    check(t[7].empty() && t[8].empty(), "ids past the table have none");
    check(strata::poly::gpt2_token_texts("llama", pieces, types, 9).empty(), "another tokenizer: no texts");
}

// A GGUF with no tensors and a tokenizer, written to a temporary file: the reader keeps 64 items of an array in its
// metadata, and string_array / int_array read all of it.
void test_reader() {
    std::string bytes = "GGUF";
    auto u32 = [&](uint32_t v) { bytes.append((const char*) &v, 4); };
    auto u64 = [&](uint64_t v) { bytes.append((const char*) &v, 8); };
    auto str = [&](const std::string& v) { u64(v.size()); bytes += v; };
    u32(3);
    u64(0);   // tensors
    u64(3);   // keys
    str("tokenizer.ggml.model"), u32(8), str("gpt2");
    const int n = 200;
    str("tokenizer.ggml.tokens"), u32(9), u32(8), u64(n);
    for (int i = 0; i < n; ++i) str("t" + std::to_string(i));
    str("tokenizer.ggml.token_type"), u32(9), u32(5), u64(n);
    for (int i = 0; i < n; ++i) u32((uint32_t) (i % 7));
    const char* path = "grammar_test_tokenizer.gguf";
    {
        std::ofstream out(path, std::ios::binary);
        out.write(bytes.data(), (std::streamsize) bytes.size());
        check(out.good(), "a temporary file");
    }
    {
        const strata::GgufFile g(path);
        const auto tokens = g.string_array("tokenizer.ggml.tokens");
        const auto types = g.int_array("tokenizer.ggml.token_type");
        check(tokens.size() == (size_t) n && tokens[0] == "t0" && tokens[n - 1] == "t199", "every token is read");
        check(types.size() == (size_t) n && types[6] == 6 && types[n - 1] == (n - 1) % 7, "every token type is read");
        check(g.get("tokenizer.ggml.tokens")->items.size() == 64, "the metadata still keeps its sample");
        check(g.string_array("tokenizer.ggml.nope").empty() && g.int_array("tokenizer.ggml.tokens").empty(),
              "a missing key, or an array of another type, is empty");
        check(g.string_array("tokenizer.ggml.token_type").empty(), "a string read of integers is empty");
    }
    std::remove(path);
}

}  // namespace

// grammar_test --dump model.gguf: "id hex-of-the-text" for every token that has a text, for
// tools/test_poly_vocab_texts.py to compare with the server's tokenizer.
int dump(const char* path) {
    const strata::GgufFile g(path);
    const strata::MetaValue* model = g.get("tokenizer.ggml.model");
    const auto tokens = g.string_array("tokenizer.ggml.tokens");
    const auto texts = strata::poly::gpt2_token_texts(model ? model->s : "", tokens,
                                                      g.int_array("tokenizer.ggml.token_type"), (int) tokens.size());
    if (texts.empty()) {
        std::fprintf(stderr, "no texts: the tokenizer is %s\n", model ? model->s.c_str() : "missing");
        return 1;
    }
    for (size_t i = 0; i < texts.size(); ++i) {
        std::printf("%zu ", i);
        for (unsigned char c : texts[i]) std::printf("%02x", c);
        std::printf("\n");
    }
    return 0;
}

// grammar_test --parse a.gbnf b.gbnf ...: whether each file is a grammar the engine takes, for serve/test_grammar.py
// (the grammars the server builds from JSON schemas).
int parse_files(int argc, char** argv) {
    llama_vocab v;
    v.pieces = {"a"};
    v.eog = {0};
    int bad = 0;
    for (int i = 2; i < argc; ++i) {
        std::ifstream in(argv[i], std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::string err;
        const bool ok = in.good() || in.eof() ? Grammar::parse(text, v, err) != nullptr : false;
        std::printf("%s %s %s\n", ok ? "ok" : "ERR", argv[i], err.c_str());
        bad += !ok;
    }
    return bad ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--dump") return dump(argv[2]);
    if (argc >= 3 && std::string(argv[1]) == "--parse") return parse_files(argc, argv);
    test_parse_errors();
    test_literal();
    test_repetition();
    test_class();
    test_utf8_split();
    test_accepts_changes_nothing();
    test_gpt2_texts();
    test_reader();
    std::printf("grammar_test: %d checks passed\n", checks);
    return 0;
}

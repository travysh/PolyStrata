// src/poly/grammar.hpp - llama.cpp's GBNF grammar (src/llama-grammar.h, MIT, Copyright (c) 2023-2026 The ggml
// authors, commit 3cf03257f219afbe7334045ff7c6a06ac68c627d, the checkout CMakeLists.txt pins), and
// src/poly/grammar.cpp (the same commit's src/llama-grammar.cpp).  The dialect and the matching are
// llama.cpp's, so a grammar that works there works here.  What changed, besides this header:
//  1. llama.h, llama-impl.h, llama-vocab.h and llama-sampler.h are not included.  The few types the grammar takes
//     from them are defined below: llama_token, llama_token_data(_array) and a llama_vocab that is a list of
//     decoded token texts and the end tokens (token_to_piece, is_eog, tokenize keep their names and meaning).
//  2. GGML_ASSERT and GGML_ABORT throw std::logic_error; LLAMA_LOG_ERROR is collected in grammar_log_text() so that
//     the caller can say why a grammar did not parse; LLAMA_LOG_DEBUG is dropped.
//  3. A repetition {m,n} with n below m is a parse error.  Upstream computes n - m unsigned and loops 2^64 times.
//  4. strata::poly::Grammar (at the end of both files) is new: parse, accepts, accept, mask, complete.
// The lazy-grammar triggers are kept as they were and not used.
#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <regex>

using llama_token = int32_t;

struct llama_token_data {
    llama_token id;
    float logit;
    float p;
};

struct llama_token_data_array {
    llama_token_data * data;
    size_t size;
    int64_t selected;
    bool sorted;
};

// What the grammar needs of a vocabulary: the bytes of each token (empty: it matches nothing) and the tokens
// that end an answer.
struct llama_vocab {
    std::vector<std::string> pieces;
    std::vector<llama_token> eog;

    const std::string & token_to_piece(llama_token id) const { return pieces.at((size_t) id); }
    bool is_eog(llama_token id) const {
        for (llama_token t : eog) {
            if (t == id) {
                return true;
            }
        }
        return false;
    }
    // `<name>` in a grammar: the one token whose text is exactly name (the angle brackets and ids are not tokenized)
    int32_t tokenize(const char * text, int32_t len, llama_token * out, int32_t max, bool, bool) const {
        const std::string want(text + 1, (size_t) len - 2);   // without the < and >
        int32_t n = 0;
        for (size_t i = 0; i < pieces.size() && n < max; ++i) {
            if (pieces[i] == want) {
                out[n++] = (llama_token) i;
            }
        }
        return n;
    }
};

#define GGML_ABORT(...) throw std::logic_error("grammar: internal error")
#define GGML_ASSERT(x) if (!(x)) GGML_ABORT()
std::string & grammar_log_text();
void grammar_log(const char * fmt, ...);
#define LLAMA_LOG_ERROR(...) grammar_log(__VA_ARGS__)
#define LLAMA_LOG_DEBUG(...) ((void) 0)

// grammar element type
enum llama_gretype {
    // end of rule definition
    LLAMA_GRETYPE_END            = 0,

    // start of alternate definition for rule
    LLAMA_GRETYPE_ALT            = 1,

    // non-terminal element: reference to rule
    LLAMA_GRETYPE_RULE_REF       = 2,

    // terminal element: character (code point)
    LLAMA_GRETYPE_CHAR           = 3,

    // inverse char(s) ([^a], [^a-b] [^abc])
    LLAMA_GRETYPE_CHAR_NOT       = 4,

    // modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to
    // be an inclusive range ([a-z])
    LLAMA_GRETYPE_CHAR_RNG_UPPER = 5,

    // modifies a preceding LLAMA_GRETYPE_CHAR or
    // LLAMA_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
    LLAMA_GRETYPE_CHAR_ALT       = 6,

    // any character (.)
    LLAMA_GRETYPE_CHAR_ANY       = 7,

    // terminal element: token (<[token-id]>)
    LLAMA_GRETYPE_TOKEN          = 8,

    // inverse token (!<[token-id]>)
    LLAMA_GRETYPE_TOKEN_NOT      = 9,
};

typedef struct llama_grammar_element {
    enum llama_gretype type;
    uint32_t           value; // Unicode code point, rule ID, or token ID
} llama_grammar_element;

struct llama_partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct llama_grammar_candidate {
    size_t               index;
    const uint32_t     * code_points;
    llama_partial_utf8   partial_utf8;
    llama_token          id;
};

using llama_grammar_rule  = std::vector<      llama_grammar_element>;
using llama_grammar_stack = std::vector<const llama_grammar_element *>;

using llama_grammar_rules      = std::vector<llama_grammar_rule>;
using llama_grammar_stacks     = std::vector<llama_grammar_stack>;
using llama_grammar_candidates = std::vector<llama_grammar_candidate>;

// TODO: remove, needed for tests atm
const llama_grammar_rules  & llama_grammar_get_rules (const struct llama_grammar * grammar);
      llama_grammar_stacks & llama_grammar_get_stacks(      struct llama_grammar * grammar);

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `llama_grammar_advance_stack`), and
// produces the N possible stacks if the given char is accepted at those
// positions
void llama_grammar_accept(struct llama_grammar * grammar, uint32_t chr);

std::vector<llama_grammar_candidate> llama_grammar_reject_candidates_for_stack(
        const llama_grammar_rules      & rules,
        const llama_grammar_stack      & stack,
        const llama_grammar_candidates & candidates);

struct llama_grammar_parser {
    const llama_vocab * vocab;
    std::map<std::string, uint32_t> symbol_ids;

    llama_grammar_rules rules;

    llama_grammar_parser(const struct llama_vocab * vocab = nullptr) : vocab(vocab) {}

    llama_grammar_stack c_rules() const;

    uint32_t get_symbol_id(const char * src, size_t len);
    uint32_t generate_symbol_id(const std::string & base_name);

    void add_rule(uint32_t rule_id, const llama_grammar_rule & rule);

    const char * parse_alternates(
            const char        * src,
            const std::string & rule_name,
            uint32_t            rule_id,
            bool                is_nested);

    const char * parse_sequence(
            const char         * src,
            const std::string  & rule_name,
            llama_grammar_rule & rule,
            bool               is_nested);

    const char * parse_rule(const char * src);

    bool parse(const char * src);
    void print(FILE * file);
};

struct llama_grammar_trigger_pattern {
    std::string pattern;
    std::regex  regex;

    size_t find(const std::string & input) const;
};

struct llama_grammar {
    // maintain a list of llama_tokens and their positions in the trigger_buffer
    using token_pos = std::pair<llama_token, std::pair<size_t, size_t>>;

    // note: allow null vocab for testing (not great)
    const llama_vocab * vocab;

    const llama_grammar_rules  rules;  // TODO: shared ptr
          llama_grammar_stacks stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    llama_partial_utf8 partial_utf8;

    // lazy grammars wait for trigger words or tokens before constraining the sampling.
    // we still have trigger_tokens for non-lazy grammars to force printing of special trigger tokens.
    // (useful e.g. for tool_choice=required)
    bool                     lazy             = false;
    bool                     awaiting_trigger = false; // Initialized to true for lazy grammars only
    std::string              trigger_buffer;           // Output buffered by lazy grammar. Will be cleared once trigger is found.
    std::vector<token_pos>   trigger_buffer_positions; // Tokens buffered by lazy grammar. Used to replay when a trigger is found.
    std::vector<llama_token> trigger_tokens;           // Tokens that trigger a lazy grammar, or tokens to force printing of (even if special).
    std::vector<llama_grammar_trigger_pattern>
                             trigger_patterns;         // Regular expressions that trigger a lazy grammar. Must be a full match of the entire generated
                                                       // string, and the grammar will be given the string from the first match group onwards.

};

//
// internal API
//

// note: needed for tests (not great)
struct llama_grammar * llama_grammar_init_impl(
        const struct llama_vocab * vocab,
        const llama_grammar_element ** rules,
        size_t n_rules,
        size_t start_rule_index);

struct llama_grammar * llama_grammar_init_impl(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                              bool lazy,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens);

void llama_grammar_free_impl(struct llama_grammar * grammar);

struct llama_grammar * llama_grammar_clone_impl(const struct llama_grammar & grammar);

// TODO: move the API below as member functions of llama_grammar
void llama_grammar_apply_impl(
        const struct llama_grammar & grammar,
            llama_token_data_array * cur_p);

void llama_grammar_accept_impl(
              struct llama_grammar & grammar,
                       llama_token   token);

void llama_grammar_accept_str(
              struct llama_grammar & grammar,
                 const std::string & piece);

void llama_grammar_accept_token(
              struct llama_grammar & grammar,
                       llama_token   token,
                 const std::string & piece);

namespace strata::poly {

// A GBNF grammar and where in it the answer so far has got.  The vocabulary must outlive it.
class Grammar {
public:
    // Null, and `err` says why, when the text is not a grammar.
    static std::unique_ptr<Grammar> parse(const std::string & gbnf, const llama_vocab & vocab, std::string & err);

    // Whether the answer so far plus this token's text is a prefix the grammar accepts; an end token only when the
    // grammar is complete.  Changes nothing.
    bool accepts(int32_t token) const;
    // The token is now part of the answer.  An end token ends it and changes nothing.
    void accept(int32_t token);
    // Sets the logit of every token the grammar rejects to -infinity.  Returns how many are left.
    int mask(float * logits, int n) const;
    // The answer so far is a whole sentence of the grammar (it may still go on).
    bool complete() const;
    // A token that ends the answer, for a pick when the grammar leaves nothing else.
    int32_t end_token() const { return vocab_->eog.empty() ? 0 : vocab_->eog[0]; }

    ~Grammar();

private:
    Grammar() = default;
    const llama_vocab * vocab_ = nullptr;
    struct llama_grammar * g_ = nullptr;
};

}  // namespace strata::poly

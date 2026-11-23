// src/poly/vocab_text.hpp - the bytes of each token of a byte-level BPE vocabulary (GPT-2's, as Qwen and GLM use),
// for the grammar to match.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::poly {

// The token texts of the GGUF's tokenizer.ggml.tokens: each piece is written in GPT-2's alphabet, one printable
// character for each byte, and decodes to the bytes it stands for.  A control token (type 3), a token past the
// table and a piece that is not in the alphabet have no text, so that no grammar matches them.  Empty (no grammar
// support) when the tokenizer is not "gpt2".
inline std::vector<std::string> gpt2_token_texts(const std::string& model, const std::vector<std::string>& pieces,
                                                 const std::vector<int32_t>& types, int n_vocab) {
    if (model != "gpt2" || pieces.empty() || n_vocab <= 0) return {};
    // GPT-2's bytes_to_unicode, backwards: the bytes that print as themselves, and the others in order from U+0100
    int byte_of[324];
    for (int& b : byte_of) b = -1;
    int next = 256;
    for (int b = 0; b < 256; ++b) {
        const bool plain = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
        byte_of[plain ? b : next++] = b;
    }
    std::vector<std::string> out((size_t) n_vocab);
    for (size_t i = 0; i < pieces.size() && i < out.size(); ++i) {
        if (i < types.size() && types[i] == 3) continue;
        const std::string& p = pieces[i];
        std::string& text = out[i];
        for (size_t k = 0; k < p.size();) {
            const unsigned char c = (unsigned char) p[k];
            const size_t len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : 0;   // the alphabet ends below U+0144
            const int cp = len == 0 || k + len > p.size() ? -1
                         : len == 1 ? c : ((c & 0x1F) << 6) | ((unsigned char) p[k + 1] & 0x3F);
            if (cp < 0 || cp >= 324 || byte_of[cp] < 0) {
                text.clear();
                break;
            }
            text.push_back((char) byte_of[cp]);
            k += len;
        }
    }
    return out;
}

}  // namespace strata::poly

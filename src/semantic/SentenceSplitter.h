/*
 * SentenceSplitter.h
 *
 * Sentence spans in UTF-8 byte offsets, using Pango log attributes
 * (is_sentence_start). Whitespace-only spans are dropped. The text
 * field is the raw slice, including its surrounding whitespace, so a
 * cache key matches the bytes that get painted.
 */
#ifndef NPP_SENTENCE_SPLITTER_H
#define NPP_SENTENCE_SPLITTER_H

#include <cstdint>
#include <string>
#include <vector>

struct SentenceSpan {
    int64_t byte_start;
    int64_t byte_length;
    std::string text;
};

std::vector<SentenceSpan> split_sentences(const std::string &utf8);

#endif

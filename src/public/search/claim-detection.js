// Use binary LR classifier to detect if claim or not
// Split post into sentences via Intl.Segmenter, then run claim detection on each sentence.

/** Segmenter instance for sentence splitting (locale-aware). */
const sentenceSegmenter = new Intl.Segmenter("en", { granularity: "sentence" });

/**
 * Splits post text into sentences for claim detection using Intl.Segmenter.
 * Input: raw string (caption for Instagram, text for Twitter, or combined title + body for Reddit).
 * Returns only sentences with at least 3 words (to avoid trivial fragments).
 *
 * @param {string} input - Raw post text (or combined text, e.g. Reddit title + "\n\n" + body)
 * @returns {string[]} Array of sentences (each with 3+ words), trimmed
 */
function getClauses(input) {
    if (input == null || typeof input !== "string") return [];
    const text = input.trim();
    if (!text) return [];

    const segments = [...sentenceSegmenter.segment(text)];
    const sentences = segments
        .map((s) => s.segment.trim())
        .filter(Boolean)
        .filter((s) => s.split(/\s+/).filter(Boolean).length >= 3);
    return sentences;
}

// Use binary LR classifier to detect if claim or not
// Transform sentence first: split post into sentences, then split into clauses

/**
 * Sentence-ending punctuation: ! and ? always; . only when not part of an abbreviation.
 * Abbreviation = period after a single letter (U.S., U.K.) or after letter+period (the second . in U.S.).
 */
const SENTENCE_END_RE = /(?:[!?]|(?<!(?:^|\s)[A-Za-z])(?<![A-Za-z]\.)\.)\s+/;

/**
 * Splits post text into clauses for claim detection.
 * Input: raw string (caption for Instagram, text for Twitter, or combined title + body for Reddit).
 * 1. Split by sentence (. ! ?), avoiding abbreviation periods (e.g. U.S., U.K., Mr., Dr.)
 * 2. Split each sentence by clause punctuation (, ; : and dash with spaces)
 * 3. Return only clauses with at least 3 words.
 *
 * @param {string} input - Raw post text (or combined text, e.g. Reddit title + "\n\n" + body)
 * @returns {string[]} Array of clauses (each with 3+ words), trimmed
 */
function getClauses(input) {
    if (input == null || typeof input !== "string") return [];
    const text = input.trim();
    if (!text) return [];

    // 1. Split into sentences by sentence-ending punctuation (preserve abbreviations like U.S.)
    const sentences = text
        .split(SENTENCE_END_RE)
        .map((s) => s.trim())
        .filter(Boolean);

    const clauses = [];
    // 2. Split each sentence by clause punctuation: comma, semicolon, colon, or dash (with spaces)
    const clauseSplit = /\s*[,;:]+\s*|\s+[-–—]\s+/;
    for (const sentence of sentences) {
        const parts = sentence.split(clauseSplit).map((s) => s.trim()).filter(Boolean);
        for (const part of parts) {
            const wordCount = part.split(/\s+/).filter(Boolean).length;
            if (wordCount >= 3) {
                clauses.push(part);
            }
        }
    }
    return clauses;
}

# Roundabout

## A social media extension to verify sources

This extension seeks to identify claims on social media, and then verify them.

It does this through the use of machine learning to identify claims and then find the most closely-related sources, sorted by apparent reliability and exactitude.

Reliability and exactitude are not objective, but we seek to do the best we can by considering the quality of the article itself.

This extension was made with React+Vite for the Spring 2026 CAHSI LREU program.

This extension uses the following ML models:
- all-MiniLM-L6-v2 for sentence embeddings when scoring source articles for relevance
- `Xenova/bert-base-NER` (via `@xenova/transformers`) for on-device named-entity signals used in Exactitude

## Exactitude (runs in browser)

**Claim detection** is determined by the **Exactitude** score only. Exactitude is **fully in-browser**: **token-classification NER** (`Xenova/bert-base-NER`, ONNX loaded in-browser) plus **compromise** for noun phrases (`#Adjective? #Noun+`), modals (`#Modal`), and term-level checks, together with numeric/date/attribution heuristics. The seven dimensions (A–G) and total out of 12 are: quantification, time specificity, location scope, defined terms, source clarity, falsifiability, (each worth 0, 1, or 2), and personal relativity (a penalty of max 2). The threshold is set to 6 to count as a claim, and thus be searched by the extension.


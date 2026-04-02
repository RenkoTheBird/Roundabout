# Roundabout

## A social media extension to verify sources

This extension seeks to identify claims on social media, and then verify them.

It does this through the use of machine learning to identify claims and then find the most closely-related sources, sorted by apparent reliability and exactitude.

Reliability and exactitude are not objective, but we seek to do the best we can by considering the quality of the article itself.

This extension was made with React+Vite for the Spring 2026 CAHSI LREU program.

This extension uses the following datasets:

```
Claims list:
@inproceedings{Thorne18Fever,
    author = {Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Mittal, Arpit},
    title = {{FEVER}: a Large-scale Dataset for Fact Extraction and {VERification}},
    booktitle = {NAACL-HLT},
    year = {2018}
}
```

Opinions list (reviews):
The "Reviews" dataset by Jyoti Kushwaha at https://www.kaggle.com/datasets/jyotikushwaha545/reviews on Kaggle.

Questions list:
SQuAD Dataset (Stanford Question Answering Dataset); found at https://rajpurkar.github.io/SQuAD-explorer/

This extension uses the following ML models:
- all-MiniLM-L6-v2 for sentence transformation
- `Xenova/bert-base-NER` (via `@xenova/transformers`) for on-device named-entity signals used in Exactitude

## Exactitude (pure extension, no API key)

Claim gating uses a **logistic-regression “claim” classifier in the browser** (MiniLM via `@xenova/transformers`) plus an **Exactitude** score. Exactitude is **fully in-browser**: **token-classification NER** (`Xenova/bert-base-NER`, ONNX loaded from the Hugging Face model hub on first use—no token required) plus **compromise** for noun phrases (`#Adjective? #Noun+`), modals (`#Modal`), and term-level checks, together with the same numeric/date/attribution heuristics as before. The six dimensions (A–F) and total out of 12 are unchanged: quantification, time specificity, location scope, defined terms, source clarity, and falsifiability.


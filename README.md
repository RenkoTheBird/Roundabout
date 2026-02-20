# Roundabout

## A social media extension to verify sources

This extension seeks to identify claims on social media, and then verify them.

It does this through the use of machine learning to identify claims and then find the most closely-related sources, sorted by apparent reliability and exactitude.

Reliability and exactitude are not objective, but we seek to do the best we can by considering the quality of the article itself.

This extension was made with React+Vite for the Spring 2026 CAHSI LREU program.

This extension uses the following datasets:

Claims list:
@inproceedings{Thorne18Fever,
    author = {Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Mittal, Arpit},
    title = {{FEVER}: a Large-scale Dataset for Fact Extraction and {VERification}},
    booktitle = {NAACL-HLT},
    year = {2018}
}

Opinions list (reviews):
The "Reviews" dataset by Jyoti Kushwaha at https://www.kaggle.com/datasets/jyotikushwaha545/reviews on Kaggle.

This extension uses the following ML models:
- all-MiniLM-L6-v2 for sentence transformation
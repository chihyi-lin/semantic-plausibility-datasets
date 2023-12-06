Authors (alphabetical order): Chih-Yi Lin, Quy Nguyen and Wen Wen

# semantic_plausibility

## DATASET ANALYSIS
We explore and perform basic analysis on three datasets available to model semantic plausiblity of s-v-o triples
* PEP-3K
* ADEPT
* PAP

### PEP-3K 
We analyzed five aspects of pep-3k data, including: 
* Word Count and Frequency
* Abstractness and Concreteness
* Part-of-Speech Tag Distribution
* Words Similarity
* Word Overlap Across Train, Dev, and Test Datasets

In general, this dataset reveals a restricted vocabulary, characterized by a notable recurrence of specific words. Subjects and objects within this dataset predominantly exhibit concrete characteristics, showcasing a subtle differentiation between 'plausible' and 'implausible'. Nouns take the lead in frequency, closely followed by verbs. The misclassification rate for words tagged as adjectives is relatively low and doesn't display any discernible bias towards 'plausible' or 'implausible'. Across various categories, there exists a slightly similarity between 'plausible' and 'implausible' words. Additionally, a small percentage of new words is observed in the dev dataset and test dataset, indicating a level of consistency across different dataset splits.

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

### ADEPT
We analyzed the following five aspects:
* 5-Class Label Distribution
* 3-Class Label Distribution
* Modifiers Rank-Frequency Distribution
* Investigate Modifer-Noun Pairs (across all classes and two extremes classes)
* Label Distribution of Concreteness of Modifiers

**Strengths:**
1. ADEPT provides a rich amount of modifiers, modifier-noun pairs, as well as contexts for researchers to investigate how they interact with each other and affect semantic plausibility.
2. Based on the aforementioned data nature, the labels are not a fixed feature of a triple but rather how plausibility changes given a context, which is more aligned with human capability (that we are good at adding context to make an event plausible or implausible).
3. Demonstrates that non-subsective modifiers are possible to occur in any range of plausibility, indicating that context should also be considered.
4. Modifiers have varieties in concreteness, allowing us to investigate how concretness affects plausibility.

**Weaknesses:**
1. The labels are highly imbalanced for both 5-class and 3-class labels.
2. The distribution of modifiers is also imbalanced, with many duplicated non-subsective adjectives, as well as adj-noun pairs.
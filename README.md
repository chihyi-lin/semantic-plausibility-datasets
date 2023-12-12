Authors (alphabetical order): Chih-Yi Lin, Quy Nguyen and Wen Wen

# Modeling semantic plausibility
## Table of Contents
- [Installation](#installation)
- [Data analysis](#dataset-analysis)
    - [PEP-3K](#pep-3k)  
    - [PAP](#pap) 
    - [ADEPT](#adept)

## Installation
Welcome to our project. Follow these steps to setup and run the code using Conda.

### 1. Clone the Repository

Clone the project repository to your local machine using the following command:

```bash
git clone https://github.com/chihyi-lin/semantic_plausibility.git
```

### 2. Set Up Conda Environment

If you're in our workspace at IMS server, you could use our environment

```bash
conda activate ./falconenv
```

Otherwise, create a Conda environment and activate it:

```bash
conda create --prefix [/path/to/environment]
conda env update --prefix [/path/to/environment] --file [/path/to/environment.yml] 
conda activate [/path/to/environment]
```

### 3. Accessing the Dataset

We assume that dataset directory is `/mount/studenten/semantic-plausibility/datasets`.

### 4. Get word2vec 

We assume that the `word2vec` model is inside the directory `gensim-data`, which is located at the same level of this repository, that is`[our_working_space]/submission/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz`


## DATASET ANALYSIS
We explore and perform basic analysis on three datasets available to model semantic plausiblity of s-v-o triples
* PEP-3K
* PAP
* ADEPT

### PEP-3K 
We analyzed five aspects of pep-3k data, including: 
* Word Count and Frequency
* Abstractness and Concreteness
* Part-of-Speech Tag Distribution
* Words Similarity
* Word Overlap Across Train, Dev, and Test Datasets

**Strengths:**
* The dataset maintains a consistent S-V-O structure without incorporating modifiers.
* It exhibits an equitable distribution between plausible and implausible events, ensuring a balanced representation within the dataset.
* Subjects and objects solely embody concrete characteristics, entirely excluding abstract concepts. Able to exclude concrete/abstract interference when we analyze other aspects.
* A small percentage of new words is observed in the dev dataset and test dataset, indicating a level of consistency across different dataset splits.

**Weaknesses:**
* This dataset reveals a restricted vocabulary, characterized by a notable recurrence of specific words.
* Only contain nouns and verbs, no modifiers involved.
* In various categories, it shows a slightly similarity between plausible words and implausible words. There is no evidence supporting a preference between 'plausible' and 'implausible' for specific words or topics.


### PAP
We explore the following aspecs
* Raw annotation
    * Basic statistics
    * The skewness toward Plausibility of the dataset (based on Average Median Ratings)
* Entropy in ratings
    * Concreteness rating of each constituent of s-v-o triples
    * Weak Correlation between constituent concreteness rating and disagreement (entropy) between annotators
* Label distribution in Aggregation labels
* Label distribution in Train-Dev-Test splits, in binary and multi-class setting.

**Strengths**
* Carefully curated dataset, balanced in term of concreteness - abstractness
* Provide raw annotation, majority votings, MACE aggregation
* Provide both fine-grained multiclass and binary classification
* Similar label distributions for Train-Dev-Test Splits

**Weakness**
* Small number of triples
* Extracted from Wikipedia - might have biases and not aware of diversity
* Unbalanced in final label distribution - more skewed to Positive class
* Crowdsource annotation - might have noise and hard to model automatically

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

## Contributions
- Analysis of PEP-3K: Wen Wen
- Analysis of PAP: Quy Nguyen
- Analysis of ADEPT: Chih-Yi Lin
- Project installation guide: Quy Nguyen

import json
import pandas as pd
import torch
import random
import numpy as np
from openprompt.utils.reproduciblity import set_seed
from openprompt.data_utils.utils import InputExample
from openprompt.prompts import ManualTemplate
from openprompt.plms import load_plm

set_seed(0)
train_path = 'datasets/adept/train-dev-test-split/train.json'

val_path = '../datasets/adept/train-dev-test-split/val.json'
test_path = '../datasets/adept/train-dev-test-split/test.json'

def get_examples(file_path):
    examples = []
    with open(file_path, "r") as file:
        data = json.load(file)
        df = pd.DataFrame(data)
        sentences1 = df['sentence1']
        sentences2 = df['sentence2']
        labels = df['label']
        for i in range(len(sentences1)):
            sentence1, sentence2, label = sentences1[i], sentences2[i], int(labels[i])
            example = InputExample(guid=i, text_a=sentence1, text_b=sentence2, label=label)
            examples.append(example)
    return examples


train_0 = get_examples(train_path)[0]
print(train_0)

# DataProcessor(train_path)
# get_train_examples(train_path)

# Load the pretrained LM
plm, tokenizer, model_config, bertTokenizerWrapper = load_plm("bert", "bert-base-cased")
my_prompt_template = ManualTemplate(
    text = '{"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.',
    tokenizer = tokenizer,
)
wrapped_example = my_prompt_template.wrap_one_example(train_0)
print(wrapped_example)
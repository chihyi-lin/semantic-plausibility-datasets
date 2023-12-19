import json
import pandas as pd
import torch
import random
import numpy as np
from openprompt.utils.reproduciblity import set_seed
from openprompt.data_utils.utils import InputExample
from openprompt.prompts import ManualTemplate
from openprompt.plms import load_plm
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
from openprompt.prompts import SoftVerbalizer
from openprompt import PromptForClassification

"""This script is adapted from OpenPrompt tutorials (https://github.com/thunlp/OpenPrompt). """

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"

set_seed(0)
train_path = 'datasets/adept/train-dev-test-split/train.json'
val_path = 'datasets/adept/train-dev-test-split/val.json'
test_path = 'datasets/adept/train-dev-test-split/test.json'

class PromptLearning:

    def __init__(self, train_path, val_path, test_path):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        
        self.train_set = self.get_examples(self.train_path)
        self.val_set = self.get_examples(self.val_path)
        self.test_set = self.get_examples(self.test_path)

        self.plm, self.tokenizer, self.model_config, self.robertaTokenizerWrapper = load_plm("roberta", "roberta-base")
        self.prompt_template = None
        self.prompt_verbalizer = None
        self.prompt_model = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None


    def get_examples(self, file_path):
        """convert the dataset into the input format"""
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

    
    def define_manual_template(self):
        """define manual prompt temeplate with a mask token for predictions"""
        self.prompt_template = ManualTemplate(
            text = '{"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.',
            tokenizer = self.tokenizer,
        )
        return None


    def define_manual_verbalizer(self):
        """define verbalizer that maps the original class labels to the words that we condiser are valid predictions. 
            classes: 0: Impossible, 1: Less likely, 2: Equally likely, 3: More likely", 4: Necessarily true"""
        classes = [0, 1, 2, 3, 4]
        self.prompt_verbalizer = ManualVerbalizer(
            classes = classes,
            label_words = {
                0: ['impossible', 'no', 'incorrect', 'invalid'],
                1: ['less likely', 'less correct'],
                2: ['same likely', 'equally likely', 'same', 'no change'],
                3: ['more likely', 'more possible'],
                4: ['yes', 'correct', 'true', 'valid']},
            tokenizer=self.tokenizer,
        )

    
    def define_soft_verbalizer(self):
        self.prompt_verbalizer = SoftVerbalizer(self.tokenizer, self.plm, num_classes=4)
        return None
    
    def init_dataloader(self):
        self.train_loader = PromptDataLoader(
            dataset = self.train_set,
            tokenizer = self.tokenizer,
            tokenizer_wrapper_class = self.robertaTokenizerWrapper,
            template = self.prompt_template, 
            max_seq_length=256, decoder_max_length=3,
            batch_size=4, shuffle=True, teacher_forcing=False, predict_eos_token=False,
            truncate_method="head"
        )
        self.val_loader = PromptDataLoader(
            dataset = self.val_set,
            tokenizer = self.tokenizer,
            tokenizer_wrapper_class = self.robertaTokenizerWrapper,
            template = self.prompt_template, 
            max_seq_length=256, decoder_max_length=3,
            batch_size=4, shuffle=True, teacher_forcing=False, predict_eos_token=False,
            truncate_method="head"
        )
        self.test_loader = PromptDataLoader(
            dataset = self.test_set,
            tokenizer = self.tokenizer,
            tokenizer_wrapper_class = self.robertaTokenizerWrapper,
            template = self.prompt_template, 
            max_seq_length=256, decoder_max_length=3,
            batch_size=4, shuffle=True, teacher_forcing=False, predict_eos_token=False,
            truncate_method="head"
        )
        return None


    def init_prompt_model(self):
        # Initialize the prompt model
        self.prompt_model = PromptForClassification(
            template = self.prompt_template,
            plm = self.plm,
            verbalizer = self.prompt_verbalizer,
        )
        self.prompt_model = self.prompt_model.to(device)
        return None

    def zero_shot(self):
        """zero-shot inference using pretrained model with prompting"""
        # Set the model to eval mode
        self.prompt_model.eval()

        # Run the inference loop
        allpreds = []
        alllabels = []
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(device)
                logits = self.prompt_model(batch)
                labels = batch['label']
                outputs = torch.argmax(logits, dim = -1)
                alllabels.extend(labels.to(device).tolist())
                allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        return acc
    

pr = PromptLearning(train_path, val_path, test_path)
pr.define_manual_template()
# pr.define_manual_verbalizer()
pr.define_soft_verbalizer()
pr.init_dataloader()
pr.init_prompt_model()
acc = pr.zero_shot()

print(acc)
print(pr.train_loader)
# zero-shot 5 classes with manual verbalizer result: 0.010552451893234015 XD
# zero-shot 5 classes with soft verbalizer result: 0.23339540657976413
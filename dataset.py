import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class DSTDatasetForBIO(Dataset):

    def __init__(
        self,
        data_dir: str = None,
        ontology_dir: str = None,
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 512,
    ):
        
        self.data = json.load(open(data_dir, 'r', encoding="utf-8-sig"))
        self.ontology = json.load(open(ontology_dir, 'r', encoding="utf-8-sig"))
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.domain = self.data[0]["info"]["topic"]

        self.special_slots = self.ontology["special_slots"][self.domain]
        self.bio_slots = self.ontology["slots"][self.domain]

        self.special_classes = ["none", "yes", "no", "soso", "dontcare"]
        self.bio_classes = ["O"]

        for slot in self.bio_slots:
            self.bio_classes.append("B-" + slot)
            self.bio_classes.append("I-" + slot)

        
        self.sample_idx_to_dialog_idx = dict()

        def unzip_dialogs(dialogs):

            samples = []

            for dialog_idx, dialog in enumerate(dialogs):
                prev_utterance = None

                for utterance in dialog["utterances"]:
                    
                    samples.append({
                        "dialog_id": dialog["info"]["id"],
                        "utterance_id": utterance["utterance_id"],
                        "utterance": utterance["text"],
                        "slots": [{slot["key"]: slot["value"]} for slot in utterance["slot"] if str(slot["value"]) != "nan"],
                        "prev_utterance": prev_utterance
                    })

                    prev_utterance = utterance["text"]

            return samples

        self.samples = unzip_dialogs(self.data)


    def get_dataloader(self, batch_size: int = 64, shuffle: bool = False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):

        sample = self.samples[idx]

        utterance = sample["utterance"]
        slots = sample["slots"]
        prev_utterance = sample["prev_utterance"]

        input_text = utterance
        
        if prev_utterance is not None:    
            input_text = input_text + self.tokenizer.sep_token + prev_utterance

        inputs = self.tokenizer(
            input_text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(input_text) + [self.tokenizer.sep_token]
        tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(input_text)
        
        if '[SEP]' in tokens:
            sep_idx = tokens.index('[SEP]')
        else:
            sep_idx = len(tokens)

        value2slot = {}
        for slot in slots:
            
            if type(list(slot.values())[0]) == float:
                continue

            values = [value.strip() for value in list(slot.values())[0].rstrip('; ').split(";")]
            for value in values:
                if value == "":
                    continue
                value2slot[value] = list(slot.keys())[0]
                value2slot['##' + value] = list(slot.keys())[0]

        bio_labels = [-100] * self.max_length
        bio_labels[:len(tokens)] = [0] * len(tokens)

        for value in sorted(value2slot.keys(), key=lambda x: len(x)):
            value_tokens = self.tokenizer.tokenize(value)
            _value_tokens = ["##" + value_tokens[0]] + value_tokens[1:]
            
            # slot tokenizer not in text
            if self.tokenizer.unk_token in value_tokens:
                continue
            
            length = len(value_tokens)
            for i in range(len(tokens[:sep_idx]) - length):
                if value_tokens == tokens[i:i+length] or _value_tokens == tokens[i:i+length]:
                    bio_labels[i:i+length] = [self.bio_classes.index("I-" + value2slot[value])] * length
                    bio_labels[i] = self.bio_classes.index("B-" + value2slot[value])
                    

        special_labels = [0] * len(self.special_slots)

        for slot in slots:
            key = list(slot.keys())[0]
            value = list(slot.values())[0]
            
            if key in self.special_slots:
                value = value.replace("don’t care", "dontcare")
                special_labels[self.special_slots.index(key)] = self.special_classes.index(value)

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "token_type_ids": inputs.token_type_ids.squeeze(),
            "special_labels": torch.LongTensor(special_labels),
            "bio_labels": torch.LongTensor(bio_labels),
        }
    
if __name__=="__main__":

    from transformers import AutoTokenizer

    data_dir = "data/ex_건강_및_식음료.json"
    ontology_dir = "data/ex_ontology.json"
    tokenizer = AutoTokenizer.from_pretrained("yeongjoon/Kconvo-roberta")
    max_length = 512

    dataset = DSTDatasetForBIO(
        data_dir, ontology_dir, tokenizer, max_length
    )

    print(dataset.samples[1000])
    exit()

    sample = dataset[1]

    pad_idx = torch.where(sample["input_ids"] == tokenizer.pad_token_id)[0][0]

    for i in range(pad_idx + 10):
        print("{} : {}".format(sample["input_ids"][i], sample["bio_labels"][i]))

    print(sample["special_labels"])

    exit()

    for sample in dataset:

        if "input_ids" in sample and "attention_mask" in sample and "special_labels" in sample and "bio_labels" in sample:
            pass
        else:
            print(sample)
            break
    
    print("complete!")

import json
from torch import nn
import random
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoConfig
import shutil
import numpy as np
import torch
from torch.optim import AdamW
from model import RobertaForBioDst, BertForBioDst
from dataset import DSTDatasetForBIO
from utils import train, evaluate, get_scores
import os

os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1, 2, 3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def parse_args():

    parser = ArgumentParser()

    parser.add_argument("-td", "--train_dir", type=str, default="./Training/라벨링데이터/건강_및_식음료.json")
    parser.add_argument("-vd", "--valid_dir", type=str, default="./Validation/라벨링데이터/건강_및_식음료.json")
    parser.add_argument("-od", "--ontology_dir", type=str, default="./Others/건강_및_식음료_ontology.json")
    parser.add_argument("-pm", "--pretrained_model", type=str, default="yeongjoon/Kconvo-roberta")

    parser.add_argument("-e", "--num_epochs", type=int, default=100)
    parser.add_argument("-p", "--patience", type=int, default=3)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-ml", "--max_length", type=int, default=512)
    parser.add_argument("-lr", "--learning_rate", type=int, default=5e-5)

    return parser.parse_args()

def main(args):

    save_dir = './Others/model/' + args.train_dir.split('/')[-1][:-5]

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    train_path = args.train_dir
    eval_path = args.valid_dir

    train_set = DSTDatasetForBIO(
        data_dir=train_path,
        ontology_dir=args.ontology_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    eval_set = DSTDatasetForBIO(
        data_dir=eval_path,
        ontology_dir=args.ontology_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    domain = train_set.domain

    print("The number of training data : ", len(train_set))
    print("The number of validation data : ", len(eval_set))
    print("epoch : ", args.num_epochs)
    print("batch size : ", args.batch_size)
    print("token max length : ", args.max_length)
    print("learning rate : ", args.learning_rate)

    train_loader = train_set.get_dataloader(batch_size=args.batch_size, shuffle=True)
    eval_loader = eval_set.get_dataloader(batch_size=args.batch_size, shuffle=False)

    config = AutoConfig.from_pretrained(args.pretrained_model)
    config.num_special_slots = len(train_set.special_slots)
    config.num_special_classes = len(train_set.special_classes)
    config.num_bio_classes = len(train_set.bio_classes)

    if config.model_type == "bert":
        model = BertForBioDst.from_pretrained(args.pretrained_model, config=config)
    elif config.model_type == "roberta":
        model = RobertaForBioDst.from_pretrained(args.pretrained_model, config=config)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)

    best_f1 = 0.
    patience = 0

    best_model = None

    for epoch in range(args.num_epochs):
        epoch += 1

        train_result = train(model, optimizer, train_loader, device)

        model = train_result["model"]
        special_loss = train_result["special_loss"]
        bio_loss = train_result["bio_loss"]

        print("special loss: {} | bio loss: {}".format(special_loss, bio_loss))
        
        with torch.no_grad():
            eval_result = evaluate(model, eval_set, eval_loader, device)
            
        scores = get_scores(eval_result, eval_set)

        print("[{}] eval total f1 score: {}".format(epoch, scores["total_f1"]))

        if scores["total_f1"] > best_f1:

            best_f1 = scores["total_f1"]

            if best_model is not None:
                shutil.rmtree(best_model)

            tokenizer.save_pretrained(save_dir)
            model_without_parallel = model.module if isinstance(model, nn.DataParallel) else model
            model_without_parallel.save_pretrained(save_dir)
            best_model = save_dir

            patience = 0
        else:
            patience += 1

            if patience == 3:
                break

if __name__=="__main__":

    args = parse_args()
    main(args)

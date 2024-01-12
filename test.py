import json
from torch import nn
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoConfig
import torch
from torch.optim import AdamW
from model import RobertaForBioDst, BertForBioDst
from dataset import DSTDatasetForBIO
from utils import train, evaluate, get_scores
import os

os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1, 2, 3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():

    parser = ArgumentParser()

    parser.add_argument("-d", "--data_dir", type=str, default="./Test/라벨링데이터/건강_및_식음료.json")
    parser.add_argument("-od", "--ontology_dir", type=str, default="./Others/건강_및_식음료_ontology.json")
    parser.add_argument("-sd", "--save_dir", type=str, default="./result_test/")

 
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-ml", "--max_length", type=int, default=512)
    parser.add_argument("-lr", "--learning_rate", type=int, default=5e-5)

    return parser.parse_args()



def main(args):

    pretrained_model = './Others/model/' + args.data_dir.split('/')[-1][:-5]

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    test_set = DSTDatasetForBIO(
        data_dir=args.data_dir,
        ontology_dir=args.ontology_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    domain = test_set.domain

    test_loader = test_set.get_dataloader(batch_size=args.batch_size, shuffle=False)

    config = AutoConfig.from_pretrained(pretrained_model)
    config.num_special_slots = len(test_set.special_slots)
    config.num_special_classes = len(test_set.special_classes)
    config.num_bio_classes = len(test_set.bio_classes)

    if config.model_type == "bert":
        model = BertForBioDst.from_pretrained(pretrained_model, config=config)
    elif config.model_type == "roberta":
        model = RobertaForBioDst.from_pretrained(pretrained_model, config=config)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)

    with torch.no_grad():
        test_result = evaluate(model, test_set, test_loader, device)
    
    scores = get_scores(test_result, test_set)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    with open(args.save_dir + "pred_{}".format(args.data_dir.split('/')[-1]), 'w') as f:
        json.dump(scores["dialogs"], f, ensure_ascii=False, indent='\t')

    print("test tp, fp, fn: {}, {}, {}".format(scores["total_tp"], scores["total_fp"], scores["total_fn"]))
    print("test precision, recall: {}, {}".format(scores["total_precision"], scores["total_recall"]))
    print("test f1 score: {}".format(scores["total_f1"]))
    
    performance_file = open(args.save_dir + "performance_{}.txt".format(args.data_dir.split('/')[-1][:-5]), 'w')
    performance_file.write("test tp, fp, fn: {}, {}, {}\n".format(scores["total_tp"], scores["total_fp"], scores["total_fn"]))
    performance_file.write("test precision, recall: {}, {}\n".format(scores["total_precision"], scores["total_recall"]))
    performance_file.write("test f1 score: {}".format(scores["total_f1"]))
    performance_file.close()
        

if __name__=="__main__":

    args = parse_args()
    main(args)

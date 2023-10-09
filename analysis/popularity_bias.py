import argparse
from math import log2
from pprint import pprint
from typing import List
import json
from pathlib import Path

from finetune import load_data

Item = int


class Namespace():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="Data directory")
    parser.add_argument("--data_name", required=True, help="Data directory")

    return parser.parse_args()


def recall(predictions: List[List[Item]], targets: List[Item], k: int = 10) -> float:
    scores = []

    for prediction, target in zip(predictions, targets):
        if target in prediction[:k]:
            scores.append(1)
        else:
            scores.append(0)

    return sum(scores) / len(scores)


def ndcg(predictions: List[List[Item]], targets: List[Item], k: int = 10) -> float:
    scores = []

    for prediction, target in zip(predictions, targets):
        if target not in prediction[:k]:
            scores.append(0)
            continue

        scores.append(1 / log2(prediction[:k].index(target) + 2))

    return sum(scores) / len(scores)


def popularity_bias(data_name, prediction):
    item_split = json.load(open(Path('../finetune_data') / Path(f"{data_name}_ours") / Path('item_split.json'), 'r'))
    user_split = json.load(open(Path('../finetune_data') / Path(f"{data_name}_ours") / Path('user_split.json'), 'r'))

    overall_prediction, overall_target = [], []
    head_item_prediction, head_target = [], []
    tail_item_prediction, tail_target = [], []
    head_user_prediction, head_user_target = [], []
    tail_user_prediction, tail_user_target = [], []

    for user, user_prediction in prediction.items():
        overall_prediction.append(user_prediction['predictions'])
        overall_target.append(user_prediction['target'])

        if int(user) in user_split['head']:
            head_user_prediction.append(user_prediction['predictions'])
            head_user_target.append(user_prediction['target'])
        else:
            tail_user_prediction.append(user_prediction['predictions'])
            tail_user_target.append(user_prediction['target'])

        if int(user_prediction['target']) in item_split['head']:
            head_item_prediction.append(user_prediction['predictions'])
            head_target.append(user_prediction['target'])
        else:
            tail_item_prediction.append(user_prediction['predictions'])
            tail_target.append(user_prediction['target'])

    bias = {
        'overall': {
            'recall@10': recall(overall_prediction, overall_target, 10),
            'ndcg@10': ndcg(overall_prediction, overall_target, 10),
        },
        'head_item': {
            'recall@10': recall(head_item_prediction, head_target, 10),
            'ndcg@10': ndcg(head_item_prediction, head_target, 10),
            'recall@20': recall(head_item_prediction, head_target, 20),
            'ndcg@20': ndcg(head_item_prediction, head_target, 20),
            'recall@50': recall(head_item_prediction, head_target, 50),
            'ndcg@50': ndcg(head_item_prediction, head_target, 50),
        },
        'tail_item': {
            'recall@10': recall(tail_item_prediction, tail_target, 10),
            'ndcg@10': ndcg(tail_item_prediction, tail_target, 10),
            'recall@20': recall(tail_item_prediction, tail_target, 20),
            'ndcg@20': ndcg(tail_item_prediction, tail_target, 20),
            'recall@50': recall(tail_item_prediction, tail_target, 50),
            'ndcg@50': ndcg(tail_item_prediction, tail_target, 50),
        },
        'head_user': {
            'recall@10': recall(head_user_prediction, head_user_target, 10),
            'ndcg@10': ndcg(head_user_prediction, head_user_target, 10),
            'recall@20': recall(head_user_prediction, head_user_target, 20),
            'ndcg@20': ndcg(head_user_prediction, head_user_target, 20),
            'recall@50': recall(head_user_prediction, head_user_target, 50),
            'ndcg@50': ndcg(head_user_prediction, head_user_target, 50),
        },
        'tail_user': {
            'recall@10': recall(tail_user_prediction, tail_user_target, 10),
            'ndcg@10': ndcg(tail_user_prediction, tail_user_target, 10),
            'recall@20': recall(tail_user_prediction, tail_user_target, 20),
            'ndcg@20': ndcg(tail_user_prediction, tail_user_target, 20),
            'recall@50': recall(tail_user_prediction, tail_user_target, 50),
            'ndcg@50': ndcg(tail_user_prediction, tail_user_target, 50),
        }
    }

    print("\n\n\n\n")
    print("=" * 50)

    print(f"percentage of head item: {len(head_item_prediction) / len(overall_prediction)}")
    print(f"percentage of head user: {len(head_user_prediction) / len(overall_prediction)}")

    print(f"overall | recall@10: {bias['overall']['recall@10']:.4f} , ndcg@10: {bias['overall']['ndcg@10']:.4f}")
    print(f"head item | recall@10: {bias['head_item']['recall@10']:.4f} , ndcg@10: {bias['head_item']['ndcg@10']:.4f}")
    print(f"tail item | recall@10: {bias['tail_item']['recall@10']:.4f} , ndcg@10: {bias['tail_item']['ndcg@10']:.4f}")
    print(f"head user | recall@10: {bias['head_user']['recall@10']:.4f} , ndcg@10: {bias['head_user']['ndcg@10']:.4f}")
    print(f"tail user | recall@10: {bias['tail_user']['recall@10']:.4f} , ndcg@10: {bias['tail_user']['ndcg@10']:.4f}")


def asin2id(file, data):
    args = Namespace(
        data_path= data,
        train_file="train.json",
        dev_file="val.json",
        test_file="test.json",
        item2id_file="smap.json",
        meta_file="meta_data.json",
    )
    train, val, test, item_meta_dict, item2id, id2item = load_data(args)
    user2id = json.load(open(Path(data) / Path('umap.json'), 'r'))
    new_prediction = dict()
    for user, user_prediction in file.items():
        new_prediction[user2id[user]] = {
            'predictions': [item2id[item] for item in user_prediction['predictions']],
            'target': item2id[user_prediction['target']]
        }
    return new_prediction


if __name__ == "__main__":

    file_name = "SASRec_Instruments_42.json"
    model, data, _ = file_name.split('_')
    prediction = json.load(open(Path('../prediction') / Path(file_name), 'r'))
    if model == 'MARS':
        prediction = asin2id(prediction, f"../finetune_data/{data}_ours")
    popularity_bias(data, prediction)

import argparse
from math import log2
from pprint import pprint
from typing import List
import json
from pathlib import Path

Item = int


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


def popularity_bias(model_name, data_name):
    item_split = json.load(open(Path('../finetune_data') / Path(f"{data_name}_ours") / Path('item_split.json'), 'r'))
    user_split = json.load(open(Path('../finetune_data') / Path(f"{data_name}_ours") / Path('user_split.json'), 'r'))
    prediction_path = Path('../prediction') / Path(f"{model_name}") / Path(f"{data_name}") / Path(
        'predictions.json')
    prediction = json.load(open(prediction_path, 'r'))

    head_item_prediction, head_target = [], []
    tail_item_prediction, tail_target = [], []
    head_user_prediction, head_user_target = [], []
    tail_user_prediction, tail_user_target = [], []

    for p, t, u in zip(prediction['predictions'], prediction['targets'], prediction['users']):
        if t in item_split['head']:
            head_item_prediction.append(p)
            head_target.append(t)
        else:
            tail_item_prediction.append(p)
            tail_target.append(t)

        if u in user_split['head']:
            head_user_prediction.append(p)
            head_user_target.append(t)
        else:
            tail_user_prediction.append(p)
            tail_user_target.append(t)

    bias = {
        'overall': {
            'recall@10': recall(prediction['predictions'], prediction['targets'], 10),
            'ndcg@10': ndcg(prediction['predictions'], prediction['targets'], 10),
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
    print(f"model: {model_name}, data: {data_name}")

    print(f"percentage of head item: {len(head_item_prediction) / len(prediction['predictions'])}")
    print(f"percentage of head user: {len(head_user_prediction) / len(prediction['predictions'])}")

    print(f"overall | recall@10: {bias['overall']['recall@10']:.4f} , ndcg@10: {bias['overall']['ndcg@10']:.4f}")
    print(f"head item | recall@10: {bias['head_item']['recall@10']:.4f} , ndcg@10: {bias['head_item']['ndcg@10']:.4f}")
    print(f"tail item | recall@10: {bias['tail_item']['recall@10']:.4f} , ndcg@10: {bias['tail_item']['ndcg@10']:.4f}")
    print(f"head user | recall@10: {bias['head_user']['recall@10']:.4f} , ndcg@10: {bias['head_user']['ndcg@10']:.4f}")
    print(f"tail user | recall@10: {bias['tail_user']['recall@10']:.4f} , ndcg@10: {bias['tail_user']['ndcg@10']:.4f}")

    json.dump(bias,
              open(Path('../prediction') / Path(f"{model_name}") / Path(f"{data_name}") / Path('bias.json'),
                   'w'), indent=1, ensure_ascii=False)


if __name__ == "__main__":

    model_list = ["BERT4Rec", "FDSA", "SASRec", "UniSRec"]
    data = ["Scientific", "Instruments"]

    for model in model_list:
        for d in data:
            popularity_bias(model, d)

from collections import Counter
import argparse
from pathlib import Path

import numpy as np
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="Data directory")
    parser.add_argument("--data_name", required=True, help="Data directory")

    return parser.parse_args()


def gini(item_probability):
    item_probability.sort()
    n = len(item_probability)
    gini_index = 0
    for i, item in enumerate(item_probability):
        gini_index += (2 * (i + 1) - n - 1) * item

    return gini_index / (n - 1)


def shannon_entropy(item_probability):
    return -sum([p * np.log(p) for p in item_probability])


def diversity(prediction):
    diversity_result = dict()
    print("\n\n")
    for cutoff in [10, 20, 50]:
        print("=" * 20)
        print(f"cutoff: {cutoff}")
        item_counter = Counter()
        for user_id, user_prediction in prediction.items():
            item_counter.update(user_prediction['predictions'][:cutoff])

        item_probability = [v / sum(item_counter.values()) for v in item_counter.values()]

        gini_index = gini(item_probability)
        entropy = shannon_entropy(item_probability)

        print(f"gini: {gini_index}")
        print(f"entropy: {entropy}")

        diversity_result[cutoff] = {"gini": gini_index, "shannon_entropy": entropy}


if __name__ == "__main__":
    file_name = "MARS_Pantry_1398.json"
    prediction_path = Path('../prediction') / Path(file_name)
    prediction = json.load(open(prediction_path, 'r'))
    print(f"prediction: {prediction_path}")
    diversity(prediction)

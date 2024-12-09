import json
import os
from collections import namedtuple
from sys import argv

Dataset = namedtuple("Dataset", ["name", "train", "val", "test", "smap", "umap", "meta_data", "cum"])

dataset_names = argv[1:]


def load_dataset(dataset: str):
    with open(os.path.join(dataset, "train.json"), "r") as f:
        train = json.load(f)
        train = {int(k): v for k, v in train.items()}

    with open(os.path.join(dataset, "val.json"), "r") as f:
        val = json.load(f)
        val = {int(k): v for k, v in val.items()}

    with open(os.path.join(dataset, "test.json"), "r") as f:
        test = json.load(f)
        test = {int(k): v for k, v in test.items()}

    with open(os.path.join(dataset, "smap.json"), "r") as f:
        smap = json.load(f)

    with open(os.path.join(dataset, "umap.json"), "r") as f:
        umap = json.load(f)

    with open(os.path.join(dataset, "meta_data.json"), "r") as f:
        meta_data = json.load(f)

    return Dataset(dataset, train, val, test, smap, umap, meta_data, 0)


datasets = [load_dataset(dataset) for dataset in dataset_names]

# Check overlapping users
from functools import reduce

print("Overlapping items:", len(reduce(lambda x, y: x & y, [set(dataset.smap.keys()) for dataset in datasets])))
print("Overlapping users:", len(reduce(lambda x, y: x & y, [set(dataset.umap.keys()) for dataset in datasets])))

# Merge datasets
new_user_maps = [{} for _ in datasets]
new_item_maps = [{} for _ in datasets]
smap_joined = {}
umap_joined = {}
meta_data_joined = {}

cum = 0
for i, dataset in enumerate(datasets):
    for asin, num_item_id in sorted(dataset.smap.items(), key=lambda x: x[1]):
        smap_joined[asin + f"_{dataset.name}"] = cum
        new_item_maps[i][num_item_id] = cum
        cum += 1
        meta_data_joined[asin + f"_{dataset.name}"] = dataset.meta_data[asin]

cum = 0
for i, dataset in enumerate(datasets):
    for uid, num_user_id in sorted(dataset.umap.items(), key=lambda x: x[1]):
        umap_joined[uid + f"_{dataset.name}"] = cum
        new_user_maps[i][num_user_id] = cum
        cum += 1

train_joined = {}
val_joined = {}
val_separate = []
test_joined = {}
test_separate = []

for i, dataset in enumerate(datasets):
    train = {}
    for num_user_id, num_item_ids in dataset.train.items():
        train[new_user_maps[i][num_user_id]] = [new_item_maps[i][num_item_id] for num_item_id in num_item_ids]

    val = {}
    for num_user_id, num_item_ids in dataset.val.items():
        val[new_user_maps[i][num_user_id]] = [new_item_maps[i][num_item_id] for num_item_id in num_item_ids]

    test = {}
    for num_user_id, num_item_ids in dataset.test.items():
        test[new_user_maps[i][num_user_id]] = [new_item_maps[i][num_item_id] for num_item_id in num_item_ids]

    train_joined.update(train)
    val_joined.update(val)
    test_joined.update(test)

    val_separate.append(val)
    test_separate.append(test)

os.makedirs(output_path := "_".join(dataset_names), exist_ok=True)

with open(os.path.join(output_path, "smap.json"), "w") as f:
    json.dump(smap_joined, f, indent=1)

with open(os.path.join(output_path, "umap.json"), "w") as f:
    json.dump(umap_joined, f, indent=1)

with open(os.path.join(output_path, "new_user_maps.json"), "w") as f:
    json.dump(new_user_maps, f, indent=1)

with open(os.path.join(output_path, "new_item_maps.json"), "w") as f:
    json.dump(new_item_maps, f, indent=1)

with open(os.path.join(output_path, "meta_data.json"), "w") as f:
    json.dump(meta_data_joined, f, indent=1)

with open(os.path.join(output_path, "train.json"), "w") as f:
    json.dump(train_joined, f, indent=1)

with open(os.path.join(output_path, "val_joined.json"), "w") as f:
    json.dump(val_joined, f, indent=1)

with open(os.path.join(output_path, "val.json"), "w") as f:
    json.dump(val_separate, f, indent=1)

with open(os.path.join(output_path, "test_joined.json"), "w") as f:
    json.dump(test_joined, f, indent=1)

with open(os.path.join(output_path, "test.json"), "w") as f:
    json.dump(test_separate, f, indent=1)

with open(os.path.join(output_path, "join_info.json"), "w") as f:
    json.dump({"datasets": dataset_names, "user_count": [len(umap) for umap in new_user_maps]}, f, indent=1)

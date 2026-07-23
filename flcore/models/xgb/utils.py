import json
import os
import uuid
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import xgboost as xgb
from flwr.common import (
    NDArray,
    bytes_to_ndarray,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.typing import Parameters
from matplotlib import pyplot as plt  # pylint: disable=E0401
from xgboost import XGBClassifier, XGBRegressor

from flcore.metrics import calculate_metrics


class TreeDataset:
    """Plain container for tabular / tree-encoded data.

    Replaces the torch Dataset. Exposes `.data` (n, features) and `.labels` (n,)
    as NumPy arrays. Kept indexable for any legacy callers.
    """

    def __init__(self, data: NDArray, labels: NDArray) -> None:
        self.data = np.asarray(data)
        self.labels = np.asarray(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return {0: self.data[idx], 1: self.labels[idx]}


def get_dataloader(
    dataset: TreeDataset, partition: str, batch_size: Union[int, str]
) -> TreeDataset:
    """Previously wrapped a torch DataLoader. Training was always full-batch
    (`batch_size == "whole"`), so we just return the dataset container."""
    return dataset


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def do_fl_partitioning(
    trainset: TreeDataset,
    testset: TreeDataset,
    pool_size: int,
    batch_size: Union[int, str],
    val_ratio: float = 0.0,
) -> Tuple[List[TreeDataset], Optional[List[TreeDataset]], TreeDataset]:
    """Split the training set into `pool_size` partitions (numpy, torch-free)."""
    rng = np.random.default_rng(0)
    n = len(trainset)
    idx = rng.permutation(n)
    partition_size = n // pool_size

    trainloaders: List[TreeDataset] = []
    val_list: List[TreeDataset] = []
    start = 0
    for p in range(pool_size):
        end = n if p == pool_size - 1 else start + partition_size
        part_idx = idx[start:end]
        start = end

        len_val = int(len(part_idx) * val_ratio)
        if len_val > 0:
            v_idx, t_idx = part_idx[:len_val], part_idx[len_val:]
            trainloaders.append(
                get_dataloader(
                    TreeDataset(trainset.data[t_idx], trainset.labels[t_idx]),
                    "train",
                    batch_size,
                )
            )
            val_list.append(
                get_dataloader(
                    TreeDataset(trainset.data[v_idx], trainset.labels[v_idx]),
                    "val",
                    batch_size,
                )
            )
        else:
            trainloaders.append(
                get_dataloader(
                    TreeDataset(trainset.data[part_idx], trainset.labels[part_idx]),
                    "train",
                    batch_size,
                )
            )

    valloaders = val_list if val_list else None
    testloader = get_dataloader(testset, "test", batch_size)
    return trainloaders, valloaders, testloader


def plot_xgbtree(tree: Union[XGBClassifier, XGBRegressor], n_tree: int) -> None:
    """Visualize the built xgboost tree."""
    xgb.plot_tree(tree, num_trees=n_tree)
    plt.rcParams["figure.figsize"] = [50, 10]
    plt.show()


def construct_tree(
    dataset: NDArray, label: NDArray, n_estimators: int, tree_type: str
) -> Union[XGBClassifier, XGBRegressor]:
    """Construct a xgboost tree from a tabular dataset."""
    tree = get_tree(n_estimators, tree_type)
    tree.fit(dataset, label)
    return tree


def get_tree(n_estimators: int, tree_type: str) -> Union[XGBClassifier, XGBRegressor]:
    """Instantiate XGBoost model."""
    if tree_type == "REG":
        tree = xgb.XGBRegressor(
            objective="reg:squarederror",
            learning_rate=0.1,
            max_depth=8,
            n_estimators=n_estimators,
            subsample=0.8,
            colsample_bylevel=1,
            colsample_bynode=1,
            colsample_bytree=1,
            alpha=5,
            gamma=5,
            num_parallel_tree=1,
            min_child_weight=1,
        )
    else:
        if tree_type == "BINARY":
            objective = "binary:logistic"
        elif tree_type == "MULTICLASS":
            objective = "multi:softprob"
        else:
            raise ValueError("Unknown tree type.")

        tree = xgb.XGBClassifier(
            objective=objective,
            learning_rate=0.1,
            max_depth=8,
            n_estimators=n_estimators,
            subsample=0.8,
            colsample_bylevel=1,
            colsample_bynode=1,
            colsample_bytree=1,
            alpha=5,
            gamma=5,
            num_parallel_tree=1,
            min_child_weight=1,
            scale_pos_weight=50,
        )

    return tree


def construct_tree_from_loader(
    dataset_loader: TreeDataset, n_estimators: int, tree_type: str
) -> Union[XGBClassifier, XGBRegressor]:
    """Construct a xgboost tree from a dataset container."""
    return construct_tree(
        dataset_loader.data, dataset_loader.labels, n_estimators, tree_type
    )


def single_tree_prediction(
    tree: Union[XGBClassifier, XGBRegressor], n_tree: int, dataset: NDArray
) -> Optional[NDArray]:
    """Extract the prediction result of a single tree in the xgboost tree
    ensemble."""
    # How to access a single tree
    # https://github.com/bmreiniger/datascience.stackexchange/blob/master/57905.ipynb
    num_t = len(tree.get_booster().get_dump())
    if n_tree > num_t:
        print(
            "The tree index to be extracted is larger than the total number of trees."
        )
        return None

    return tree.predict(  # type: ignore
        dataset, iteration_range=(n_tree, n_tree + 1), output_margin=True
    )


def tree_encoding(  # pylint: disable=R0914
    trainloader: TreeDataset,
    client_trees: Union[
        Tuple[XGBClassifier, int],
        Tuple[XGBRegressor, int],
        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
    ],
    client_tree_num: int,
    client_num: int,
) -> Optional[Tuple[NDArray, NDArray]]:
    """Transform the tabular dataset into prediction results using the
    aggregated xgboost tree ensembles from all clients.

    Returns (X_enc, y) as NumPy arrays:
      X_enc: float32, shape (n_samples, client_num * client_tree_num)
      y:     float32, shape (n_samples,)
    """
    if trainloader is None:
        return None

    x_train, y_train = trainloader.data, trainloader.labels

    x_train_enc = np.zeros(
        (x_train.shape[0], client_num * client_tree_num), dtype=np.float32
    )

    temp_trees: Any = None
    if isinstance(client_trees, list) is False:
        temp_trees = [client_trees[0]] * client_num
    elif isinstance(client_trees, list) and len(client_trees) != client_num:
        temp_trees = [client_trees[0][0]] * client_num
    else:
        cids = []
        temp_trees = []
        for i, _ in enumerate(client_trees):
            temp_trees.append(client_trees[i][0])  # type: ignore
            cids.append(client_trees[i][1])  # type: ignore
        sorted_index = np.argsort(np.asarray(cids))
        temp_trees = np.asarray(temp_trees)[sorted_index]

    for i, _ in enumerate(temp_trees):
        for j in range(client_tree_num):
            predictions = single_tree_prediction(temp_trees[i], j, x_train)
            if len(predictions.shape) != 1:
                predictions = np.argmax(predictions, 1)
            x_train_enc[:, i * client_tree_num + j] = predictions

    x_train_enc = x_train_enc.astype(np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).ravel()
    return x_train_enc, y_train


def tree_encoding_loader(
    dataloader: TreeDataset,
    batch_size: int,
    client_trees: Union[
        Tuple[XGBClassifier, int],
        Tuple[XGBRegressor, int],
        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
    ],
    client_tree_num: int,
    client_num: int,
) -> Optional[TreeDataset]:
    encoding = tree_encoding(dataloader, client_trees, client_tree_num, client_num)
    if encoding is None:
        return None
    data, labels = encoding
    tree_dataset = TreeDataset(data, labels)
    return get_dataloader(tree_dataset, "tree", batch_size)


def serialize_objects_to_parameters(objects_list: List, tmp_dir="") -> Parameters:
    net_weights = objects_list[0]
    if type(net_weights) is Parameters:
        net_weights = parameters_to_ndarrays(net_weights)
    net_json = json.dumps(net_weights, cls=NumpyEncoder)

    if type(objects_list[1]) is list:
        trees_json = []
        cids = []
        for tree, cid in objects_list[1]:
            trees_json.append(tree_to_json(tree, tmp_dir))
            cids.append(cid)
        tree_json = trees_json
        cid = cids
    else:
        tree_json = tree_to_json(objects_list[1][0], tmp_dir)
        cid = objects_list[1][1]

    parameters = ndarrays_to_parameters([net_json, tree_json, cid])

    return parameters


def parameters_to_objects(parameters: Parameters, tree_config_dict, tmp_dir="") -> List:
    # Begin data deserialization
    weights_binary = parameters.tensors[0]
    tree_binary = parameters.tensors[1]
    cid_binary = parameters.tensors[2]

    weights_json = bytes_to_ndarray(weights_binary)
    tree_json = bytes_to_ndarray(tree_binary)
    cid_data = bytes_to_ndarray(cid_binary)

    weights_json = json.loads(str(weights_json))
    weights_array = [np.asarray(layer_weights) for layer_weights in weights_json]
    weights_parameters = ndarrays_to_parameters(weights_array)

    client_tree_num = tree_config_dict["client_tree_num"]
    task_type = tree_config_dict["task_type"]

    if len(tree_json.shape) != 0:
        trees = []
        cids = []
        for tree_from_ensemble, cid in zip(tree_json, cid_data):
            cids.append(cid)
            trees.append(
                json_to_tree(tree_from_ensemble, client_tree_num, task_type, tmp_dir)
            )
        tree_parameters = [(tree, cid) for tree, cid in zip(trees, cids)]
    else:
        cid = int(cid_data.item())
        tree = json_to_tree(tree_json, client_tree_num, task_type, tmp_dir)
        tree_parameters = (tree, cid)

    return [weights_parameters, tree_parameters]


def tree_to_json(tree, tmp_directory=""):
    tmp_path = os.path.join(tmp_directory, str(uuid.uuid4()) + ".json")
    tree.get_booster().save_model(tmp_path)
    with open(tmp_path, "r") as fr:
        tree_params_obj = json.load(fr)
        tree_json = json.dumps(tree_params_obj)
    os.remove(tmp_path)

    return tree_json


def json_to_tree(tree_json, client_tree_num, task_type, tmp_directory=""):
    tree_json = json.loads(str(tree_json))
    tmp_path = os.path.join(tmp_directory, str(uuid.uuid4()) + ".json")
    with open(tmp_path, "w") as fw:
        json.dump(tree_json, fw)
    tree = get_tree(
        client_tree_num,
        task_type,
    )
    tree.load_model(tmp_path)
    os.remove(tmp_path)

    return tree


def train_test(data, client_tree_num):
    (X_train, y_train), (X_test, y_test) = data

    X_train.flags.writeable = True
    y_train.flags.writeable = True
    X_test.flags.writeable = True
    y_test.flags.writeable = True

    print("Size of the trainset:", X_train.shape[0])
    print("Size of the testset:", X_test.shape[0])
    assert X_train.shape[1] == X_test.shape[1]

    # Try to automatically determine the type of task
    n_classes = np.unique(y_train).shape[0]
    if n_classes == 2:
        task_type = "BINARY"
    elif n_classes > 2 and n_classes < 100:
        task_type = "MULTICLASS"
    else:
        task_type = "REG"

    if task_type == "BINARY":
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0

    # Build global XGBoost tree for comparison
    global_tree = construct_tree(X_train, y_train, client_tree_num, task_type)
    preds_test = global_tree.predict(X_test)

    metrics = calculate_metrics(y_test, preds_test, task_type)
    return metrics
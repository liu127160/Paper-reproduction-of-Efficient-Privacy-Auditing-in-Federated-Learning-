# TODO: Utilities functionsd
import os
import pickle
from pathlib import Path
import numpy as np
import scipy
import torch
import torch.nn as nn
from functorch import grad, make_functional_with_buffers, vmap
from model import Net
from scipy.special import softmax
import sklearn.metrics as metrics
import pandas as pd
from sklearn.metrics import auc, roc_curve


def load_pkl(path):
    with open(path, "rb") as f:
        print(f"file is loaded from {path}")
        all_data = pickle.load(f)

    return all_data


def save_pkl(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"file is saved to {path}")


def config_dir(args):
    if args.fl.partitioning == "iid":
        partition_detail = f"train_num-{args.data.train_num}__pop_num-{args.data.pop_num}"
    elif args.fl.partitioning in ("dirichlet", "dirichletk"):
        partition_detail = (
            f"alpha-{args.fl.dirichlet_alpha}"
            f"__train_num-{args.data.train_num}__pop_num-{args.data.pop_num}"
        )
    elif args.fl.partitioning == "iidk" or "new_iidk" in args.fl.partitioning:
        partition_detail = (
            f"train_ratio-{args.data.train_ratio}__pop_ratio-{args.data.pop_ratio}"
        )
    else:
        raise ValueError(f"Unsupported partitioning: {args.fl.partitioning}")

    experiment_name = (
        f"partition-{args.fl.partitioning}__{partition_detail}"
        f"__clients-{args.fl.num_parties}__seed-{args.random_seed}"
        f"__optimizer-{args.fl.client_optimizer}__model-{args.fl.model}"
        f"__augment-{args.fl.data_argu}__local_epochs-{args.fl.epochs}"
        f"__batch-{args.fl.batch_size}__lr-{args.fl.learning_rate}"
        f"__wd-{args.fl.weight_decay}__rounds-{args.fl.comm_round}"
    )
    log_dir = (
        f"{args.server_dir_path}/results/{args.fl.federated_optimizer}"
        f"/{args.data.dataset}/{experiment_name}"
    )
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return log_dir

def get_save_dir(args, random_seed=None, dataset=None, is_train=False):
    if random_seed is None:
        random_seed = args.random_seed
    if dataset is None:
        dataset = args.dataset

    if args.partitioning == "iid":
        partition_detail = f"train_num-{args.train_num}__pop_num-{args.pop_num}"
    elif args.partitioning in ("dirichlet", "dirichletk"):
        partition_detail = (
            f"alpha-{args.dirichlet_alpha}"
            f"__train_num-{args.train_num}__pop_num-{args.pop_num}"
        )
    elif args.partitioning == "iidk" or "new_iidk" in args.partitioning:
        partition_detail = (
            f"train_ratio-{args.train_ratio}__pop_ratio-{args.pop_ratio}"
        )
    else:
        raise ValueError(f"Unsupported partitioning: {args.partitioning}")

    experiment_name = (
        f"partition-{args.partitioning}__{partition_detail}"
        f"__clients-{args.num_parties}__seed-{random_seed}"
        f"__optimizer-{args.client_optimizer}__model-{args.model}"
        f"__augment-{args.data_argu}__local_epochs-{args.epochs}"
        f"__batch-{args.batch_size}__lr-{args.learning_rate}"
        f"__wd-{args.weight_decay}__rounds-{args.comm_round}"
    )
    log_dir = (
        f"{args.server_dir_path}/results/{args.federated_optimizer}"
        f"/{dataset}/{experiment_name}"
    )

    if os.path.exists(f"{log_dir}/global_at_0.pkl") and is_train:
        raise ValueError(f"the log_dir {log_dir} already exists")

    print("Information will be saved into ", log_dir)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return log_dir

class AverageMetric:
    def __init__(self, init_value=0):
        self.value = init_value
        self.average = init_value
        self.size = 0
        self.all_values = []

    def add(self, v):
        # v is 1-d
        self.value += v
        self.size += 1
        self.average = self.value / self.size

    def add_list(self, v_list):
        self.value += np.sum(v_list, axis=1)
        self.size += v_list.shape[1]
        self.average = self.value / self.size

    def set_list(self, v_list):
        # set the all values based on v_list
        self.all_values = v_list
        self.value = np.sum(v_list, axis=1)
        self.size = v_list.shape[1]
        self.average = self.value / self.size

    def get(self):
        return self.average

    def update(self, v):
        self.all_values.append(v)

    def get_list(self):
        return np.stack(self.all_values, axis=0)



def compute_p_value(target_value, observed_distribution):
    # Handle arrays element-wise
    # return scipy.stats.percentileofscore(observed_distribution, target_value) / 100
    p_values = np.array([scipy.stats.percentileofscore(observed_distribution, x) / 100 for x in target_value])
    return p_values


def get_low(tpr_list, fpr_list, fpr_t=0.01):
    tpr_list = np.array(tpr_list)
    fpr_list = np.array(fpr_list)
    low = tpr_list[np.where(fpr_list < fpr_t)[0][-1]]
    tfpr = fpr_list[np.where(fpr_list < fpr_t)[0][-1]]
    return low, tfpr



def get_roc(train, test):
    all_t = np.concatenate([train, test])
    all_t = np.append(all_t, [np.max(all_t) + 0.0001, np.min(all_t) - 0.0001])
    all_t.sort()
    tpr = []
    fpr = []
    for t in all_t:
        tpr.append(np.sum(train < t) / len(train))
        fpr.append(np.sum(test < t) / len(test))
    return tpr, fpr, metrics.auc(fpr, tpr)


def population_attack(train, test, pop):
    """
    Compute the p value given the population distribution and then perform.
    """
    train_p = compute_p_value(train, pop)
    test_p = compute_p_value(test, pop)
    return get_roc(train_p, test_p)


def get_list_dict(args, random_seed):
    # get the hydra config and return the list dict
    if args.fl.partitioning == "iid":
        list_dict = load_pkl(
            f"{args.server_dir_path}/{args.data.dataset}/iid_{args.data.train_num}_{args.data.pop_num}_{random_seed}.pkl"
        )
    elif args.fl.partitioning == "dirichlet":
        data_partitioning = f"{args.server_dir_path}/{args.data.dataset}/{args.fl.partitioning}_{args.fl.dirichlet_alpha}_{args.data.train_num}_{args.data.pop_num}_{random_seed}.pkl"
        list_dict = load_pkl(data_partitioning)
    elif args.fl.partitioning == "iidk" or "new_iidk" in args.fl.partitioning:
        data_partitioning = f"{args.server_dir_path}/{args.data.dataset}/{args.fl.partitioning}_{args.data.train_ratio}_{args.data.pop_ratio}_{args.fl.num_parties}_{random_seed}.pkl"
        list_dict = load_pkl(data_partitioning)
    elif args.fl.partitioning == "dirichletk":
        data_partitioning = f"{args.server_dir_path}/{args.data.dataset}/{args.fl.partitioning}_{args.fl.dirichlet_alpha}_{args.data.train_ratio}_{args.data.pop_ratio}_{args.fl.num_parties}_{random_seed}.pkl"
        list_dict = load_pkl(data_partitioning)
    return list_dict


def get_dataset_classes(args):
    if args.data.dataset == "cifar10":
        num_classes = 10
    elif args.data.dataset == "cifar100":
        num_classes = 100
    elif args.data.dataset == "purchase":
        num_classes = 100
    elif "texas" in args.data.dataset:
        num_classes = 100
    elif args.data.dataset == "medicalmnist":
        num_classes = 6
    elif args.data.dataset == "pneumonia":
        num_classes = 2
    elif args.data.dataset == "retinal":
        num_classes = 4
    elif args.data.dataset == "skin":
        num_classes = 23
    elif args.data.dataset == "kidney":
        num_classes = 4
    else:
        raise ValueError(f"dataset {args.dataset} not supported")
    return num_classes


def get_trend_slope(y):
    y = y.T
    # y should be trend* number of samples
    x = np.arange(1, y.shape[0] + 1)
    X = np.repeat(x[:, np.newaxis], y.shape[1], axis=1)
    slope = np.sum(
        (X - np.mean(X, axis=0)) * (y - np.mean(y, axis=0)), axis=0
    ) / np.sum((X - np.mean(X, axis=0)) ** 2, axis=0)
    return slope

def save_info_results_combined(
    args, alg, random_seed, client_idx, auc, train_acc, test_acc, train_loss, test_loss, low1, low5, low10, low20, low50
):
    all_dict = {}
    all_dict["dataset"] = args.data.dataset
    all_dict["num_parties"] = args.fl.num_parties
    all_dict["train_num"] = args.data.train_num
    all_dict["partitioning"] = args.fl.partitioning
   
    all_dict["dirichlet_alpha"] = args.fl.dirichlet_alpha
    all_dict["epochs"] = args.fl.epochs
    all_dict["batch_size"] = args.fl.batch_size
    all_dict["client_optimizer"] = args.fl.client_optimizer
    all_dict["federated_optimizer"] = args.fl.federated_optimizer
    all_dict["model"] = args.fl.model
    all_dict["learning_rate"] = args.fl.learning_rate
    all_dict["weight_decay"] = args.fl.weight_decay
    all_dict["data_argu"] = args.fl.data_argu
    all_dict["alg"] = alg
    all_dict["start_round"] = args.audit.start_round
    all_dict["target_round"] = args.audit.target_round
    all_dict["random_seed"] = random_seed
    all_dict["client_idx"] = client_idx
    all_dict["train_acc"] = train_acc
    all_dict["test_acc"] = test_acc
    all_dict["train_loss"] = train_loss
    all_dict["test_loss"] = test_loss
    all_dict["auc"] = auc
    all_dict["low_01"] = low1
    all_dict["low_05"] = low5
    all_dict["low_1"] = low10
    all_dict["low_2"] = low20
    all_dict["low_5"] = low50
    all_dict["target_model_type"] = args.audit.target_model_type
    all_dict["comm_round"] = args.fl.comm_round

    df_to_append = pd.DataFrame([all_dict])
    result_file = Path(config_dir(args)) / (
        f"audit_{args.audit.target_model_type}"
        f"_from_R{args.audit.start_round + 1}"
        f"_to_R{args.audit.target_round + 1}.csv"
    )
    df_to_append.to_csv(
        result_file,
        mode="a",
        header=not result_file.exists(),
        index=False,
    )
    print(f"Auditing result is saved to {result_file}")
    
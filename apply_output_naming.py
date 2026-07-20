"""Apply the result-naming-only update to utilies.py.

Run this script from the project root after first restoring the supplied
backup.  It changes no model, training, or auditing calculation.
"""

from pathlib import Path
import re
import sys


target = Path("utilies.py")
if not target.is_file():
    sys.exit("Run this script from the project root (the folder containing utilies.py).")

source = target.read_text(encoding="utf-8")
if "experiment_name" in source:
    sys.exit("The naming update already appears to be applied; no change was made.")

new_config_dir = '''def config_dir(args):
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
'''

new_get_save_dir = '''def get_save_dir(args, random_seed=None, dataset=None, is_train=False):
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
'''

source, config_count = re.subn(
    r"def config_dir\(args\):.*?(?=\n\ndef get_save_dir)",
    new_config_dir.rstrip(),
    source,
    flags=re.DOTALL,
)
source, save_dir_count = re.subn(
    r"def get_save_dir\(args, random_seed=None, dataset=None, is_train=False\):.*?(?=\n\nclass AverageMetric)",
    new_get_save_dir.rstrip(),
    source,
    flags=re.DOTALL,
)

old_tail = '''    all_dict["target_model_type"] = args.audit.target_model_type

    df_to_append = pd.DataFrame([all_dict])
    df_to_append.to_csv(
        "auditing_results.csv",
        mode="a",
        header=not pd.io.common.file_exists("auditing_results.csv"),
        index=False,
    )
'''
new_tail = '''    all_dict["target_model_type"] = args.audit.target_model_type
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
'''

if config_count != 1 or save_dir_count != 1 or old_tail not in source:
    sys.exit(
        "utilies.py does not match the expected original version. "
        "Restore utilies.py from utilies.py.before_output_naming, then rerun this script."
    )

target.write_text(source.replace(old_tail, new_tail), encoding="utf-8")
print("Updated utilies.py: experiment folders and audit CSV names are now parameter-specific.")

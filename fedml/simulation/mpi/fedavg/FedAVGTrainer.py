from .utils import transform_tensor_to_list
import csv
import pickle
import os
import time

import jsonlines


def append_to_jsonl(file_path, new_data):
    with jsonlines.open(file_path, mode="a") as writer:
        writer.write(new_data)


class FedAVGTrainer(object):
    def __init__(
        self,
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    ):
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None

        self.device = device
        self.args = args

    def _synchronize_device(self):
        """Wait for queued CUDA work so elapsed wall time includes GPU work."""
        try:
            import torch

            if self.device is not None and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except Exception:
            # Timing must never interrupt a federated-learning run.
            pass

    def _write_timing(self, round_idx, stage, seconds):
        """Append one timing record for this client to its experiment folder."""
        file_path = f"{self.args.save_dir}/{self.client_index}_timing.csv"
        is_new_file = not os.path.exists(file_path)
        with open(file_path, "a", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["round", "stage", "seconds", "client_idx"],
            )
            if is_new_file:
                writer.writeheader()
            writer.writerow(
                {
                    "round": round_idx,
                    "stage": stage,
                    "seconds": f"{seconds:.6f}",
                    "client_idx": self.client_index,
                }
            )

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index

        if self.train_data_local_dict is not None:
            self.train_local = self.train_data_local_dict[client_index]
        else:
            self.train_local = None

        if self.train_data_local_num_dict is not None:
            self.local_sample_number = self.train_data_local_num_dict[client_index]
        else:
            self.local_sample_number = 0

        if self.test_data_local_dict is not None:
            self.test_local = self.test_data_local_dict[client_index]
        else:
            self.test_local = None

    def train(self, round_idx=None):
        self.args.round_idx = round_idx
        self._synchronize_device()
        start_time = time.perf_counter()
        self.trainer.train(self.train_local, self.device, self.args)
        self._synchronize_device()
        self._write_timing(round_idx, "local_train", time.perf_counter() - start_time)

        weights = self.trainer.get_model_params()
        return weights, self.local_sample_number

    def test(self, round_idx=None, is_global=False):
        # train data
        train_metrics = self.trainer.test(self.train_local, self.device, self.args)
        train_tot_correct, train_num_sample, train_loss = (
            train_metrics["test_correct"],
            train_metrics["test_total"],
            train_metrics["test_loss"],
        )

        # test data
        self._synchronize_device()
        start_time = time.perf_counter()
        test_metrics = self.trainer.test(
            self.test_local, self.device, self.args, is_test=True
        )
        self._synchronize_device()
        stage = "snapshot_eval_global" if is_global else "snapshot_eval_local"
        self._write_timing(round_idx, stage, time.perf_counter() - start_time)
        test_tot_correct, test_num_sample, test_loss = (
            test_metrics["test_correct"],
            test_metrics["test_total"],
            test_metrics["test_loss"],
        )

        print(test_metrics.keys())
        if (
            self.args.save_models
            and (
                self.args.save_client == "all"
                or self.client_index == self.args.save_client
            )
            and "save_information" in test_metrics
        ):
            file_path = (
                f"{self.args.save_dir}/{self.client_index}_global_pred.jsonl"
                if is_global
                else f"{self.args.save_dir}/{self.client_index}_local_pred.jsonl"
            )

            data_to_append = {round_idx: test_metrics["save_information"]}

            # Call the function
            append_to_jsonl(file_path, data_to_append)

        return (
            train_tot_correct,
            train_loss,
            train_num_sample,
            test_tot_correct,
            test_loss,
            test_num_sample,
        )

import csv
import os
from collections import defaultdict

class Logger:
    def __init__(self, run_dir, filename="train.csv"):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)

        self.csv_path = os.path.join(run_dir, filename)
        self.metrics = defaultdict(list)

    def log(self, key, value):
        self.metrics[key].append(float(value))

    def log_dict(self, metric_dict):
        for k, v in metric_dict.items():
            self.log(k, v)

    def summarize(self):
        summary = {}
        for k, values in self.metrics.items():
            if len(values) > 0:
                summary[k] = sum(values) / len(values)
        return summary

    def write(self, step=None, prefix=""):
        summary = self.summarize()

        if step is not None:
            summary = {"step": step, **summary}

        # terminal print
        msg = []
        for k, v in summary.items():
            if isinstance(v, float):
                msg.append(f"{prefix}{k}: {v:.4f}")
            else:
                msg.append(f"{prefix}{k}: {v}")
        print(" | ".join(msg))

        # csv write
        # write header only if file is empty / new
        file_exists = os.path.exists(self.csv_path)
        file_empty = (not file_exists) or (os.path.getsize(self.csv_path) == 0)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())
            if file_empty:
                writer.writeheader()
            writer.writerow(summary)

        self.metrics.clear()
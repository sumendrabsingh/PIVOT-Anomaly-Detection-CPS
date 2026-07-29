import subprocess
import os
import sys

DATASETS = ["hai", "wadi"]  # Goal mode activated, running all datasets

MODELS = [
    ("dae", 32),
    ("vae", 32),
    ("svdd", 32),
    ("transformer", 32)
]

def run_cmd(cmd, name):
    print(f"\n================ {name} ================")
    try:
        subprocess.run(" ".join(cmd), shell=True, check=True)
    except Exception as e:
        print(f"Failed {name}: {e}")

if __name__ == "__main__":
    for ds in DATASETS:
        train_data = rf"Datasets/{ds}_train_preprocessed.csv"
        for m, b in MODELS:
            out_model = rf"Oracles/{ds}_{m}.pth"
            name = f"TRAIN {m.upper()} on {ds.upper()}"
            cmd = [
                "conda", "run", "-p", r"env", "python", "PIVOT_Oracle_Training.py",
                "--data", train_data,
                "--output", out_model,
                "--bottleneck", str(b),
                "--type", m,
                "--epochs", "20",   # Reduced epochs for fast proof of concept
                "--patience", "5",
                "--batch_size", "4096"
            ]
            run_cmd(cmd, name)

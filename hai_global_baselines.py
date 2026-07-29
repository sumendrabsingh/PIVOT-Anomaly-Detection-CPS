import os
import subprocess

MODELS = [("dae", 32), ("vae", 32), ("svdd", 32), ("transformer", 32)]
DATASET = "hai"
TRAIN_DATA = r"Datasets/hai_train_preprocessed.csv"
TEST_DATA = r"Datasets/hai_test_preprocessed.csv"
RESULTS_FILE = r"Oracles/hai_baselines_evaluation_results.txt"

if os.path.exists(RESULTS_FILE):
    os.remove(RESULTS_FILE)

print("=== STARTING HAI GLOBAL BASELINES TRAINING ===")
for m, b in MODELS:
    out_model = rf"Oracles/hai_{m}.pth"
    print(f"\nTraining {m.upper()} on HAI...")
    train_cmd = [
        "conda", "run", "-p", r"env", "python", "PIVOT_Oracle_Training.py",
        "--data", TRAIN_DATA,
        "--output", out_model,
        "--bottleneck", str(b),
        "--type", m,
        "--epochs", "20",
        "--patience", "5",
        "--batch_size", "4096"
    ]
    subprocess.run(" ".join(train_cmd), shell=True, check=True)

print("\n=== STARTING HAI GLOBAL BASELINES EVALUATION ===")
for m, b in MODELS:
    out_model = rf"Oracles/hai_{m}.pth"
    print(f"\nEvaluating {m.upper()} on HAI...")
    eval_cmd = [
        "conda", "run", "-p", r"env", "python", "PIVOT_Oracle_Evaluation.py",
        "--data", TEST_DATA,
        "--model", out_model,
        "--threshold_mode", "train_p99"
    ]
    with open(RESULTS_FILE, 'a') as f:
        f.write(f"\n================ {m} ================\n")
        subprocess.run(" ".join(eval_cmd), shell=True, stdout=f, stderr=subprocess.STDOUT)

print("Done! Results saved to", RESULTS_FILE)

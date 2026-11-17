import subprocess
import os

DATASETS = ["hai", "wadi"]
MODELS = ["dae", "vae", "svdd", "transformer"]

# SWaT was already done, but can include if needed
# DATASETS.append("swat")

results_file = r"Oracles/global_baselines_evaluation_results.txt"

def run_evaluation(dataset, model_type):
    model_path = rf"Oracles/{dataset}_{model_type}.pth"
    if dataset == "swat":
        test_data = r"Datasets/swat_test_preprocessed.csv"
    else:
        test_data = rf"Datasets/{dataset}_test_preprocessed.csv"
        
    if not os.path.exists(model_path):
        print(f"[SKIP] Model {model_path} not found.")
        return
        
    print(f"\n[{dataset.upper()}] Evaluating {model_type.upper()}...")
    cmd = [
        "conda", "run", "-p", r"env", "python", "PIVOT_Oracle_Evaluation.py",
        "--data", test_data,
        "--model", model_path,
        "--threshold_mode", "train_p99" # Use P99 threshold by default for conservative baseline matching
    ]
    
    with open(results_file, "a") as f:
        f.write(f"\n{'='*60}\n DATASET: {dataset.upper()} | MODEL: {model_type.upper()}\n{'='*60}\n")
        
    try:
        # Run and append output to results file
        with open(results_file, "a") as f:
            subprocess.run(" ".join(cmd), shell=True, check=True, stdout=f, stderr=subprocess.STDOUT)
    except Exception as e:
        print(f"Failed to evaluate {model_type} on {dataset}: {e}")

if __name__ == "__main__":
    if os.path.exists(results_file):
        os.remove(results_file)
        
    for ds in DATASETS:
        for m in MODELS:
            run_evaluation(ds, m)
            
    print(f"\nAll evaluations complete! Results saved to: {results_file}")

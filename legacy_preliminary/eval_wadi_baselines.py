import subprocess
import os

MODELS = ['dae', 'vae', 'svdd', 'transformer']
results_file = r'Oracles/wadi_baselines_evaluation_results.txt'

if os.path.exists(results_file):
    os.remove(results_file)

for m in MODELS:
    model_path = rf'Oracles/wadi_{m}.pth'
    test_data = r'Datasets/wadi_test_preprocessed.csv'
    cmd = ['conda', 'run', '-p', r'env', 'python', 'PIVOT_Oracle_Evaluation.py', '--data', test_data, '--model', model_path, '--threshold_mode', 'train_p99']
    print(f'Running {m}...')
    with open(results_file, 'a') as f:
        f.write(f'\n================ {m} ================\n')
        subprocess.run(" ".join(cmd), shell=True, check=True, stdout=f, stderr=subprocess.STDOUT)
print('Done!')

import os
import pandas as pd
from PIVOT_Adversarial_Evaluation import run_final_evaluation

def run_batch():
    data_path = r'SWaT_Orig_DataSets/Original Dataset Backup/SWaT_Dataset_Attack_v0_Clean.csv'
    
    models = [
        r'Oracles\clean_standard_model.pth',
        r'Oracles\clean_dae_model.pth',
        r'Oracles\clean_vae_model.pth',
        r'Oracles\clean_svdd_model.pth',
        r'Oracles\clean_contrastive_model.pth',
        r'Oracles\clean_transformer_model.pth'
    ]
    
    results = []
    
    print("="*60)
    print(" BATCH ADVERSARIAL EVALUATION STARTING")
    print("="*60)
    
    for model_path in models:
        print(f"\nEvaluating: {model_path}")
        try:
            res = run_final_evaluation(
                data_path=data_path,
                bundle_path=model_path,
                threshold_mode='f1_opt',
                window_size=1,
                align_baseline=False,
                error_mode='mean',
                run_adversarial=True,
                epsilon=0.1,
                alpha=0.02,
                num_iter=10
            )
            
            f1_clean = res['f1_clean']
            f1_adv = res.get('f1_adv', 0.0)
            recall_clean = res['recall_clean']
            recall_adv = res.get('recall_adv', 0.0)
            
            results.append({
                'Model': os.path.basename(model_path),
                'Accuracy (Clean)': res['accuracy_clean'],
                'Accuracy (Adv)': res.get('accuracy_adv', 0.0),
                'Precision (Clean)': res['precision_clean'],
                'Precision (Adv)': res.get('precision_adv', 0.0),
                'Recall (Clean)': recall_clean,
                'Recall (Adv)': recall_adv,
                'F1 (Clean)': f1_clean,
                'F1 (Adv)': f1_adv,
                'False Alarm (Clean)': res['false_alarm_clean'],
                'False Alarm (Adv)': res.get('false_alarm_adv', 0.0)
            })
            print(f"  [SUCCESS] {os.path.basename(model_path)} | Clean F1: {f1_clean:.4f} | Adv F1: {f1_adv:.4f}")
        except Exception as e:
            print(f"  [ERROR] Failed to evaluate {model_path}: {e}")
            
    df_results = pd.DataFrame(results)
    df_results.to_csv('adversarial_benchmark_results.csv', index=False)
    print("\nBatch Evaluation Complete. Results saved to adversarial_benchmark_results.csv")
    print(df_results.to_markdown())

if __name__ == "__main__":
    run_batch()

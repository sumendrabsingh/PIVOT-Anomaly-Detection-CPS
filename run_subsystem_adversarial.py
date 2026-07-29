import os
import pandas as pd
from run_preprocessed_subsystem_experiments import evaluate_subsystems_or_voting

def run_subsystem_batch():
    results = []
    print("="*60)
    print(" SUBSYSTEM ADVERSARIAL EVALUATION STARTING")
    print("="*60)
    
    try:
        # Run Baseline (Clean)
        print("\n--- Running Baseline Subsystem OR-Voting ---")
        res_clean = evaluate_subsystems_or_voting(window_size=1, run_adversarial=False)
        f1_clean = res_clean['f1']
        recall_clean = res_clean['recall']
        
        # Run Adversarial
        print("\n--- Running Adversarial Subsystem OR-Voting ---")
        res_adv = evaluate_subsystems_or_voting(window_size=1, run_adversarial=True, epsilon=0.1, alpha=0.02, num_iter=10)
        f1_adv = res_adv['f1']
        recall_adv = res_adv['recall']
        
        results.append({
            'Model': 'Subsystem_OR_Voting',
            'Accuracy (Clean)': res_clean['accuracy'],
            'Accuracy (Adv)': res_adv['accuracy'],
            'Precision (Clean)': res_clean['precision'],
            'Precision (Adv)': res_adv['precision'],
            'Recall (Clean)': res_clean['recall'],
            'Recall (Adv)': res_adv['recall'],
            'F1 (Clean)': res_clean['f1'],
            'F1 (Adv)': res_adv['f1'],
            'False Alarm (Clean)': res_clean['false_alarm'],
            'False Alarm (Adv)': res_adv['false_alarm']
        })
        print(f"  [SUCCESS] Subsystem_OR_Voting | Clean F1: {f1_clean:.4f} | Adv F1: {f1_adv:.4f}")
    except Exception as e:
        print(f"  [ERROR] Failed to evaluate Subsystem OR Voting: {e}")
        import traceback
        traceback.print_exc()
        
    df_results = pd.DataFrame(results)
    df_results.to_csv('subsystem_adversarial_benchmark_results.csv', index=False)
    print("\nSubsystem Evaluation Complete. Results saved to subsystem_adversarial_benchmark_results.csv")
    print(df_results.to_markdown())

if __name__ == "__main__":
    run_subsystem_batch()

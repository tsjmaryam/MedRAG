import pandas as pd
from config_loader import get_config

cfg = get_config()
cols = cfg['columns']

results_path = './dataset/test_run_results.csv'
ground_truth_path = cfg['paths']['patient_csv']

results_df = pd.read_csv(results_path)
gt_df = pd.read_csv(ground_truth_path, encoding='ISO-8859-1')

correct = 0
total = 0
mismatches = []

for _, row in results_df.iterrows():
    participant_no = str(row['Participant No.'])
    generated = str(row['Generated Diagnosis']).lower()
    gt_row = gt_df[gt_df[cols['participant_id']].astype(str) == participant_no]
    if gt_row.empty:
        continue
    true_diagnosis = str(gt_row[cols['diagnosis']].values[0]).lower().strip()
    total += 1
    if true_diagnosis in generated:
        correct += 1
    else:
        mismatches.append({
            'Participant': participant_no,
            'True': true_diagnosis,
            'Generated': generated[:200]
        })

accuracy = correct / total * 100 if total > 0 else 0
print(f"\nResults: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
print(f"\nMismatches:")
for m in mismatches:
    print(f"  Participant {m['Participant']}: True={m['True']} | Generated={m['Generated'][:100]}")
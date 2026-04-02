import pandas as pd
import json
import os
import ast
from config_loader import get_config

def run_preprocessing(config_path=None):
    cfg = get_config(config_path)
    cols = cfg['columns']
    paths = cfg['paths']

    evidences_path = './zip_contents/release_evidences.json'
    conditions_path = './zip_contents/release_conditions.json'

    with open(evidences_path) as f:
        evidences = json.load(f)
    with open(conditions_path) as f:
        conditions = json.load(f)

    level3_to_level2 = cfg.get('level3_to_level2', {
        'acute_copd_exacerbation_infection': 'respiratory_system',
        'atrial_fibrillation': 'cardiovascular_system',
        'gerd': 'gastrointestinal',
        'anemia': 'hematologic',
        'viral_pharyngitis': 'respiratory_system',
        'inguinal_hernia': 'gastrointestinal',
        'myasthenia_gravis': 'neurological',
        'spontaneous_pneumothorax': 'respiratory_system',
        'cluster_headache': 'neurological',
        'boerhaave': 'gastrointestinal',
        'spontaneous_rib_fracture': 'musculoskeletal',
        'hiv_initial_infection': 'infectious',
        'panic_attack': 'psychiatric',
        'anaphylaxis': 'immunological',
        'sle': 'immunological',
        'sarcoidosis': 'immunological',
        'tuberculosis': 'infectious',
        'influenza': 'infectious',
        'pneumonia': 'respiratory_system',
        'unstable_angina': 'cardiovascular_system',
        'stable_angina': 'cardiovascular_system',
        'possible_nstemi_stemi': 'cardiovascular_system',
        'psvt': 'cardiovascular_system',
        'myocarditis': 'cardiovascular_system',
        'pericarditis': 'cardiovascular_system',
        'pulmonary_embolism': 'respiratory_system',
        'pulmonary_neoplasm': 'respiratory_system',
        'acute_pulmonary_edema': 'respiratory_system',
        'bronchitis': 'respiratory_system',
        'bronchiectasis': 'respiratory_system',
        'bronchiolitis': 'respiratory_system',
        'bronchospasm_acute_asthma_exacerbation': 'respiratory_system',
        'urti': 'respiratory_system',
        'whooping_cough': 'respiratory_system',
        'acute_laryngitis': 'respiratory_system',
        'croup': 'respiratory_system',
        'epiglottitis': 'respiratory_system',
        'acute_otitis_media': 'ear_nose_throat',
        'acute_rhinosinusitis': 'ear_nose_throat',
        'allergic_sinusitis': 'ear_nose_throat',
        'chronic_rhinosinusitis': 'ear_nose_throat',
        'guillain_barre_syndrome': 'neurological',
        'acute_dystonic_reactions': 'neurological',
        'chagas': 'infectious',
        'ebola': 'infectious',
        'scombroid_food_poisoning': 'gastrointestinal',
        'pancreatic_neoplasm': 'gastrointestinal',
    })

    def decode_evidences(evidence_list, evidences_dict):
        symptoms = []
        for ev in evidence_list:
            if '_@_' in ev:
                code, value = ev.split('_@_')
            else:
                code, value = ev, None
            if code in evidences_dict:
                ev_info = evidences_dict[code]
                q = ev_info.get('question_en', '')
                if value and 'value_meaning' in ev_info and value in ev_info['value_meaning']:
                    val_en = ev_info['value_meaning'][value].get('en', value)
                    symptoms.append(f'{q}: {val_en}')
                else:
                    symptoms.append(q)
        return ', '.join(symptoms)

    def process_file(filepath, output_json_dir, output_csv_path):
        df = pd.read_csv(filepath)
        os.makedirs(output_json_dir, exist_ok=True)
        rows = []
        for idx, row in df.iterrows():
            participant_no = str(idx + 1)
            pathology = str(row['PATHOLOGY'])
            pathology_key = pathology.lower().replace(' ', '_').replace('(', '').replace(')', '')
            level2 = level3_to_level2.get(pathology_key, 'general')
            try:
                ev_list = ast.literal_eval(row['EVIDENCES'])
            except:
                ev_list = []
            symptoms_text = decode_evidences(ev_list, evidences)
            csv_row = {
                cols['participant_id']: participant_no,
                'Age': row['AGE'],
                'Sex': row['SEX'],
                cols['pain_location']: symptoms_text[:300],
                cols['pain_symptoms']: symptoms_text[300:600],
                cols['pain_restriction']: '',
                cols['diagnosis']: pathology,
                cols['level2']: level2,
                'Level 1': 'medical'
            }
            rows.append(csv_row)
            json_record = {
                'Participant No.': participant_no,
                'Pain Presentation and Description Areas of pain as per physiotherapy input': symptoms_text[:300],
                'Pain descriptions and assorted symptoms (self-report) Associated symptoms include: parasthesia, numbness, weakness, tingling, pins and needles': symptoms_text[300:600],
                cols['diagnosis']: pathology,
                cols['level2']: level2
            }
            with open(os.path.join(output_json_dir, f'participant_{participant_no}.json'), 'w') as f:
                json.dump(json_record, f, indent=2)
        pd.DataFrame(rows).to_csv(output_csv_path, index=False)
        print(f'Done: {len(rows)} records written to {output_csv_path}')

    process_file(
        paths['train_folder'].replace('/df/train', '') + '/df/train/release_train_patients',
        paths['train_folder'],
        './dataset/AI Data Set with Categories.csv'
    )

    process_file(
        paths['test_folder'].replace('/df/test', '') + '/df/test/release_test_patients',
        paths['test_folder'],
        './dataset/AI Data Set with Categories_test.csv'
    )

if __name__ == '__main__':
    run_preprocessing()
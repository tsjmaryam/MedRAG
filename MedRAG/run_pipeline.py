import os
import json
import numpy as np
from main_MedRAG_v2 import (
    Faiss,
    get_query_embedding,
    extract_features_from_json,
    generate_diagnosis_report,
    save_results_to_csv,
    document_embeddings,
    documents
)
from config_loader import get_config

cfg = get_config()
paths = cfg['paths']

kg_path = paths['kg_file']
folder_path = paths['train_folder']

top_n = 2
match_n = 5
k = 3
model = 'qwen'

results = []

test_patients = [f for f in os.listdir(folder_path) if f.endswith('.json')][:50]

for file_name in test_patients:
    file_path = os.path.join(folder_path, file_name)
    participant_no = file_name.replace('participant_', '').replace('.json', '')

    print(f"Step 1: Extracting features for participant {participant_no}...")
    pain_location, pain_symptoms = extract_features_from_json(file_path)
    query = f"Pain Location: {pain_location}\nPain Symptoms: {pain_symptoms}"

    print("Step 2: Getting query embedding...")
    query_embedding = get_query_embedding(query)

    print("Step 3: Running FAISS search...")
    indices = Faiss(document_embeddings, query_embedding, k)
    retrieved_documents = [documents[idx] for idx in indices[0]]

    print("Step 4: Calling Qwen model...")
    report = generate_diagnosis_report(
        path=kg_path,
        query=query,
        retrieved_documents=retrieved_documents,
        i=participant_no,
        top_n=top_n,
        match_n=match_n,
        model=model
    )
    print("Step 5: Done.")
    print(report)
    results.append([participant_no, report, '', ''])

save_results_to_csv(results, './dataset/test_run_results.csv')
print("\nResults saved.")
import openai
import faiss
import numpy as np
import os
import re
import json
import pandas as pd
from tqdm import tqdm
from huggingface_hub import InferenceClient
from KG_Retrieve_v2 import build_kg_retrieve
from authentication import api_key, hf_token
from config_loader import get_config
import nltk

for pkg in ["punkt_tab", "punkt", "stopwords"]:
    try:
        nltk.data.find(
            f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}"
        )
    except LookupError:
        nltk.download(pkg)

cfg = get_config()
paths = cfg['paths']
cols = cfg['columns']
diagnoses_list = cfg['diagnoses_list']

client = openai.OpenAI(api_key=api_key)

main_get_category_and_level3 = build_kg_retrieve()


from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts):
    return embedding_model.encode(texts, show_progress_bar=True)

def get_query_embedding(query):
    return embedding_model.encode([query])[0]


def Faiss(document_embeddings, query_embedding, k):
    index = faiss.IndexFlatIP(document_embeddings.shape[1])
    index.add(document_embeddings)
    _, indices = index.search(np.array([query_embedding]), k)
    print("index: ", indices)
    return indices


def extract_diagnosis(generated_text):
    return re.findall(r'\*\*Diagnosis\*\*:\s(.*?)\n', generated_text)


def remove_parentheses(text):
    return re.sub(r'\(.*?\)', '', text).strip()


def KG_preprocess(file_path):
    kg_data = pd.read_excel(file_path, usecols=['subject', 'relation', 'object'])
    kg_data['subject'] = kg_data['subject'].apply(remove_parentheses)
    kg_data['object'] = kg_data['object'].apply(remove_parentheses)
    knowledge_graph = {}
    for _, row in kg_data.iterrows():
        subject, relation, obj = row['subject'], row['relation'], row['object']
        knowledge_graph.setdefault(subject, []).append((relation, obj))
        knowledge_graph.setdefault(obj, []).append((relation, subject))
    return knowledge_graph


def extract_features_from_json(file_path):
    with open(file_path, 'r') as file:
        patient_case = json.load(file)
    pain_location = patient_case.get("Pain Presentation and Description Areas of pain as per physiotherapy input", "")
    pain_symptoms = patient_case.get(
        "Pain descriptions and assorted symptoms (self-report) Associated symptoms include: parasthesia, numbness, weakness, tingling, pins and needles", "")
    return pain_location, pain_symptoms


def get_additional_info_from_level_2(participant_no, kg_path, top_n, match_n):
    data = pd.read_csv(paths['patient_csv'], encoding='ISO-8859-1')
    row = data[data[cols['participant_id']].astype(str) == str(participant_no)]
    if row.empty:
        print(f"No data found for Participant No.: {participant_no}")
        return None

    level_2_value = row[cols['level2']].values[0]

    kg_data = pd.read_excel(kg_path, usecols=['subject', 'relation', 'object'])
    if kg_data.empty:
        return None

    relevant_level_3 = [
        desc for desc, lv2 in cfg.get('level3_to_level2', {}).items()
        if lv2 == level_2_value
    ]
    print("Relevant Level 3 Descriptions:", relevant_level_3)
    if not relevant_level_3:
        return None

    merged_info = {}
    for level_3 in relevant_level_3:
        related_info = kg_data[kg_data['subject'] == level_3]
        for _, row2 in related_info.iterrows():
            subject = row2['subject']
            relation = row2['relation'].replace('_', ' ')
            obj = row2['object']
            merged_info.setdefault((subject, relation), []).append(obj)

    additional_info = []
    for (subject, relation), objects in merged_info.items():
        additional_info.append(f"{subject} {relation} {', '.join(objects)}")

    if not additional_info:
        return None
    return ', '.join(additional_info)

def get_system_prompt():
    diagnoses_str = ', '.join(diagnoses_list)
    return f'''
        You are a knowledgeable medical assistant with expertise in pain management.
        Your tasks are:
        1. Analyse the retrieved similar patients cases and knowledge graph to assist with the new patient case.
        2. Output EXACTLY ONE diagnosis. The diagnosis must come from this list: {diagnoses_str}. Do not list multiple diagnoses.
        3. Do not repeat any question more than once.
        4. You are given differences of diagnoses of similar symptoms or pain locations. Read that information as a reference to your diagnostic if applicable.
        5. Do mind the nuance between these factors of similar diagnosis with knowledge graph information and consider it when diagnose new patient information.
        6. Ensure that the recommendations are evidence-based and consider the most recent and effective practices in pain management.
        7. The output should include:
           - "Diagnoses (related to pain)"
           - Explanations of diagnose
           - "Pain/General Physiotherapist Treatments"
           - "Pain Psychologist Treatments"
           - "Pain Medicine Treatments"
        8. In "Diagnoses", only output the diagnosis itself.
        9. Leave Psychologist Treatments blank if not applicable, with text "Not applicable".
        10. If information is needed, guide the doctor to ask further questions.
        11. Follow this structured format:

    ### Diagnoses
    1. **Diagnosis**: Answer.
    2. **Explanations of diagnose**: Answer.

    ### Instructive question
    1. **Questions**: Answer.

    ### Pain/General Physiotherapist Treatments
    1. **Session No.: General Overview**
        - **Specific interventions/treatments**:
        - **Goals**:
        - **Exercises**:
        - **Manual Therapy**:
        - **Techniques**:

    ### Pain Psychologist Treatments(if applicable)
    1. **Treatment 1**:

    ### Pain Medicine Treatments

    ### Recommendations for Further Evaluations
    1. **Evaluation 1**:
    '''


def generate_diagnosis_report(path, query, retrieved_documents, i, top_n, match_n, model):
    system_prompt = get_system_prompt()
    additional_info = get_additional_info_from_level_2(i, path, top_n=top_n, match_n=match_n)
    prompt = f"{query}\nRetrieved Documents: {retrieved_documents}\nInformation from knowledge graph: {additional_info}. Now complete the tasks in that format"

    if model in ('gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo-0125'):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    else:
        LLMclient = InferenceClient("mistralai/Mistral-Nemo-Instruct-2407", token=hf_token)
        response = LLMclient.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800
        )
        return response.choices[0].message.content


def save_results_to_csv(results, output_file):
    df = pd.DataFrame(results, columns=['Participant No.', 'Generated Diagnosis', 'True Diagnosis', 'Original Diagnosis'])
    df.to_csv(output_file, index=False)


folder_path = paths['train_folder']
document_embeddings_file_path = paths['document_embeddings']

documents = [
    os.path.join(folder_path, f) for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]


def save_embeddings(embeddings, file_path):
    np.save(file_path, embeddings)


def load_embeddings(file_path):
    return np.load(file_path)


if os.path.exists(document_embeddings_file_path):
    document_embeddings = load_embeddings(document_embeddings_file_path)
else:
    document_embeddings = get_embeddings(documents)
    save_embeddings(document_embeddings, document_embeddings_file_path)
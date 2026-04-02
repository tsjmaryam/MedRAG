import pandas as pd
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import nltk
import string
import os
from sentence_transformers import SentenceTransformer
from config_loader import get_config

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def build_kg_retrieve(config_path=None):
    cfg = get_config(config_path)
    cols = cfg['columns']
    paths = cfg['paths']
    categories = cfg['categories']

    KG_file_path = paths['kg_file']
    file_path = paths['patient_csv']
    embedding_save_path = paths['kg_embeddings']

    def preprocess_text(text):
        if pd.isna(text):
            return ''
        text = re.sub(r'\(.*?\)', '', text).strip()
        text = text.replace('_', ' ').lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(text.split())

    kg_data = pd.read_excel(KG_file_path, usecols=['subject', 'relation', 'object'])

    knowledge_graph = {}
    for _, row in kg_data.iterrows():
        subject, relation, obj = row['subject'], row['relation'], row['object']
        knowledge_graph.setdefault(subject, []).append((relation, obj))
        knowledge_graph.setdefault(obj, []).append((relation, subject))

    kg_data['object_preprocessed'] = kg_data.apply(
        lambda row: preprocess_text(row['object']) if row['relation'] != 'is_a' else None,
        axis=1
    )
    symptom_nodes = kg_data['object_preprocessed'].dropna().unique().tolist()

    def get_symptom_embeddings(symptom_nodes, save_path):
        embeddings_path = os.path.join(save_path, 'KG_embeddings.npy')
        if os.path.exists(embeddings_path):
            return np.load(embeddings_path)
        os.makedirs(save_path, exist_ok=True)
        symptom_embeddings = embedding_model.encode(symptom_nodes, show_progress_bar=True)
        np.save(embeddings_path, symptom_embeddings)
        return np.array(symptom_embeddings)

    symptom_embeddings = get_symptom_embeddings(symptom_nodes, embedding_save_path)

    G = nx.Graph()
    for node, edges in knowledge_graph.items():
        for relation, neighbor in edges:
            G.add_edge(node, neighbor, relation=relation)

    def find_top_n_similar_symptoms(query, n):
        if pd.isna(query) or not query:
            return []
        query_embedding = embedding_model.encode([preprocess_text(query)])
        emb = symptom_embeddings[:len(symptom_nodes)]
        similarities = cosine_similarity(query_embedding, emb).flatten()
        top_n, seen = [], set()
        for i in similarities.argsort()[::-1]:
            if similarities[i] > 0.5 and symptom_nodes[i] not in seen:
                top_n.append(symptom_nodes[i])
                seen.add(symptom_nodes[i])
            if len(top_n) == n:
                break
        return top_n

    def get_diagnoses_for_symptom(symptom):
        diagnoses = []
        if symptom in G:
            for neighbor in G.neighbors(symptom):
                edge_data = G.get_edge_data(neighbor, symptom)
                if edge_data and edge_data.get('relation') != 'is_a':
                    diagnoses.append(neighbor)
        return diagnoses

    def find_closest_category(top_symptoms, top_n):
        category_votes = {c: 0 for c in categories}
        for symptom in list(set(top_symptoms)):
            if symptom not in G:
                continue
            for diagnosis in get_diagnoses_for_symptom(symptom):
                for single in diagnosis.split(','):
                    single = single.strip().replace(' ', '_').lower()
                    if single not in G:
                        continue
                    min_dist, closest = float('inf'), None
                    for category in categories:
                        if category not in G:
                            continue
                        try:
                            d = nx.shortest_path_length(G, source=single, target=category)
                        except nx.NetworkXNoPath:
                            d = float('inf')
                        if d < min_dist:
                            min_dist, closest = d, category
                    if closest:
                        category_votes[closest] += 1
        sorted_cats = sorted(category_votes.items(), key=lambda x: x[1], reverse=True)
        return [sorted_cats[i][0] for i in range(top_n)]

    def main_get_category_and_level3(n, participant_no, top_n):
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
        row = data.loc[data[cols['participant_id']].astype(str) == str(participant_no)]
        if row.empty:
            print(f"Participant No. {participant_no} not found!")
            return None

        pain_location = row[cols['pain_location']].values[0] or ''
        pain_symptoms = row[cols['pain_symptoms']].values[0] or ''
        pain_restriction = row[cols['pain_restriction']].values[0] or ''

        top_loc = find_top_n_similar_symptoms(pain_location, n)
        top_sym = find_top_n_similar_symptoms(pain_symptoms, n)
        top_res = find_top_n_similar_symptoms(pain_restriction, n)

        top_loc_orig = kg_data.loc[kg_data['object_preprocessed'].isin(top_loc), 'object'].drop_duplicates().tolist()
        top_sym_orig = kg_data.loc[kg_data['object_preprocessed'].isin(top_sym), 'object'].drop_duplicates().tolist()
        top_res_orig = kg_data.loc[kg_data['object_preprocessed'].isin(top_res), 'object'].drop_duplicates().tolist()

        return find_closest_category(top_loc_orig + top_sym_orig + top_res_orig, top_n)

    return main_get_category_and_level3
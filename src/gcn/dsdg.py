import logging
from sentence_transformers import SentenceTransformer
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)

class DSDGGenerator:
    def __init__(self, excel_path, embd_model_name='all-MiniLM-L6-v2', tau=0.1, k=5):
        self.excel_path = excel_path
        self.embd_model = SentenceTransformer(embd_model_name)
        self.tau = tau
        self.k = k
        self.G = nx.Graph()
        self.dsdg_dict = self.generate_dsdg_dict()
        self._initialize_graph()

    def generate_dsdg_dict(self):
        logging.info("Generating DSDG dictionary from Excel.")
        df = pd.read_excel(self.excel_path)
        dsdg_dict = {}
        for _, row in df.iterrows():
            disease = row['Disease']
            drug = row['Drug']
            dsdg_dict[disease] = {f"Disease {cat}": row[f"Disease {cat}"] for cat in ['symptoms', 'causes', 'diagnosis', 'treatment', 'complications']}
            dsdg_dict[drug] = {f"Drug {cat}": row[f"Drug {cat}"] for cat in ['description', 'dosage', 'effects', 'toxicity', 'food_interaction', 'drug_interaction', 'pharmacodynamics', 'experimental_results']}
        return dsdg_dict

    def get_knowledge_category(self, name, category):
        return self.dsdg_dict.get(name, {}).get(category, 'Unknown')

    def calculate_distance(self, embedding1, embedding2):
        cosine_sim = cosine_similarity(embedding1, embedding2)
        numerator = np.exp(cosine_sim / self.tau)
        denominator = np.sum(np.exp(cosine_sim / self.tau))
        return numerator / denominator

    def get_top_k(self, arr):
        return arr.argsort()[-self.k:][::-1]

    def _initialize_graph(self):
        logging.info("Initializing graph from Excel.")
        df = pd.read_excel(self.excel_path)
        self.diseases = df['Disease'].unique()
        self.drugs = df['Drug'].unique()

        logging.info("Creating disease and drug nodes with embeddings.")
        for disease in self.diseases:
            description = self.get_knowledge_category(disease, 'Disease description')
            embedding = self.embd_model.encode(description)
            self.G.add_node(disease, embedding=embedding, type='disease')

        for drug in self.drugs:
            description = self.get_knowledge_category(drug, 'Drug description')
            embedding = self.embd_model.encode(description)
            self.G.add_node(drug, embedding=embedding, type='drug')

        logging.info("Creating edges between disease and drug nodes.")
        for _, row in df.iterrows():
            disease = row['Disease']
            drug = row['Drug']
            disease_embedding = self.G.nodes[disease]['embedding']
            drug_embedding = self.G.nodes[drug]['embedding']
            distance = self.calculate_distance(disease_embedding, drug_embedding)
            self.G.add_edge(disease, drug, weight=distance)

        logging.info("Updating edges based on top-K.")
        for disease in self.diseases:
            neighbors = list(self.G.neighbors(disease))
            neighbor_embeddings = np.array([self.G.nodes[neighbor]['embedding'] for neighbor in neighbors])
            disease_embedding = self.G.nodes[disease]['embedding']
            distances = self.calculate_distance(disease_embedding, neighbor_embeddings)
            top_k_indices = self.get_top_k(distances)

            for idx, neighbor in enumerate(neighbors):
                if idx not in top_k_indices:
                    self.G.remove_edge(disease, neighbor)

    def get_graph(self):
        return self.G


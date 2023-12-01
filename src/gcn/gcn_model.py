import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class GraphConvolutionalNetwork(nn.Module):
    """
        A class implementing a Graph Convolutional Network (GCN) for obtaining the prefix embedding.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphConvolutionalNetwork, self).__init__()

        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)

        return x


def get_initial_prefix(G, symptoms_embedding, hidden_dim=128, output_dim=3404, n_soft_prompts=100):
    """
        Generates an initial prefix for soft prompt tuning using a GCN.

        This function initializes a GCN model to process the graph data and compute node embeddings.
        It then combines global graph embeddings with disease-specific embeddings to form an initial
        prefix for soft prompt tuning.

        Args:
            G (nx.Graph): A NetworkX graph representing the data.
            symptoms_embedding (torch.Tensor): Embedding of the symptoms.
            hidden_dim (int): The hidden dimension size for the GCN. Defaults to 128.
            output_dim (int): The output dimension size for the GCN. Defaults to 3404.
            n_soft_prompts (int): Number of soft prompts to concatenate. Defaults to 100.

        Returns:
            torch.Tensor: The initial prefix for soft prompt tuning.
        """
    # Initialize GCN model
    input_dim = G.nodes[list(G.nodes())[0]]['embedding'].shape[0]
    model = GraphConvolutionalNetwork(input_dim, hidden_dim, output_dim/2)

    # Prepare input for GCN
    node_ids = {node: i for i, node in enumerate(G.nodes())}
    node_embeddings = torch.tensor([G.nodes[node]['embedding'] for node in G.nodes()])
    edge_index = torch.tensor([[node_ids[src], node_ids[dst]] for src, dst in G.edges()]).t().contiguous()
    data = pyg_utils.Data(x=node_embeddings, edge_index=edge_index)

    # Forward pass through GCN
    out = model(data)

    # Calculate the global graph embedding by averaging all node embeddings
    g_a = torch.mean(out, dim=0)

    # Calculate cosine similarity between symptoms_embedding and all disease nodes
    disease_nodes = [node for node, attr in G.nodes(data=True) if attr['type'] == 'disease']
    disease_embeddings = torch.tensor([G.nodes[node]['embedding'] for node in disease_nodes])
    disease_similarities = cosine_similarity(symptoms_embedding.reshape(1, -1), disease_embeddings)
    most_similar_disease_idx = np.argmax(disease_similarities)
    g_d = torch.tensor(G.nodes[disease_nodes[most_similar_disease_idx]]['embedding'])

    # Combine g_a and g_d to get the initial prefix g for the soft prompt tuning
    g = torch.cat((g_a, g_d), dim=0)

    # Concatenate n_soft_prompts of g to create the final soft prompt
    final_soft_prompt = torch.cat([g for _ in range(n_soft_prompts)], dim=0)

    return final_soft_prompt

# Example usage:
# G is the graph obtained from DSDGGenerator
# symptoms_embedding is the embedding of the extracted symptoms, obtained using embd_model from DSDGGenerator

# g = get_initial_prefix(G, symptoms_embedding, embd_model)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class GraphConvolutionalNetwork(nn.Module):
    """
    Implements a Graph Convolutional Network (GCN) for generating graph-based embeddings.

    Attributes:
        conv1 (pyg_nn.GCNConv): First graph convolutional layer.
        conv2 (pyg_nn.GCNConv): Second graph convolutional layer.

    Methods:
        forward(data): Performs a forward pass through the network.
        generate_prefix(G, identified_entities, n_soft_prompts): Generates the soft prompt prefix using the GCN.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, configs = None):
        super(GraphConvolutionalNetwork, self).__init__()
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, output_dim // 2)
        self.confic = configs

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def generate_prefix(self, G, identified_entities, n_soft_prompts=100):
        """
        Generates a soft prompt prefix using the GCN model.

        Args:
            G (nx.Graph): The graph representing the data.
            identified_entities (list): List of identified drugs and diseases.
            n_soft_prompts (int): Number of soft prompts to concatenate.

        Returns:
            torch.Tensor: The soft prompt prefix.
        """
        # Prepare and process graph data
        node_ids = {node: i for i, node in enumerate(G.nodes())}
        node_embeddings = torch.tensor([G.nodes[node]['embedding'] for node in G.nodes()])
        edge_index = torch.tensor([[node_ids[src], node_ids[dst]] for src, dst in G.edges()]).t().contiguous()
        data = pyg_utils.Data(x=node_embeddings, edge_index=edge_index)

        # Forward pass through GCN
        out = self.forward(data)

        # Calculate global graph embedding (g_a)
        g_a = torch.mean(out, dim=0)

        # Calculate the average embedding for identified entities (g_d)
        entity_embeddings = torch.tensor([G.nodes[entity]['embedding'] for entity in identified_entities if entity in G.nodes])
        g_d = torch.mean(entity_embeddings, dim=0) if len(entity_embeddings) > 0 else torch.zeros(out.shape[1])

        # Combine g_a and g_d to form the initial prefix
        g = torch.cat((g_a, g_d), dim=0)

        # Concatenate n_soft_prompts copies of g, default to 100
        final_soft_prompt = torch.cat([g for _ in range(n_soft_prompts)], dim=0)

        return final_soft_prompt


# Example usage:
# G is the graph obtained from DSDGGenerator
# symptoms_embedding is the embedding of the extracted symptoms, obtained using embd_model from DSDGGenerator

# g = get_initial_prefix(G, symptoms_embedding, embd_model)


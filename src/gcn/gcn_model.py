import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import Sequential, GCNConv
import torch_geometric.utils as pyg_utils
import networkx as nx


class GraphConvolutionalNetwork(nn.Module):
    """
    Implements an Graph Convolutional Network (GCN) with skip connections for generating graph-based embeddings.

    Attributes:
        conv1 (GCNConv): First graph convolutional layer.
        conv2 (GCNConv): Second graph convolutional layer.
        conv3 (GCNConv): Third graph convolutional layer for deeper feature extraction.
        configs (dict): Optional configurations for the GCN model.

    Methods:
        forward(data): Performs a forward pass through the network.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, configs=None):
        super(GraphConvolutionalNetwork, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim // 2)
        self.configs = configs

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First layer
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, training=self.training)

        # Second layer with skip connection from the first layer
        x2 = F.relu(self.conv2(x1, edge_index)) + x1
        x2 = F.dropout(x2, training=self.training)

        # Third layer with skip connection from the second layer
        x3 = self.conv3(x2, edge_index) + x2

        return x3

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
        entity_embeddings = torch.tensor(
            [G.nodes[entity]['embedding'] for entity in identified_entities if entity in G.nodes])
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

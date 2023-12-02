import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import re


def extract_entities(ia_response):
    # Regular expressions to find drugs and diseases in the response
    drugs_pattern = r"Drugs: \[(.*?)\]"
    diseases_pattern = r"Diseases: \[(.*?)\]"

    # Find all matches for drugs and diseases
    drugs_match = re.search(drugs_pattern, ia_response)
    diseases_match = re.search(diseases_pattern, ia_response)

    # Extract and split the entities into lists, removing spaces and handling empty cases
    drugs = [drug.strip() for drug in drugs_match.group(1).split(',')] if drugs_match else []
    diseases = [disease.strip() for disease in diseases_match.group(1).split(',')] if diseases_match else []

    # Combine drugs and diseases into a single list
    identified_entities = drugs + diseases

    return identified_entities


class SoftPromptTuner:
    """
    A class dedicated to fine-tuning the Graph Convolutional Network (GCN) for generating soft prompts.

    This class facilitates the training of the GCN, which is responsible for generating the soft prompts
    used in conjunction with the LLaMA model for knowledge acquisition tasks.

    Attributes:
        llama_utils (LLaMAUtils): An instance of LLaMAUtils for model operations.
        gcn_model (GraphConvolutionalNetwork): The GCN model to be trained.
        prompt_manager (PromptManager): Manager for generating prompts for IA (Inquiry Analysis).
        config (dict): Configuration dictionary containing training parameters.
        criterion (torch.nn.Module): Loss function for GCN training.
        writer (SummaryWriter): Tensorboard writer for logging training metrics.
        best_val_loss (float): Best validation loss achieved during training.

    Methods:
        train_gcn(train_loader, val_loader): Trains the GCN model.
        save_gcn_checkpoint(filename): Saves the GCN model checkpoint.
        load_gcn_checkpoint(filename): Loads a GCN model checkpoint.
    """

    def __init__(self, llama_utils, gcn_model, prompt_manager, config):
        self.llama_utils = llama_utils
        self.gcn_model = gcn_model
        self.prompt_manager = prompt_manager
        self.config = config
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.writer = SummaryWriter()
        self.best_val_loss = float('inf')

    def train_gcn(self, train_loader, val_loader):
        optimizer = optim.Adam(self.gcn_model.parameters(),
                               lr=self.config['learning_rate'],
                               weight_decay=self.config['weight_decay'])
        num_training_steps = len(train_loader) * self.config['epochs']
        num_warmup_steps = int(num_training_steps * self.config['warmup_ratio'])
        lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=num_warmup_steps,
                                                       num_training_steps=num_training_steps)

        for epoch in range(self.config['epochs']):
            self.gcn_model.train()
            epoch_loss = 0
            for batch_idx, (input_ids, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                identified_entities = extract_entities(self.run_inference(input_ids, use_openai=False))
                soft_prompt_prefix = self.gcn_model.generate_prefix(self.knowledge_base.get_graph(),
                                                                    identified_entities)

                # Detach the LLaMA model from the computational graph
                with torch.no_grad():
                    llama_output = self.llama_utils.llama_training_forward(input_ids, soft_prompt_prefix.detach())

                # Compute loss using the detached output
                loss = self.criterion(llama_output.view(-1, llama_output.size(-1)), labels.view(-1))
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                epoch_loss += loss.item()
                self.writer.add_scalar("Training Loss", loss.item(), epoch * len(train_loader) + batch_idx)

            val_loss = self.validate_epoch(val_loader, epoch)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_gcn_checkpoint(f"gcn_checkpoint_epoch_{epoch}.pth")

        self.writer.close()

    def save_gcn_checkpoint(self, filename):
        """Saves the current state of the GCN model to a file."""
        torch.save(self.gcn.state_dict(), filename)

    def load_gcn_checkpoint(self, filename):
        # Load GCN model checkpoint
        self.gcn_model.load_state_dict(torch.load(filename))

    def run_inference(self, input_data, use_openai=False):
        # IA using GPT-3.5 or LLaMA
        ia_combined_prompt = self.prompt_manager.generate_combined_prompt("inquiry_analysis")
        if use_openai:
            ia_response = self.llama_utils.openai_inference(ia_combined_prompt + input_data)
        else:
            ia_response = self.llama_utils.llama_inference(ia_combined_prompt + input_data, use_soft_prompt=False)
        return ia_response

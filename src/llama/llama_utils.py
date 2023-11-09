import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoftEmbedding(nn.Module):
    """Class for managing soft embeddings."""

    def __init__(self, wte: nn.Embedding, n_tokens: int = 20, random_range: float = 0.5,
                 initialize_from_vocab: bool = True):
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.Parameter(
            self.initialize_embedding(wte, n_tokens, random_range, initialize_from_vocab)
        )

    def initialize_embedding(self, n_tokens: int = 20,
                             random_range: float = 0.5, initialize_from_vocab: bool = True):
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        else:
            return torch.FloatTensor(n_tokens, self.wte.weight.size(1)).uniform_(-random_range, random_range)

    def forward(self, input_ids):
        input_embedding = self.wte(input_ids)
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)


class LLaMAUtils:
    """Class for managing LLaMA models and utilities."""

    def __init__(self, configs, soft_embedding=None):
        self.tokenizer = None
        self.model = None
        logger.info("Initializing LLaMA Utils...")
        self.configs = configs
        self.initialize_model()
        self.initialize_tokenizer()
        self.soft_embedding = soft_embedding
        self.max_length = configs.get('max_length', 512)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.soft_embedding is not None:
            self.set_input_embeddings()

    def initialize_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.configs['model_name'],
            use_auth_token=self.configs.get('use_auth_token', False)
        )

    def initialize_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.configs['model_name'])

    def set_input_embeddings(self):
        if self.soft_embedding:
            logger.info("Setting input embeddings for the model.")
            self.model.set_input_embeddings(self.soft_embedding)

    def llama_inference(self, prompt: str, use_soft_prompt=True):
        logger.info(f"Running inference for prompt: {prompt[:50]}...")
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        if use_soft_prompt and self.soft_embedding is not None:
            full_embedding = self.soft_embedding(input_ids)
            output = self.model.generate(inputs_embeds=full_embedding, max_length=self.max_length)
        else:
            output = self.model.generate(input_ids=input_ids, max_length=self.max_length)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded_output

    def llama_training_forward(self, input_ids):
        logger.info("Running forward pass for training...")
        full_embedding = self.soft_embedding(input_ids.to(self.device))
        logits = self.model(inputs_embeds=full_embedding).logits
        return logits

    def save_model(self, save_path):
        logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_model(self, load_path):
        logger.info(f"Loading model from {load_path}")
        self.model = AutoModelForCausalLM.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)

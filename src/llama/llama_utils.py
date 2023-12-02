import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoftEmbedding(nn.Module):
    """
    A class for managing soft embeddings.

    Attributes:
        wte (nn.Embedding): The word token embeddings of the base model.
        n_tokens (int): Number of tokens for the soft embeddings. Defaults to 100.
        learned_embedding (nn.Parameter): The learnable soft embeddings.

    Methods:
        initialize_embedding(): Initializes the soft embeddings, either from
                                the vocabulary or randomly.
        forward(input_ids): Computes the forward pass, concatenating the soft
                            embeddings with the input embeddings.
    """

    def __init__(self, wte: nn.Embedding, n_tokens: int = 100, random_range: float = 0.5,
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
    """
    A utility class for managing LLaMA models and related operations.

    LLaMAUtils facilitates the initialization, management, and use of LLaMA models, particularly in
    conjunction with soft embeddings for prompt tuning tasks.

    Attributes:
        soft_embedding (SoftEmbedding): The soft embedding module.
        tokenizer (AutoTokenizer): Tokenizer for the LLaMA model.
        model (AutoModelForCausalLM): The LLaMA model.
        configs (dict): Configuration parameters for the model.
        max_length (int): Maximum length for model generation.
        device (torch.device): The device on which the model operates.

    Methods:
        initialize_soft_embedding(): Initializes the soft embedding module.
        load_soft_prompt(checkpoint_file): Loads soft prompt parameters from a checkpoint file.
        llama_inference(prompt, use_soft_prompt): Runs inference using the LLaMA model with an optional soft prompt.
        initialize_model(): Initializes the LLaMA model based on the configuration.
        initialize_tokenizer(): Initializes the tokenizer for the model.
        llama_training_forward(input_ids): Performs a forward pass for training purposes.
        save_model(save_path): Saves the model and tokenizer to the specified path.
        load_model(load_path): Loads the model and tokenizer from the specified path.
    """

    def __init__(self, configs, soft_prompt_checkpoint=None):
        self.soft_embedding = None
        self.tokenizer = None
        self.model = None
        logger.info("Initializing LLaMA Utils...")
        self.configs = configs
        self.initialize_model()
        self.initialize_tokenizer()
        self.max_length = configs.get('max_length', 512)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Initialize or load soft embedding
        if soft_prompt_checkpoint:
            self.load_soft_prompt(soft_prompt_checkpoint)
        else:
            self.initialize_soft_embedding()

    def initialize_soft_embedding(self):
        wte = self.model.get_input_embeddings()
        self.soft_embedding = SoftEmbedding(wte)
        logger.info("Soft embedding initialized.")

    def load_soft_prompt(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        self.soft_embedding = SoftEmbedding(self.model.get_input_embeddings())
        self.soft_embedding.load_state_dict(checkpoint['soft_prompt_params'])
        logger.info("Soft prompt loaded from checkpoint.")

    def llama_inference(self, prompt: str, use_soft_prompt=True):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        if use_soft_prompt and self.soft_embedding is not None:
            # Generate embeddings using soft prompt
            soft_prompt_embedding = self.soft_embedding(input_ids)
            output = self.model.generate(input_ids=input_ids,
                                         inputs_embeds=soft_prompt_embedding,
                                         max_length=self.max_length)
        else:
            # Use default model embeddings
            output = self.model.generate(input_ids=input_ids, max_length=self.max_length)

        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded_output

    def initialize_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.configs['model_name'],
            use_auth_token=self.configs.get('use_auth_token', False)
        )

    def initialize_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.configs['model_name'])

    def llama_training_forward(self, input_ids, soft_prompt_prefix):
        logger.info("Running forward pass for training with soft prompt...")
        self.set_input_embeddings(soft_prompt_prefix)  # Update model's input embeddings with soft prompt
        logits = self.model(input_ids=input_ids.to(self.device)).logits
        return logits

    def save_model(self, save_path):
        logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_model(self, load_path):
        logger.info(f"Loading model from {load_path}")
        self.model = AutoModelForCausalLM.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)

    def freeze_llama_weights(self):
        """Freezes the weights of the LLaMA model to prevent them from being updated during training."""
        for param in self.model.parameters():
            param.requires_grad = False

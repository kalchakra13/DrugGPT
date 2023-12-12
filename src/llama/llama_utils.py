import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import numpy as np
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
import dataclasses
from typing import Optional
import math

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

    def initialize_model(self, local=False):
        if local:
            # Load the local LLaMA-2 model
            config = LlamaConfig.from_dict(LlamaConfig)
            self.model = LlamaForCasualLM.from_pretrained(config, 'path/to/model_weights')
        else:
            # Load model from Hugging Face
            self.model = AutoModelForCausalLM.from_pretrained(
                self.configs['model_name'],
                use_auth_token=self.configs.get('use_auth_token', False)
            )
        self.model.to(self.device)

    def initialize_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.configs['model_name'], trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

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


class LlamaConfig:
    hidden_act: str
    hidden_size: int
    intermediate_size: int
    max_sequence_length: int
    num_attention_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    position_embedding_base: int

    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    initializer_range: float
    model_type: str
    torch_dtype: str
    num_key_value_heads: int = 0

    def __post_init__(self):
        if self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0

    @classmethod
    def from_dict(cls, d: dict) -> "LlamaConfig":
        field_names = (field.name for field in dataclasses.fields(cls))
        return cls(**{k: v for k, v in d.items() if k in field_names})


class RotaryEmbedding(nn.Module):
    """
        Implements rotary positional embeddings.

        Attributes:
            dim (int): The dimensionality of the embeddings.
            max_seq_len (int): The maximum sequence length.

        Methods:
            forward(x, offset=0): Applies rotary embeddings to the input tensor.
        """
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('sin', freqs.sin())
        self.register_buffer('cos', freqs.cos())

    def forward(self, x, offset=0):
        dim = self.dim
        sin, cos = self.sin[offset:offset + x.shape[1]], self.cos[offset:offset + x.shape[1]]
        sin, cos = sin.unsqueeze(-1), cos.unsqueeze(-1)
        return torch.cat((x[..., :dim // 2] * cos + x[..., dim // 2:] * sin,
                          x[..., :dim // 2] * sin - x[..., dim // 2:] * cos), dim=-1)


class LlamaFFN(nn.Module):
    """
        Implements the feed-forward network (FFN) for the LLaMA architecture.

        Attributes:
            gate_proj (nn.Linear): Linear layer for gating.
            up_proj (nn.Linear): Linear layer for up-projection.
            down_proj (nn.Linear): Linear layer for down-projection.

        Methods:
            forward(x): Passes the input through the feed-forward network.
        """
    def __init__(self, hidden_size, intermediate_size):
        super(LlamaFFN, self).__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        x1 = self.gate_proj(x)
        x2 = self.up_proj(x)
        return self.down_proj(F.silu(x1) * x2)


class LlamaAttention(nn.Module):
    """
        Implements the attention mechanism for the LLaMA architecture.

        Attributes:
            hidden_size (int): Size of the hidden layer.
            num_attention_heads (int): Number of attention heads.
            num_key_value_heads (int): Number of key/value heads.
            head_dim (int): Dimension of each attention head.
            rotary_embedding (RotaryEmbedding): Rotary embedding module.
            q_proj, k_proj, v_proj, o_proj (nn.Linear): Linear projection layers for query, key, value, and output.
            k_cache, v_cache (torch.Tensor): Caches for key and value tensors.

        Methods:
            forward(hidden_states, attention_mask, total_seq_len): Computes the attention mechanism.
        """
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, max_sequence_length, rotary_embedding):
        super(LlamaAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.rotary_embedding = rotary_embedding

        # Ensure the hidden size is divisible by the number of attention heads
        assert self.hidden_size % self.num_attention_heads == 0

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.k_cache = torch.zeros(max_sequence_length, self.num_key_value_heads, self.head_dim)
        self.v_cache = torch.zeros(max_sequence_length, self.num_key_value_heads, self.head_dim)

    def forward(self, hidden_states, attention_mask, total_seq_len):
        batch_size, seq_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # Apply rotary embeddings
        q, k = self.rotary_embedding(q, k, total_seq_len - seq_len)

        # Append to cache
        self.k_cache = torch.cat((self.k_cache, k.squeeze(0)), dim=0)[-total_seq_len:]
        self.v_cache = torch.cat((self.v_cache, v.squeeze(0)), dim=0)[-total_seq_len:]

        # Compute attention scores
        attn_output, _ = F.multi_head_attention_forward(
            query=q,
            key=self.k_cache.unsqueeze(0).expand(batch_size, -1, -1, -1),
            value=self.v_cache.unsqueeze(0).expand(batch_size, -1, -1, -1),
            embed_dim_to_check=self.hidden_size,
            num_heads=self.num_attention_heads,
            dropout_p=0,
            attention_mask=attention_mask,
            need_weights=False,
        )

        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaDecoderLayer(nn.Module):
    """
        Represents a single decoder layer in the LLaMA model.

        Attributes:
            attn (LlamaAttention): Attention mechanism for the layer.
            ffn (LlamaFFN): Feed-forward network for the layer.
            input_norm, post_attention_norm (nn.LayerNorm): Layer normalization modules.

        Methods:
            forward(hidden_states, attention_mask, total_seq_len): Processes input through the decoder layer.
        """
    def __init__(self, config, rotary_embedding):
        super(LlamaDecoderLayer, self).__init__()
        self.attn = LlamaAttention(config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
                                   config.max_sequence_length, rotary_embedding)
        self.ffn = LlamaFFN(config.hidden_size, config.intermediate_size)
        self.input_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask, total_seq_len):
        # Apply input normalization and attention layer
        attn_output = self.attn(self.input_norm(hidden_states), attention_mask, total_seq_len)
        hidden_states = attn_output + hidden_states

        # Apply FFN on post-attention normalization
        ffn_output = self.ffn(self.post_attention_norm(hidden_states))
        hidden_states = ffn_output + hidden_states

        return hidden_states


class LlamaModel(nn.Module):
    """
        The core LLaMA model comprising multiple decoder layers.

        Attributes:
            embed_tokens (nn.Embedding): Token embedding layer.
            layers (nn.ModuleList): List of LLaMA decoder layers.
            norm (nn.LayerNorm): Layer normalization for the final output.

        Methods:
            forward(inputs, total_seq_len, attention_mask): Processes input through the entire model.
        """
    def __init__(self, config):
        super(LlamaModel, self).__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        rotary_embedding = RotaryEmbedding(config.position_embedding_base, config.max_sequence_length,
                                           config.hidden_size // config.num_attention_heads)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, rotary_embedding)
                                     for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, inputs, total_seq_len, attention_mask):
        hidden_states = self.embed_tokens(inputs)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, total_seq_len)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCasualLM(nn.Module):
    """
        LLaMA model for causal language modeling.

        Attributes:
            model (LlamaModel): The core LLaMA model.
            lm_head (nn.Linear): Linear layer for language modeling predictions.
            vocab_size (int): Size of the vocabulary.
            dtype (str): Data type of the model.

        Methods:
            forward(inputs, attention_mask=None, total_seq_len=None, inputs_embeds=None): Processes input for language modeling.
            save_pretrained(save_path): Saves the model's state dictionary to a file.
            from_pretrained(config, load_path): Class method to load a pretrained model from a file.
        """
    def __init__(self, config, dtype='float32'):
        super(LlamaForCasualLM, self).__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.dtype = dtype

    def forward(self, inputs, attention_mask=None, total_seq_len=None, inputs_embeds=None):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.model.embed_tokens(inputs)

        if attention_mask is None:
            attention_mask = torch.ones((hidden_states.shape[0], 1, hidden_states.shape[1], total_seq_len),
                                        dtype=torch.float32, device=hidden_states.device)
            attention_mask = (attention_mask - 1) * 1e9  # Apply mask

        for layer in self.model.layers:
            hidden_states = layer(hidden_states, attention_mask, total_seq_len)

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    def save_pretrained(self, save_path):
        torch.save(self.state_dict(), save_path)

    @classmethod
    def from_pretrained(cls, config, load_path):
        # Create an instance of the model
        model = cls(config)
        # Load the saved weights into the model
        model.load_state_dict(torch.load(load_path))
        return model

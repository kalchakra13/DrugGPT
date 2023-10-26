import torch
import torch.nn as nn
from models.basic import BasicContainer


class NaivePromptLeaner(BasicContainer):
    def __init__(self, embed_dim, hidden_size, hidden_dropout_prob=0.1) -> None:
        super().__init__()
        self.net = nn.Linear(embed_dim, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.apply(self._init_weights)
    
    def forward(self, embs):
        out = self.net(embs)
        out = self.LayerNorm(out)
        out = self.dropout(out)
        if out.dim() == 2:
            out = out.unsqueeze(1)
        return out


class Prompter(BasicContainer):
    def __init__(self, embed_dim, hidden_size, num_virtual_tokens, hidden_dropout_prob=0.1, conditional=False) -> None:
        super().__init__()
        self.conditional = conditional
        if self.conditional:
            self.net = nn.Linear(embed_dim, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.apply(self._init_weights)

        self.virtual_tokens = nn.Parameter(torch.FloatTensor(num_virtual_tokens, hidden_size))
        nn.init.trunc_normal_(self.virtual_tokens, std=0.02)
    
    def forward(self, embs):
        if self.conditional:
            out = self.net(embs)
            if out.dim() == 2:
                out = out.unsqueeze(1)    
            out = out + self.virtual_tokens[None, :, :]
        else:
            out = self.virtual_tokens[None, :, :]
        out = self.LayerNorm(out)
        out = self.dropout(out)
        return out


class ConceptPromptPrefixer(BasicContainer):
    def __init__(self, embed_dim, hidden_size, n_prompts, hidden_dropout_prob=0.1, ap=False) -> None:
        super().__init__()
        self.net = nn.Linear(embed_dim, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.apply(self._init_weights)
        
        self.n_prompts = n_prompts
        self.ap = ap
    
    def forward(self, concept_prompts):
        assert concept_prompts.size(1) == self.n_prompts
        
        if self.ap:
            concept_prompts = concept_prompts.mean(1, keepdims=True)
        
        return self.dropout(self.LayerNorm(self.net(concept_prompts)))


class PromptLearnerWithVirtualTokens(BasicContainer):
    def __init__(self, embed_dim, hidden_size, num_virtual_tokens, hidden_dropout_prob=0.1) -> None:
        super().__init__()
        self.net = nn.Linear(embed_dim, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.apply(self._init_weights)

        self.virtual_tokens = nn.Parameter(torch.FloatTensor(num_virtual_tokens, hidden_size))
        nn.init.trunc_normal_(self.virtual_tokens, std=0.02)
    
    def forward(self, embs):
        out = self.net(embs)
        if out.dim() == 2:
            out = out.unsqueeze(1)
        
        out = out + self.virtual_tokens[None, :, :]
        out = self.LayerNorm(out)
        out = self.dropout(out)
        return out


class PromptLearnerWithPrefix(BasicContainer):
    def __init__(self, embed_dim, hidden_size, num_virtual_tokens, hidden_dropout_prob=0.1) -> None:
        super().__init__()
        self.net = nn.Linear(embed_dim, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.apply(self._init_weights)

        self.virtual_tokens = nn.Parameter(torch.FloatTensor(num_virtual_tokens, hidden_size))
        nn.init.trunc_normal_(self.virtual_tokens, std=0.02)
    
    def forward(self, embs):
        out = self.net(embs)
        if out.dim() == 2:
            out = out.unsqueeze(1)
        
        prefix = torch.nn.functional.normalize(self.virtual_tokens, dim=-1)
        prefix = prefix[None, :, :].repeat(out.size(0), 1, 1)
        out = torch.cat((prefix, out), dim=1)

        out = self.LayerNorm(out)
        out = self.dropout(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dim_query=None, num_attention_heads=None):
        super().__init__()
        
        if num_attention_heads is None:
            num_attention_heads = hidden_size // 64

        if dim_query is None:
            dim_query = hidden_size

        self.q = nn.Linear(dim_query, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)

        self.q_norm = nn.LayerNorm(hidden_size)
        self.k_norm = nn.LayerNorm(hidden_size)

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        assert self.attention_head_size * self.num_attention_heads == self.hidden_size
        self.scale = self.attention_head_size ** -0.5
        
    def forward(self, q, kv=None, attn_mask=None):
        if kv is None:
            kv = q

        B = q.size(0)
        L = kv.size(1)

        query = self.q(q).view(B, -1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        key = self.k(kv).view(-1, L, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        value = self.v(kv).view(-1, L, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)

        attn = (query @ key.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn = attn + attn_mask
        
        attn = attn.softmax(dim=-1)
        out = (attn @ value).transpose(1, 2).view(B, -1, self.hidden_size)
        out = self.o(out)
        return out


class Dictionary(BasicContainer):
    def __init__(self, embed_dim, hidden_size, num_tokens=10000, residual=False, concat=False) -> None:
        super().__init__()

        self.net = nn.Linear(embed_dim, hidden_size)
        self.attn = MultiHeadAttention(hidden_size)
        self.LN = nn.LayerNorm(hidden_size)
        
        self.diction = nn.Parameter(torch.FloatTensor(1, num_tokens, hidden_size))
        nn.init.trunc_normal_(self.diction, std=0.02)
        self.diction_norm = nn.LayerNorm(hidden_size)

        self.apply(self._init_weights)

        self.residual = residual
        self.concat = concat

        assert not (self.residual and self.concat)
    
    def forward(self, x):
        x = self.net(x)

        if x.dim() == 2:
            x = x.unsqueeze(1)

        out = self.attn(q=x, kv=self.diction_norm(self.diction))

        if self.residual:
            out = x + out
        
        if self.concat:
            out = torch.cat((x, out), dim=1)

        out = self.LN(out)
        return out


def build_prompt_learner(config: dict, embed_dim: int, hidden_size: int):
    if config.get('add_cross_attention'):
        if config.get('prompter'):
            print(f'### Using prompter (prompt_length: {config["prompt_length"]}  conditional: {config["conditional"]})')
            return Prompter(embed_dim, hidden_size, config['prompt_length'], conditional=config['conditional'])

        if config.get('ConceptPromptPrefixer'):
            print(f'### Using ConceptPromptPrefixer, with n_prompts = {config["n_prompts"]} and average_pooling = {config["prompt_length"] == 1}')
            return ConceptPromptPrefixer(embed_dim, hidden_size, config['n_prompts'], ap=config['prompt_length'] == 1)

        print('### Not using soft prompts')
        return None
    else:
        if config.get('with_virtual_tokens', False):
            print('### Using (CLIP\'s emb + virtual tokens) to prompt the decoder')
            return PromptLearnerWithVirtualTokens(embed_dim, hidden_size, config['prompt_length'])
        elif config.get('with_prefix'):
            print('### Using (virtual tokens, CLIP\'s emb) to prompt the decoder')
            return PromptLearnerWithPrefix(embed_dim, hidden_size, config['prompt_length'] - 1)
        elif config.get('with_dict'):
            num_tokens = config.get('num_tokens', 10000)
            prompt_length = config['prompt_length']
            assert prompt_length in [1, 2]
            residual = config.get('residual', False)
            concat = prompt_length == 2
            print(f'### Using a dictionary of size {num_tokens} (residual: {residual}, concat: {concat})')
            return Dictionary(embed_dim, hidden_size, num_tokens, residual=residual, concat=concat)
        else:
            print('### Using only the CLIP\'s emb to prompt the decoder')
            return NaivePromptLeaner(embed_dim, hidden_size)

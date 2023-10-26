# Configuration options for LLaMA models
LLAMA_CONFIGS = {
    'model_name': 'meta-llama/Llama-2-7b-chat-hf',
    'quantization_config': {
        'load_in_4bit': True,
        'bnb_4bit_quant_type': 'nf4',
        'bnb_4bit_compute_dtype': 'float16',
        'bnb_4bit_use_double_quant': False,
    },
    'device_map': {'': 0},
    'use_auth_token': True,
    'max_length': 500,
}

# Configuration options for Soft Prompt Tuning
SOFT_PROMPT_CONFIGS = {
    'epochs': 10,
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    'weight_decay': 0.0,
    'lr_scheduler_type': 'linear',
    'warmup_ratio': 0.1,
    'gradient_clip': None,
    'logging_interval': 100,
}

DATA_LOADER_CONFIGS = {
    'batch_size': 32,
    'max_length': 512,
    'val_split': 0.2,
    'shuffle': True,
    'random_state': 42
}
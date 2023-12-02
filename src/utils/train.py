import logging
from transformers import AutoTokenizer
from ..data.data_loader import get_data_loaders
from ..prompt_tuning.soft_prompt_tuning import SoftPromptTuner, extract_entities
from ..llama.llama_utils import LLaMAUtils, SoftEmbedding
from ..gcn.gcn_model import GraphConvolutionalNetwork
from ..gcn.dsdg import DSDGGenerator
import argparse
from pathlib import Path
import yaml

# Load configurations from model.yaml
with open('model.yaml', 'r') as file:
    configs = yaml.safe_load(file)

LLAMA_CONFIGS = configs['LLAMA_CONFIGS']
SOFT_PROMPT_CONFIGS = configs['SOFT_PROMPT_CONFIGS']
DATA_LOADER_CONFIGS = configs['DATA_LOADER_CONFIGS']
GCN_CONFIGS = configs['GCN_CONFIGS']


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize tokenizer and DSDG Generator
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_CONFIGS['model_name'])
    dsdg_generator = DSDGGenerator('path/to/excel', embd_model_name='all-MiniLM-L6-v2')

    # Initialize LLaMAUtils
    llama_utils = LLaMAUtils(LLAMA_CONFIGS)
    llama_utils.freeze_llama_weights()

    # Initialize Graph Convolutional Network (GCN)
    gcn_model = GraphConvolutionalNetwork(GCN_CONFIGS['input_dim'], GCN_CONFIGS['hidden_dim'], GCN_CONFIGS['output_dim'])

    # Load dataset
    train_loader, val_loader = get_data_loaders(tokenizer, '../../data/FT1.csv', DATA_LOADER_CONFIGS)

    # Initialize Soft Prompt Tuner with GCN
    soft_prompt_tuner = SoftPromptTuner(llama_utils, gcn_model, dsdg_generator, SOFT_PROMPT_CONFIGS)

    # Start training the GCN
    logger.info("Starting GCN training...")
    soft_prompt_tuner.train_gcn(train_loader, val_loader)

    logger.info("Training complete.")


def create_config(config_path):
    """
    Load the YAML configuration file.

    Args:
    config_path (str): Path to the configuration file.

    Returns:
    dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_folder_name(config):
    """
    Generate a unique folder name based on configuration.

    Args:
    config (dict): Configuration dictionary.

    Returns:
    str: Generated folder name.
    """
    # Example: Generate a folder name by concatenating some config parameters
    folder_name_parts = [
        config.get('model_name', 'model'),
        config.get('learning_rate', 'lr'),
        config.get('batch_size', 'bs')
    ]
    folder_name = '_'.join(str(part) for part in folder_name_parts)
    return folder_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for medical datasets")

    # Basic configurations
    parser.add_argument('--ckpt_name', type=str, default='model_state_latest.pth', help='Checkpoint filename')
    parser.add_argument('--config', type=str, default='configs/model.yaml', help='Path to the configuration file')
    parser.add_argument('--output_root', type=str, default='output/training', help='Root directory for output files')

    # Dataset options
    parser.add_argument('--dataset', type=str, choices=['FT1', 'FT2', 'FT3'], default='FT1', help='Dataset to use')
    parser.add_argument('--train_file', type=str, help='Path to the training file')
    parser.add_argument('--val_file', type=str, help='Path to the validation file')
    parser.add_argument('--test_file', type=str, help='Path to the test file')

    # Training configurations
    parser.add_argument('--device', default='cuda', help='Device to use for training')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint')

    # Additional options
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation only')
    parser.add_argument('--msg', type=str, default='', help='Additional message to log')

    args = parser.parse_args()

    # Load and display configurations
    config = create_config(args.config)
    print("### Configuration")
    print(yaml.dump(config))

    # Set output directory
    if args.output_dir is None:
        folder_name = args.dataset + '_' + get_folder_name(config)
        args.output_dir = Path(args.output_root) / folder_name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize result directory
    args.result_dir = args.output_dir / 'result'
    args.result_dir.mkdir(parents=True, exist_ok=True)

    # Save the configuration to the output directory
    with open(args.output_dir / 'config.yaml', 'w') as file:
        yaml.dump(config, file)

    # Call the main training function
    main(args, config)

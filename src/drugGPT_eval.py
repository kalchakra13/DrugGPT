import pandas as pd
from tqdm import tqdm
import argparse
import os
from .ensemble.ensemble_model import EnsembleModel
from .prompt.prompt_manager import PromptManager
from .llama.llama_utils import LLaMAUtils, SoftEmbedding
from .gcn.dsdg import DSDGGenerator
from .utils.parser import binary_parser, text_parser, mc_parser
import logging
import yaml


class Evaluation:
    def __init__(self, ensemble_model, parser_dict, log_results=True, store_results=False,
                 log_wrong_answers_only=False, useopenai=False):
        self.ensemble_model = ensemble_model
        self.log_results = log_results
        self.parser_dict = parser_dict
        self.useopenai = useopenai
        self.store_results = store_results
        self.log_wrong_answers_only = log_wrong_answers_only

    @staticmethod
    def check_text_accuracy(prediction, actual):
        prediction = prediction.rstrip('.')
        prediction_initials = ''.join(word[0] for word in prediction.split())
        actual_words = actual.split()
        return all(word in prediction for word in actual_words) or prediction_initials == actual

    @staticmethod
    def calculate_f1_metrics(prediction, label):
        prediction_set = set(prediction.split(', '))
        label_set = set(label.split(', '))
        true_positives = len(prediction_set & label_set)
        false_positives = len(prediction_set - label_set)
        false_negatives = len(label_set - prediction_set)

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        return precision, recall, f1_score

    def log_answer(self, i, input_data, prediction, actual_label):
        if self.log_wrong_answers_only and prediction.lower() == actual_label.lower():
            return
        logging.info(f"Index: {i} Question: {input_data}")
        logging.info(f"Predicted: {prediction}, Actual: {actual_label}")

    def evaluate(self, dataset_name, evaluation_set):
        print(f"\nEvaluating: {dataset_name}")
        slice_size = len(evaluation_set['sample'])

        accurate_predictions = 0
        precision_list = []
        recall_list = []
        f1_list = []
        wrong_answers = []

        for i in tqdm(range(slice_size), desc="Processing"):
            input_data = evaluation_set['sample'][i]
            full_response = self.ensemble_model.run_inference(input_data, use_openai=self.useopenai)
            parsed_response = self.parser_dict[dataset_name](full_response)

            correct_answer = self.check_text_accuracy(parsed_response, evaluation_set['label'][i].lower())
            if correct_answer:
                accurate_predictions += 1
            else:
                wrong_answers.append((i, input_data, parsed_response, evaluation_set['label'][i]))

            if dataset_name == 'chatDoctor':
                precision, recall, f1 = self.calculate_f1_metrics(parsed_response, evaluation_set['label'][i].lower())
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)

            self.log_answer(i, input_data, parsed_response, evaluation_set['label'][i])

        results = {
            'Accuracy': accurate_predictions / slice_size
        }

        if dataset_name == 'chatDoctor':
            results['Average Precision'] = sum(precision_list) / len(precision_list)
            results['Average Recall'] = sum(recall_list) / len(recall_list)
            results['Average F1 Score'] = sum(f1_list) / len(f1_list)

        if self.store_results:
            df = pd.DataFrame(wrong_answers, columns=['Index', 'Question', 'Predicted', 'Actual'])
            df.to_csv(f'evaluation_wrong_answers_{dataset_name}.csv', index=False)

        return results


def load_evaluation_set(dataset_name):
    """
    Load the evaluation set for a given dataset.

    Args:
    dataset_name (str): Name of the dataset to load.

    Returns:
    DataFrame: The loaded dataset.
    """

    # Dictionary mapping dataset names to their file paths
    data_paths_dict = {
        'pubmedqa': {
            'type': 'binary',
            'data': f'../../data/pubmedqa_data.csv',
            'answer': f'../../data/pubmedqa_answer.csv'
        },
        'ade': {
            'type': 'text',
            'data': f'../../data/ade_data.csv',
            'answer': f'../../data/ade_answer.csv'
        },
        'chatDoctor': {
            'type': 'text',
            'data': f'../../data/chatDoctor_data.csv',
            'answer': f'../../data/chatDoctor_answer.csv'
        },
        'DDI_binary': {
            'type': 'binary',
            'data': f'../../data/DDI_binary_data.csv',
            'answer': f'../../data/DDI_binary_answer.csv'
        },
        'drug_usage': {
            'type': 'text',
            'data': f'../../data/drug_usage_data.csv',
            'answer': f'../../data/drug_usage_answer.csv'
        },
        'medmcqa': {
            'type': 'mc',
            'data': f'../../data/medmcqa_data.csv',
            'answer': f'../../data/medmcqa_answer.csv'
        },
        'mmlu_mc': {
            'type': 'mc',
            'data': f'../../data/mmlu_mc_data.csv',
            'answer': f'../../data/mmlu_mc_answer.csv'
        },
        'usmle_mc': {
            'type': 'mc',
            'data': f'../../data/usmle_mc_data.csv',
            'answer': f'../../data/usmle_mc_answer.csv'
        },
        'moderna_interactions': {
            'type': 'binary',
            'data': f'../../data/moderna_interactions_data.csv',
            'answer': f'../../data/moderna_interactions_answer.csv'
        }
    }

    # Get the paths for the specified dataset
    dataset_paths = data_paths_dict.get(dataset_name, {})
    if not dataset_paths:
        raise ValueError(f"Dataset {dataset_name} not found in data paths dictionary")

    # Load the dataset
    data_df = pd.read_csv(dataset_paths['data'])
    answer_df = pd.read_csv(dataset_paths['answer'])

    # Merge data and answers into a single DataFrame (assuming they can be merged directly)
    evaluation_set = pd.merge(data_df, answer_df, on='some_common_column')

    return evaluation_set


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ensemble Model")
    parser.add_argument('--openai_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--hf_key', type=str, required=True, help='Hugging Face API key')
    parser.add_argument('--excel_path', type=str, required=True, help='Path to DSDG Excel file')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['pubmedqa', 'ade', 'chatDoctor', 'DDI_binary', 'drug_usage', 'medmcqa', 'mmlu_mc',
                                 'usmle_mc', 'moderna_interactions'], help='Name of the dataset for evaluation')
    parser.add_argument('--evaluation_set_path', type=str, required=True, help='Path to the evaluation dataset')
    parser.add_argument('--log_results', action='store_true', help='Enable logging of results')
    parser.add_argument('--store_results', action='store_true', help='Enable storing of results')
    parser.add_argument('--log_wrong_answers_only', action='store_true', help='Log only wrong answers')
    parser.add_argument('--use_open_ai', action='store_true', help='Use OpenAI API for inference of last generation '
                                                                   'model for better conversational alignment')
    args = parser.parse_args()

    # Set environment variables for API keys
    os.environ["OPENAI_API_KEY"] = args.openai_key
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = args.hf_key

    # Load configurations from model.yaml
    with open('model.yaml', 'r') as file:
        configs = yaml.safe_load(file)

    LLAMA_CONFIGS = configs['LLAMA_CONFIGS']

    # Initialize components
    prompt_manager = PromptManager()
    dsdg_generator = DSDGGenerator(args.excel_path, embd_model_name='all-MiniLM-L6-v2')
    llama_utils = LLaMAUtils(LLAMA_CONFIGS)

    # Load soft embedding checkpoint
    checkpoint_file = "../../data/soft_prompt_checkpoint.pth"
    soft_embedding = SoftEmbedding.load_checkpoint(checkpoint_file)
    llama_utils.set_input_embeddings(soft_embedding)

    # Initialize Ensemble Model
    ensemble_model = EnsembleModel(prompt_manager, soft_embedding, dsdg_generator, llama_utils, args.openai_key)

    # Define parser dictionary for different datasets
    parser_dict = {
        'pubmedqa': binary_parser,
        'ade': text_parser,
        'chatDoctor': text_parser,
        'DDI_binary': binary_parser,
        'drug_usage': text_parser,
        'medmcqa': mc_parser,
        'mmlu_mc': mc_parser,
        'usmle_mc': mc_parser,
        'moderna_interactions': binary_parser
    }

    # Load evaluation dataset
    evaluation_set = load_evaluation_set(args.evaluation_set_path)

    # Initialize and run evaluation
    evaluator = Evaluation(ensemble_model, args.log_results, args.store_results, args.log_wrong_answers_only,
                           args.use_open_ai)
    results = evaluator.evaluate(args.dataset_name, evaluation_set, parser_dict)

    # Log and store results
    logging.info(f"Evaluation results: {results}")
    if args.store_results:
        with open(f"evaluation_results_{args.dataset_name}.txt", 'w') as file:
            file.write(str(results))


if __name__ == "__main__":
    main()

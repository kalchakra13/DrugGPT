import pandas as pd
from tqdm import tqdm


class Evaluation:
    def __init__(self, ensemble_inference):
        self.ensemble_inference = ensemble_inference

    @staticmethod
    def check_text_accuracy(prediction, actual):
        prediction = prediction.rstrip('.')
        prediction_initials = ''.join(word[0] for word in prediction.split())
        actual_words = actual.split()
        prediction_contains_all_words = all(word in prediction for word in actual_words)
        return prediction_contains_all_words or prediction_initials == actual

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

    def evaluate(self, dataset_name, evaluation_set, full_responses=None):
        print(f"\nEvaluating: {dataset_name}")
        slice_size = len(evaluation_set['sample'])

        accurate_predictions = 0
        precision_list = []
        recall_list = []
        f1_list = []

        for i in tqdm(range(slice_size), desc="Processing"):
            full_response = self.ensemble_inference(evaluation_set['sample'][i])
            prediction = full_response.split(".")[0].lower()

            if dataset_name == 'ade':
                if self.check_text_accuracy(prediction, evaluation_set['label'][i].lower()):
                    accurate_predictions += 1
            elif dataset_name == 'drug_usage':
                answer1, answer2 = prediction.split(", ")
                label1, label2 = evaluation_set['label'][i].lower().split(", ")
                if answer1 == label1 and answer2 == label2:
                    accurate_predictions += 1
            elif dataset_name == 'chatDoctor':
                precision, recall, f1 = self.calculate_f1_metrics(prediction, evaluation_set['label'][i].lower())
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
            else:
                if prediction == evaluation_set['label'][i].lower():
                    accurate_predictions += 1

        if dataset_name == 'chatDoctor':
            average_precision = sum(precision_list) / len(precision_list)
            average_recall = sum(recall_list) / len(recall_list)
            average_f1 = sum(f1_list) / len(f1_list)
            print(f"\nAverage Precision: {average_precision}")
            print(f"Average Recall: {average_recall}")
            print(f"Average F1 Score: {average_f1}")
        else:
            accuracy = accurate_predictions / slice_size
            print(f"\nAccuracy: {accurate_predictions}/{slice_size} ({accuracy * 100:.2f}%)")

        # Constructing the results data
        data = {
            'Questions': evaluation_set['sample'],
            'Long Answer': [response.split('.')[0] for response in full_responses],
            'Short Answer': [response.split('.')[0].lower() for response in full_responses],
            'Official Answer': evaluation_set['label']
        }

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(data)

        # Save the DataFrame to a csv file
        df.to_csv(f'evaluation_results_{dataset_name}.csv', index=False)

        return {
            'Long Answer': [response.split('.')[0] for response in full_responses],
            'Short Answer': [response.split('.')[0].lower() for response in full_responses],
            'Official Answer': evaluation_set['label'],
            'Precision': precision_list,
            'Recall': recall_list,
            'F1 Score': f1_list,
            'Accuracy': accuracy if dataset_name != 'chatDoctor' else None,
            'Average Precision': average_precision if dataset_name == 'chatDoctor' else None,
            'Average Recall': average_recall if dataset_name == 'chatDoctor' else None,
            'Average F1 Score': average_f1 if dataset_name == 'chatDoctor' else None
        }

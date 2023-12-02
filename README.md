# DrugGPT
This repo contains the original implementation of our paper ***DrugGPT: A Knowledge-Grounded Collaborative Large Language Model for Evidence-based Drug Analysis***.

![DrugGPT Architecture](./img/flowchart.png)
## Updates
[15/11/2023] We clean up our codebase to improve readability!

[26/10/2023] We make all the codebases available!

[12/10/2023] We release our code on the pre-processing and generation of data and model. 

## Pre-trained Model
The trained model is available at [Google Drive](https://drive.google.com/file/d/1jyavc13OdwzVZaTDdo6oEm4_adjr_nO8/view?usp=sharing).

## Full Model
To access the full model, visit our demo [DrugGPT Demo](https://demo-drug-gpt.vercel.app/en).

### Instruction on how to use the demo for drug analysis and inquiry

1. There are 4 modes accessible for downstream tasks: 
   1. General: This mode is intended for general drug inquiry. User is prompted to input symptom, disease (if diagnosed) and medication info (if prescribed). The model will generate information about the drug, including its name, usage, side effects, etc. This model is recommended for general conversation about drug and disease.
   2. Multiple Choice: This mode is intended for drug related multiple choice questions. User is prompted to input the question and the options. The model will generate the answer to the question. This mode is not recommended for continuous conversation but for accurate, evidence-based MC Q&A.
   3. Yes/No: This mode is intended for drug related yes/no questions. User is prompted to input the question. The model will generate the answer to the question. This mode is not recommended for continuous conversation but for accurate, evidence-based binary Q&A.
   4. Text Q&A: This mode is intended for drug related text Q&A. User is prompted to input the question. The model will generate the answer to the question. This mode is not recommended for continuous conversation but for accurate, evidence-based text Q&A.
2. After selecting the desired mode and inputting the information, click the 'Submit' button at the bottom of the form to initiate the conversation.
3. DrugGPT should never be used as medical consultant at the current stage. Please consult to licensed medical professionals for any medical advice.

### Demos on downstream tasks
The demo videos showing DrugGPT performing downstream tasks will be available at: 
1. [Multiple Choice Q&A](https://www.loom.com/share/6528968b9b804db19f5d7c5e1554197a?sid=6b6019e4-d8d5-46b3-bf7d-7c4895ec91bb)
2. [Drug and Dosage Recommendation](https://www.loom.com/share/81e82eb651bc4a208097e5f8dc56e28d?sid=a4ca2143-6e6f-48f0-9863-b6e691f40273)
3. [Adverse Reaction](https://www.loom.com/share/2b4b91726dfe4a38afe99a56cacd5170?sid=9f0e4480-0b60-48f9-909c-4f1f02c58a82)
4. [Drug-drug Interaction](https://www.loom.com/share/e0b43a02862248da937cd10b8c7b0284?sid=3eb53ad4-b248-46a0-8ce4-d1f17389071d)
5. [Pharmacology Q&A](https://www.loom.com/share/5dbac6d2cac9406da717db0572e9d5b6?sid=c8b2827d-533d-42af-ae16-88ecdbb35acc)
6. [Generalization Study](https://www.loom.com/share/cc7a476209a444fa851fd4b0bd1ea1fd?sid=18c0f238-1260-4af5-a557-4ca9fb04fef6)

## Clone the repo
```
git clone https://github.com/AI-in-Health/DrugGPT.git

# clone the following repo to calculate automatic metrics
cd DrugGPT
git clone https://github.com/ruotianluo/coco-caption.git 
```

## Codebase structure
```
DrugGPT/ # the root of the repo
    ├── README.md
    ├── _init_.ipynb # scripts for logging, loading, etc.
    ├── configs
    │   ├── finetune.yaml      # config file for fine-tuning
    │   ├── methods.yaml       # config file for methods
    │   ├── model.yaml         # config file for llama and soft prompt models
    │   └── train.yaml         # config file for training
    ├── data
    │   └──source.md          # links to the source datasets and preprocessed datasets
    │   
    ├── notebooks              # Folder for notebooks
    │   └── evaluation.ipynb   # Notebook for evaluation of benchmark models
    src
    ├── data
    │   ├── data_loader.py     # scripts for loading data
    ├── ensemble
    │   ├── ensemble_model.py  # the ensemble model structure
    ├── evaluation
    │   ├── evaluation_metrics.py # script for evalaution
    ├── gcn
    │   ├── dsdg.py # contains code for generating dsdg graph
    │   ├── gcn_model.py # gcn model used to obtain the graph embedding of dsdg
    ├── llama
    │   ├── llama_utils.py # the llama model and the soft prompt
    ├── prompt
    │   ├── prompt_manager.py # manages hard prompts
    ├── prompt_tuning
    │   ├── soft_prompt_tuning.py # fine-tuning soft prompt
    ├── utils
    │   ├── basic.py # basec container
    │   ├── checkpointer.py # checkpointer
    │   ├── train.py # fine-tuning
    │   ├── language_model.py # language model
    │   ├── optim.py # optimizer
    │   ├── parser.py # parser for different types of outputs
    │   └── scheduler.py # scheduler
    └── drugGPT_eval # script for evaluating DrugGPT
```

## Environment

```
conda create -n pi python==3.9
conda activate pi
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.34.0
pip install langchain==0.0.314
pip install pytorch-lightning==1.5.1
pip install pandas rouge scipy
pip install networkx==2.5.1
pip install torch_geometric==1.7.2
pip install nltk
pip install tqdm
pip install openai==0.28.1
pip instal installed tiktoken==0.5.1
pip install huggingface-hub==0.17.3 
pip install safetensors==0.4.0 
pip install sentence-transformers==2.2.2 
pip install sentencepiece==0.1.99 
pip install tokenizers==0.14.1
pip install accelerate==0.23.0
pip install einops==0.7.0
pip install re
pip install pandas


# if you want to re-produce our data preparation process
pip install scikit-learn plotly
```
Higher version of `torch` and `cuda` can also work.


## Download the data
The source data can be accessed at:
1. **MedQA-USMLE**: [GitHub](https://github.com/jind11/MedQA) | [PapersWithCode](https://paperswithcode.com/dataset/medqa-usmle)
2. **MedMCQA**: [MedMCQA Homepage](https://medmcqa.github.io/)
3. **MMLU-Medicine**: [Hugging Face Datasets](https://huggingface.co/datasets/cais/mmlu)
4. **ChatDoctor**: [GitHub](https://github.com/Kent0n-Li/ChatDoctor)
5. **ADE-Corpus-v2**: [Hugging Face Datasets](https://huggingface.co/datasets/ade_corpus_v2)
6. **Drug-Effects**: [Kaggle](https://www.kaggle.com/datasets/jithinanievarghese/drugs-side-effects-and-medical-condition)
7. **DDI-Corpus**: [Hugging Face Datasets](https://huggingface.co/datasets/bigbio/ddi_corpus)
8. **PubMedQA**: [PubMedQA Homepage](https://pubmedqa.github.io/)

The preprocessed data might be too large for direct access, you can download the pre-processed data from [Google Drive](https://drive.google.com/drive/folders/1Cd1KdWzCdD0iOUE2HB9BZa8ReVoHVc11?usp=drive_link) (please send us an email for access).

### Data Preprocessing Process

The data preprocessing for DrugGPT involves several crucial steps to ensure the quality and relevance of the data used for training and fine-tuning the model. Below is an overview of the process:

1. **Data Cleaning**: 
   - Remove duplicate and contaminated data to ensure the uniqueness and purity of the dataset.

2. **Relevance Filtering**: 
   - For datasets not entirely drug-related, irrelevant data is filtered out to maintain focus on drug-related content.

3. **Data Organization**: 
   - Organize the data into columns for queries, answers, and explanations (if available). The explanation column is particularly useful for hallucination assessment during model training.

4. **Expert Review**: 
   - Conduct a manual inspection with medical experts to verify that the data quality aligns with drug analysis processes in medical settings.

5. **Data Storage**: 
   - Store the preprocessed files in CSV formats, tagged with either _data or _answer to indicate their content type.

6. **Sample Collection**: 
   - Collect 1000 data samples curated from various datasets, including PubmedQA, MedMCQA, ADE-Corpus-V2, DDI-corpus, and Drug-Effects. These datasets cover the five downstream tasks of DrugGPT.

7. **Preparation for Knowledge-based Instruction Prompt Tuning**: 
   - Store the 1000 data samples specifically prepared for Knowledge-based Instruction Prompt Tuning, a novel process based on PEFT (refer to [PEFT paper](https://arxiv.org/abs/2104.08691)) modified to incorporate our DSDG graph in the KA-LLM inference.

8. **Dataset Variations**: 
   - Randomly sample the data into three distinct datasets: FT1, FT2, and FT3.csv, to provide diverse training scenarios.

9. **Data Storage Location**: 
   - All prepared datasets are stored in the `data` folder within the project structure.

This meticulous preprocessing ensures that DrugGPT is trained on high-quality, relevant data, laying a strong foundation for accurate and reliable drug-related predictions and analysis.


## Training
The training is only applicable to the finetuning the KA-LLM (Knowledge Acquisition) model. Which is a component of the ensembled DrugGPT specialized in locating specific knowledge from the DSDG (Drug and Symptom Disease Graph). 
The training is intended to align the features in DSDG with the natural language input which KA-LLM takes as the input for downstream tasks. 

### Fine-Tuning Hyperparameters

Below is a table of the hyperparameters used for fine-tuning the Knowledge Acquisition Language Model (KA-LLM):

| Parameter                | Value           | Description                                                              |
|--------------------------|-----------------|--------------------------------------------------------------------------|
| Model                    | LLaMA-7B        | The base language model used for fine-tuning.                            |
| Soft Prompt Length       | 100             | Length of the soft prompt used in tuning.                                |
| Epochs                   | 20              | Number of training epochs.                                               |
| Learning Rate            | 1e-3            | Learning rate for the optimizer.                                         |
| Optimizer                | AdamW           | The optimization algorithm used.                                         |
| Weight Decay             | 0.01            | Weight decay parameter for the optimizer.                                |
| Frozen LLM Parameters    | True            | Indicates if the LLM parameters are kept frozen.                         |
| Number of Data Samples   | 1000            | Total number of data samples used for tuning.                            |
| Data Sample Distribution | 200 per dataset | Each dataset contributes 200 samples.                                    |
| Datasets Used            | Various         | Includes PubmedQA, MedMCQA, ADE-Corpus-V2, DDI-corpus, and Drug-Effects. |
| τ (Tau)                  | 0.1             | Hyperparameter for DSDG edge weight calculations.                        |
| K                        | 5               | Hyperparameter for DSDG edge weight calculations.                        |

These settings were selected to optimize the performance of DrugGPT on various downstream tasks while considering computational efficiency.

### Arguments:
Here are some key argument to run `train.py`:
- `--ckpt_name`: The filename for the model checkpoint. Default is `model_state_latest.pth`.
- `--config`: Path to the configuration file. Default is set to `configs/model.yaml`.
- `--output_root`: Root directory for saving output files. Default is `output/training`.

- `--dataset`: Choose the dataset to use for training.
- `--train_file`: Path to the training dataset file.
- `--val_file`: Path to the validation dataset file.
- `--test_file`: Path to the test dataset file.

- `--device`: Specify the device for training, typically `cuda` for GPU training.
- `--seed`: Set a random seed for reproducibility. Default is `42`.
- `--distributed`: Enable this flag to use distributed training.
- `--dist_url`: URL used to set up distributed training. Default is `env://`.
- `--world_size`: Number of distributed processes. Default is `1`.

- `--resume`: Resume training from the latest checkpoint.
- `--evaluate`: Use this flag to perform evaluation only.
- `--msg`: An additional message for logging purposes.

### Examples:
Example usage:
To access the train.py script, run the following command:
```bash
cd src
cd utils
```
To run the training process, use the following command:
```bash
python3 train.py --dataset FT1 --train_file path/to/FT1_train.xml --val_file path/to/FT1_val.xml --config configs/model.yaml --output_root output/FT1_training
```

## Evaluation
### Prerequisites:
Hugging Face API for running inference with DrugGPT, which is built upon the [LLaMA](https://huggingface.co/docs/transformers/model_doc/llama2) architecture. Please refer to [Hugging Face API](https://huggingface.co/docs/api-inference/quicktour) for more details.
OpenAI key if you plan to use the latest GPT models for conversational generation. Please refer to the [OpenAI API](https://openai.com/blog/openai-api). For one-shot generation, we recommend set use_open_ai to false as OpenAI is not a necceary component for DrugGPT.
The LLaMA implementation can be accessed in the [LLaMA GitHub repo](https://github.com/facebookresearch/llama), however, it might be computational expensive to run the inference.
If you decide to use the llama inference api, here is the [link](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) to require access, in addition to the Hugging Face API key.

### Arguements:
Here are some key argument to run `drugGPT_eval.py`:
- `--openai_key`: Your OpenAI API key for accessing GPT models.
- `--hf_key`: Your Hugging Face API key for accessing models from the Hugging Face Hub.
- `--excel_path`: Path to the Excel file containing the DSDG (Drug and Disease Graph) data.
- `--dataset_name`: The name of the dataset for evaluation. Choose from predefined options like pubmedqa, ade, etc.
- `--evaluation_set_path`: Path to the evaluation dataset file.
- `--log_results`: Enable this flag to log evaluation results.
- `--store_results`: Enable this flag to store evaluation results in a file.
- `--log_wrong_answers_only`: Log only the wrong answers for a concise output.
- `--use_open_ai`: Use this flag to enable the use of OpenAI API for inference.

### Examples:
To evaluate the model, use the following command:
```bash
python3 drugGPT_eval.py \
  --openai_key YOUR_OPENAI_API_KEY \
  --hf_key YOUR_HUGGINGFACE_API_KEY \
  --excel_path path/to/DSDG_excel.xlsx \
  --dataset_name pubmedqa \
  --evaluation_set_path path/to/pubmedqa_evaluation_data.csv \
  --log_results \
  --store_results

```
### Baseline models
To evaluate other models, use the template provided in notebooks/evaluation.ipynb.

## Bugs or Questions?

If you encounter any problems when using the code, or want to report a bug, you can open an issue or email {hongjian.zhou@cs.ox.ac.uk, fenglin.liu@eng.ox.ac.uk}. Please try to specify the problem with details so we can help you better and quicker!

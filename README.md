# DrugGPT
This repo contains the original implementation of our paper 
## Updates
[26/10/2023] We make all the codebases available!

[12/10/2023] We release our code on the pre-processing and generation of data and model. 

## Pre-trained Model
The trained model is available at [Google Drive](https://drive.google.com/file/d/1jyavc13OdwzVZaTDdo6oEm4_adjr_nO8/view?usp=sharing).

## Demo
To access the full model, visit our demo [DrugGPT Demo](https://demo-drug-gpt.vercel.app/en).

### Instruction on how to use the demo for drug analysis and inquiry

1. There are 4 modes accessible for downstream tasks: 
   1. General: This mode is intended for general drug inquiry. User is prompted to input symptom, disease (if diagnosed) and medication info (if prescribed). The model will generate information about the drug, including its name, usage, side effects, etc. This model is recommended for general conversation about drug and disease.
   2. Multiple Choice: This mode is intended for drug related multiple choice questions. User is prompted to input the question and the options. The model will generate the answer to the question. This mode is not recommended for continuous conversation but for accurate, evidence-based MC Q&A.
   3. Yes/No: This mode is intended for drug related yes/no questions. User is prompted to input the question. The model will generate the answer to the question. This mode is not recommended for continuous conversation but for accurate, evidence-based binary Q&A.
   4. Text Q&A: This mode is intended for drug related text Q&A. User is prompted to input the question. The model will generate the answer to the question. This mode is not recommended for continuous conversation but for accurate, evidence-based text Q&A.
2. After selecting the desired mode and inputting the information, click the 'Submit' button at the bottom of the form to initiate the conversation.
3. DrugGPT should never be used as medical consultant at the current stage. Please consult to licensed medical professionals for any medical advice.

## Clone the repo
```
git clone https://github.com/AI-in-Health/DrugGPT.git

# clone the following repo to calculate automatic metrics
cd DrugGPT
git clone https://github.com/ruotianluo/coco-caption.git 
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



## Bugs or Questions?

If you encounter any problems when using the code, or want to report a bug, you can open an issue or email {hongjian.zhou@cs.ox.ac.uk, fenglin.liu@eng.ox.ac.uk}. Please try to specify the problem with details so we can help you better and quicker!



## Citation

Please consider citing our papers if our code or datasets are useful to your work, thanks sincerely!

```bibtex
```

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate


class PromptManager:
    def __init__(self):
        self.sys_prompts = {
            'inquiry_analysis': {
                'task': "Your task is to identify any relevant drugs, symptoms, and diseases in the user's inquiry.",
                'answer_format': "Question: {question}\nAnswer: Drugs {drugs}, Symptoms {symptoms}, Disease {disease}"
            },
            'knowledge_acquisition': {
                'task': "Your task is to identify the knowledge base numbers needed to answer the question.",
                'answer_format': "Question: {question}\nIdentified: Drugs {drugs}, Symptoms {symptoms}, Disease {"
                                 "disease}\nAnswer: Knowledge Needed {knowledge_needed}"
            },
            'evidence_generation': {
                'task': "Your task is to answer multiple-choice questions. Understand the question, analyze it, "
                        "and provide a concise and accurate answer.",
                'answer_format': """
Analysis: Provide an analysis that logically leads to the answer based on the relevant information.
Final Answer: Provide the final answer, which should be a single letter in the alphabet representing the best option among the multiple choices provided in the question."""
            }
        }
        self.fs_examples = {
            'inquiry_analysis': [
                {
                    'question': "A 29-year-old woman develops painful swelling of both hands. She is also very stiff "
                                "in the morning. Physical examination reveals involvement of the proximal "
                                "interphalangeal joints and metacarpophalangeal (MCP) joints. Her RF is positive and "
                                "ANA is negative. Which of the following medications is most likely to improve her "
                                "joint pain symptoms?",
                    'drugs': ["D-penicillamine", "an anti-malarial", "methotrexate", "NSAID or aspirin"],
                    'symptoms': ["Painful swelling of both hands", "stiffness in the morning",
                                 "involvement of proximal interphalangeal joints and metacarpophalangeal joints."],
                    'diseases': ["None"]
                }
            ],
            'knowledge_acquisition': [
                {
                    'question': "A 29-year-old woman develops painful swelling of both hands. She is also very stiff "
                                "in the morning. Physical examination reveals involvement of the proximal "
                                "interphalangeal joints and metacarpophalangeal (MCP) joints. Her RF is positive and "
                                "ANA is negative. Which of the following medications is most likely to improve her "
                                "joint pain symptoms?",
                    'drugs': ["D-penicillamine", "an anti-malarial", "methotrexate", "NSAID or aspirin"],
                    'symptoms': ["Painful swelling of both hands", "stiffness in the morning",
                                 "involvement of proximal interphalangeal joints and metacarpophalangeal joints."],
                    'diseases': ["None"],
                    'knowledge_needed': [1, 8]
                }
            ]
        }

    def generate_combined_prompt(self, task_type, prompt_type):
        if prompt_type == "sys_prompt":
            return self.sys_prompts[task_type]['task'] + '\n' + self.sys_prompts[task_type]['answer_format']
        elif prompt_type == "fs_prompt":
            fs_template = FewShotPromptTemplate(
                examples=self.fs_examples[task_type],
                example_prompt=PromptTemplate(
                    input_variables=["question", "drugs", "symptoms", "diseases"],
                    template=self.sys_prompts[task_type]['answer_format']
                ),
                suffix="Question: {input_question}",
                input_variables=["input_question"]
            )
            return fs_template
        else:
            raise ValueError(f"Invalid prompt_type: {prompt_type}")


# Initialize the PromptManager
prompt_manager = PromptManager()

# Generate system prompt for Inquiry Analysis
sys_prompt_inquiry = prompt_manager.generate_combined_prompt("inquiry_analysis", "sys_prompt")
print("System Prompt for Inquiry Analysis:\n", sys_prompt_inquiry)

# Generate few-shot prompt for Inquiry Analysis
fs_prompt_inquiry = prompt_manager.generate_combined_prompt("inquiry_analysis", "fs_prompt")
print("\nFew-Shot Prompt for Inquiry Analysis:\n", fs_prompt_inquiry)

# Generate system prompt for Knowledge Acquisition
sys_prompt_knowledge = prompt_manager.generate_combined_prompt("knowledge_acquisition", "sys_prompt")
print("\nSystem Prompt for Knowledge Acquisition:\n", sys_prompt_knowledge)

# Generate few-shot prompt for Knowledge Acquisition
fs_prompt_knowledge = prompt_manager.generate_combined_prompt("knowledge_acquisition", "fs_prompt")
print("\nFew-Shot Prompt for Knowledge Acquisition:\n", fs_prompt_knowledge)

# Generate system prompt for Evidence Generation
sys_prompt_evidence = prompt_manager.generate_combined_prompt("evidence_generation", "sys_prompt")
print("\nSystem Prompt for Evidence Generation:\n", sys_prompt_evidence)

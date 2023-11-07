from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate


class PromptManager:
    def __init__(self):
        self.sys_prompts = {
            'inquiry_analysis': {
                'task': 'You are tasked with identifying the drug, symptom, and disease from user inquiry:Drugs: List '
                        'any. If none, your answer for this section should be ’[]’. Answer should be in form ’[drug '
                        'a, drug b, ...]’.Symptoms: List any. If none, your answer for this section should be ’[]’. '
                        'Answer should be in form ’[symptom a, symptomb, ...]’.Diseases: List any. If none, '
                        'your answer for this section should be ’[]’. Answer should be in form ’[disease a, '
                        'disease b, ...].',
                'answer_format': "Question: {question}\nAnswer: Drugs {drugs}, Symptoms {symptoms}, Disease {disease}"
            },
            'knowledge_acquisition': {
                'task': 'Task: You are tasked with extracting the knowledge to answer a medical inquiry accurately. '
                        'Step 1: Identify the categories of knowledge needed (List the numbers corresponding to the '
                        'knowledge categories necessary) to answer the inquiry correctly. If none, your answer for '
                        'this section should be ’[]’. Answer should be in form ’[1, 2, 3, ...]’.The knowledge '
                        'categories of drugs are:1. Drug description and indication.2. Drug dosage recommendation.3. '
                        'Drug adverse effect.4. Drug toxicity.5. Drug-food interaction.6. Drug-drug interaction.7. '
                        'Drug pharmacodynamics.8. Pubmed experimental summaries.The knowledge categories of diseases '
                        'and symptoms are:1. Common symptoms.2. Disease causes.3. Disease diagnosis.4. Disease '
                        'treatment.5. Disease complications.Step 2: Extract the specific knowledge from the '
                        'identified knowledge categories to answer the inquiry correctly.',
                'answer_format': "Question: {question}\nIdentified: Drugs {drugs}, Symptoms {symptoms}, Disease {"
                                 "disease}\nAnswer: Knowledge Needed {knowledge_needed}"
            },
            'evidence_generation': {
                'task': """Your task is to answer questions. Understand the question, analyze it step by step, 
                and provide a concise and accurate answer. Among the provided choices, choose the one that best fits 
                the criteria below:TO DO:Only use the knowledge provided to answer the inquiryNOT TO DO: 1. Do not 
                make assumptions not supported by the provided content. 2. Avoid providing personal opinions or 
                interpretations. 3. Summarize and interpret the knowledge provided objectively and accurately.""",
                'answer_format': """Analysis: Provide an analysis that logically leads to the answer based on the 
                relevant information. Final Answer: Provide the final answer, which should be a single letter in the 
                alphabet representing the best option among the multiple choices provided in the question. When 
                analyzing each choice, include the relevant knowledge relied upon and display its source link (
                provided as Link[https://...]) to the relevant part of your output"""
            }
        }
        self.fs_examples = {
            'inquiry_analysis':
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
            ,
            'knowledge_acquisition':
                {
                    'question': """A 29-year-old woman develops painful swelling of both hands. She is also very stiff 
                                in the morning. Physical examination reveals involvement of the proximal 
                                interphalangeal joints and metacarpophalangeal (MCP) joints. Her RF is positive and 
                                ANA is negative. Which of the following medications is most likely to improve her 
                                joint pain symptoms?
                                Drugs Identified: ["D-penicillamine", "an anti-malarial", "methotrexate", "NSAID or aspirin"]
                                Symptoms Identified: ["Painful swelling of both hands", "stiffness in the morning",
                                 "involvement of proximal interphalangeal joints and metacarpophalangeal joints."]
                                 Diseases Identified: ["None"]""",
                    'knowledge_needed': "[1, 8]",
                },
            'evidence_generation': {
                'task': """Question: A 29-year-old woman develops painful swelling of both hands. She is also very 
                stiff in the morning. Physical examination reveals involvement of the proximal interphalangeal joints 
                and metacarpophalangeal (MCP) joints. Her RF is positive and ANA is negative. Which of the following 
                medications is most likely to improve her joint pain symptoms? Knowledge: Key embedding: 
                D-penicillamine Knowledge Block: Penicillamine is a chelating (KEE-late-ing) agent that binds to 
                excess copper and removes it from the blood stream. Penicillamine is used to remove excess copper in 
                people with an inherited condition called Wilson's disease. Penicillamine is also used to treat 
                severe rheumatoid arthritis after other medicines have been tried without success. Penicillamine is 
                not approved to treat juvenile rheumatoid arthritis. Link[
                https://www.drugs.com/mtm/penicillamine.html] Key embedding: Anti-malarial Knowledge Block: 
                Hydroxychloroquine is a quinoline medicine used to treat or prevent malaria, a disease caused by 
                parasites that enter the body through the bite of a mosquito. Hydroxychloroquine is also used to 
                treat symptoms of rheumatoid arthritis and discoid or systemic lupus erythematosus. Link[
                https://www.drugs.com/hydroxychloroquine.html] Key embedding: Methotrexate Knowledge Block: 
                Methotrexate interferes with the growth of certain cells of the body, especially cells that reproduce 
                quickly, such as cancer cells, bone marrow cells, and skin cells. Methotrexate is used to treat 
                leukemia and certain types of cancer of the breast, skin, head and neck, lung, or uterus. 
                Methotrexate is also used to treat severe psoriasis and rheumatoid arthritis in adults. It is also 
                used to treat active polyarticular-course juvenile rheumatoid arthritis in children. Link[
                https://www.drugs.com/methotrexate.html] Key embedding: NSAID or aspirin Knowledge Block: Aspirin, 
                a salicylate, is used for immediate relief of pain, fever, inflammation, arthritis, migraines, 
                and reduce the risk of major adverse cardiovascular events. It provides relief for various symtoms 
                such as the flu, the common cold, neck and back pain, rheumatoid arthritis, bursitis, burns, 
                and various injuries. Link[https://www.drugs.com/aspirin.html] Key Embedding: ['Painful swelling of 
                both hands', 'stiffness in the morning', 'involvement of proximal interphalangeal joints and 
                metacarpophalangeal joints.']  Knowledge Block: rheumatoid arthritis: Rheumatoid arthritis is a 
                long-term condition that causes pain, swelling and stiffness in the joints. The condition usually 
                affects the hands, feet and wrists.Common medications: methotrexate, leflunomide, hydroxychloroquine, 
                sulfasalazine. DMARD such as methotrexate is usually the first medicine given for rheumatoid 
                arthritis, often with another NSAID to relieve any pain. DMARDs help slow the progress of RA, 
                and reduce pain, stiffness, and inflammation, however, they do not provide short-term pain relief and 
                may take several weeks or months to demonstrate a clinical effect. NSAIDs , such as aspirin, 
                ibuprofen, are used to help decrease swelling, pain, and fever and relieve joint pain symptoms 
                instead. Link[https://www.nhs.uk/conditions/rheumatoid-arthritis/] """,
                'answer_format': """Analysis: Provide an analysis that logically leads to the answer based on the 
                        relevant information. Final Answer: Provide the final answer, which should be a single letter in the 
                        alphabet representing the best option among the multiple choices provided in the question."""
            }

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

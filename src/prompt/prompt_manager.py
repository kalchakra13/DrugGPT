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
                'answer_format': "Question: {question}\nAnswer: Knowledge Needed {knowledge_needed}"
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
                    provided as Link[https://...]) to the relevant part of your output""",
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
                    'answer': """
                    Drugs: [D-penicillamine, an anti-malarial, methotrexate, NSAID or aspirin],
                    Symptoms: [Painful swelling of both hands, stiffness in the morning,
                                 involvement of proximal interphalangeal joints and metacarpophalangeal joints],
                    Diseases: [None]""",
                }
            ,
            'knowledge_acquisition':
                {
                    'question': """A 29-year-old woman develops painful swelling of both hands. She is also very stiff 
                                in the morning. Physical examination reveals involvement of the proximal 
                                interphalangeal joints and metacarpophalangeal (MCP) joints. Her RF is positive and 
                                ANA is negative. Which of the following medications is most likely to improve her 
                                joint pain symptoms?
                                Drugs: ["D-penicillamine", "an anti-malarial", "methotrexate", "NSAID or aspirin"]
                                Symptoms: ["Painful swelling of both hands", "stiffness in the morning",
                                 "involvement of proximal interphalangeal joints and metacarpophalangeal joints."]
                                 Diseases: ["None"]""",
                    'answer': "Knowledge Needed [1, 8]",
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

                'answer': """Analysis: The patient's symptoms and positive RF (Rheumatoid Factor) suggest a diagnosis 
                of rheumatoid arthritis. The question asks for the medication that would most likely improve her 
                joint pain symptoms. A: D-penicillamine is used to treat severe rheumatoid arthritis after other 
                medicines have been tried without success, suggesting it is not the first-line treatment for pain 
                relief in rheumatoid arthritis. More information is avaliable here(
                https://www.drugs.com/mtm/penicillamine.html). B: An anti-malarial such as Hydroxychloroquine is used 
                to treat symptoms of rheumatoid arthritis, but it is not primarily used for pain relief. More 
                information is avaliable here(https://www.drugs.com/hydroxychloroquine.html).C: Methotrexate is a 
                first-line treatment for rheumatoid arthritis, but it is used to slow the progression of the  disease 
                and does not provide immediate pain relief. More information is avaliable here(https://www.drugs.com/ 
                methotrexate.html).D: NSAIDs or aspirin are commonly used to provide immediate relief from pain, 
                inflammation, and fever, making them suitable for relieving joint pain symptoms in rheumatoid 
                arthritis. More information is avaliable here(https://www.drugs.com/aspirin.html). Final Answer: D"""
            }

        }

    def generate_combined_prompt(self, task_type):
        sys_template = PromptTemplate(
            input_variables=["task", "answer_format"],
            template=(
                "You are given a task and an answer format, complete the task and respond following the answer format\n"
                "Task: {task}\n"
                "Answer format: {answer_format}\n"
            )
        )
        fs_template = PromptTemplate(
            input_variables=["question", "answer"],
            template=(
                "Question: {question}\n"
                "Answer: {answer}\n"
            )
        )

        return sys_template.format(**self.sys_prompts[task_type]) + '\n' + fs_template.format(**self.fs_examples[task_type])


# Initialize the PromptManager
prompt_manager = PromptManager()

# Generate system prompt for Inquiry Analysis
sys_prompt_inquiry = prompt_manager.generate_combined_prompt("inquiry_analysis")
print("System Prompt for Inquiry Analysis:\n", sys_prompt_inquiry)

# Generate few-shot prompt for Inquiry Analysis
fs_prompt_inquiry = prompt_manager.generate_combined_prompt("inquiry_analysis")
print("\nFew-Shot Prompt for Inquiry Analysis:\n", fs_prompt_inquiry)

# Generate system prompt for Knowledge Acquisition
sys_prompt_knowledge = prompt_manager.generate_combined_prompt("knowledge_acquisition")
print("\nSystem Prompt for Knowledge Acquisition:\n", sys_prompt_knowledge)


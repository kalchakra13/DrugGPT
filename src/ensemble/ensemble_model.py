import openai
import re


class EnsembleModel:
    """
        This class facilitates the process of inquiry analysis, knowledge acquisition, and evidence generation
        by orchestrating the interaction between different models and knowledge sources.

        Attributes:
            prompt_manager (PromptManager): Manages and generates prompts for various tasks.
            soft_prompt (torch.Tensor): The soft prompt to be used with the LLaMA model.
            knowledge_base (DSDGGenerator): A knowledge base containing medical information.
            llama_utils (LLaMAUtils): Utility class for LLaMA model operations.
            openai_api_key (str): API key for OpenAI services.

        Methods:
            openai_inference(prompt): Performs inference using OpenAI's GPT-3.5 model.
            llama_inference(prompt): Performs inference using the LLaMA model with soft prompts.
            extract_knowledge(ka_response): Extracts relevant knowledge entries based on the KA response.
            run_inference(input_data): Orchestrates the complete inference process involving IA, KA, and EG steps.
        """
    def __init__(self, prompt_manager, soft_prompt, knowledge_base, llama_utils, openai_api_key):
        self.prompt_manager = prompt_manager
        self.soft_prompt = soft_prompt
        self.knowledge_base = knowledge_base
        self.llama_utils = llama_utils
        self.openai_api_key = openai_api_key
        openai.api_key = self.openai_api_key

    def openai_inference(self, prompt):
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=150  # Adjust tokens as needed
        )
        return response.choices[0].text.strip()

    def llama_inference(self, prompt, use_soft_prompt=True):
        return self.llama_utils.llama_inference(prompt, use_soft_prompt=use_soft_prompt)

    def extract_knowledge(self, ka_response):
        # Regex to extract the knowledge numbers needed
        drug_knowledge_needed = re.findall(r'Drug Knowledge Needed \[([\d, ]+)\]', ka_response)
        disease_knowledge_needed = re.findall(r'Disease Knowledge Needed \[([\d, ]+)\]', ka_response)

        # Convert string numbers to lists of integers
        drug_numbers = [int(num) for num in drug_knowledge_needed[0].split(',')] if drug_knowledge_needed else []
        disease_numbers = [int(num) for num in
                           disease_knowledge_needed[0].split(',')] if disease_knowledge_needed else []

        # Retrieve the knowledge entries
        knowledge_entries = []
        for num in drug_numbers:
            knowledge_entries.append(self.knowledge_base.dsdg_dict['drug'][f'Drug {num}'])
        for num in disease_numbers:
            knowledge_entries.append(self.knowledge_base.dsdg_dict['disease'][f'Disease {num}'])

        # Combine all entries into a single string
        combined_knowledge = '\n'.join(knowledge_entries)
        return combined_knowledge

    def run_inference(self, input_data, use_openai=False):
        # Inquiry Analysis (IA) using GPT-3.5
        ia_combined_prompt = self.prompt_manager.generate_combined_prompt("inquiry_analysis")
        if use_openai:
            ia_response = self.openai_inference(ia_combined_prompt + input_data)
        else:
            ia_response = self.llama_inference(ia_combined_prompt + input_data, use_soft_prompt=False)

        # Knowledge Acquisition (KA) using LLaMA model with soft prompts
        ka_combined_prompt = self.prompt_manager.generate_combined_prompt("knowledge_acquisition")
        ka_response = self.llama_inference(self.soft_prompt + ka_combined_prompt + ia_response)

        # Accessing the Knowledge Base after KA
        dsdg_enriched_input = self.extract_knowledge(ka_response)

        # Evidence Generation (EG) using GPT-3.5, integrating the knowledge from the Knowledge Base
        eg_combined_prompt = self.prompt_manager.generate_combined_prompt("evidence_generation")
        if use_openai:
            eg_response = self.openai_inference(eg_combined_prompt + '\nKnowledge:\n' + dsdg_enriched_input)
        else:
            eg_response = self.llama_inference(eg_combined_prompt + '\nKnowledge:\n' + dsdg_enriched_input,
                                               use_soft_prompt=False)

        return eg_response

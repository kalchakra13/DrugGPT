class EnsembleModel:
    def __init__(self, prompt_manager, soft_prompt, knowledge_base, llama_utils):
        self.prompt_manager = prompt_manager
        self.soft_prompt = soft_prompt
        self.knowledge_base = knowledge_base
        self.llama_utils = llama_utils

    def inference(self, prompt, input_data, use_soft_prompt=False):
        # If soft prompts are to be used, they would be prepended to the prompt here.
        if use_soft_prompt:
            prompt = f"{self.soft_prompt}\n{prompt}"
        # The inference is conducted using the LLaMAUtils class.
        return self.llama_utils.llama_inference(prompt, input_data)

    def run_inference(self, input_data):
        # Inquiry Analysis (IA) using the combined prompt
        ia_combined_prompt = self.prompt_manager.generate_combined_prompt("inquiry_analysis")
        ia_response = self.inference(ia_combined_prompt, input_data)

        # Knowledge Acquisition (KA) using Few-shot with soft prompts and DSDG knowledge
        ka_combined_prompt = self.prompt_manager.generate_combined_prompt("knowledge_acquisition")
        # The DSDG knowledge base is accessed here to enrich the KA step
        dsdg_enriched_input = self.knowledge_base.access_knowledge(ia_response)
        ka_response = self.inference(ka_combined_prompt, dsdg_enriched_input, use_soft_prompt=True)

        # Evidence Generation (EG) using the output from KA
        eg_combined_prompt = self.prompt_manager.generate_combined_prompt("evidence_generation")
        eg_response = self.inference(eg_combined_prompt, ka_response)

        return {
            "ia_response": ia_response,
            "ka_response": ka_response,
            "eg_response": eg_response
        }

from prompt_manager import PromptManager  # Assuming PromptManager is in a separate module now
from llama_utils import inference
class EnsembleModel:
    def __init__(self, prompt_manager, soft_prompt, knowledge_base):
        self.prompt_manager = prompt_manager
        self.soft_prompt = soft_prompt
        self.knowledge_base = knowledge_base

    def run_inference(self, input_data):
        # Inquiry Analysis (IA) using the standard prompt
        ia_sys_prompt = self.prompt_manager.generate_combined_prompt("inquiry_analysis", "sys_prompt")
        ia_fs_prompt = self.prompt_manager.generate_combined_prompt("inquiry_analysis", "fs_prompt")
        ia_response = inference(ia_sys_prompt, ia_fs_prompt, input_data)

        # Knowledge Acquisition (KA) using Few-shot with soft prompts and DSDG knowledge
        ka_sys_prompt = self.prompt_manager.generate_combined_prompt("knowledge_acquisition", "sys_prompt")
        ka_fs_prompt = self.prompt_manager.generate_combined_prompt("knowledge_acquisition", "fs_prompt")
        combined_ka_prompt = f"{self.soft_prompt}\n{ka_sys_prompt}\n{ka_fs_prompt}"
        # The DSDG knowledge base is accessed here to enrich the KA step
        dsdg_enriched_input = self.knowledge_base.access_knowledge(ia_response)
        ka_response = inference(combined_ka_prompt, dsdg_enriched_input)

        # Evidence Generation (EG) using the output from KA
        eg_sys_prompt = self.prompt_manager.generate_combined_prompt("evidence_generation", "sys_prompt")
        eg_fs_prompt = self.prompt_manager.generate_combined_prompt("evidence_generation", "fs_prompt")
        eg_response = inference(eg_sys_prompt, eg_fs_prompt, ka_response)

        return {
            "ia_response": ia_response,
            "ka_response": ka_response,
            "eg_response": eg_response
        }
# Placeholder

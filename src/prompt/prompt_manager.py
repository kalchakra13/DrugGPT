from typing import Dict, List, Tuple
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate


class PromptManager:
    def __init__(self):
        # Define templates for each type of instruction prompt
        self.instruction_templates = {
            'Inquiry_Analysis': PromptTemplate(
                input_variables=["task", "answer_format"],
                template="Task: {task}\nAnswer Format:\n{answer_format}\n----Below are some examples-----\n"
            ),
            'Knowledge_Acquisition': PromptTemplate(
                input_variables=["task", "answer_format"],
                template="Task: {task}\nAnswer Format:\n{answer_format}\n----Below are some examples-----\n"
            ),
            'Evidence_Generation': PromptTemplate(
                input_variables=["task", "answer_format", "not_to_dos"],
                template="Task: {task}\nAnswer Format:\n{answer_format}\nNot to do:\n{not_to_dos}\n----Below are some examples-----\n"
            )
        }

        self.KNOWLEDGE_AGENT_FS_EXAMPLES = {
            "example": [
                {
                    "question": "A 29-year-old woman develops painful swelling of both hands. She is also very stiff in the morning. Physical examination reveals involvement of the proximal interphalangeal joints and metacarpophalangeal (MCP) joints. Her RF is positive and ANA is negative. Which of the following medications is most likely to improve her joint pain symptoms?",
                    "drugs": ["D-penicillamine", "an anti-malarial", "methotrexate", "NSAID or aspirin"],
                    "symptoms": ["Painful swelling of both hands", "stiffness in the morning",
                                 "involvement of proximal interphalangeal joints and metacarpophalangeal joints."],
                    "diseases": ["None"],
                    "knowledge_needed": [1, 8]
                },
            ],
        }

        self.FS_EXAMPLES = {
            "pubmedqa": [
                {
                    "question": "Does intermittent warm blood cardioplegia provide adequate myocardial resuscitation after global ischaemia?",
                    "content": "Intermittent warm blood cardioplegia is controversial, and many surgeons consider it inadequate for myocardial protection. The purpose of this study was to compare intermittent and continuous warm blood cardioplegia as resuscitation in hearts exposed to global ischaemia. Pigs were put on cardiopulmonary bypass (CPB) and subjected to 30 min of warm, unprotected, global ischaemia, followed by continuous (n = 7) or intermittent (n = 10, 12 ml/kg every 10 min) warm (34 degrees C) antegrade blood cardioplegia for 45 min (delivery pressure 75-80 mmHg) and weaned from CPB 45 to 60 min later. Indices of left ventricular function were acquired with the conductance catheter technique and pressure-volume loops at baseline and after 90 min of reperfusion. Cardioplegia was delivered during 17% of the cross-clamp time. Global left ventricular function, evaluated by preload recruitable stroke work (PRSW), was unchanged after continuous cardioplegia; 95 (76-130) (median (quartile interval)) to 91 (90-104) erg/ml x 10(3), but decreased after intermittent cardioplegia; 122 (100-128) to 64 (23-93) erg/ml x 10(3). Two pigs in the intermittent group weaned from CPB, but died before post-bypass measurement. A 95% confidence interval for the difference in post-bypass mean PRSW was estimated as 32 +/- 30 erg/ml x 10(3) (corresponding to P = 0.04 for comparison between treatments). The end-diastolic pressure-volume relation (EDPVR) increased from 0.17 (0.14-0.20) (continuous) and 0.15 (0.12-0.22) (intermittent) mmHg/ml to 0.27 (0.22-0.33) (P = 0.018) and 0.39 (0.25-0.66) (P = 0.005) mmHg/ml, respectively, indicating deterioration in diastolic function. No difference between groups was found in EDPVR, stiffness constant, troponin T release or myocardial water content.",
                    "analysis": "Based on the provided information from the study, intermittent warm blood cardioplegia does not seem to provide adequate myocardial resuscitation after global ischaemia.",
                    "final_answer": "no"
                },
            ],
            "ade": [
                {
                    "question": "Identify the adverse drug reaction related to azithromycin in this context: Intravenous azithromycin-induced ototoxicity.",
                    "content": "Intravenous azithromycin-induced ototoxicity has been identified as an adverse drug reaction of azithromycin.",
                    "analysis": "It can be concluded that the adverse drug reaction related to azithromycin is ototoxicity.",
                    "final_answer": "ototoxicity"
                },
            ],
            "chatDoctor": [
                {
                    "question": "What are the recommended medications for Panic disorder?",
                    "content": "The recommended medications for Panic disorder are lorazepam, alprazolam, clonazepam, paroxetine, venlafaxine, mirtazapine, buspirone, fluvoxamine, imipramine, desvenlafaxine, clomipramine, acamprosate",
                    "analysis": "The medications listed in the content are the recommended ones for treating Panic disorder.",
                    "final_answer": "lorazepam, alprazolam, clonazepam, paroxetine, venlafaxine, mirtazapine, buspirone, fluvoxamine, imipramine, desvenlafaxine, clomipramine, acamprosate"
                },
            ],
            "DDI_binary": [
                {
                    "question": "Is there a reaction between Sibutramine and Icatibant?",
                    "content": "There is a reaction effect identified between Sibutramine and Icatibant.",
                    "analysis": "From the given context, it can be concluded that there is a reaction effect identified between Sibutramine and Icatibant",
                    "final_answer": "yes"
                },
            ],
            "drug_usage": [
                {
                    "question": "Answer the following two questions about acetaminophen:\n Have studies shown adverse effects on preganancy?\n Have studies shown an interaction with alchohol?",
                    "content": "The FDA label for acetaminophen considers it a pregnancy category C drug, meaning this drug has demonstrated adverse effects in animal studies. Acetaminophen interacts with alchohol.",
                    "analysis": "The content shows that acetaminophen has shwon adverse effects on preganancy demonstrated in animal studies. The content has shown that acetaminophen interacts with alchohol",
                    "final_answer": "yes, yes"
                },
            ],
            "usmle_mc": [
                {
                    "question": "A 62-year-old woman presents for a regular check-up. She complains of lightheadedness and palpitations which occur episodically. Past medical history is significant for a myocardial infarction 6 months ago and NYHA class II chronic heart failure. She also was diagnosed with grade I arterial hypertension 4 years ago. Current medications are aspirin 81 mg, atorvastatin 10 mg, enalapril 10 mg, and metoprolol 200 mg daily. Her vital signs are a blood pressure of 135/90 mm Hg, a heart rate of 125/min, a respiratory rate of 14/min, and a temperature of 36.5¬∞C (97.7¬∞F). Cardiopulmonary examination is significant for irregular heart rhythm and decreased S1 intensity. ECG is obtained and is shown in the picture (see image). Echocardiography shows a left ventricular ejection fraction of 39%. Which of the following drugs is the best choice for rate control in this patient? A:Atenolol, B:Diltiazem, C:Propafenone, D:Digoxin",
                    "content": "In a patient with chronic heart failure and palpitations, the first-line treatment for rate control is a medication that targets the atrioventricular (AV) node. This is because the AV node is responsible for regulating the heart rate by controlling the conduction of electrical impulses from the atria to the ventricles. Medications that are commonly used for rate control include beta-blockers, calcium channel blockers, and digoxin. In this case, the patient is already on a high dose of a beta-blocker, so adding another medication from the same class may not be effective. Calcium channel blockers can be used as an alternative, but they may worsen heart failure symptoms. Digoxin is a medication that has been used for decades for rate control in patients with heart failure and atrial fibrillation. It works by increasing the strength of the heart's contractions and slowing down the electrical impulses in the AV node, which results in a slower heart rate. Therefore, digoxin is the best choice for rate control in this patient.",
                    "analysis": "The 62-year-old woman's presentation of lightheadedness and palpitations, an irregular heart rhythm on cardiopulmonary examination, and her history of myocardial infarction and NYHA class II chronic heart failure suggest she may be suffering from a rate control issue related to her heart. The patient is already on a regimen that includes metoprolol, a beta-blocker used for rate control, and the dose is relatively high. This makes adding another beta-blocker like atenolol less effective. The use of calcium channel blockers, like diltiazem, might worsen her heart failure symptoms. Propafenone, a class IC antiarrhythmic, is mainly used to treat conditions that cause a fast heart rate, such as atrial fibrillation and atrial flutter, and ventricular arrhythmias, but may not be suitable for a patient with heart failure. On the other hand, digoxin has been traditionally used for rate control in patients with heart failure and atrial fibrillation, as it increases the strength of the heart's contractions and slows down the electrical impulses in the AV node.",
                    "final_answer": "D",
                },
            ],
            "mmlu_mc": [
                {
                    "question": "A 55-year-old man is brought to the emergency department by his wife because he told her he did not want to live anymore. During the past 6 weeks, he has experienced fatigue, loss of interest in usual activities, a 7-kg (15-lb) weight loss, and insomnia. He has no history of serious medical or psychiatric illness. Vital signs are temperature 37.0¬∞C (98.6¬∞F), pulse 80/min, respirations 16/min, and blood pressure 140/82 mm Hg. Physical examination discloses no abnormalities. Beck Depression Inventory score is 35 (severely depressed). He says he feels guilty that his investments have not done well and that he has ruined his family finances. He reports he hears voices at night telling him he has sinned. In addition to olanzapine, which of the following is the best treatment option for this patient? A:Divalproex, B:Fluoxetine, C:Lamotrigine, D:Lithium carbonate",
                    "content": "The patient is experiencing symptoms of depression, including fatigue, loss of interest, weight loss, and insomnia. He also has feelings of guilt and hears voices at night. These symptoms suggest a major depressive episode with psychotic features. The best treatment option for this condition is a combination of an antidepressant and an antipsychotic medication. While antipsychotics can help with the psychotic symptoms, they do not treat the underlying depression. Therefore, an antidepressant medication is necessary to address the patient's depressive symptoms.",
                    "analysis": "The patient presents with symptoms indicative of a major depressive episode with possible psychotic features, as indicated by his hearing voices. The severity of his depression is confirmed by his high Beck Depression Inventory score. As such, his treatment should address both the depressive symptoms and the psychotic symptoms.",
                    "final_answer": "B",
                },
            ],
            "medmcqa": [
                {
                    "question": "A 42year old female presents with diazepam and alcohol overdose. She is comatose. Temperature is 34.5degC. BP is 100/80 mmHg. Creatinine is 2.4mg/dL, AST -500, GGT- 35 IU. Urine dipstick showed 3+ for blood but urine analysis was normal. USG abdomen was normal. What is the most likely diagnosis? A:Hypothermia, B: Alcoholic hallucinosis, C: Rhabdomyolysis, D: Acute interstitial nephritis",
                    "content": "The patient's presentation suggests a drug overdose with diazepam and alcohol, which can cause respiratory depression and lead to hypoxia. The low body temperature indicates hypothermia, which can be a complication of drug overdose. However, the elevated creatinine level and abnormal liver function tests suggest renal and hepatic dysfunction. The presence of blood in the urine dipstick without any abnormalities on urine analysis suggests myoglobinuria, which can occur in rhabdomyolysis. Rhabdomyolysis is a medical emergency that can be caused by drug overdose, leading to muscle breakdown and release of myoglobin into the bloodstream, which can cause renal failure. Therefore, the most likely diagnosis in this case is rhabdomyolysis.",
                    "analysis": "The patient has presented following an overdose of diazepam and alcohol. Given the symptoms and clinical findings, we are considering several potential diagnoses: Hypothermia, Alcoholic hallucinosis, Rhabdomyolysis, and Acute interstitial nephritis.Hypothermia (option A) could be suggested by the low body temperature, but it doesn't account for all of the patient's symptoms and test results.Alcoholic hallucinosis (option B) could be a possibility given the alcohol overdose, but the patient is comatose and not displaying signs of hallucinations.Acute interstitial nephritis (option D) might cause an increase in creatinine, but it generally does not result in blood in the urine without abnormality in urine analysis.Rhabdomyolysis (option C), on the other hand, can result from a drug overdose. This condition causes muscle breakdown, which releases myoglobin into the bloodstream. Myoglobin can cause renal failure, which could explain the increased creatinine. It also can show up as blood on a urine dipstick test while other aspects of a urine analysis remain normal, due to the fact that common urine dipstick tests cannot differentiate between myoglobin and hemoglobin.",
                    "final_answer": "C",
                },
            ],
            "moderna_interactions": [
                {
                    "question": "Does the therapeutic efficacy of Moderna COVID-19 Vaccine decrease when used in combination with abatacept?",
                    "content": "The therapeutic efficacy of Moderna COVID-19 Vaccine decrease when used in combination with abatacept?",
                    "analysis": "From the given context, it can be concluded that the therapeutic efficacy of Moderna COVID-19 Vaccine does decrease when used with abatacept",
                    "final_answer": "yes"
                },
            ],

        }

        # Define templates for few-shot examples
        self.few_shot_templates = {
            'Inquiry_Analysis': FewShotPromptTemplate(
                examples=self.KNOWLEDGE_AGENT_FS_EXAMPLES["example"],
                example_prompt=PromptTemplate(
                    input_variables=["question", "drugs", "symptoms", "diseases", "knowledge_needed"],
                    template="Question: {question}\nStep 1: Identifying the drug, symptom, and disease:\nDrugs: {drugs}\nSymptoms: {symptoms}\nDiseases: {diseases}\nStep 2: Automated knowledge base matching:\nKnowledge needed: {knowledge_needed}"
                ),
                suffix="Question: {input_question}",
                input_variables=["input_question"]
            ),
            'Knowledge_Acquisition': FewShotPromptTemplate(
                examples=self.FS_EXAMPLES["pubmedqa"],
                example_prompt=PromptTemplate(
                    input_variables=["question", "content", "analysis", "final_answer"],
                    template="Question: {question}\nContent: {content}\nAnalysis: {analysis}\nFinal Answer: {final_answer}"
                ),
                suffix="Question: {input_question}\nContent: {input_content}",
                input_variables=["input_question", "input_content"]
            ),
            'Evidence_Generation': FewShotPromptTemplate(
                examples=self.FS_EXAMPLES["chatDoctor"],
                example_prompt=PromptTemplate(
                    input_variables=["question", "content", "analysis", "final_answer"],
                    template="Question: {question}\nContent: {content}\nAnalysis: {analysis}\nFinal Answer: {final_answer}"
                ),
                suffix="Question: {input_question}\nContent: {input_content}",
                input_variables=["input_question", "input_content"]
            )
        }

    def generate_instruction_prompt(self, prompt_type: str, instruction_data: Dict[str, str]) -> str:
        if prompt_type not in self.instruction_templates:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
        return self.instruction_templates[prompt_type].format(**instruction_data)

    def generate_few_shot_prompt(self, prompt_type: str, input_data: Dict[str, str]) -> str:
        if prompt_type not in self.few_shot_templates:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
        return self.few_shot_templates[prompt_type].format(**input_data)

    def generate_combined_prompt(self, instruction_type: str, instruction_data: Dict[str, str], few_shot_type: str,
                                 few_shot_data: Dict[str, str]) -> str:
        instruction_prompt = self.generate_instruction_prompt(instruction_type, instruction_data)
        few_shot_prompt = self.generate_few_shot_prompt(few_shot_type, few_shot_data)

        combined_prompt = f"{instruction_prompt}\n{few_shot_prompt}"
        return combined_prompt


# Initialize the PromptManager
prompt_manager = PromptManager()

# Example usage
instruction_data = {
    "task": "You are tasked with identifying...",
    "answer_format": "Step 1: Identifying the drug...",
    # Add "not_to_dos" for Evidence Generation type
}

few_shot_data = {
    "input_question": "A 29-year-old woman develops...",
    "input_content": "The recommended medications for Panic disorder..."
    # Add this only for Knowledge Acquisition and Evidence Generation types
}

combined_prompt = prompt_manager.generate_combined_prompt("Inquiry_Analysis", instruction_data, "Inquiry_Analysis",
                                                          few_shot_data)
print(combined_prompt)

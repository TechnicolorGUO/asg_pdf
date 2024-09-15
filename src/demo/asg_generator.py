import subprocess
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
print(result.stdout.decode('utf-8'))



from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # server
model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# not yet tested
def generate_old(context, question, temp=0.7):

    # template = """You are provided with several context excerpts from a research paper. Based on these excerpts, provide a concise and professional summary of the methods described in the paper.	
    # ============================================================
    # {context}
    # ============================================================
    # Question: {question}
    # ============================================================
    # ANSWER:
    # """

    
    template = """You are provided with several context excerpts from a research paper. 
    From these excerpts, identify and summarize the methods used in the paper (not methods from related work or other sources). 
    Based on these excerpts, provide a concise and professional summary of the methods described in the paper. 
    The summary must be a single paragraph of no more than 150 words, and all content must be derived from the provided excerpts.	
    ============================================================
    {context}
    ============================================================
    Question: {question}
    ============================================================
    ANSWER:
    """    

    prompt_template = PromptTemplate.from_template(template)
    formatted_prompt = prompt_template.format(context=context, question=question)

    # generate answer
    inputs = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=1000, temperature=temp)
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the answer by detecting the "ANSWER:" tag
    answer_start = "ANSWER:"
    start_index = res.find(answer_start)
    if start_index != -1:
        answer = res[start_index + len(answer_start):].strip()
    else:
        answer = context
    return answer

def generate(context, temp=0.7):

    template = """
    You are provided with several context excerpts from a research paper. 
    Using only the information in these excerpts, identify the methods used in the paper (excluding methods from related work). 
    Directly provide a concise and professional description in a single paragraph of no more than 120 words. 
    ============================================================
    {context}
    ============================================================
    ANSWER:
    """

    # keyword = "application"
    # template = """
    # You are provided with several context excerpts from a research paper. 
    # Using only the information in these excerpts, identify the {keyword} used in the paper (excluding {keyword} from related work). 
    # Directly provide a concise and professional description in a single paragraph of no more than 120 words. 
    # ============================================================
    # {context}
    # ============================================================
    # ANSWER:
    # """    

    prompt_template = PromptTemplate.from_template(template)
    formatted_prompt = prompt_template.format(context=context)

    # generate answer
    inputs = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=1000, temperature=temp)
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the answer by detecting the "ANSWER:" tag
    answer_start = "ANSWER:"
    start_index = res.find(answer_start)
    if start_index != -1:
        answer = res[start_index + len(answer_start):].strip()
    else:
        answer = context
    return answer

# context = """
# on the learning process. Specifically, it first divides the text embedding into real and imaginary parts in a complex space. Then, it follows the division rule in complex space to compute the angle difference between two text embeddings. After normalization, the angle difference becomes an//
# and syntactic information in language, which broadly affects the performance of downstream tasks, such as text classification (Li et al., 2021), sentiment analysis (Suresh & Ong, 2021; Zhang et al., 2022), semantic matching (Grill et al., 2020; Lu et al., 2020), clustering (Reimers & Gurevych,//
# becomes an objective to be optimized. It is intuitive to optimize the normalized angle difference, because if the normalized angle difference between two text embeddings is smaller, it means that the two text embeddings are closer to each other in the complex space, i.e., their similarity is//
# We first experimented with both short and longtext datasets and showed that AnglE outperforms the SOTA STS models in both transfer and nontransfer STS tasks. For example, AnglE shows an average Spearman correlation of 73.55% in nontransfer STS tasks, compared to 68.03% for SBERT. Then, an ablation//
# Then, an ablation study shows that all components contribute positively to AnglE’s superior performance. Next, we discuss the domainspecific scenarios with limited annotated data that are challenging for AnglElike supervised STS, where it is observed that AnglE can work well with LLMsupervised//
# samples that are dissimilar are selected from different texts within the same minibatch (inbatch negatives). However, supervised negatives are underutilized, and the correctness of inbatch negatives is difficult to guarantee without annotation, which can lead to performance degradation. Although//
# For supervised STS (Reimers & Gurevych, 2019; Su, 2022), most efforts to date employed the cosine function in their training objective to measure the pairwise semantic similarity. However, the cosine function has saturation zones, as shown in Figure 1. It can impede the optimization due to the//
# """

# # question = "Summarize the methods used in this research paper."
# # result = generate_old(context, question)

# result = generate(context)
# print(result)
# print(len(result))

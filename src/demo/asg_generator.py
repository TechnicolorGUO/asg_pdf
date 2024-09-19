import subprocess
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
print(result.stdout.decode('utf-8'))



from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
import torch
import os
import re

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # server
# model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

def generate(context, pipeline, keyword, temp=0.7):
    # template = """
    # Use the following pieces of context to answer the question at the end.
    # Provide a precise answer based on the context in a single paragraph of no more than 120 words:
    # ```
    # ------------------------------------------------------------
    # on the learning process. Specifically, it first divides the text embedding into real and imaginary parts in a complex space. Then, it follows the division rule in complex space to compute the angle difference between two text embeddings. After normalization, the angle difference becomes an//
    # and syntactic information in language, which broadly affects the performance of downstream tasks, such as text classification (Li et al., 2021), sentiment analysis (Suresh & Ong, 2021; Zhang et al., 2022), semantic matching (Grill et al., 2020; Lu et al., 2020), clustering (Reimers & Gurevych,//
    # becomes an objective to be optimized. It is intuitive to optimize the normalized angle difference, because if the normalized angle difference between two text embeddings is smaller, it means that the two text embeddings are closer to each other in the complex space, i.e., their similarity is//
    # We first experimented with both short and longtext datasets and showed that AnglE outperforms the SOTA STS models in both transfer and nontransfer STS tasks. For example, AnglE shows an average Spearman correlation of 73.55% in nontransfer STS tasks, compared to 68.03% for SBERT. Then, an ablation//
    # Then, an ablation study shows that all components contribute positively to AnglE’s superior performance. Next, we discuss the domainspecific scenarios with limited annotated data that are challenging for AnglElike supervised STS, where it is observed that AnglE can work well with LLMsupervised//
    # samples that are dissimilar are selected from different texts within the same minibatch (inbatch negatives). However, supervised negatives are underutilized, and the correctness of inbatch negatives is difficult to guarantee without annotation, which can lead to performance degradation. Although//
    # For supervised STS (Reimers & Gurevych, 2019; Su, 2022), most efforts to date employed the cosine function in their training objective to measure the pairwise semantic similarity. However, the cosine function has saturation zones, as shown in Figure 1. It can impede the optimization due to the//
    # ------------------------------------------------------------

    # Question: What are the methods used in the paper?
    # ```
    # Respond in several complete and logical sentences: 
    # ```
    # Answer: The paper introduces a method called AnglE that computes the angle difference between text embeddings in a complex space for semantic similarity tasks. This approach involves dividing text embeddings into real and imaginary parts, followed by the normalization of these angle differences to optimize performance. The paper also discusses the effectiveness of AnglE in outperforming state-of-the-art models like SBERT across various datasets in both transfer and non-transfer tasks. Additionally, the paper emphasizes the significance of an ablation study that confirms the positive contribution of all components of AnglE, making it particularly effective in challenging scenarios with limited annotated data.
    # ```

    # Now do the real task below:
    # ------------------------------------------------------------
    # {context}
    # ------------------------------------------------------------
    # Question: Given above context, what are the {keyword} used in the paper?
    # ------------------------------------------------------------
    # Answer:
    # """

    template = """
According to the following context, answer the question: What {keyword} are used in the paper?
Please provide a direct answer in a single paragraph of no more than 120 words.
{context}
------------------------------------------------------------
Answer:
"""

    prompt_template = PromptTemplate.from_template(template)
    formatted_prompt = prompt_template.format(context=context, keyword=keyword)

    # 通过传入的 pipeline 生成答案
    outputs = pipeline(
        formatted_prompt, 
        max_new_tokens=150, 
        temperature=temp
    )

    print("Full Output:", outputs[0]["generated_text"])
    print("\n\n\n\n")

    # 从生成的结果中提取答案
    res = outputs[0]["generated_text"]
    answer_start = "Answer:"
    start_index = res.rfind(answer_start)
    if start_index != -1:
        answer = res[start_index + len(answer_start):].strip().split("\n")[0].strip()
    else:
        answer = context
    last_period_pos = answer.rfind('.')
    if last_period_pos != -1:
        answer = answer[:last_period_pos + 1]
    else:
        pass
    return answer



def extract_query_list(text):
    pattern = re.compile(
        r'\[\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*\]'
    )
    match = pattern.search(text)
    if match:
        return match.group(0)
    return None



def generate_sentence_patterns(keyword, pipeline, num_patterns=10, temp=0.7):
    prompt = f"""
You are a helpful assistant that provides only the output requested, without any additional text.

Please generate {num_patterns} commonly used sentence templates in academic papers to describe the keyword '{keyword}'.
- Do not include any explanations, sign-offs, or additional text.
- The list should be in the following format:
[
    "First template should be here",
    "Second template should be here",
    ...
    "Nineth template should be here",
    "Tenth template should be here"
]

Begin your response immediately with the list, and do not include any other text.
"""

    outputs = pipeline(
        prompt,
        max_new_tokens=200,
        temperature=0.3,  # Lower temperature for more focused output
        num_return_sequences=1,
        eos_token_id=pipeline.tokenizer.eos_token_id,
    )

    res = outputs[0]["generated_text"]
    # print("Model Output:\n", res)  # Add this line to see the raw output
    # print("\n\n\n\n")

    query_list = extract_query_list(res)

    return query_list

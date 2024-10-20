'''
1. only retrieve information in (abstract)+introduction+(conclusion)
'''
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
import transformers
import torch
import os
import re
import ast

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
Global_pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    token = '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',
    device_map="auto",
)

# def generate(context, pipeline, temp=0.7):
#     """
#     This version does not include the elements of categorization in the prompt.
#     """

#     template = '''
# Directly generate an abstract based on the following context (a survey paper). The abstract includes 4 elements: background of the general topic, main problems mentioned in the paper, contributions of the survey paper, and the aim of the survey paper. The abstract should be in a single paragraph of no more than 200 words, following the style of a standard academic abstract. Do not include any headings (words like "Background:", "Problems:") or extra explanations except for the abstract.

# Context:
# {context}

# Abstract:
# '''

#     prompt_template = PromptTemplate.from_template(template)
#     formatted_prompt = prompt_template.format(context=context)

#     outputs = pipeline(
#         formatted_prompt, 
#         max_new_tokens=512,  # Adjust as needed
#         temperature=temp,
#         top_p=0.95,
#         repetition_penalty=1.2,
#         eos_token_id=Global_pipeline.tokenizer.eos_token_id
#     )

#     # print("Full Output:", outputs[0]["generated_text"])
#     # print("\n\n\n\n\n")

#     # 从生成的结果中提取答案
#     res = outputs[0]["generated_text"]
#     answer_start = "Abstract:"
#     start_index = res.find(answer_start)
#     if start_index != -1:
#         answer = res[start_index + len(answer_start):].strip()
#         first_double_newline = answer.find("\n\n")
#         if first_double_newline != -1:
#             answer = answer[:first_double_newline].strip()

#     else:
#         answer = res.strip()
#     return answer

abstract1 = """
Large language models (LLMs) augmented with external data have demonstrated remarkable capabilities in completing real-world tasks. External data not only bolsters the models’ domain-specific expertise and temporal relevance but also diminishes incidences of hallucination, thereby enhancing both the controllability and interpretability of outputs. Techniques for integrating external data into LLMs, such as Retrieval-Augmented Generation (RAG) and fine-tuning, are gaining increasing attention and widespread application. Nonetheless, the effective deployment of data-augmented LLMs across various specialized fields presents substantial challenges. These challenges encompass a wide range of issues, from retrieving relevant data and accurately interpreting user intent to fully harnessing the reasoning capabilities of LLMs for complex tasks. We believe that there is no one-size-fits-all solution for data-augmented LLM applications. In practice, underperformance often arises from a failure to correctly identify the core focus of a task or because the task inherently requires a blend of multiple capabilities that must be disentangled for better resolution. In this survey, we propose a RAG task categorization method, classifying user queries into four levels based on the type of external data required and the task’s primary focus: explicit fact queries, implicit fact queries, interpretable rationale queries, and hidden rationale queries. We define these levels of queries, provide relevant datasets, and summarize the key challenges and most effective techniques for addressing these challenges. Finally, we discuss three main forms of integrating external data into LLMs: context, small model, and fine-tuning, highlighting their respective strengths, limitations, and the types of problems they are suited to solve. This work aims to help readers thoroughly understand and decompose the data requirements and key bottlenecks in building LLM applications, offering solutions to the different challenges and serving as a guide to systematically developing such applications.
"""
introduction1 = """
1 Introduction Large Language Models (LLMs) have demonstrated remarkable capabilities, including extensive world knowledge and sophisticated reasoning skills. Despite these advancements, there are significant challenges in effectively deploying them across various specialized fields. These challenges include issues like model hallucinations, misalignment with domain-specific knowledge, among others. Incorporating domain-specific data, particularly private or on-premise data that could not be included in their initial training corpus, is crucial for tailoring LLM applications to meet specific industry needs. Through techniques like RAG and fine tuning, data augmented LLM applications have demonstrated advantages over applications built solely on generic LLMs, in several aspects: • Enhanced Professionalism and Timeliness: The data used to train LLMs often lags in timeliness and may not cover all domains comprehensively, especially proprietary data owned by users. Data augmented LLM applications address this issue by providing more detailed and accurate answers for complex questions, allowing for data updates and customization. • Alignment with Domain Experts: Through the use of and learning from domain-specific data, data augmented LLM applications can exhibit capabilities more like domain experts, such as doctors and lawyers. • Reduction in Model Hallucination: Data augmented LLM applications generate responses based on real data, grounding their reactions in facts and significantly minimizing the possibility of hallucinations. • Improved Controllability and Explainability: The data used can serve as a reference for the model’s predictions, enhancing both controllability and explainability. Despite the enthusiasm for these advancements, developers often struggle and have to invest a significant amount of human labor to meet its expectations (e.g., achieving a high success rate in question answering). Numerous studies [1, 2, 3, 4, 5] highlight the challenges and frustrations involved in constructing a data augmented LLM applications based on technologies like RAG and fine-tuning, particularly in specialized domains such as the legal field, healthcare, manufacturing, and others. These challenges span a wide range, from constructing data pipelines (e.g., data processing and indexing) to leveraging LLMs’ capabilities to achieve complex intelligent reasoning. For example, in applications of finance, there is a frequent need to understand and utilize high-dimensional time series data, whereas in healthcare, medical images or time-series medical records are often essential. Enabling LLMs to comprehend these varied forms of data represents a recurring challenge. On the other hand, in legal and mathematical applications, LLMs typically struggle to grasp long-distance dependencies between different structures. Additionally, depending on the specific application domain, there are increased demands for the interpretability and consistency of LLM responses. The inherent nature of LLMs tends to be characterized by low interpretability and high uncertainty, which poses significant challenges. Enhancing the transparency of LLMs and reducing their uncertainty are critical for increasing trust and reliability in their outputs, especially in fields where precision and accountability are paramount. Through extensive discussions with domain experts and developers, and by carefully analyzing the challenges they face, we have gained a deep understanding that data augmented LLM applications is not a one-size-fits-all solution. The real-world demands, particularly in expert domains, are highly complex and can vary significantly in their relationship with given data and the reasoning difficulties they require. However, developers often do not realize these distinctions and end up with a solution full of performance pitfalls (akin to a house with leaks everywhere). In contrast, if we could fully understand the demands at different levels and their unique challenges, we could build applications accordingly and make the application steadily improve (like constructing a solid and reliable house step by step). Yet, research efforts and existing relevant surveys [6, 7, 8, 9, 10, 11, 12, 13] frequently focus on only one of these levels or a particular topic of technologies. This has motivated us to compile this comprehensive survey, which aims to clearly define these different levels of queries, identify the unique challenges associated with each (Figure 1), and list related works and efforts addressing them. This survey is intended to help readers construct a bird’s-eye view of data augmented LLM applications and also serve as a handbook on how to approach the development of such applications systematically.
"""

abstract2 = """
Large Language Models (LLMs) have revolutionized various applications in natural language processing (NLP) by providing unprecedented text generation, translation, and comprehension capabilities. However, their widespread deployment has brought to light significant concerns regarding biases embedded within these models. This paper presents a comprehensive survey of biases in LLMs, aiming to provide an extensive review of the types, sources, impacts, and mitigation strategies related to these biases. We systematically categorize biases into several dimensions. Our survey synthesizes current research findings and discusses the implications of biases in real-world applications. Additionally, we critically assess existing bias mitigation techniques and propose future research directions to enhance fairness and equity in LLMs. This survey serves as a foundational resource for researchers, practitioners, and policymakers concerned with addressing and understanding biases in LLMs.
"""
introduction2 = """
1. Introduction 1.1 Overview of LLMs and Their Significance Large Language Models (LLMs) have emerged as a cornerstone of contemporary natural language processing (NLP), offering transformative capabilities in text generation, comprehension, and translation. Models such as GPT-3, GPT-4, and their successors have demonstrated remarkable proficiency in generating coherent, contextually relevant text across diverse applications, including conversational agents, automated content creation, and language translation (Brown et al., 2020; Achiam et. al., 2023). These models leverage extensive datasets and advanced deep learning architectures to achieve their performance, enabling unprecedented levels of automation and efficiency in processing and generating human language (Vaswani et al., 2017). LLMs have advanced various domains, including customer service automation, educational tools, and creative industries. Their ability to understand, produce human-like text and even understand the deeper sentiments (Gupta et. al., 2024) has significantly improved user interaction experiences and expanded technological capabilities in everyday applications. 1.2 Motivation for Studying Biases in LLMs Despite their impressive capabilities, LLMs are not without their challenges, notably the issue of inherent biases. Research has shown that these models can perpetuate and even exacerbate existing societal biases present in their training data (Bolukbasi et al., 2016; Caliskan et al., 2017). These biases can manifest in various forms, such as gender bias, racial bias, and contextual bias, potentially leading to unfair or discriminatory outcomes when the models are deployed in real-world scenarios (Binns et. al., 2017; Gupta et. al., 2024). The persistence of bias in LLMs raises critical ethical and operational concerns. For instance, biased outputs from these models can adversely affect marginalized groups, contribute to misinformation, and undermine user trust (Barocas & Selbst, 2016; O'Neil et. al., 2016). Understanding and addressing these biases is crucial to ensuring that LLMs are used responsibly and equitably, fostering broader acceptance and minimizing negative impacts. 1.3 Objectives and Scope of the Survey Although multiple research papers highlight the biases in LLM, the field of LLM is evolving so exponentially that a comprehensive survey of the biases incorporating a deeper fundamental understanding of biases, sources of biases in LLM, current challenges in the field, and future research direction would be a valuable handbook for researchers and practitioner. The current survey intends to fill the gap that exists today in terms of a holistic survey of biases in LLM capturing the recent advancement in the field. This survey aims to provide a comprehensive overview of biases in LLMs by systematically reviewing the existing literature and current practices related to bias detection and mitigation. Our objectives include: 1. Categorizing Biases: We aim to classify various types of biases observed in LLMs, such as demographic, contextual, and algorithmic biases, and analyze their sources and implications (Mehrabi et al., 2019; Zhao et al., 2019). 2. Assessing Impact: This survey will evaluate the impact of these biases and discuss the broader social and ethical implications (Dastin, 2018). 3. Evaluating Mitigation Strategies: We will review current approaches and methodologies for detecting and mitigating biases in LLMs (Zhao et al., 2019). 4. Identifying Future Directions: By synthesizing current research findings, we aim to identify gaps and propose future research directions to advance the field of bias mitigation in LLMs The scope of this survey encompasses an examination of both theoretical and practical aspects of bias in LLMs to provide a holistic view of the challenges and solutions related to this critical issue.
"""

abstract3 = """
Large language models (LLMs) have demonstrated great success in various fields, benefiting from their huge amount of parameters that store knowledge. However, LLMs still suffer from several key issues, such as hallucination problems, knowledge update issues, and lacking domain-specific expertise. The appearance of retrieval augmented generation (RAG), which leverages an external knowledge database to augment LLMs, makes up those drawbacks of LLMs. This paper reviews all significant techniques of RAG, especially in the retriever and the retrieval fusions. Besides, tutorial codes are provided for implementing the representative techniques in RAG. This paper further discusses the RAG training, including RAG with/without datastore update. Then, we introduce the application of RAG in representative natural language processing tasks and industrial scenarios. Finally, this paper discusses the future directions and challenges of RAG for promoting its development.
"""
introduction3 = """
1 INTRODUCTION Large language models (LLMs) [71, 108, 114, 138, 164] have achieved significant advancements in recent years and have become the cornerstone of various applications in the field of natural language processing (NLP). These LLMs are typically pre-trained on a large amount of natural language corpus and then fine-tuned on the specific downstream tasks’ datasets. Recent works [3, 53, 106, 117] demonstrate the success of LLMs can be explained by the fact that language models act as knowledge bases, which refers to implicitly storing the knowledge learned from training datasets in the parameters as internal memory and generating responses by retrieving answers from memory. To store more knowledge for better generation performance, existing works generally enlarge the memory capacity by increasing the volume of parameters [1, 11, 54, 78]. Although existing LLMs have shown great power, there are still several challenges hindering the development of LLMs. One of the most prominent challenges is the hallucination problem [23, 69, 70], which refers to the tendency of LLMs to generate responses that are coherent and fluent but factually incorrect. Another big challenge is the knowledge update issue. To update the knowledge stored in the LLMs’ internal memory [106, 146, 168], it is necessary to retrain/fine-tune LLMs with new data, which is a costly process. Another challenge for general LLMs is lacking of domain-specific expertise [20, 133, 134, 166]. Training a domain-specific LLM, however, demands considerable manpower for dataset collection. To address these challenges, recent works [10, 49, 87] have proposed leveraging an external knowledge database to augment LLMs, known as retrieval-augmented generation (RAG). By supplying LLMs with retrieved relevant factual information, the hallucination problem can be alleviated to some extent. Besides, the knowledge update issue can also be addressed by updating the external knowledge database, which can augment LLMs with up-to-date knowledge. RAG can also convert a general LLM into a domain-specific LLM by constructing and utilizing a domain-specific knowledge database. Therefore, RAG plays an important role in augmenting the functionality of LLMs, making them more accurate, knowledgeable, and reliable in a wide range of applications. Contributions: This paper reviews all techniques involved in RAG for natural language processing. Although there are several survey papers for RAG [41, 59, 88, 162, 171], our survey still has some key insights, (1) This paper systematically introduces each component of RAG, including details about the retriever from building to querying, and techniques of the retrieval fusions with tutorial codes. (2) This paper exhibits different RAG training strategies, including RAG with/without datastore update. (3) This paper further discusses the applications of RAG on downstream NLP tasks and practical NLP scenarios. (4) This paper finally identifies promising future directions for exploring and main challenges for addressing. The remainder of this paper is organized as follows. Section 2 gives an overview of RAG. Section 3 and Section 4 comprehensively introduce all technical details used in retrievers and retrieval fusions. Section 6 presents how to train the RAG with/without new knowledge. Section 7 presents the techniques used in representative NLP tasks. Section 8 shows the applications of RAG in practical NLP scenarios. Section 9 discusses the future directions of RAG. Section 10 makes a final conclusion of this paper.
"""

# result = generate(introduction3, Global_pipeline)
# print(result)

outline1 = """
[
    [1, '1 Abstract'], 
    [1, '2 Introduction'],
        [2, '2.1 Predictive Modeling on Imbalance Data: Problem Overview'], 
        [2, '2.2 Class Imbalance: A Major Challenge in Predictive Modeling'], 
        [2, '2.3 Survey Scope and Organization'], 
            [3, '2.3.1 Dealing with Class Imbalance'], 
            [3, '2.3.2 Clustering Machine Learning Methods'], 
            [3, '2.3.3 Microtubule Dynamics and Behavior'], 
            [3, '2.3.4 Other Methods'], 
        [2, '2.4 Conclusion'], 
    [1, '3 Dealing with Class Imbalance'], 
        [2, '3.1 An Experimental Design to Evaluate Class Imbalance Treatment Methods'], 
        [2, '3.2 An Improved SMOTE Imbalanced Data Classification Method Based on Support Degree'], 
    [1, '4 Clustering Machine Learning Methods'], 
        [2, '4.1 An unsupervised learning approach to resolving the data imbalanced issue in supervised learning problems in functional genomics'], 
    [1, '5 Microtubule Dynamics and Behavior'], 
        [2, '5.1 Active Learning for Class Imbalance Problem'], 
    [1, '6 Conclusion'], 
    [1, '7 Future Research Directions'], 
    [1, '8 References']
]
"""
outline1_no_endline = "[[1, '1 Abstract'], [1, '2 Introduction'], [2, '2.1 Predictive Modeling on Imbalance Data: Problem Overview'], [2, '2.2 Class Imbalance: A Major Challenge in Predictive Modeling'], [2, '2.3 Survey Scope and Organization'], [3, '2.3.1 Dealing with Class Imbalance'], [3, '2.3.2 Clustering Machine Learning Methods'], [3, '2.3.3 Microtubule Dynamics and Behavior'], [3, '2.3.4 Other Methods'], [2, '2.4 Conclusion'], [1, '3 Dealing with Class Imbalance'], [2, '3.1 An Experimental Design to Evaluate Class Imbalance Treatment Methods'], [2, '3.2 An Improved SMOTE Imbalanced Data Classification Method Based on Support Degree'], [1, '4 Clustering Machine Learning Methods'], [2, '4.1 An unsupervised learning approach to resolving the data imbalanced issue in supervised learning problems in functional genomics'], [1, '5 Microtubule Dynamics and Behavior'], [2, '5.1 Active Learning for Class Imbalance Problem'], [1, '6 Conclusion'], [1, '7 Future Research Directions'], [1, '8 References']]"
outline2 = """
[
    [1, '1 Abstract'], 
    [1, '2 Introduction'],
        [2, '2.1 Predictive Modeling Challenges on Imbalance Data'], 
        [2, '2.2 Class Imbalance Treatment Methods'], 
        [2, '2.3 Genomic Machine Learning Techniques'], 
        [2, '2.4 Applications of Class Imbalance Treatment Methods in Real-World Problems'], 
        [2, '2.5 Future Directions'], 
    [1, '3 Dealing with Class Imbalance Issues'], 
        [2, '3.1 SMOTE: Synthetic Minority Over-sampling Technique'], 
        [2, '3.2 An Improved SMOTE Imbalanced Data Classification Method Based on Support Degree'], 
        [2, '3.3 Experimental Design to Evaluate Class Imbalance Treatment Methods'], 
    [1, '4 Genomic Machine Learning Techniques'], 
        [2, '4.1 Boosting for Learning Multiple Classes with Imbalanced Class Distribution'], 
        [2, '4.2 An Unsupervised Learning Approach to Resolving the Data Imbalanced Issue in Supervised Learning Problems in Functional Genomics'], 
    [1, '5 Nanomaterials in Electronic Devices'], 
        [2, '5.1 Active Learning for Class Imbalance Problem'], 
    [1, '6 Conclusion'], 
    [1, '7 References']
]
"""

# original version
def generate_subsection_content(subsection_title, description="", temp=0.7):
    prompt = f"Write a detailed section about: {subsection_title}.\n{description}\n. Do not answer anything else except for the detailed section"
    outputs = Global_pipeline(
        prompt, 
        max_new_tokens=512,  # Adjust as needed
        temperature=temp,
        top_p=0.95,
        repetition_penalty=1.2
    )
    
    generated_content = outputs[0]["generated_text"].strip()
    return generated_content

# 处理解析输入的 outline
def parse_outline(outline):
    # 假设输入的是字符串格式的列表，比如你的 outline1
    outline_list = ast.literal_eval(outline)
    return outline_list

# 将解析的 outline 转换为实际内容生成的过程
def process_outline(outline_list):
    content = ""
    
    # 遍历 outline 的每一部分，生成对应的内容
    for level, section_title in outline_list:
        # 生成每个 section 或 subsection 的内容
        if level == 1:  # 顶级 section 处理
            content += f"# {section_title}\n"
            section_content = generate_subsection_content(section_title)
            content += f"{section_content}\n\n"
        elif level == 2:  # 二级 subsection 处理
            content += f"## {section_title}\n"
            subsection_content = generate_subsection_content(section_title)
            content += f"{subsection_content}\n\n"
        elif level == 3:  # 三级 subsection 处理
            content += f"### {section_title}\n"
            subsection_content = generate_subsection_content(section_title)
            content += f"{subsection_content}\n\n"
    
    return content

# 解析 outline
parsed_outline = parse_outline(outline1)
print(parsed_outline, "\n\n\n\n")

# 处理生成内容
generated_content = process_outline(parsed_outline)
print(generated_content)

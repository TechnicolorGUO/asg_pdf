# 0*. 提取md文本过程中 最好是有# ##等标志 但UnstructuredMarkdownLoader中没有
# 0*. 在部分pdf中abstract标志会出问题 abstract之后没有双换行符
# 0*. authors中是否用spacy或其他库提取人名？
# 1. 部分格式问题如首字母大写导致解析出md有额外空格
# 2. pdf转md过程会产生多余符号 如Test4.md
# 3. 部分introduction提取不准 regex还需根据demo用pdf完善 如有的pdf中没有related work

import re
import os
import json
import subprocess
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

def convert_pdf_to_md(pdf_file, output_dir="output", method="auto"):
    base_name = os.path.splitext(os.path.basename(pdf_file))[0]
    target_dir = os.path.join(output_dir, base_name)

    if os.path.exists(target_dir):
        print(f"Folder for {pdf_file} already exists in {output_dir}. Skipping conversion.")
    else:
        command = ["magic-pdf", "-p", pdf_file, "-o", output_dir, "-m", method]
        try:
            subprocess.run(command, check=True)
            print(f"Successfully converted {pdf_file} to markdown format in {target_dir}.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")

def extract_information_from_md(md_text):
    # Title: 在第一个双换行符之前的内容。
    title_match = re.search(r'^(.*?)(\n\n|\Z)', md_text, re.DOTALL)
    title = title_match.group(1).strip() if title_match else "N/A"

    # Authors: 从第一个双换行符之后，到abstract标志（\n\nAbstract\n\n或者\n\nABSTRACT\n\n 或者\n\nAbstract.\n\n或者\n\nAbstract-\n\n或者\n\nA bstract\n\n(注意这里a和b之间有一个空格)或者\n\nA BSTRACT\n\n(注意这里a和b之间有一个空格)或者\n\nA bstract.\n\n或者\n\nA bstract-\n\n) 之前的内容。这里abstract标识符可以大小写不敏感。
    authors_match = re.search(r'\n\n(.*?)(\n\n[Aa]bstract[\.\- ]?\n\n|\n\n[Aa]bstract[\.\- ]?\n\n|\n\n[Aa]bstract[\.\- ]?\n\n|\n\n[Aa] bstract[\.\- ]?\n\n|\n\n[Aa] BSTRACT[\.\- ]?\n\n)', md_text, re.DOTALL)
    authors = authors_match.group(1).strip() if authors_match else "N/A"

    # Abstract: 从 abstract标志（\n\nAbstract\n\n或者\n\nABSTRACT\n\n 或者\n\nAbstract.\n\n或者\n\nAbstract-\n\n或者\n\nA bstract\n\n(注意这里a和b之间有一个空格)或者\n\nA BSTRACT\n\n(注意这里a和b之间有一个空格)或者\n\nA bstract.\n\n或者\n\nA bstract-\n\n)之后，到 下一个\n\n之前的内容。
    abstract_match = re.search(r'(\n\n[Aa]bstract[\.\- ]?\n\n|\n\n[Aa]bstract[\.\- ]?\n\n|\n\n[Aa]bstract[\.\- ]?\n\n|\n\n[Aa] bstract[\.\- ]?\n\n|\n\n[Aa] BSTRACT[\.\- ]?\n\n)(.*?)(\n\n|\Z)', md_text, re.DOTALL)
    abstract = abstract_match.group(2).strip() if abstract_match else "N/A"

    # Introduction: 从introduction标志（\n\n\1 Introduction\n\n或者\n\n1 INTRODUCTION\n\n 或者\n\n1 Introduction.\n\n或者\n\n1 Introduction-\n\n或者\n\n1 I ntroduction\n\n(注意这里i和n之间有一个空格)或者\n\n1 I NTRODUCTION\n\n或者\n\n1 I ntroduction.\n\n或者\n\nI ntroduction-\n\n)之后，到related work标志（\n\n2 Related Work\n\n或者\n\n2 Related Work.\n\n或者\n\n2 Related Work-\n\n或者\n\n2 R elated Work\n\n(注意这里r和n\e之间有一个空格)或者\n\n2 R ELATED WORK\n\n或者\n\n2 R elated Work.\n\n或者\n\n2 R elated Work-\n\n\n\n2 R ELATED  W ORK\n\n或者\n\n2 R ELATED  W ORK.\n\n或者\n\n2 R ELATED  W ORK-\n\n）之前。introduction标志和related work标志的大小写都不敏感
    introduction_match = re.search(
        r'\n\n[1iI][\.\- ]?\s*[Ii]\s*[nN]\s*[tT]\s*[rR]\s*[oO]\s*[dD]\s*[uU]\s*[cC]\s*[tT]\s*[iI]\s*[oO]\s*[nN][\.\- ]?\n\n(.*?)'
        r'(?=\n\n[2rR][\.\- ]?\s*[Rr]\s*[Ee]\s*[Ll]\s*[Aa]\s*[Tt]\s*[Ee]\s*[Dd]\s*[Ww]\s*[Oo]\s*[Rr]\s*[Kk][\.\- ]?\n\n)',
        md_text, re.DOTALL
    )
    introduction = introduction_match.group(1).strip() if introduction_match else "N/A"

    extracted_data = {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "introduction": introduction
    }
    return extracted_data
# convert_pdf_to_md("Test9.pdf")
# convert_pdf_to_md("Test4.pdf")

def process_md_file(md_file_path):
    loader = UnstructuredMarkdownLoader(md_file_path)
    data = loader.load()
    assert len(data) == 1, "Expected exactly one document in the markdown file."
    assert isinstance(data[0], Document), "The loaded data is not of type Document."
    extracted_text = data[0].page_content

    extracted_data = extract_information_from_md(extracted_text)
    json_file_path = 'extracted_info.json'
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)

    return extracted_text

def process_pdf(pdf_file):
    output_dir = "output"
    base_name = os.path.splitext(os.path.basename(pdf_file))[0]
    target_dir = os.path.join(output_dir, base_name, "auto")

    # 1. Convert PDF to markdown if the folder doesn't exist
    convert_pdf_to_md(pdf_file, output_dir)

    # 2. Process the markdown file in the output directory
    md_file_path = os.path.join(target_dir, f"{base_name}.md")
    if not os.path.exists(md_file_path):
        raise FileNotFoundError(f"Markdown file {md_file_path} does not exist. Conversion might have failed.")

    return process_md_file(md_file_path)

# md_text = process_md_file("./Test2.md")
# md_text = process_md_file("./Test3.md")
md_text = process_md_file("./Test4.md")
print(md_text)










# convert_pdf_to_md("A Hybrid Re-sampling Method for SVM Learning from Imbalanced Data Sets.pdf")
# convert_pdf_to_md("A Multistrategy Approach for Digital Text Categorization from Imbalanced Documents.pdf")
# convert_pdf_to_md("A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data.pdf")
# convert_pdf_to_md("Active Learning for Class Imbalance Problem.pdf")
# convert_pdf_to_md("Active Learning from Positive and Unlabeled Data.pdf")
# convert_pdf_to_md("An Active Under-sampling Approach for Imbalanced Data Classification.pdf")
# convert_pdf_to_md("An Experimental Design to Evaluate Class Imbalance Treatment Methods.pdf")
# convert_pdf_to_md("An Improved AdaBoost Algorithm for Unbalanced Classification Data.pdf")
# convert_pdf_to_md("An Improved SMOTE Imbalanced Data Classification Method Based on Support Degree.pdf")
# convert_pdf_to_md("An Unsupervised Learning Approach to Resolving the Data Imbalanced Issue in Supervised Learning Problems in Functional Genomics.pdf")
# convert_pdf_to_md("Asymmetric Bagging and Random Subspace for Support Vector Machines-Based Relevance Feedback in Image Retrieval.pdf")
# convert_pdf_to_md("Boosting for Learning Multiple Classes with Imbalanced Class Distribution.pdf")
# convert_pdf_to_md("Class Probability Estimates are Unreliable for Imbalanced Data (and How to Fix Them).pdf")
# convert_pdf_to_md("Combating the Small Sample Class Imbalance Problem Using Feature Selection.pdf")
# convert_pdf_to_md("Disturbing Neighbors Ensembles of Trees for Imbalanced Data.pdf")
# convert_pdf_to_md("Evaluation Measures for Ordinal Regression.pdf")
# convert_pdf_to_md("Experimental Perspectives on Learning from Imbalanced Data.pdf")
# convert_pdf_to_md("Explicitly Representing Expected Cost: An Alternative to ROC Representation.pdf")
# convert_pdf_to_md("Exploratory Undersampling for Class-Imbalance Learning.pdf")
# convert_pdf_to_md("FAST: A ROC-based Feature Selection Metric for Small Samples and Imbalanced Data Classification Problems.pdf")
# convert_pdf_to_md("FSVM-CIL: Fuzzy Support Vector Machines for Class Imbalance Learning.pdf")
# convert_pdf_to_md("Feature Selection for Text Categorization on Imbalanced Data.pdf")
# convert_pdf_to_md("Fuzzy Support Vector Machines.pdf")
# convert_pdf_to_md("Learning on the Border: Active Learning in Imbalanced Data Classification.pdf")
# convert_pdf_to_md("Learning from Imbalanced Data.pdf")
# convert_pdf_to_md("RAMOBoost: Ranked Minority Oversampling in Boosting.pdf")
# convert_pdf_to_md("RUSBoost: A Hybrid Approach to Alleviating Class Imbalance.pdf")
# convert_pdf_to_md("Regression Error Characteristic Surfaces.pdf")
# convert_pdf_to_md("The Relationship Between Precision-Recall and ROC Curves.pdf")
# convert_pdf_to_md("Theoretical Analysis of a Performance Measure for Imbalanced Data.pdf")
# convert_pdf_to_md("Training Cost-Sensitive Neural Networks with Methods Addressing the Class Imbalance Problem.pdf")
# convert_pdf_to_md("Wrapper-based Computation and Evaluation of Sampling Methods for Imbalanced Datasets.pdf")
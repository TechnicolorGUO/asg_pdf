import transformers
import torch
import pandas as pd
import os
import json
import re
import ast
from .survey_generator_api import *
from .asg_abstract import AbstractGenerator
from .asg_conclusion import ConclusionGenerator
import pandas as df

class OutlineGenerator():
    def __init__(self, pipeline, df, cluster_names, mode='desp'):
        self.pipeline = pipeline
        # self.pipeline.model.load_adapter("technicolor/llama3.1_8b_outline_generation")
        self.pipeline.model.set_adapter("outline")
        self.df = df
        self.cluster = [{'label': i, 'name': cluster_names[i]} for i in range(len(cluster_names))]
        self._add_cluster_info()
        self.mode = mode

    def _add_cluster_info(self):
        label_to_info = {label: self.df[self.df['label'] == label] for label in range(len(self.cluster))}
        for cluster in self.cluster:
            cluster['info'] = label_to_info[cluster['label']]

    def get_cluster_info(self):
        return self.cluster

    def generate_claims(self):
        result = []
        if self.mode == 'desp':
            for i in range(len(self.cluster)):
                cluster = self.cluster[i]
                claims = ''
                for j in range(len(cluster['info'])):
                    # claims = cluster['info'].iloc[j]['retrieval_result'] + '\n' + claims
                    claims = cluster['info'].iloc[j]['ref_title'] + '\n' + claims
                result.append(claims)
        else:
            for i in range(len(self.cluster)):
                cluster = self.cluster[i]
                claims = ''
                data = cluster['info']
                for j in range(len(data)):
                    entry = data.iloc[j]
                    title = entry['title']
                    abstract = entry['abstract']
                    prompt = f'''
                        Title:
                        {title}
                        Abstract:
                        {abstract}
                        Task:
                        Conclude new findings and null findings from the abstract in one sentence in the atomic format. Do not separate
                        new findings and null findings. The finding must be relevant to the title. Do not include any other information.
                        Definition:
                        A scientific claim is an atomic verifiable statement expressing a finding about one aspect of a scientific entity or
                        process, which can be verified from a single source.'''
                    
                    messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                    ]
                    
                    outputs = self.pipeline(
                        messages,
                        max_new_tokens=256,
                    )
                    claim = outputs[0]["generated_text"][-1]['content']
                    print(claim)
                    print('+++++++++++++++++++++++++++++++++')
                    claims = claims + '\n' + claim
                result.append(claims)
        return result

    def generate_outline(self, survey_title):
        claims = self.generate_claims()
        cluster_with_claims = ""
        for i in range(len(self.cluster)):
            cluster = self.cluster[i]
            cluster_with_claims = cluster_with_claims + f'Cluster {i}: {cluster["name"]}\n' + "Descriptions for entities in this cluster: \n" + claims[i] + '\n\n'
        # system_prompt = f'''
        #     You are a helpful assistant who is helping a researcher to generate an outline for a survey paper.
        #     The references used by this survey paper have been clustered into different categories.
        #     The researcher will provides you with the title of the survey paper
        #     together with the cluster names and the descriptions for entities in each cluster.
        #     '''
        system_prompt = f'''Generate the outline of the survey paper following the format of the example : [[1, '1 Introduction'], [1, '2 Perturbations of (co)differentials'], [2, '2.1 Derivations of the tensor algebra'], [more sections...]].\
        The first element in the sub-list refers to the hierachy of the section name (from 1 to 3).\
        The second element in the sub-list refers to the section name.
        '''

        example_json = {"title":"A Survey of Huebschmann and Stasheff's Paper: Formal Solution of the Master Equation via HPT and Deformation Theory","outline":[{"title":"1 Introduction","outline":[]},{"title":"2 Perturbations of (co)differentials","outline":[{"title":"2.1 Derivations of the tensor algebra","outline":[]},{"title":"2.2 Coderivations of the tensor coalgebra","outline":[]},{"title":"2.3 Coderivations of the symmetric coalgebra","outline":[]},{"title":"2.4 DGLA\u2019s and perturbations of the codifferential","outline":[]},{"title":"2.5 Strongly homotopy Lie algebras","outline":[]},{"title":"2.6 The Hochschild chain complex and DGA\u2019s","outline":[]},{"title":"2.7 Strongly homotopy associative algebras","outline":[]}]},{"title":"3 Master equation","outline":[]},{"title":"4 Twisting cochain","outline":[{"title":"4.1 Differential on Hom","outline":[]},{"title":"4.2 Cup product and cup bracket","outline":[]},{"title":"4.3 Twisting cochain","outline":[]}]},{"title":"5 Homological perturbation theory (HPT)","outline":[{"title":"5.1 Contraction","outline":[]},{"title":"5.2 The first main theorem.","outline":[]}]},{"title":"6 Corollaries and the second main theorem","outline":[{"title":"6.1 Other corollaries of Theorem\u00a01.","outline":[]},{"title":"6.2 The second main theorem","outline":[]}]},{"title":"7 Differential Gerstenhaber and BV algebras","outline":[{"title":"7.1 Differential Gerstenhaber algebras","outline":[]},{"title":"7.2 Differential BV algebras","outline":[]},{"title":"7.3 Formality","outline":[{"title":"7.3.1 Formality of differential graded P\ud835\udc43Pitalic_P-algebras","outline":[]},{"title":"7.3.2 Examples","outline":[]}]},{"title":"7.4 Differential BV algebras and formality","outline":[]}]},{"title":"8 Deformation theory","outline":[]},{"title":"References","outline":[]}]}
        # user_prompt = {"survey_title":survey_title, "claims":cluster_with_claims}
        user_prompt = f'''Generate the outline of the survey paper given the title:{survey_title}, and three lists of sentences describing each cluster of the references used by this survey:{cluster_with_claims}'''

        messages = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content":"[[1, '1 Abstract'], [1, '2 Introduction'], "}
            ] 

        outputs = self.pipeline(
            messages,
            max_new_tokens=4096,
        )
        result = outputs[0]["generated_text"][-1]['content']

        self.pipeline.model.disable_adapters()

        return messages, result

    
def parseOutline(survey_id):
    file_path = f'./src/static/data/txt/{survey_id}/outline.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    prefix = data['messages'][2]['content']   
    response = data['outline']
    print(response)

    # Extract content between the first '[' and the last ']'
    def extract_first_last(text):
        first_match = re.search(r'\[', text)
        last_match = re.search(r'\](?!.*\])', text)  # Negative lookahead to find the last ']'
        if first_match and last_match:
            return '[' + text[first_match.start()+1:last_match.start()] + ']'
        return None

    # prefix_extracted = extract_first_last(prefix)
    response_extracted = extract_first_last(response)
    print(response_extracted)

    # if prefix_extracted:
    #     prefix_list = ast.literal_eval(prefix_extracted)
    # else:
    #     prefix_list = None

    if response_extracted:
        outline_list = ast.literal_eval(response_extracted)
    else:
        outline_list = None

    result = []
    # for item in prefix_list:
    #     outline_list.append(item)
    for item in outline_list:
        result.append(item)
    
    print("The result is: ", result)
    return result

def generateOutlineHTML(survey_id):
    outline_list = parseOutline(survey_id)
    print(outline_list)
    html = '''
    <div class="container-fluid w-50 d-flex flex-column justify-content-center align-items-center">

        <style>
            /* 不同层级的样式 */
            .level-1 {
                font-size: 20px;
                font-weight: bold;
                position: relative;
                padding-right: 40px; /* 为箭头留出空间 */
            }
            .level-2 {
                font-size: 18px;
                padding-left: 40px;
            }
            .level-3 {
                font-size: 16px;
                padding-left: 80px;
            }
            .list-group-item {
                border: none;
            }
            
            /* 自定义卡片样式 */
            .custom-card {
                background-color: #fff;
                border-radius: 8px;
                padding: 20px;
                margin-top: 20px;
                width: 100%;
                max-width: 800px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 
                            0 6px 20px rgba(0, 0, 0, 0.1);
            }

            /* 自定义卡片主体样式 */
            .custom-card-body {
                padding: 20px;
            }

            /* 折叠图标样式 */
            .collapse-icon {
                background: none;
                border: none;
                padding: 0;
                position: absolute;
                right: 10px;
                top: 50%;
                transform: translateY(-50%) rotate(0deg);
                cursor: pointer;
                font-size: 16px;
                /* 旋转过渡效果 */
                transition: transform 0.2s;
            }
            /* 去除按钮聚焦时的轮廓 */
            .collapse-icon:focus {
                outline: none;
            }
            /* 当折叠展开时旋转图标 */
            .collapsed .collapse-icon {
                transform: translateY(-50%) rotate(0deg);
            }
            .in .collapse-icon {
                transform: translateY(-50%) rotate(90deg);
            }
        </style>

        <div class="custom-card">
            <div class="custom-card-body">
                <ul class="list-group list-group-flush">
    '''

    # 添加默认的一级标题内容
    default_items = [[1, '1. Abstract'], [1, '2. Introduction']]

    # 将默认项与解析出的纲要列表合并
    combined_list = default_items + outline_list

    # 构建树形结构，以便检测一级标题是否有子标题
    def build_outline_tree(outline_list):
        sections = []
        stack = []
        for level, content in outline_list:
            level = int(level)
            node = {'level': level, 'content': content, 'subitems': []}
            if level == 1:
                sections.append(node)
                stack = [node]
            elif level == 2:
                if stack:
                    parent = stack[-1]
                    parent['subitems'].append(node)
                    # stack.append(node)
                else:
                    sections.append(node)
            elif level == 3:
                if stack:
                    parent = stack[-1]
                    parent['subitems'].append(node)
                else:
                    sections.append(node)
        return sections

    sections = build_outline_tree(combined_list)

    # 生成 HTML
    def generate_html_from_sections(sections):
        html = ''
        section_index = 1  # 用于生成唯一的 ID

        def generate_node_html(node):
            nonlocal section_index
            level = node['level']
            content = node['content']
            has_subitems = len(node['subitems']) > 0
            if level == 1:
                # 一级标题
                if has_subitems:
                    # 如果有子标题，添加下拉图标和可折叠功能
                    section_id = f"outline_collapseSection{section_index}"
                    section_index += 1
                    node_html = f'''
                        <li class="list-group-item level-1">
                            {content}
                            <a class="collapsed" data-toggle="collapse" data-target="#{section_id}" aria-expanded="true" aria-controls="{section_id}">
                                &#9654; <!-- 右箭头表示折叠状态 -->
                            </a>
                            <ul class="list-group collapse in" id="{section_id}">
                    '''
                    for subitem in node['subitems']:
                        node_html += generate_node_html(subitem)
                    node_html += '''
                            </ul>
                        </li>
                    '''
                else:
                    # 如果没有子标题，不显示下拉图标
                    node_html = f'''
                        <li class="list-group-item level-1">
                            {content}
                        </li>
                    '''
            elif level == 2:
                    node_html = f'<li class="list-group-item level-2">{content}</li>'
            elif level == 3:
                # 三级标题直接显示，已经在二级标题中处理
                node_html = f'<li class="list-group-item level-3">{content}</li>'
            return node_html

        for section in sections:
            html += generate_node_html(section)

        return html

    html += generate_html_from_sections(sections)

    html += '''
                </ul>
            </div>
        </div>
        <!-- 添加 Bootstrap v3.3.0 的 JavaScript 来处理折叠功能 -->
        <script>
        $(document).ready(function(){
            // 切换箭头方向
            $('.collapsed').click(function(){
                $(this).toggleClass('collapsed');
            });
        });
        </script>
    </div>
    '''
    print(html)
    print('+++++++++++++++++++++++++++++++++')
    return html



def generateSurvey(survey_id, pipeline, title, context):
    outline = parseOutline(survey_id)
    default_items = [[1, '1. Abstract'], [1, '2. Introduction']]
    outline = str(default_items + outline)
    
    client = getQwenClient()

    temp = {"survey_id":survey_id, "outline":outline, "survey_title":title, "context": context, "abstract": "", "introduction": "", "content":"", "conclusion": "","html":""}

    generated_survey_paper = generate_survey_paper(outline, context, client)
    print("Generated Survey Paper:\n", generated_survey_paper)

    generated_introduction = generate_introduction(generated_survey_paper, client)
    print("\nGenerated Introduction:\n", generated_introduction)

    abs_generator = AbstractGenerator(pipeline)
    abstract = abs_generator.generate(title, generated_introduction)
    print("\nGenerated Abstract:\n", abstract)
    con_generator = ConclusionGenerator(pipeline)
    conclusion = con_generator.generate(title, generated_introduction)
    print("\nGenerated Conclusion:\n", conclusion)

    temp["abstract"] = abstract
    temp["introduction"] = generated_introduction
    temp["content"] = generated_survey_paper
    temp["conclusion"] = conclusion

    with open(f'./src/static/data/txt/{survey_id}/generated_result.json', 'w') as f:
        json.dump(temp, f)

    print("Survey has been saved to static.")

    return

        

if __name__ == '__main__':
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    context = '''
    Many paradigms have been proposed to asses informativeness of data samples for active learning. One of the popular approaches is selecting the most uncertain data sample, i.e the data sample in which current classifier is least confident. Some other approaches are selecting the sample which yields a model with minimum risk or the data sample which yields fastest convergence in gradient based methods.//
    An active under-sampling approach is presented in this paper to change the data distribution of training datasets, and improve the classification accuracy of minority classes while maintaining overall classification performance.//
    In this paper, we propose an uncertainty-based active learning algorithm which requires only samples of one class and a set of unlabeled data in order to operate.//
    The principal contribution of our work is twofold: First, we use Bayes’ rule and density estimation to avoid the need to have a model of all classes for computing the uncertainty measure.//
    This technique reduces the number of input parameters of the problem. At the rest of this paper, we first review recent related works in the fields of active learning and active one-class learning (section II).//
    The classifier predicts that all the samples are non-fraud, it will have a quite high accuracy. However, for problems like fraud detection, minority class classification accuracy is more critical.//
    The algorithm used and the features selected are always the key points at design time, and many experiments are needed to select the final algorithm and the best suited feature set.//
    Active learning works by selecting among unlabeled data, the most informative data sample. The informativeness of a sample is the amount of accuracy gain achieved after adding it to the training set.//
    Some other approaches are selecting the sample which yields a model with minimum risk or the data sample which yields fastest convergence in gradient based methods.//
    In this paper, we propose a novel approach reducing each within group error, BABoost, that is a variant of AdaBoost.//
    Simulations on different unbalanced distribution data and experiments performed on several real datasets show that the new method is able to achieve a lower within group error.//
    Active learning with early stopping can achieve a faster and scalable solution without sacrificing prediction performance.//
    We also propose an efficient Support Vector Machine (SVM) active learning strategy which queries a small pool of data at each iterative step instead of querying the entire dataset.//
    The second part consists of applying a treatment method and inducing a classifier for each class distribution.//
    This time we measured the percentage of the performance loss that was recovered by the treatment method.//
    We used two well-known over-sampling methods, random over-sampling and SMOTE.//
    We tested our proposed technique on a sample of three representative functional genomic problems: splice site, protein subcellular localization and phosphorylation site prediction problems.//
    Among the possible PTMs, phosphorylation is the most studied and perhaps the most important.//
    The second part consists of applying a treatment method and inducing a classifier for each class distribution.//
    We show that Active Learning (AL) strategy can be a more efficient alternative to resampling methods to form a balanced training set for the learner in early stages of the learning.//
    '''

    Global_pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    token = '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',
    device_map="auto",
)
    Global_pipeline.model.load_adapter(peft_model_id = "technicolor/llama3.1_8b_outline_generation", adapter_name="outline")
    Global_pipeline.model.load_adapter(peft_model_id ="technicolor/llama3.1_8b_conclusion_generation", adapter_name="conclusion")
    Global_pipeline.model.load_adapter(peft_model_id ="technicolor/llama3.1_8b_abstract_generation", adapter_name="abstract")


    # generateOutlineHTML('test')
    generateSurvey("test", Global_pipeline,"Predictive modeling of imbalanced data",context)





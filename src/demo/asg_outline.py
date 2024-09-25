import transformers
import torch
import pandas as pd
import os
import json
import re
import ast

class OutlineGenerator():
    def __init__(self, pipeline, df, cluster_names, mode='desp'):
        self.pipeline = pipeline
        self.pipeline.model.load_adapter("technicolor/llama3.1_8b_outline_generation")
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
    
    return result

def generateOutlineHTML(survey_id):
    outline_list = parseOutline(survey_id)
    print(outline_list)
    html = '''
    <div class="outline-container">
        <style>
            body, html {
                margin: 0;
                padding: 0;
                height: 100%; /* Ensures the full height of the web page */
                display: flex;
                justify-content: center; /* Centers content horizontally */
                align-items: center; /* Centers content vertically */
                background-color: #f0f0f0; /* Light grey background */
            }
            .outline {
                width: 60%; /* Responsive width */
                padding: 20px;
                border: 1px solid #ccc;
                border-radius: 8px;
                background-color: #fff; /* White background for the outline */
                box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Subtle shadow */
                text-align: left;
            }
            .outline ul {
                list-style-type: none; /* Removes list numbering */
                padding-left: 0; /* Removes default padding */
            }
            .outline li {
                margin-bottom: 10px; /* More space between items */
                font-size: 16px; /* Larger font for better readability */
                line-height: 1.5; /* Improved line spacing */
            }
        </style>
        <div class="outline">
            <ul>
                <li>1. Abstract</li>
                <li>2. Introduction</li>
    '''
    for item in outline_list:
        html += f'<li>{item[1]}</li>'
    html += '</ul></div></div>'
    print(html)
    print('+++++++++++++++++++++++++++++++++')
    return html
        

if __name__ == '__main__':
    generateOutlineHTML('test')


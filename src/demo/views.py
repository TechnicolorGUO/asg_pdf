





from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
import os
import json
import requests
import time
import pandas as pd
import numpy as np

import traceback

from demo.category_and_tsne import ref_category_desp
from demo.ref_paper_desp import ref_desp
import hashlib
import pdb
import re
import pke
import networkx as nx
from collections import defaultdict
import os

DATA_PATH = 'static/data/'
TXT_PATH = 'static/reftxtpapers/overall/'

Survey_dict = {
    '2742488' : 'Energy Efficiency in Cloud Computing',
    '2830555' : 'Cache Management for Real-Time Systems',
    '2907070' : 'Predictive Modeling on Imbalanced Data',
    '3073559' : 'Malware Detection with Data Mining',
    '3274658' : 'Analysis of Handwritten Signature'
}



Survey_Topic_dict = {
    '2742488' : ['energy'],
    '2830555' : ['cache'],
    '2907070' : ['imbalanced'],
    '3073559' : ['malware', 'detection'],
    '3274658' : ['handwritten', 'signature']
}


Survey_n_clusters = {
    '2742488' : 3,
    '2830555' : 3,
    '2907070' : 3,
    '3073559' : 3,
    '3274658' : 2
}

Global_survey_id = ""
Global_ref_list = []
Global_category_description = []
Global_category_label = []
Global_df_selected = ""


from demo.taskDes import absGen, introGen,introGen_supervised, methodologyGen, conclusionGen
from demo.category_and_tsne import clustering, get_cluster_description, clustering_with_criteria


class reference_collection(object):
    def __init__(
            self,
            input_df
    ):
        self.input_df = input_df

    def full_match_with_entries_in_pd(self, query_paper_titles):
        entries_in_pd = self.input_df.copy()
        entries_in_pd['ref_title'] = entries_in_pd['ref_title'].apply(str.lower)
        query_paper_titles = [i.lower() for i in query_paper_titles]

        # matched_entries = entries_in_pd[entries_in_pd['ref_title'].isin(query_paper_titles)]
        matched_entries = self.input_df[entries_in_pd['ref_title'].isin(query_paper_titles)]
        #print(matched_entries.shape)
        return matched_entries,matched_entries.shape[0]

    # select the sentences that can match with the topic words
    def match_ref_paper(self, query_paper_titles,match_mode='full', match_ratio=70):
        # query_paper_title = query_paper_title.lower()
        # two modes for str matching
        if match_mode == 'full':
            matched_entries, matched_num = self.full_match_with_entries_in_pd(query_paper_titles)
        return matched_entries, matched_num


def generate_uid():
    uid_str=""
    hash = hashlib.sha1()
    hash.update(str(time.time()).encode('utf-8'))
    uid_str= hash.hexdigest()[:10]

    return uid_str

def index(request):
    return render(request, 'demo/index.html')



class PosRank(pke.unsupervised.PositionRank):
    def __init__(self):
        """Redefining initializer for PositionRank."""
        super(PosRank, self).__init__()
        self.positions = defaultdict(float)
        """Container the sums of word's inverse positions."""
    def candidate_selection(self,grammar=None,maximum_word_number=3,minimum_word_number=2):
        if grammar is None:
            grammar = "NP:{<ADJ>*<NOUN|PROPN>+}"

        # select sequence of adjectives and nouns
        self.grammar_selection(grammar=grammar)

        # filter candidates greater than 3 words
        for k in list(self.candidates):
            v = self.candidates[k]
            #pdb.set_trace()
            #if len(k) < 3:
            #    del self.candidates[k]
            if len(v.lexical_form) > maximum_word_number or len(v.lexical_form) < minimum_word_number:
                #if len(v.lexical_form) < minimum_word_number:
                #    pdb.set_trace()
                del self.candidates[k]

def clean_str(input_str):
    input_str = str(input_str).strip().lower()
    if input_str == "none" or input_str == "nan" or len(input_str) == 0:
        return ""
    input_str = input_str.replace('\\n',' ').replace('\n',' ').replace('\r',' ').replace('——',' ').replace('——',' ').replace('__',' ').replace('__',' ').replace('........','.').replace('....','.').replace('....','.').replace('..','.').replace('..','.').replace('..','.').replace('. . . . . . . . ','. ').replace('. . . . ','. ').replace('. . . . ','. ').replace('. . ','. ').replace('. . ','. ')
    input_str = re.sub(r'\\u[0-9a-z]{4}', ' ', input_str).replace('  ',' ').replace('  ',' ')
    return input_str

def PosRank_get_top5_ngrams(input_pd):

    pos = {'NOUN', 'PROPN', 'ADJ'}
    #extractor = pke.unsupervised.TextRank()
    #extractor = pke.unsupervised.PositionRank()
    extractor = PosRank()

    #input_str=input_pd["abstract"][0].replace('-','')#.value()

    #pdb.set_trace()

    #for (keyphrase, score) in extractor.get_n_best(n=5, stemming=True):#stemming=False
    #    print(keyphrase, score)
    abs_top5_unigram_list_list = []
    abs_top5_bigram_list_list = []
    abs_top5_trigram_list_list = []
    intro_top5_unigram_list_list = []
    intro_top5_bigram_list_list = []
    intro_top5_trigram_list_list = []

    for line_index,pd_row in input_pd.iterrows():

        input_str=pd_row["abstract"].replace('-','')
        extractor.load_document(input=input_str,language='en',normalization=None)
        #extractor.load_document(input=input_str,language="en",normalization='stemming')

        #unigram
        unigram_extractor=extractor
        #unigram_extractor.candidate_weighting(window=1,pos=pos,top_percent=0.33)
        unigram_extractor.candidate_selection(maximum_word_number=1,minimum_word_number=1)
        unigram_extractor.candidate_weighting(window=6,pos=pos,normalized=False)
        abs_top5_unigram_list = []
        for (keyphrase, score) in unigram_extractor.get_n_best(n=5, stemming=True):
            keyphrase = keyphrase.replace('-','')
            if len(keyphrase)>2:
                abs_top5_unigram_list.append(keyphrase)
        #pdb.set_trace()
        #bigram
        bigram_extractor=extractor
        #bigram_extractor.candidate_weighting(window=2,pos=pos,top_percent=0.33)
        #abs_top5_bigram = extractor.get_n_best(n=5, stemming=True)#stemming=False
        bigram_extractor.candidate_selection(maximum_word_number=2,minimum_word_number=2)
        bigram_extractor.candidate_weighting(window=6,pos=pos,normalized=False)
        abs_top5_bigram_list = []
        for (keyphrase, score) in bigram_extractor.get_n_best(n=5, stemming=True):
            keyphrase = keyphrase.replace('-','')
            if len(keyphrase)>2:
                abs_top5_bigram_list.append(keyphrase)

        #trigram
        trigram_extractor=extractor
        #trigram_extractor.candidate_weighting(window=3,pos=pos,top_percent=0.33)
        trigram_extractor.candidate_selection(maximum_word_number=3,minimum_word_number=3)
        trigram_extractor.candidate_weighting(window=6,pos=pos,normalized=False)
        abs_top5_trigram_list = []
        for (keyphrase, score) in trigram_extractor.get_n_best(n=5, stemming=True):
            keyphrase = keyphrase.replace('-','')
            if len(keyphrase)>2:
                abs_top5_trigram_list.append(keyphrase)

        '''
        input_str=pd_row["intro"].replace('-','')
        extractor.load_document(input=input_str,language='en',normalization=None)

        #unigram
        extractor.candidate_weighting(window=1,pos=pos,top_percent=0.33)
        intro_top5_unigram_list = []
        for (keyphrase, score) in extractor.get_n_best(n=5, stemming=True):
            intro_top5_unigram_list.append(keyphrase)

        #bigram
        extractor.candidate_weighting(window=2,pos=pos,top_percent=0.33)
        #intro_top5_bigram = extractor.get_n_best(n=5, stemming=True)#stemming=False
        intro_top5_bigram_list = []
        for (keyphrase, score) in extractor.get_n_best(n=5, stemming=True):
            intro_top5_bigram_list.append(keyphrase)

        #trigram
        extractor.candidate_weighting(window=3,pos=pos,top_percent=0.33)
        intro_top5_trigram_list = []
        for (keyphrase, score) in extractor.get_n_best(n=5, stemming=True):
            intro_top5_trigram_list.append(keyphrase)
        '''

        abs_top5_unigram_list_list.append(abs_top5_unigram_list)
        abs_top5_bigram_list_list.append(abs_top5_bigram_list)
        abs_top5_trigram_list_list.append(abs_top5_trigram_list)
        '''
        intro_top5_unigram_list_list.append(intro_top5_unigram_list)
        intro_top5_bigram_list_list.append(intro_top5_bigram_list)
        intro_top5_trigram_list_list.append(intro_top5_trigram_list)
        '''
    return abs_top5_unigram_list_list,abs_top5_bigram_list_list,abs_top5_trigram_list_list




@csrf_exempt
def upload_refs(request):
    is_valid_submission = True
    has_label_id = False
    has_ref_link = False

    file_dict = request.FILES
    if len(list(file_dict.keys()))>0:
        file_name = list(file_dict.keys())[0]
        file_obj = file_dict[file_name]
    else:
        is_valid_submission = False

    if is_valid_submission == True:
        ## get uid
        global Global_survey_id
        uid_str = generate_uid()
        Global_survey_id = uid_str
        print('Uploaded survey id', Global_survey_id)

        global Survey_dict
        survey_title = file_name.split('.')[-1].title()
        Survey_dict[uid_str] = survey_title

        new_file_name = Global_survey_id
        csvfile_name = new_file_name + '.'+ file_name.split('.')[-1]
        with open(DATA_PATH + csvfile_name, 'wb+') as f:
            for chunk in file_obj.chunks():
                f.write(chunk)

        input_pd = pd.read_csv(DATA_PATH + csvfile_name, sep = ',', encoding='utf-8') #sep = '\t' #
        #print(input_pd.keys())
        #pdb.set_trace()

        clusters_topic_words = []
        ref_paper_num = input_pd.shape[0]
        if ref_paper_num>0:

            ## change col name
            try:
                # required columns
                input_pd["ref_title"] = input_pd["reference paper title"].apply(lambda x: clean_str(x) if len(str(x))>0 else '')
                input_pd["ref_context"] = [""]*ref_paper_num
                input_pd["ref_entry"] = input_pd["reference paper citation information (can be collected from Google scholar/DBLP)"]
                input_pd["abstract"] = input_pd["reference paper abstract (Please copy the text AND paste here)"].apply(lambda x: clean_str(x) if len(str(x))>0 else '')
                input_pd["intro"] = input_pd["reference paper introduction (Please copy the text AND paste here)"].apply(lambda x: clean_str(x) if len(str(x))>0 else '')

                # optional columns
                input_pd["ref_link"] = input_pd["reference paper doi link (optional)"].apply(lambda x: x if len(str(x))>0 else '')
                input_pd["label"] = input_pd["reference paper category label (optional)"].apply(lambda x: str(x) if len(str(x))>0 else '')
                #input_pd["label"] = input_pd["reference paper category id (optional)"].apply(lambda x: str(x) if len(str(x))>0 else '')
            except:
                print("Cannot convert the column name")
                is_valid_submission = False

            ## get cluster_num, check has_label_id
            stat_input_pd_labels = input_pd["label"].value_counts()
            #pdb.set_trace()
            if len(stat_input_pd_labels.keys())>1:
                cluster_num = len(stat_input_pd_labels.keys())
                clusters_topic_words = stat_input_pd_labels.keys().tolist()
                has_label_id = True
            else:
                #pdb.set_trace()
                cluster_num = 3 # as default
            global Survey_n_clusters
            Survey_n_clusters[uid_str] = cluster_num
            global Survey_Topic_dict
            Survey_Topic_dict[uid_str] = clusters_topic_words

            ## check has_ref_link
            if len(input_pd["ref_link"].value_counts().keys())>1:
                has_ref_link = True


            ## get keywords
            try:
                #pdb.set_trace()
                input_pd["topic_word"],input_pd["topic_bigram"],input_pd["topic_trigram"] = ['']*ref_paper_num,['']*ref_paper_num,['']*ref_paper_num
                # input_pd["topic_word"],input_pd["topic_bigram"],input_pd["topic_trigram"] = PosRank_get_top5_ngrams(input_pd)
                # input_pd["topic_word"],input_pd["topic_bigram"],input_pd["topic_trigram"] = abs_top5_unigram_list_list, abs_top5_bigram_list_list, abs_top5_trigram_list_list

                #Survey_Topic_dict[uid_str] = input_pd["topic_word"]
            except:
                print("Cannot select keywords")
                is_valid_submission = False
                #Survey_Topic_dict[uid_str] = []

            #pdb.set_trace()
            ## generate reference description
            try:
                ref_desp_gen = ref_desp(input_pd)
                description_list = ref_desp_gen.ref_desp_generator()
                ref_desp_list=[]
                for ref_desp_set in description_list:
                    ref_desp_list.append(ref_desp_set[1])
                #pdb.set_trace()
                input_pd["description"]=ref_desp_list
            except:
                print("Cannot generate reference paper's description")
                is_valid_submission = False

            #pdb.set_trace()
            ## output tsv
            try:
                output_tsv_filename = DATA_PATH + new_file_name + '.tsv'

                #output_df = input_pd[["ref_title","ref_context","ref_entry","abstract","intro","description"]]
                output_df = input_pd[["ref_title","ref_context","ref_entry","abstract","intro","topic_word","topic_bigram","topic_trigram","description"]]

                if has_label_id == True:
                    output_df["label"]=input_pd["label"]
                else:
                    output_df["label"]=[""]*input_pd.shape[0]
                if has_ref_link == True:
                    output_df["ref_link"]=input_pd["ref_link"]
                else:
                    output_df["ref_link"]=[""]*input_pd.shape[0]

                #pdb.set_trace()
                output_df.to_csv(output_tsv_filename, sep='\t')
            except:
                print("Cannot output tsv")
                is_valid_submission = False
            #Survey_dict[Global_survey_id] = topic
            #Survey_Topic_dict[Global_survey_id] = [topic.lower()]

        else:
            # no record in submitted file
            is_valid_submission = False


    if is_valid_submission == True:
        if len(clusters_topic_words) == 0:
            references = output_df['ref_title'].tolist()
            ref_links = output_df['ref_link'].tolist()
            ref_ids = [i for i in range(output_df['ref_title'].shape[0])]

        elif len(clusters_topic_words)>0:
            references = []
            ref_links = []
            ref_ids = []
            for df in output_df.groupby('label'):
                references.append(list(df[1]['ref_title']))
                ref_links.append(list(df[1]['ref_link']))
                ref_ids.append(df[1].index.tolist())
                #pdb.set_trace()
                #ref_ids.append(list(df[1]['ref_id']))

        ref_list = {'references':[i.title() for i in references],
                    'ref_links':ref_links,
                    'ref_ids':ref_ids,
                    'is_valid_submission':is_valid_submission,
                    "uid":uid_str,
                    "tsv_filename":output_tsv_filename,
                    'topic_words': clusters_topic_words}

    else:
        ref_list = {'references':[],'ref_links':[],'ref_ids':[],'is_valid_submission':is_valid_submission,"uid":uid_str,"tsv_filename":output_tsv_filename,'topic_words': []}
        #ref_list = {'references':[],'ref_links':[],'ref_ids':[]}
    #pdb.set_trace()
    ref_list = json.dumps(ref_list)
    return HttpResponse(ref_list)

@csrf_exempt
def annotate_categories(request):
    # Global_survey_id = request.POST.get('uid', False)
    # global Global_survey_id
    POST_dict = dict(request.POST)
    topic_word_list = POST_dict['topic_word_list[]']
    annotated_paper_ids_list_list= []
    for i in range(len(POST_dict)-1):
        annotated_paper_ids_list_list.append([int(i) for i in POST_dict['ref_lists['+str(i)+'][]']])

    assert(len(topic_word_list)==len(annotated_paper_ids_list_list))

    tsvfile_name = Global_survey_id + '.tsv'
    input_pd = pd.read_csv(DATA_PATH + tsvfile_name, sep = '\t')
    #assert(len(annotated_topic_word_list)==input_pd.shape[0])

    annotated_topic_word_list = ['']*input_pd.shape[0]
    for category_id,annotated_paper_ids_list in enumerate(annotated_paper_ids_list_list):
        for annotated_paper_id in annotated_paper_ids_list:
            annotated_topic_word_list[annotated_paper_id] = topic_word_list[category_id]

    input_pd['label'] = annotated_topic_word_list
    #pdb.set_trace()
    output_tsv_filename = DATA_PATH + tsvfile_name
    os.remove(output_tsv_filename)
    input_pd.to_csv(output_tsv_filename, sep='\t')

    return HttpResponse('')

@csrf_exempt
def get_topic(request):
    topic = request.POST.get('topics', False)
    references, ref_links, ref_ids = get_refs(topic)
    global Global_survey_id
    Global_survey_id = topic
    ref_list = {
        'references' : references,
        'ref_links'  : ref_links,
        'ref_ids'    : ref_ids
    }
    ref_list = json.dumps(ref_list)
    return HttpResponse(ref_list)

@csrf_exempt
def automatic_taxonomy(request):
    ref_dict = dict(request.POST)
    print(ref_dict)
    ref_list = ref_dict['refs']
    query = ref_dict['taxonomy_standard'][0]
    global Global_ref_list
    Global_ref_list = ref_list

    print('Categorization survey id', Global_survey_id)

    colors, category_label, category_description =  Clustering_refs(n_clusters=3) # fix with 3
    # colors, category_label, category_description = Clustering_refs_with_criteria(n_clusters=Survey_n_clusters[Global_survey_id], query=query)

    global Global_category_description
    Global_category_description = category_description
    global Global_category_label
    Global_category_label = category_label

    df_tmp = Global_df_selected.reset_index()
    df_tmp['index'] = df_tmp.index
    ref_titles = list(df_tmp.groupby(df_tmp['label'])['ref_title'].apply(list))
    ref_indexs = list(df_tmp.groupby(df_tmp['label'])['index'].apply(list))
    cate_list = {
        'colors': colors,
        'category_label': ['Sampling approaches', 'Algorithmic approaches', 'Meta-learning approaches'], # category_label,
        'survey_id': Global_survey_id,
        'ref_titles': [[i.title() for i in j] for j in ref_titles],
        'ref_indexs': ref_indexs
    }
    print(cate_list)
    cate_list = json.dumps(cate_list)
    return HttpResponse(cate_list)


@csrf_exempt
def select_sections(request):

    sections = request.POST
    # print(sections)

    survey = {}

    for k,v in sections.items():
        print(Survey_dict[Global_survey_id])
        # if k == "title":
        survey['title'] = "A Survey of " + Survey_dict[Global_survey_id]

        if k == "abstract":
            survey['abstract'] = ["The issue of class imbalance is pervasive in various practical applications of machine learning and data mining, including information retrieval and filtering, and the detection of credit card fraud. The problem of imbalanced learning concerns the effectiveness of learning algorithms when faced with underrepresented data and severe class distribution skews. The classification of data with imbalanced class distribution significantly hinders the performance of most standard classifier learning algorithms that assume a relatively balanced class distribution and equal misclassification costs.",
                                  "In this survey, we present a comprehensive overview of predictive modeling on imbalanced data. We categorize existing literature into three clusters: Sampling approaches, Algorithmic approaches, and Meta-learning approaches, which we introduce in detail. Our aim is to provide readers with a thorough understanding of the different strategies proposed to tackle the class imbalance problem and evaluate their effectiveness in enhancing the performance of learning algorithms."]
        if k == "introduction":
            survey['introduction'] = [
              {
                'subtitle': 'Background',
                'content' : '''Class imbalance is a common problem in machine learning and data mining, where the distribution of classes in the training dataset is highly skewed, with one class being significantly underrepresented compared to the other(s). This issue is prevalent in many real-world applications, including fraud detection, medical diagnosis, anomaly detection, and spam filtering, to name a few.
                               \nThe problem of imbalanced data affects the performance of many learning algorithms, which typically assume a balanced class distribution and equal misclassification costs. When the data is imbalanced, standard learning algorithms tend to favor the majority class, resulting in low accuracy in predicting the minority class. This drawback can lead to serious consequences, such as false negative errors in fraud detection or misdiagnosis in medical applications.
                               \nTo address the class imbalance problem, various techniques have been proposed, including resampling methods, cost-sensitive learning, and ensemble methods, among others. Resampling methods involve creating synthetic samples or under/oversampling the minority/majority classes to balance the data. Cost-sensitive learning assigns different misclassification costs to different classes to prioritize the minority class's correct prediction. Ensemble methods combine multiple models to improve predictive performance.
                               \nThe effectiveness of these techniques varies depending on the dataset and problem at hand. Hence, it is crucial to conduct a comprehensive evaluation of the different approaches to identify the most suitable one for a specific application. As such, your survey paper aims to provide an overview of the current state-of-the-art predictive modeling techniques for imbalanced data and highlight their strengths and limitations.
                            '''
              },
             {
                'subtitle': 'Methodologies', # Sampling approaches, Algorithmic approaches, and Meta-learning approaches
                'content' : '''Exisiting works are mainly categorized into Sampling approaches, Algorithmic approaches, and Meta-learning approaches.
                              \nSampling approaches:
                              \nResampling techniques are among the most popular methods for handling imbalanced data. These techniques involve either oversampling the minority class or undersampling the majority class to create a more balanced dataset. Examples of oversampling methods include SMOTE (Synthetic Minority Over-sampling Technique), ADASYN (Adaptive Synthetic Sampling), and Borderline-SMOTE. Undersampling techniques include random undersampling and Tomek Links. Moreover, hybrid methods, which combine both oversampling and undersampling, have also been proposed.
                              \nAlgorithmic approaches:
                              \nAnother approach to address the class imbalance problem is to modify the learning algorithm itself. Examples of such algorithmic approaches include cost-sensitive learning, where different costs are assigned to different types of misclassifications. Another approach is to adjust the decision threshold of the classifier, where the threshold is shifted to increase sensitivity towards the minority class. Additionally, ensemble methods, such as bagging, boosting, and stacking, have been proposed to combine multiple classifiers to improve predictive performance.
                              \nMeta-learning approaches:
                              \nMeta-learning approaches aim to automatically select the most suitable sampling or algorithmic approach for a specific dataset and problem. These approaches involve training a meta-classifier on multiple base classifiers, each using a different sampling or algorithmic approach. The meta-classifier then selects the most appropriate approach based on the characteristics of the input dataset. Examples of meta-learning approaches include MetaCost, MetaCostNN, and RAkEL.
                              \nThese approaches have shown promising results in addressing the class imbalance problem. However, their effectiveness depends on the specific characteristics of the dataset and problem at hand. Therefore, a comprehensive evaluation of different approaches is necessary to identify the most suitable one for a particular application.
                            '''
             },
             {
                'subtitle': 'Reminder',
                'content' : 'The rest of the paper is organized as follows. In section 2, we introduce the class imbalance problem and its causes and characteristics. Evaluation metrics are addressed in section 3. Section 4 presents an overview of the existing techniques for handling imbalanced data. Applications is illustrated in Section 5. Section 6 shows challenges and open issues. Conclusion and future directions are in Section 7.'
             }
            ]

        if k == "c_and_c":
            survey['c_and_c'] = '''Imbalanced data is a common problem in many real-world applications of machine learning and data mining, where the distribution of classes is highly skewed, with one or more classes being significantly underrepresented compared to the others. This can occur due to various reasons, such as sampling bias, data collection limitations, class overlap, or natural class distribution. The causes of imbalanced data can differ across different domains and applications, and understanding them is essential for developing effective predictive modeling techniques.
                            \nIn addition to the causes, imbalanced data is characterized by several properties that make it challenging for traditional machine learning algorithms. Firstly, the data imbalance results in a class distribution bias, where the majority class dominates the data, and the minority class(es) are often overshadowed, leading to poor classification performance. Secondly, the imbalance can lead to an asymmetric misclassification cost, where misclassifying the minority class is often more costly than misclassifying the majority class, resulting in high false negative rates. Thirdly, imbalanced data can exhibit class overlap, where instances from different classes are difficult to distinguish, leading to low discriminative power of the features and classifiers. Finally, imbalanced data can pose challenges for model evaluation and comparison, as traditional performance metrics such as accuracy, precision, and recall, can be misleading or inadequate in imbalanced settings.
                            \nUnderstanding the causes and characteristics of imbalanced data is crucial for developing effective and efficient predictive modeling techniques that can handle such data. The next section of this survey will discuss the various approaches proposed in the literature to address the imbalanced learning problem, with a focus on sampling, algorithmic, and meta-learning approaches.
                            '''
        if k == "evaluation":
            survey['evaluation'] = '''Evaluation metrics are an essential aspect of machine learning and data mining, as they quantify the performance of predictive models on a given dataset. In the case of imbalanced data, traditional evaluation metrics such as accuracy, precision, and recall may not be sufficient or even appropriate due to the class imbalance and asymmetry in misclassification costs. Therefore, alternative metrics have been proposed to measure the performance of predictive models on imbalanced datasets.
                            \nOne commonly used evaluation metric for imbalanced data is the area under the receiver operating characteristic curve (AUC-ROC). The AUC-ROC is a measure of the model's ability to distinguish between positive and negative instances and is computed as the area under the curve of the ROC plot. The ROC plot is a graphical representation of the trade-off between true positive rate (TPR) and false positive rate (FPR) for different decision thresholds. A perfect classifier would have an AUC-ROC score of 1, while a random classifier would have a score of 0.5.
                            \nAnother popular evaluation metric for imbalanced data is the area under the precision-recall curve (AUC-PR). The AUC-PR measures the precision-recall trade-off of the model and is computed as the area under the curve of the precision-recall plot. The precision-recall plot shows the relationship between precision and recall for different decision thresholds. A perfect classifier would have an AUC-PR score of 1, while a random classifier would have a score proportional to the ratio of positive to negative instances.
                            \nOther evaluation metrics for imbalanced data include F-measure, geometric mean, balanced accuracy, and cost-sensitive measures such as weighted and cost-sensitive versions of traditional metrics. F-measure is a harmonic mean of precision and recall, which balances the trade-off between them. The geometric mean is another metric that balances TPR and FPR and is useful in highly imbalanced datasets. Balanced accuracy is the average of TPR and TNR (true negative rate) and is useful in datasets where the class imbalance is extreme. Cost-sensitive measures incorporate the cost of misclassification and can be tailored to the specific application domain.
                            \nChoosing an appropriate evaluation metric for imbalanced data is essential to avoid biased or misleading performance estimates. The selection of metrics should be based on the application requirements, the class distribution, and the misclassification costs. In the next section, we will discuss various sampling, algorithmic, and meta-learning approaches proposed in the literature to address the imbalanced learning problem and their associated evaluation metrics.
                            '''

        if k == "methodology":
            survey['methodology'] = [
                'Our survey categorized existing works into three types: Sampling approaches, Algorithmic approaches, and Meta-learning approaches. Sampling approaches involve oversampling or undersampling, while algorithmic approaches modify the learning algorithm itself. Meta-learning approaches aim to automatically select the most suitable approach based on the characteristics of the input dataset.',
                [{'subtitle': 'Sampling approaches',
                  'content': 'For sampling approaches, Batista, et al. [1] proposed a simple experimental design to assess the performance of class imbalance treatment methods.  E.A.P.A. et al. [2] performs a broad experimental evaluation involving ten methods, three of them proposed by the authors, to deal with the class imbalance problem in thirteen uci data sets.  Batuwita, et al. [3] presents a method to improve fsvms for cil (called fsvm-cil), which can be used to handle the class imbalance problem in the presence of outliers and noise.  V. et al. [4] implements a wrapper approach that computes the amount of under-sampling and synthetic generation of the minority class examples (smote) to improve minority class accuracy.  Chen, et al. [5] presents ranked minority oversampling in boosting (ramoboost), which is a ramo technique based on the idea of adaptive synthetic data generation in an ensemble learning system.  Chen, et al. [6] proposes a new feature selection method, feature assessment by sliding thresholds (fast), which is based on the area under a roc curve generated by moving the decision boundary of a single feature classifier with thresholds placed using an even-bin distribution.  Davis, et al. [7] shows that a deep connection exists between roc space and pr space, such that a curve dominates in roc space if and only if it dominates in pr space.  In classifying documents, the system combines the predictions of the learners by applying evolutionary techniques as well [8]. Ertekin, et al. [9] is concerns with the class imbalance problem which has been known to hinder the learning performance of classification algorithms.  Ertekin, et al. [10] demonstrates that active learning is capable of solving the problem.  Garcı́aÿ, et al. [11] analyzes a generalization of a new metric to evaluate the classification performance in imbalanced domains, combining some estimate of the overall accuracy with a plain index about how dominant the class with the highest individual accuracy is.  Ghasemi, et al. [12] proposes an active learning algorithm that can work when only samples of one class as well as a set of unlabeled data are available.  He, et al. [13] provides a comprehensive review of the development of research in learning from imbalanced data.  Li, et al. [14] proposes an oversampling method based on support degree in order to guide people to select minority class samples and generate new minority class samples.  Li, et al. [15] analyzes the intrinsic factors behind this failure and proposes a suitable re-sampling method.  Liu, et al. [16] proposes two algorithms to overcome this deficiency.  J. et al. [17] considers the application of these ensembles to imbalanced data : classification problems where the class proportions are significantly different.  Seiffert, et al. [18] presents a new hybrid sampling/boosting algorithm, called rusboost, for learning from skewed training data.  Song, et al. [19] proposes an improved adaboost algorithm called baboost (balanced adaboost), which gives higher weights to the misclassified examples from the minority class.  Sun, et al. [20] develops a cost-sensitive boosting algorithm to improve the classification performance of imbalanced data involving multiple classes.  Van et al. [21] presents a comprehensive suite of experimentation on the subject of learning from imbalanced data.  Wasikowski, et al. [22] presents a first systematic comparison of the three types of methods developed for imbalanced data classification problems and of seven feature selection metrics evaluated on small sample data sets from different applications.  an active under-sampling approach is proposed for handling the imbalanced problem in Yang, et al. [23]. Zhou, et al. [24] studies empirically the effect of sampling and threshold-moving in training cost-sensitive neural networks. \n'},
                 {'subtitle': 'Algorithmic approaches',
                  'content': 'For algorithmic approaches, Baccianella, et al. [25] proposed a simple way to turn standard measures for or into ones robust to imbalance.  Lin, et al. [26] applies a fuzzy membership to each input point and reformulate the svms such that different input points can make different constributions to the learning of decision surface. \n'},
                 {'subtitle': 'Meta-learning approaches',
                  'content': 'For meta-learning approaches, Drummond, et al. [27] proposed an alternative to roc representation, in which the expected cost of a classi er is represented explicitly.  Tao, et al. [28] develops a mechanism to overcome these problems.  Torgo et al. [29] presents a generalization of regression error characteristic (rec) curves.  C. et al. [30] demonstrates that class probability estimates attained via supervised learning in imbalanced scenarios systematically underestimate the probabilities for minority class instances, despite ostensibly good overall calibration.  Yoon, et al. [31] proposes preprocessing majority instances by partitioning them into clusters.  Zheng, et al. [32] investigates the usefulness of explicit control of that combination within a proposed feature selection framework.'}]]



        if k == "app":
            survey['app'] = '''The problem of imbalanced data is pervasive in many real-world applications of predictive modeling, where the data is often skewed towards one or more minority class or classes. Such applications include, but are not limited to, fraud detection in finance, rare disease diagnosis in healthcare, fault detection in manufacturing, spam filtering in email systems, and anomaly detection in cybersecurity. In these scenarios, accurately identifying the minority class instances is of utmost importance, as they often represent critical and rare events that have significant impact or consequences.
                            \nHowever, traditional classification algorithms tend to perform poorly on imbalanced datasets, since they are often biased towards the majority class due to its abundance in the data. This results in low accuracy, high false negative rates, and poor generalization performance, especially for the minority class(es) of interest. In addition, the cost of misclassifying the minority class is often much higher than that of the majority class, making it even more critical to achieve high accuracy and low false negative rates for these instances.
                            \nTo overcome the class imbalance problem, a variety of predictive modeling techniques have been proposed and developed in the literature, specifically designed to handle imbalanced datasets. These techniques range from simple preprocessing methods that adjust the class distribution, to more complex algorithmic modifications that incorporate class imbalance considerations into the learning process. The effectiveness of these techniques depends on the specific characteristics of the dataset and problem, and thus, their selection and evaluation require careful experimentation and analysis.
                            \nOverall, the development and application of predictive modeling techniques for imbalanced data is an active and important research area, with many practical and societal implications. Advancements in this field have the potential to improve the accuracy, efficiency, and fairness of many critical applications, and thus, benefit society as a whole.
                            '''

        if k == "app":
            survey['clg'] = '''Selecting the most appropriate sampling, algorithmic, or meta-learning approach for a specific dataset: There is no one-size-fits-all solution, and choosing the right approach can be challenging.
                            \nLack of standard evaluation metrics that can capture the performance of models on imbalanced data, especially for rare events: Existing evaluation metrics like accuracy can be misleading in imbalanced datasets, and there is a need for metrics that can capture the performance of models on rare events.
                            \nInterpretability and explainability of models trained on imbalanced data: It can be difficult to understand how a model arrives at its predictions, especially when the data is heavily skewed, and there is a need for more interpretable models.
                            \nScalability of methods to handle very large datasets with imbalanced class distributions: As datasets grow in size, it can be challenging to scale methods to handle the imbalanced class distribution efficiently.
                            \nNeed for better feature engineering techniques to handle imbalanced data: Feature engineering is an important step in predictive modeling, and there is a need for better techniques that can handle imbalanced data.
                            \nDevelopment of new learning algorithms that are specifically designed to work well on imbalanced datasets: Most standard learning algorithms assume a relatively balanced class distribution, and there is a need for new algorithms that can handle imbalanced data more effectively.
                            \nResearch into the use of semi-supervised and unsupervised learning techniques for imbalanced data: Semi-supervised and unsupervised learning techniques have shown promise in imbalanced data, and there is a need for more research to explore their potential.
                            \nPotential benefits of using ensemble methods to combine multiple models trained on imbalanced data: Ensemble methods can improve the performance of models on imbalanced data by combining multiple models, and there is a need for more research to explore their potential.
                            \nDeveloping more effective methods for dealing with concept drift and evolving class distributions over time in imbalanced datasets: As class distributions evolve over time, it can be challenging to adapt models to the new distribution, and there is a need for more effective methods to handle concept drift.
                            '''


        if k == "conclusion":
            conclusion = '''In conclusion, the class imbalance problem is a significant challenge in predictive modeling, which can lead to biased models and poor performance. In this survey, we have provided a comprehensive overview of existing works on predictive modeling on imbalanced data. We have discussed different approaches to address this problem, including sampling approaches, algorithmic approaches, and meta-learning approaches, as well as evaluation metrics and challenges in this field. We also presented some potential future research directions in this area. The insights and knowledge provided in this survey paper can help researchers and practitioners better understand the challenges and opportunities in predictive modeling on imbalanced data and design more effective approaches to address this problem in real-world applications.
            \nThere are also some potencial directions for future research:
            \n1. Incorporating domain knowledge: Incorporating domain-specific knowledge can help improve the performance of models on imbalanced data. Research can be done on developing techniques to effectively integrate domain knowledge into the modeling process.
            \n2. Explainability of models: With the increasing adoption of machine learning models in critical applications, it is important to understand how the models make predictions. Research can be done on developing explainable models for imbalanced data, which can provide insights into the reasons for model predictions.
            \n3. Online learning: Imbalanced data can evolve over time, and models need to be adapted to new data as it becomes available. Research can be done on developing online learning algorithms that can adapt to imbalanced data in real-time.
            \n4. Multi-label imbalanced classification: In many real-world scenarios, multiple classes can be imbalanced simultaneously. Research can be done on developing techniques for multi-label imbalanced classification that can effectively handle such scenarios.
            \n5. Transfer learning: In some cases, imbalanced data in one domain can be used to improve the performance of models in another domain. Research can be done on developing transfer learning techniques for imbalanced data, which can leverage knowledge from related domains to improve performance.
            \n6. Incorporating fairness considerations: Models trained on imbalanced data can have biases that can disproportionately affect certain groups. Research can be done on developing techniques to ensure that models trained on imbalanced data are fair and do not discriminate against any particular group.
            \n7. Imbalanced data in deep learning: Deep learning has shown great promise in various applications, but its effectiveness on imbalanced data is not well understood. Research can be done on developing techniques to effectively apply deep learning to imbalanced data.
            \n8. Large-scale imbalanced data: With the increasing availability of large-scale datasets, research can be done on developing scalable techniques for predictive modeling on imbalanced data.
            '''
            survey['conclusion'] = conclusion

        # for k, v in sections.items():
        #     if k == "title":
        #         survey['title'] = "A Survey of " + Survey_dict[Global_survey_id]
        #     if k == "abstract":
        #         abs, last_sent = absGen(Global_survey_id, Global_df_selected, Global_category_label)
        #         survey['abstract'] = [abs, last_sent]
        #     if k == "introduction":
        #         # intro = introGen_supervised(Global_survey_id, Global_df_selected, Global_category_label, Global_category_description, sections)
        #         intro = introGen(Global_survey_id, Global_df_selected, Global_category_label,
        #                          Global_category_description, sections)
        #         survey['introduction'] = intro
        #     if k == "methodology":
        #         proceeding, detailed_des = methodologyGen(Global_survey_id, Global_df_selected, Global_category_label,
        #                                                   Global_category_description)
        #         survey['methodology'] = [proceeding, detailed_des]
        #         print('======')
        #         print(survey['methodology'])
        #         print('======')
        #
        #     if k == "conclusion":
        #         conclusion = conclusionGen(Global_survey_id, Global_category_label)
        #         survey['conclusion'] = conclusion


        ## reference
        ## here is the algorithm part
        # df = pd.read_csv(data_path, sep='\t')

    survey['references'] = []
    try:
        # print(Global_df_selected.head())
        for ref in Global_df_selected['ref_entry']:
            entry = str(ref)
            survey['references'].append(entry)
    except:
        import traceback
        print(traceback.print_exc())
        # colors, category_label, category_description = Clustering_refs(n_clusters=Survey_n_clusters[Global_survey_id])
        # for ref in Global_df_selected['ref_entry']:
        #     entry = str(ref).encode('utf-8')
        #     survey['references'].append(entry)


    survey_dict = json.dumps(survey)

    return HttpResponse(survey_dict)


@csrf_exempt
def get_survey(request):
    survey_dict = get_survey_text()
    survey_dict = json.dumps(survey_dict)
    return HttpResponse(survey_dict)


def get_refs(topic):
    '''
    Get the references from given topic
    Return with a list
    '''
    default_references = ['ref1','ref2','ref3','ref4','ref5','ref6','ref7','ref8','ref9','ref10']
    default_ref_links = ['', '', '', '', '', '', '', '', '', '']
    default_ref_ids = ['', '', '', '', '', '', '', '', '', '']
    references = []
    ref_links = []
    ref_ids = []

    try:
        ## here is the algorithm part
        ref_path   = os.path.join(DATA_PATH, topic + '.tsv')
        df         = pd.read_csv(ref_path, sep='\t')
        for i,r in df.iterrows():
            # print(r['intro'], r['ref_title'], i)
            if not pd.isnull(r['intro']):
                references.append(r['ref_title'])
                ref_links.append(r['ref_link'])
                ref_ids.append(i)
    except:
        print(traceback.print_exc())
        references = default_references
        ref_links = default_ref_links
        ref_ids = default_ref_ids
    print(len(ref_ids))
    return references, ref_links, ref_ids


def get_survey_text(refs=Global_ref_list):
    '''
    Get the survey text from a given ref list
    Return with a dict as below default value:
    '''
    print('REFERENCES FOR GENERATING SURVEY CONTENT', refs)
    survey = {
        'Title': "A Survey of " + Survey_dict[Global_survey_id],
        'Abstract': "test "*150,
        'Introduction': "test "*500,
        'Methodology': [
            "This is the proceeding",
            [{"subtitle": "This is the first subtitle", "content": "test "*500},
             {"subtitle": "This is the second subtitle", "content": "test "*500},
             {"subtitle": "This is the third subtitle", "content": "test "*500}]
        ],
        'Conclusion': "test "*150,
        'References': []
    }

    try:
        ## abs generation
        abs, last_sent = absGen(Global_survey_id, Global_df_selected, Global_category_label)
        survey['Abstract'] = [abs, last_sent]

        ## Intro generation
        #intro = introGen_supervised(Global_survey_id, Global_df_selected, Global_category_label, Global_category_description)
        intro = introGen(Global_survey_id, Global_df_selected, Global_category_label, Global_category_description)
        survey['Introduction'] = intro

        ## Methodology generation
        proceeding, detailed_des = methodologyGen(Global_survey_id, Global_df_selected, Global_category_label, Global_category_description)
        survey['Methodology'] = [proceeding, detailed_des]

        ## Conclusion generation
        conclusion = conclusionGen(Global_survey_id, Global_category_label)
        survey['Conclusion'] = conclusion

        ## reference
        ## here is the algorithm part
        # df = pd.read_csv(data_path, sep='\t')
        try:
            for ref in Global_df_selected['ref_entry']:
                entry = str(ref)
                survey['References'].append(entry)
        except:
            colors, category_label, category_description = Clustering_refs(n_clusters=Survey_n_clusters[Global_survey_id])
            for ref in Global_df_selected['ref_entry']:
                entry = str(ref)
                survey['References'].append(entry)

    except:
        print(traceback.print_exc())
    return survey


def Clustering_refs(n_clusters):
    df = pd.read_csv(DATA_PATH + Global_survey_id + '.tsv', sep='\t', index_col=0, encoding='utf-8')
    df_selected = df.iloc[Global_ref_list]

    ## update cluster labels and keywords
    df_selected, colors = clustering(df_selected, n_clusters, Global_survey_id)
    ## get description and topic word for each cluster
    description_list = get_cluster_description(df_selected, Global_survey_id)

    global Global_df_selected
    Global_df_selected = df_selected
    category_description = [0]*len(colors)
    category_label = [0]*len(colors)


    for i in range(len(colors)):

        for j in description_list:
            print(j)
            print('\n ======= \n')
            if j['category'] == i:
                category_description[i] = j['category_desp']
                category_label[i] = j['topic_word'].replace('-', ' ').title()
    return colors, category_label, category_description
    # return 1,0,1

def Clustering_refs_with_criteria(n_clusters, query):
    df = pd.read_csv(DATA_PATH + Global_survey_id + '.tsv', sep='\t', index_col=0)
    df_selected = df.iloc[Global_ref_list]
    ## update cluster labels and keywords
    df_selected, colors = clustering_with_criteria(df_selected, n_clusters, Global_survey_id, query)
    # print(colors)
    ## get description and topic word for each cluster
    description_list = get_cluster_description(df_selected, Global_survey_id)
    # print(description_list)

    global Global_df_selected
    Global_df_selected = df_selected
    category_description = [0]*len(colors)
    category_label = [0]*len(colors)
    for i in range(len(colors)):
        for j in description_list:
            if j['category'] == i:
                category_description[i] = j['category_desp']
                category_label[i] = j['topic_word'].replace('-', ' ').title()
    return colors, category_label, category_description

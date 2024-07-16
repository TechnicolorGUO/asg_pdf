import gensim
import pandas as pd
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import strip_punctuation, remove_stopwords
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from summa import keywords
TaggededDocument = gensim.models.doc2vec.TaggedDocument

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from operator import itemgetter
import traceback


import nltk
from nltk.tokenize import word_tokenize,sent_tokenize


import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import datasets, manifold
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from sklearn.manifold import TSNE


import nltk
from fuzzywuzzy import fuzz

from summa.summarizer import summarize
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import spacy
from rank_bm25 import BM25Okapi
import torch
from sklearn.cluster import AgglomerativeClustering
nlp = spacy.load("en_core_sci_sm")

IMG_PATH = './src/static/img/'

plt.switch_backend('agg')
device = 0
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length = 128)
# model = AutoModel.from_pretrained("bert-base-uncased").to(device)
class ref_category_desp(object):

    def __init__(
            self,
            input_DF,
            survey_id
    ):
        self.input_DF = input_DF
        self.survey_id = survey_id
    # select the sentences that can match with the topic words
    def sentence_selection(self, abs, topic_list,match_ratio=30):

        matched_sentences = []

        abs = abs.lower()
        sentences = nltk.tokenize.sent_tokenize(abs)

        for sentence in sentences:

            for topic_word in topic_list:
                # match the sentences in abs with topic_words
                #ratio = fuzz.partial_ratio(sentence, topic_word)
                ratio = fuzz.token_set_ratio(sentence, topic_word)
                if (ratio >= match_ratio):
                    matched_sentences.append([sentence,topic_word,ratio])
                    break

        return matched_sentences

    # concat all the selected sentences into a string for textrank
    def sentence_concat(self, abs_list):
        sentences = ''
        for sentences_list in abs_list:
            for sentence in sentences_list:
                single_sentence = str(sentence[0])
                if ((single_sentence.startswith('keywords'))== False):
                    sentences = sentences + single_sentence

                    # print(single_sentence)
        sentences = sentences.replace('.', '. ')
        sentences = sentences.replace('.  ', '. ')
        sentences = sentences.replace('\n', ' ')
        sentences = sentences.replace('  ', ' ')
        return sentences

    # use textrank to get the category description
    def textrank_summary(self, sentences, summary_len):
        category_summary = ''
        summary = summarize(sentences, words=summary_len)
        if (summary):
            category_summary = summary
        return summary

    # 找出category description句对应的句子
    # 其实是为了找到category description对应的topic word (经过concat已无法通过序号找到)
    def summary_sentence_match(self, summary_sentence, abs_list):
        max_ratio = 0
        matched_sentence_list = ['', '']
        if summary_sentence =='':
            return matched_sentence_list

        for sentences_list in abs_list:
            for sentence in sentences_list:
                single_sentence = str(sentence[0])
                ratio = fuzz.ratio(single_sentence, summary_sentence)
                if (ratio >= max_ratio):
                    max_ratio = ratio
                    matched_sentence_list = sentence

        return matched_sentence_list

    # 按固定格式写 category_desp
    def rewrite_category_desp(self, num, category_summary, topic_word):
        num_list = ['first','sencond','third','fourth','fifth','sixth','seventh','eighth','ninth','tenth']
        topic_word = topic_word.replace('-',' ')
        try:
            text = "The "+num_list[num] +" category is about the "+topic_word+'. '
        except Exception:
            #Error in the function rewrite_category_desp: more than ten categories
            print(traceback.print_exc())

        category_desp = text + category_summary
        category_desp = category_desp.replace('  ',' ')
        category_desp = category_desp.replace('.', '. ')
        category_desp = category_desp.replace('.  ', '. ')
        category_desp = category_desp.replace('\n', ' ')

        return category_desp

    def category_summary(self,train_data, match_ratio = 70, summary_len = 30, topic_selection = 'topic_bigram'):

        topic_word_list = train_data['topic_word'].reset_index(drop=True)
        topic_word_list = topic_word_list[0]
        topic_bigram_list = train_data['topic_bigram'].reset_index(drop=True)[0]
        topic_trigram_list = train_data['topic_trigram'].reset_index(drop=True)[0]

        # train_data['abs_intro'] = train_data['abstract']+train_data['intro']
        if self.survey_id == '3073559' or self.survey_id == '2907070':
            abstract_list = train_data['abstract'].reset_index(drop=True)
            train_data.to_csv('test.tsv', sep='\t')
        else:
            abstract_list = train_data['intro'].reset_index(drop=True)

        # select the sentences matched with the topic_words/topic_bigrams/topic_trigrams
        # from the abstracts in ref papers by the function sentence_selection
        if topic_selection == 'topic_word_list':
            topic_str = topic_word_list
        elif topic_selection == 'topic_trigram_list':
            topic_str = topic_trigram_list
        else:
            topic_str = topic_bigram_list
        '''
        topic_str = topic_str.replace('[', '')
        topic_str = topic_str.replace(']', '')
        topic_str = topic_str.replace('\\', '')
        topic_str = topic_str.replace('\'', '')
        topic_str = topic_str.replace('_', '-')
        topic_list = topic_str.split(',')
        '''
        # topic_str = eval(topic_str)
        topic_str = [i.replace('[', '') for i in topic_str]
        topic_str = [i.replace(']', '') for i in topic_str]
        topic_str = [i.replace('\\', '') for i in topic_str]
        topic_str = [i.replace('\'', '') for i in topic_str]
        topic_str = [i.replace('_', '-') for i in topic_str]
        topic_list = topic_str

        matched_sent_list = []
        for abs in abstract_list:
            matched_sentences = self.sentence_selection(abs, topic_list, match_ratio)
            matched_sent_list.append(matched_sentences)



        # concat the sentences in matched_sent_list and replace some symbols
        concat_sents = self.sentence_concat(matched_sent_list)

        # use textrank to summarize the selected sentences
        category_summary = self.textrank_summary(concat_sents, summary_len)

        return category_summary,matched_sent_list

    def desp_generator(self, match_ratio = 70, summary_len = 30, topic_selection = 'topic_bigram'):
        train_tsv = self.input_DF

        data_without_NaN = train_tsv#.dropna(axis=0)

        print('data_without_NaN')
        print(data_without_NaN.head(5))
        type_info = data_without_NaN['label'].value_counts()
        type_list = list(data_without_NaN['label'].unique())


        desp_list = []

        for type_item in type_list:

            #print('___________________________')
            #print('type_item: ' + str(type_item) + '  fuzzy matched sentences: ')

            desp_dict = {}
            train_data = data_without_NaN[data_without_NaN['label'] == type_item]

            category_summary,matched_sent_list = self.category_summary(train_data, match_ratio, summary_len, topic_selection)

            # adjust the match_ratio dynamically to get a category_summary sentence
            while (category_summary == ''):
                match_ratio = match_ratio - 10
                category_summary,matched_sent_list = self.category_summary(train_data, match_ratio, summary_len, topic_selection)
                if match_ratio <= 0:
                    break

            topic_word = ''
            category_desp = ''

            if category_summary != '':
                summary_sent_info = self.summary_sentence_match(category_summary, matched_sent_list)

                # if summary_sent_info[1]!='':
                try:
                    topic_word = summary_sent_info[1]
                except Exception:
                    # failed to find the topic word in the category_summary
                    print(traceback.print_exc())
                else:
                    category_desp = self.rewrite_category_desp(type_item, category_summary, topic_word)

            else:
                topic_selection = 'topic_word_list'  # 如果bigram匹配不到，再用word匹配一轮
                category_summary, matched_sent_list = self.category_summary(train_data, match_ratio, summary_len,
                                                                            topic_selection)

                # adjust the match_ratio dynamically to get a category_summary sentence
                while (category_summary == ''):
                    match_ratio = match_ratio - 10
                    category_summary, matched_sent_list = self.category_summary(train_data, match_ratio, summary_len,
                                                                                topic_selection)
                    if match_ratio <= 0:
                        break

                if category_summary != '':
                    summary_sent_info = self.summary_sentence_match(category_summary, matched_sent_list)

                    # if summary_sent_info[1]!='':
                    try:
                        topic_word = summary_sent_info[1]
                    except Exception:
                        # failed to find the topic word in the category_summary
                        print(traceback.print_exc())
                    else:
                        category_desp = self.rewrite_category_desp(type_item, category_summary, topic_word)

            desp_dict['category'] = type_item
            desp_dict['category_desp'] = category_desp
            desp_dict['topic_word'] = topic_word


            desp_list.append(desp_dict)
        return desp_list


def clustering(df, n_cluster, survey_id):
    wordnet_lemmatizer = WordNetLemmatizer()
    text = df['abstract']
    print("原始文本数据:", text)

    wordstest_model = text
    test_model = [
        [
            wordnet_lemmatizer.lemmatize(word.lower()) 
            for word in remove_stopwords(strip_punctuation(str(words) if words is not None else "")).split()
        ] 
        for words in wordstest_model
    ]
    
    print("预处理后的测试模型:", test_model)
    
    dictionary = corpora.Dictionary(test_model, prune_at=2000000)
    print("词典内容:", dictionary.token2id)
    
    corpus_model = [dictionary.doc2bow(test) for test in test_model]
    print("语料库:", corpus_model)
    
    dictionary = corpora.Dictionary(test_model, prune_at=2000000)
    corpus_model = [dictionary.doc2bow(test) for test in test_model]
    tfidf_model = models.TfidfModel(corpus_model)
    corpus_tfidf = tfidf_model[corpus_model]
    
    top_words = []
    for testword in text:
        processed_testword = [
            wordnet_lemmatizer.lemmatize(word.lower()) 
            for word in remove_stopwords(strip_punctuation(str(testword) if testword is not None else "")).split()
        ]
        test_bow = dictionary.doc2bow(processed_testword)
        
        print("处理后的测试词:", processed_testword)
        print("测试词的BOW:", test_bow)
        
        test_tfidf = tfidf_model[test_bow]
        print("测试词的TFIDF:", test_tfidf)
        
        if test_tfidf:
            top_n_words = sorted(test_tfidf, key=lambda x: x[1], reverse=True)[:5]
            print("前N个词的索引和分数:", top_n_words)
            top_words.append([dictionary[i[0]] for i in top_n_words])
        else:
            top_words.append([])
    
    print("提取的关键词是:", top_words)

    x_train = []
    cnt = 0

    # 创建训练数据集
    for i, text in enumerate(text):
        word_list = top_words[cnt]
        document = TaggedDocument(word_list, tags=[i])
        x_train.append(document)
        cnt += 1

    min_cnt = 2 if len(df) > 20 else 1

    # 创建 Doc2Vec 模型
    model_dm = Doc2Vec(vector_size=100, min_count=min_cnt, sample=1e-3, workers=4)

    # 构建词汇表
    print("训练数据是: ", x_train)
    model_dm.build_vocab(x_train)

    # 训练模型
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=500)

    # 保存模型
    model_dm.save('model_dm')

    # 加载模型进行推理或进一步使用
    infered_vectors_list = []
    model_dm = Doc2Vec.load("model_dm")
    # print("load train vectors...")
    i = 0
    for text, label in x_train:
        vector = model_dm.infer_vector(text)
        infered_vectors_list.append(vector)
        i += 1

    sim_matrix = cosine_similarity(infered_vectors_list)
    labels = SpectralClustering(n_clusters=n_cluster, gamma=0.1).fit_predict(sim_matrix)
    df['label'] = labels

    tfidf_top_words = []
    for i in range(len(x_train)):
        string = ""
        text = x_train[i][0]
        for word in text:
            string = string + word + ' '
        tfidf_top_words.append(string)
    df['top_n_words'] = tfidf_top_words

    ## ------ get unigram ------
    n = df['label'].nunique()
    top_dicts = []
    for i in range(n):
        tmp_dict = {}
        for j, r in df.iterrows():
            if r['label'] == i:
                testword = r['top_n_words']
                for word in [
                    wordnet_lemmatizer.lemmatize(word.lower()) 
                    for word in remove_stopwords(strip_punctuation(str(testword) if testword is not None else "")).split()
                ]:
                    tmp_dict[word] = tmp_dict.get(word, 0) + 1
        top_dicts.append(tmp_dict)

    top_list = []
    for d in top_dicts:
        tmp = sorted(d.items(), key=itemgetter(1), reverse=True)
        tmp_list = []
        for i in range(min(3, len(tmp))):  # 确保不会超出tmp的长度
            if tmp[i][1] >= 2:
                tmp_list.append(tmp[i][0])
        top_list.append(tmp_list)

    topic_words = []
    for j, r in df.iterrows():
        topic_words.append(top_list[r['label']])

    df['topic_word'] = topic_words

    ## ------ get bigram ------
    n = df['label'].nunique()
    docs = []
    for i in range(n):
        tmp = []
        docs.append(tmp)

    source_docs = []
    for i in range(n):
        tmp = []
        source_docs.append(tmp)

    for i in range(n):
        for j, r in df.iterrows():
            if r['label'] == i:
                testword = r['abstract']
                for word in [
                    word.lower() 
                    for word in remove_stopwords(strip_punctuation(str(testword) if testword is not None else "")).split()
                ]:
                    docs[i].append(lancaster_stemmer.stem(wordnet_lemmatizer.lemmatize(word)))
                    source_docs[i].append(word)

    bigram_docs = []
    for i in range(n):
        tmp = []
        for j in range(len(docs[i]) - 1):
            tmp.append(docs[i][j] + "_" + docs[i][j + 1])

        bigram_docs.append(tmp)

    test_model = bigram_docs
    dictionary = corpora.Dictionary(test_model, prune_at=2000000)
    corpus_model = [dictionary.doc2bow(test) for test in test_model]
    tfidf_model = models.TfidfModel(corpus_model)
    corpus_tfidf = tfidf_model[corpus_model]

    def get_source(bigram):
        for i in range(n):
            for j in range(len(docs[i]) - 1):
                if bigram == docs[i][j] + "_" + docs[i][j + 1]:
                    return (source_docs[i][j] + "_" + source_docs[i][j + 1])

    top_list = []
    for testword in test_model:
        test_bow = dictionary.doc2bow(testword)
        test_tfidf = tfidf_model[test_bow]
       

        top_n_words = sorted(test_tfidf, key=lambda x: x[1], reverse=True)[:5]
        # print([(get_source(dictionary[i[0]]), i[1]) for i in top_n_words])
        top_list.append([(get_source(dictionary[i[0]])) for i in top_n_words])
        # print()

    topic_words = []
    for j, r in df.iterrows():
        topic_words.append(top_list[r['label']])

    df['topic_bigram'] = topic_words

    ## ------ get trigram ------
    n = df['label'].nunique()
    docs = []
    for i in range(n):
        tmp = []
        docs.append(tmp)

    source_docs = []
    for i in range(n):
        tmp = []
        source_docs.append(tmp)

    for i in range(n):
        for j, r in df.iterrows():
            if r['label'] == i:
                testword = r['abstract']
                for word in [word.lower() for word in remove_stopwords(strip_punctuation(str(testword))).split()]:
                    docs[i].append(lancaster_stemmer.stem(wordnet_lemmatizer.lemmatize(word)))
                    source_docs[i].append(word)

    trigram_docs = []
    for i in range(n):
        tmp = []
        for j in range(len(docs[i]) - 2):
            tmp.append(docs[i][j] + "_" + docs[i][j + 1] + "_" + docs[i][j + 2])

        trigram_docs.append(tmp)

    test_model = trigram_docs
    dictionary = corpora.Dictionary(test_model, prune_at=2000000)
    corpus_model = [dictionary.doc2bow(test) for test in test_model]
    tfidf_model = models.TfidfModel(corpus_model)
    corpus_tfidf = tfidf_model[corpus_model]

    def get_source(trigram):
        for i in range(n):
            for j in range(len(docs[i]) - 2):
                if trigram == docs[i][j] + "_" + docs[i][j + 1] + "_" + docs[i][j + 2]:
                    return (source_docs[i][j] + "_" + source_docs[i][j + 1] + "_" + source_docs[i][j + 2])

    top_list = []
    for testword in test_model:
        test_bow = dictionary.doc2bow(testword)
        test_tfidf = tfidf_model[test_bow]

        top_n_words = sorted(test_tfidf, key=lambda x: x[1], reverse=True)[:5]
        top_list.append([(get_source(dictionary[i[0]])) for i in top_n_words])
        # print([(get_source(dictionary[i[0]]), i[1]) for i in top_n_words])

    topic_words = []
    for j, r in df.iterrows():
        topic_words.append(top_list[r['label']])

    df['topic_trigram'] = topic_words


    ## get tsne fig

    tsne = TSNE(n_components=2, init='pca', perplexity=10)
    X_tsne = tsne.fit_transform(np.array(infered_vectors_list))
    colors = scatter(X_tsne, df['label'])

    plt.savefig(IMG_PATH + 'tsne_' + survey_id + '.png', dpi=800, transparent=True)

    plt.close()
    return df, colors

def ngram_is_topicword(input_ngram_str):
    is_topicword = False
    input_words = word_tokenize(input_ngram_str)
    if len(input_words)>1:
        pos_tags =nltk.pos_tag(input_words)
        #print(pos_tags)
        if pos_tags[0][1]=="JJ" or pos_tags[0][1][:2]=="NN":
            for tag_index, pos_tag in enumerate(pos_tags[1:]):
                #print(pos_tag[1][:2])
                if pos_tag[1][:2]!="NN":
                    break
                if tag_index==len(input_words)-2:
                    is_topicword = True
    return is_topicword




def selectSentences(query, absIntro):
    '''
    tokenized_corpus = [para.split(" ") for para in absIntro]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    para_scores = bm25.get_scores(tokenized_query)
    print(para_scores)
    '''

    sent_no = []
    sentences = []
    doc = nlp(absIntro)
    sents = list(doc.sents)
    sent_no.append(len(sents))
    sentences = [str(sent) for sent in sents]
    #print("The number of total sentences:", len(sentences))
    #print(sentences)
    tokenized_sentences = [doc.split() for doc in sentences]
    bm25Sent = BM25Okapi(tokenized_sentences)
    sent_scores = bm25Sent.get_scores(query.split())
    max_score = max(sent_scores)
    min_score = min(sent_scores)
    sent_scores = [(score - min_score) / (max_score - min_score + 0.00001) for score in sent_scores]
    

    #print(sent_scores)
    #print(len(sent_scores))
    total_sentences = [query] + sentences
    inputs = tokenizer(total_sentences, return_tensors = "pt", padding=True, truncation = True).to(device)
    outputs = model(**inputs)
    #print(len(outputs))
    pooled_outputs = outputs[1].cpu().detach()
    #print(pooled_outputs.size())
    #print(pooled_outputs[0].size())
    ptm_sent_scores = torch.mm(pooled_outputs[0].unsqueeze(0), pooled_outputs[1:].t()).squeeze().tolist()
    max_score = max(ptm_sent_scores)
    min_score = min(ptm_sent_scores)
    ptm_sent_scores = [(score - min_score) / (max_score - min_score + 0.00001) for score in ptm_sent_scores]
    #print(ptm_sent_scores)
    total_scores = [sent_score + ptm_score for sent_score, ptm_score in zip(sent_scores, ptm_sent_scores)]
    #print(total_scores)
    
    sentences_scores = sorted([(score, sentence) for score, sentence in zip(total_scores, sentences)], reverse = True)
    
    #sentences_scores = sorted([(score, sentence) for score, sentence in zip(sent_scores, sentences)], reverse = True)
    selected_sentences = [sentence for score, sentence in sentences_scores[: 10]]
    return " ".join(selected_sentences)

def clustering_with_criteria(df, n_cluster, survey_id, query):
    text = df['abstract']
    sentences = []
    for doc in text:
        selected_sentences = selectSentences(query, doc)
        sentences.append(selected_sentences)
    
    inputs = tokenizer(sentences, return_tensors = "pt", padding=True, truncation = True).to(device)
    outputs = model(**inputs)
    pooled_outputs = outputs[1].cpu().detach().numpy()
    kmeans = AgglomerativeClustering(n_clusters = 3, affinity = "cosine", linkage = "complete").fit(pooled_outputs)
    labels = kmeans.labels_
    

    wordstest_model = text
    test_model = [
        [wordnet_lemmatizer.lemmatize(word.lower()) for word in remove_stopwords(strip_punctuation(words)).split()] for
        words in wordstest_model]
    dictionary = corpora.Dictionary(test_model, prune_at=2000000)
    corpus_model = [dictionary.doc2bow(test) for test in test_model]
    tfidf_model = models.TfidfModel(corpus_model)
    corpus_tfidf = tfidf_model[corpus_model]

    top_words = []
    for testword in text:
        test_bow = dictionary.doc2bow([wordnet_lemmatizer.lemmatize(word.lower()) for word in
                                       remove_stopwords(strip_punctuation(testword)).split()])
        test_tfidf = tfidf_model[test_bow]
        top_n_words = sorted(test_tfidf, key=lambda x: x[1], reverse=True)[:5]  # [:len(test_tfidf)]
        top_words.append([(dictionary[i[0]]) for i in top_n_words])

    x_train = []
    cnt = 0
    for i, text in enumerate(text):
        word_list = top_words[cnt]
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)
        cnt += 1

    min_cnt = 2 if len(df)>20 else 1

    model_dm = Doc2Vec(x_train, min_count=min_cnt, size=100, sample=1e-3, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=500)
    model_dm.save('model_dm')
    infered_vectors_list = []
    # print("load doc2vec model...")
    model_dm = Doc2Vec.load("model_dm")
    # print("load train vectors...")
    i = 0
    for text, label in x_train:
        vector = model_dm.infer_vector(text)
        infered_vectors_list.append(vector)
        i += 1

    sim_matrix = cosine_similarity(infered_vectors_list)
    #labels = SpectralClustering(n_clusters=n_cluster, gamma=0.1).fit_predict(sim_matrix)
    
    df['label'] = labels

    tfidf_top_words = []
    for i in range(len(x_train)):
        string = ""
        text = x_train[i][0]
        for word in text:
            string = string + word + ' '
        tfidf_top_words.append(string)
    df['top_n_words'] = tfidf_top_words

    ## ------ get unigram ------
    n = df['label'].nunique()
    top_dicts = []
    for i in range(n):
        tmp_dict = {}
        for j, r in df.iterrows():
            if r['label'] == i:
                testword = r['top_n_words']
                for word in [wordnet_lemmatizer.lemmatize(word.lower()) for word in
                             remove_stopwords(strip_punctuation(testword)).split()]:
                    try:
                        tmp_dict[word] += 1
                    except:
                        tmp_dict[word] = 1
        top_dicts.append(tmp_dict)

    top_list = []
    for d in top_dicts:
        tmp = sorted(d.items(), key=itemgetter(1), reverse=True)
        tmp_list = []
        for i in [0, 1, 2]:
            if tmp[i][1] >= 2:
                tmp_list.append(tmp[i][0])
        top_list.append(tmp_list)

    topic_words = []
    for j, r in df.iterrows():
        topic_words.append(top_list[r['label']])

    df['topic_word'] = topic_words

    ## ------ get bigram ------
    n = df['label'].nunique()
    docs = []
    for i in range(n):
        tmp = []
        docs.append(tmp)

    source_docs = []
    for i in range(n):
        tmp = []
        source_docs.append(tmp)

    for i in range(n):
        for j, r in df.iterrows():
            if r['label'] == i:
                testword = r['abstract']
                for word in [word.lower() for word in remove_stopwords(strip_punctuation(testword)).split()]:
                    docs[i].append(lancaster_stemmer.stem(wordnet_lemmatizer.lemmatize(word)))
                    source_docs[i].append(word)

    bigram_docs = []
    for i in range(n):
        tmp = []
        for j in range(len(docs[i]) - 1):
            tmp.append(docs[i][j] + "_" + docs[i][j + 1])

        bigram_docs.append(tmp)

    test_model = bigram_docs
    dictionary = corpora.Dictionary(test_model, prune_at=2000000)
    corpus_model = [dictionary.doc2bow(test) for test in test_model]
    tfidf_model = models.TfidfModel(corpus_model)
    corpus_tfidf = tfidf_model[corpus_model]

    def get_source(bigram):
        for i in range(n):
            for j in range(len(docs[i]) - 1):
                if bigram == docs[i][j] + "_" + docs[i][j + 1]:
                    return (source_docs[i][j] + "_" + source_docs[i][j + 1])

    top_list = []
    for testword in test_model:
        test_bow = dictionary.doc2bow(testword)
        test_tfidf = tfidf_model[test_bow]
        
        
        top_n_words = sorted(test_tfidf, key=lambda x: x[1], reverse=True)[:10]
        tmp_list = [(get_source(dictionary[i[0]])) for i in top_n_words]
        print(tmp_list)
        print('*'*100)
        print([' '.join(i.split('_')) for i in tmp_list])
        print('*'*100)
        print([i for i in tmp_list if ngram_is_topicword(' '.join(i.split('_')))])
        tmp_list = [i for i in tmp_list if ngram_is_topicword(' '.join(i.split('_')))][:5]  
        top_list.append(tmp_list)
        
        # import time
        # time.sleep(2)
        # print(top_list)

    topic_words = []
    for j, r in df.iterrows():
        topic_words.append(top_list[r['label']])

    df['topic_bigram'] = topic_words

    ## ------ get trigram ------
    n = df['label'].nunique()
    docs = []
    for i in range(n):
        tmp = []
        docs.append(tmp)

    source_docs = []
    for i in range(n):
        tmp = []
        source_docs.append(tmp)

    for i in range(n):
        for j, r in df.iterrows():
            if r['label'] == i:
                testword = r['abstract']
                for word in [word.lower() for word in remove_stopwords(strip_punctuation(testword)).split()]:
                    docs[i].append(lancaster_stemmer.stem(wordnet_lemmatizer.lemmatize(word)))
                    source_docs[i].append(word)

    trigram_docs = []
    for i in range(n):
        tmp = []
        for j in range(len(docs[i]) - 2):
            tmp.append(docs[i][j] + "_" + docs[i][j + 1] + "_" + docs[i][j + 2])

        trigram_docs.append(tmp)

    test_model = trigram_docs
    dictionary = corpora.Dictionary(test_model, prune_at=2000000)
    corpus_model = [dictionary.doc2bow(test) for test in test_model]
    tfidf_model = models.TfidfModel(corpus_model)
    corpus_tfidf = tfidf_model[corpus_model]

    def get_source(trigram):
        for i in range(n):
            for j in range(len(docs[i]) - 2):
                if trigram == docs[i][j] + "_" + docs[i][j + 1] + "_" + docs[i][j + 2]:
                    return (source_docs[i][j] + "_" + source_docs[i][j + 1] + "_" + source_docs[i][j + 2])

    top_list = []
    for testword in test_model:
        test_bow = dictionary.doc2bow(testword)
        test_tfidf = tfidf_model[test_bow]
        top_n_words = sorted(test_tfidf, key=lambda x: x[1], reverse=True)[:5]
        top_list.append([(get_source(dictionary[i[0]])) for i in top_n_words])
        # print([(get_source(dictionary[i[0]]), i[1]) for i in top_n_words])

    topic_words = []
    for j, r in df.iterrows():
        topic_words.append(top_list[r['label']])

    df['topic_trigram'] = topic_words


    ## get tsne fig

    tsne = TSNE(n_components=2, init='pca', perplexity=10)
    #X_tsne = tsne.fit_transform(np.array(infered_vectors_list))   #np.array(pooled_outputs)
    X_tsne = tsne.fit_transform(np.array(pooled_outputs))
    colors = scatter(X_tsne, df['label'])

    plt.savefig(IMG_PATH + 'tsne_' + survey_id + '.png', dpi=800, transparent=True)

    plt.close()
    return df, colors

def scatter(x, colors):
    sns.set_style('whitegrid')
    sns.set_palette('Set1')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    # We choose a color palette with seaborn.
    palette = np.array(sns.hls_palette(8, l=0.4, s=.8))
    color_hex = sns.color_palette(sns.hls_palette(8, l=0.4, s=.8)).as_hex()
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=1,
                    c=palette[colors.astype(np.int)])
    c = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in colors]
    for i in range(x.shape[0]):
        ax.text(x[i, 0], x[i, 1], '[' + str(i) + ']', fontsize=20, color=c[i], weight='1000')
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    return color_hex[:colors.nunique()]


def get_cluster_description(df, survey_id):
    match_ratio = 70  # threshold for fuzzy matching
    summary_len = 30  # length of generated summary

    topic_selection = 'topic_bigram_list'  # topic_bigram_list #topic_word_list #topic_trigram_list
    if len(df) <= 5:
        match_ratio = 90
    input_DF = df
    category_desp = ref_category_desp(input_DF, survey_id)
    print('category_desp ', category_desp)
    description_list = category_desp.desp_generator(match_ratio=match_ratio, summary_len=summary_len, topic_selection=topic_selection)
    print('='*100)
    print(description_list)
    print('*'*100)
    # for i in range(len(description_list)):
    #     if description_list[i]['topic_word'] == "":
    #         description_list[i]['topic_word'] = ' '.join(list(input_DF['topic_bigram'])[i][i%2].split('_'))


    return description_list

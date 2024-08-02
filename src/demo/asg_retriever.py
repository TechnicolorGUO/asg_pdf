'''
0. The deletion function does not work. 多个相同检索结果是因为chroma中数据删除不干净 需手动删除
1. advanced level retriever not implemented yet (HyDE)
2. 将提取信息load到json的部分代码集成进建立向量数据库的函数process_pdf
3*. HyDE方面自己写一个预期的回答query能产生的虚拟文档 在此基础上调整HyDE效果
'''
import torch
import uuid
import os
import json
import chromadb
from langchain_chroma import Chroma
from .asg_splitter import TextSplitting
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from .asg_loader import DocumentLoading

class Retriever:
    client = None
    cur_dir = os.getcwd()  # current directory
    # cur_dir = os.path.dirname(os.path.abspath(__file__))
    chromadb_path = os.path.join(cur_dir, "chromadb")

    def __init__ (self):
        self.client = chromadb.PersistentClient(path=self.chromadb_path)

    def create_collection_chroma(self, collection_name: str):
        """
        The Collection will be created with collection_name, the name must follow the rules:\n
        0. Collection name must be unique, if the name exists then try to get this collection\n
        1. The length of the name must be between 3 and 63 characters.\n
        2. The name must start and end with a lowercase letter or a digit, and it can contain dots, dashes, and underscores in between.\n
        3. The name must not contain two consecutive dots.\n
        4. The name must not be a valid IP address.\n
        """
        try: 
            self.client.create_collection(name=collection_name)
        except chromadb.db.base.UniqueConstraintError: 
            self.get_collection_chroma(collection_name)
        return collection_name

    def get_collection_chroma (self, collection_name: str):
        collection = self.client.get_collection(name=collection_name)
        return collection

    def add_documents_chroma (self, collection_name: str, embeddings_list: list[list[float]], documents_list: list[dict], metadata_list: list[dict]) :
        """
        Please make sure that embeddings_list and metadata_list are matched with documents_list\n
        Example of one metadata: {"doc_name": "Test2.pdf", "page": "9"}\n
        The id will be created automatically as uuid v4 
        The chunks content and metadata will be logged (appended) into ./logs/<collection_name>.json
        """
        collection = self.get_collection_chroma(collection_name)
        num = len(documents_list)
        ids=[str(uuid.uuid4()) for i in range(num) ]

        collection.add(
            documents= documents_list,
            metadatas= metadata_list,
            embeddings= embeddings_list,
            ids=ids 
        )
        logpath = os.path.join(self.cur_dir, "logs", f"{collection_name}.json")
        os.makedirs(os.path.dirname(logpath), exist_ok=True)
        logs = []
        try:  
            with open (logpath, 'r') as chunklog:
                logs = json.load(chunklog)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            logs = [] # old_log does not exist or empty
       
        added_log= [{"chunk_id": ids[i], "metadata": metadata_list[i], "page_content": documents_list[i]} \
                       for i in range(num)]
      
        logs.extend(added_log)

        # write back
        with open (logpath, "w") as chunklog:
            json.dump(logs, chunklog, indent=4)
        print(f"Logged document information to '{logpath}'.")
            
    # def query_chroma (self, collection_name: str, query_embeddings: list[list[float]]) -> dict:
    #     # return n closest results (chunks and metadatas) in order
    #     collection = self.get_collection_chroma(collection_name)
    #     result = collection.query(
    #         query_embeddings=query_embeddings,
    #         n_results=5,
    #     )
    #     print(f"Query executed on collection '{collection_name}'.")
    #     return result
    
    def query_chroma(self, collection_name: str, query_embeddings: list[list[float]], n_results: int = 5) -> dict:
        # return n closest results (chunks and metadatas) in order
        collection = self.get_collection_chroma(collection_name)
        result = collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
        )
        print(f"Query executed on collection '{collection_name}'.")
        return result

    def update_chroma (self, collection_name: str, id_list: list[str], embeddings_list: list[list[float]], documents_list: list[str], metadata_list: list[dict]):
        collection = self.get_collection_chroma(collection_name)
        num = len(documents_list)
        collection.update(
            ids=id_list,
            embeddings=embeddings_list,
            metadatas=metadata_list,
            documents=documents_list,
        )
        update_list = [{"chunk_id": id_list[i], "metadata": metadata_list[i], "page_content": documents_list[i]} for i in range(num)]
       
        # update the chunk log 
        logs = []

        logpath = os.path.join(self.cur_dir, "logs", f"{collection_name}.json")
        # logpath = "{:0}/assets/log/{:1}.json".format(self.cur_dir, collection_name)
        try:  
            with open (logpath, 'r') as chunklog:
                logs = json.load(chunklog)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            logs = [] # old_log does not exist or empty, then no need to update
        else:
            for i in range(num):
                for log in logs:
                    if (log["chunk_id"] == update_list[i]["chunk_id"]):
                        log["metadata"] = update_list[i]["metadata"]
                        log["page_content"] = update_list[i]["page_content"]
                        break

        # write back
        with open (logpath, "w") as chunklog:
            json.dump(logs, chunklog, indent=4)
        print(f"Updated log file at '{logpath}'.")

    def delete_collection_entries_chroma(self, collection_name: str, id_list: list[str]):
        collection = self.get_collection_chroma(collection_name)
        collection.delete(ids=id_list)
        print(f"Deleted entries with ids: {id_list} from collection '{collection_name}'.")

    def delete_collection_chroma(self, collection_name: str):
        # delete the collection itself and all entries in the collection 
        print(f"The collection {collection_name} will be deleted forever!")    
        self.client.delete_collection(collection_name)
        try:
            logpath = os.path.join(self.cur_dir, "logs", f"{collection_name}.json")
            print(f"Collection {collection_name} has been removed, deleting log file of this collection")
            os.remove(logpath)
        except FileNotFoundError:
            print("The log of this collection did not exist!")

    def list_collections_chroma(self):
        collections = self.client.list_collections()
        print(f"Existing collections: {[col.name for col in collections]}")

def process_pdf_old(file_path: str, query_text: str):

    # load and split
    splitters = TextSplitting().pypdf_recursive_splitter(file_path)
    # splitters = TextSplitting().unstructured_recursive_splitter(file_path)

    # print(splitters) # [Document(page_content='...'), Document(page_content='...')]
    # print(len(splitters)) # 32 chunks for abstract and introduction part of Test2.pdf
    # print(splitters[0]) # page_content='ABSTRACT\nHigh-quality text embedding is pivotal in improving semantic textual similarity\n(STS) tasks, which are crucial components in Large Language Model (LLM) applications. However, a common challenge existing text embedding models face is'
    # extract the page content from the splitters
    documents_list = [document.page_content for document in splitters]
    # not sure if the replacement is helpful
    for i in range(len(documents_list)):
        documents_list[i] = documents_list[i].replace('\n', ' ')
    # print(documents_list[0])  # the first chunk
    # print(len(documents_list))  # 32 chunks for extracted parts

    # embed
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    doc_results = embedder.embed_documents(documents_list)
    if isinstance(doc_results, torch.Tensor): # tensor to list
        embeddings_list = doc_results.tolist()
    else:
        embeddings_list = doc_results
    # embeddings_list = doc_results # if no need to convert to list
    print(f"Generated embeddings for {len(embeddings_list)} chunks.")
    # print(embeddings_list[0])  # the first embedding
    # print(len(embeddings_list[0]))  # 384 dimensions

    # store
    # the metadata_list should be provided from indexing, simply use file title
    # used for multiple document indexing
    metadata_list = [{"doc_name": os.path.basename(file_path)} for i in range(len(documents_list))]
    # no need to repeatedly add documents
    retriever = Retriever()

    # retriever.delete_collection_chroma("Test2")

    retriever.list_collections_chroma()
    collection_name = retriever.create_collection_chroma(os.path.basename(file_path).split('.')[0])
    retriever.add_documents_chroma(
        collection_name=collection_name,
        embeddings_list=embeddings_list,
        documents_list=documents_list,
        metadata_list=metadata_list
    )
    # collection_name = "Test2" # need to extract title
    # collection = retriever.get_collection_chroma(collection_name)
    # document_ids = [str(uuid.uuid4()) for _ in range(documents_list)]
    # doc_id = document_ids[0]  # 替换为想查询的具体文档id
    # result = collection.get(ids=["dffa16a8-fd92-4726-b64b-8a1b958671fd"], include=["embeddings", "documents", "metadatas"])
    # print(result)
    # test for deleting the collection "Test2"
    # retriever.delete_collection_chroma("Test2")
    # retriever.delete_collection_entries_chroma("Test2", ["dffa16a8-fd92-4726-b64b-8a1b958671fd"])
    # retriever.list_collections_chroma()
    # print(result["documents"])
    # print(result["metadatas"])
    # print(result["embeddings"])
    # embeddings不在json中 过长 不易处理

    # query
    # rephrase the queries
    rephrase_query = query_text # rephrase the query (not implemented yet)
    query_embeddings = embedder.embed_query(query_text)

    # if isinstance(query_embeddings, torch.Tensor):  # tensor to list
    #     query_embeddings = query_embeddings.tolist()
    # print(query_embeddings) # 384 dimensions
    # query_result = retriever.query_chroma(collection_name, query_embeddings)
    # query_result = retriever.query_chroma(collection_name = collection_name, query_embeddings= query_embeddings) # query according to the embeddings
    # query_result = retriever.query_chroma(collection_name=collection_name, query_embeddings=query_embeddings, n_results=5) # query according to the embeddings
    query_result = retriever.query_chroma(collection_name=collection_name, query_embeddings=[query_embeddings], n_results=5)

    # print(query_result, "\n")
    print(query_result)
    # print(query_result["documents"])
    # print(query_result["distances"])
    # print(query_result["metadatas"])
    # print(query_result["ids"])
    # print(query_result["embeddings"])
    query_result_chunks = query_result["documents"][0]
    query_result_ids = query_result["ids"][0]

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    with open ("{}/retrieval/{}_{}.json".format(cur_dir, collection_name, str(uuid.uuid4())).format(), 'w') as retrieval:
        json.dump(query_result, retrieval, indent=4)

    #context = '//\n'.join(["@" + query_result_ids[i] + "//" + query_result_chunks[i].replace("\n", ".") for i in range (len(query_result_chunks))])
    context = '//\n'.join(["@" + query_result_ids[i] + "//" + query_result_chunks[i] for i in range (len(query_result_chunks))])
    print(context)
    # result = generate(context, query_text)


# process_pdf_old("./Test2.pdf", "What method is used in the paper?")


def process_pdf(file_path: str, survey_id: str):
    # Load and split the PDF
    # splitters = TextSplitting().pypdf_recursive_splitter(file_path)

    splitters = TextSplitting().unstructured_recursive_splitter(file_path, survey_id)

    documents_list = [document.page_content for document in splitters]
    for i in range(len(documents_list)):
        documents_list[i] = documents_list[i].replace('\n', ' ')
    
    # Embed the documents
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    doc_results = embedder.embed_documents(documents_list)
    if isinstance(doc_results, torch.Tensor):
        embeddings_list = doc_results.tolist()
    else:
        embeddings_list = doc_results

    # Prepare metadata
    metadata_list = [{"doc_name": os.path.basename(file_path)} for i in range(len(documents_list))]

    title = file_path.split('/')[-1].split('.')[0]

    title_new = title.strip()
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*',' ']
    for char in invalid_chars:
        title_new = title_new.replace(char, '_')
    print("============================")
    print(title_new)

    return embeddings_list, documents_list, metadata_list, title_new

# embeddings_list, documents_list, metadata_list = process_pdf("./ESE.pdf")
# print(len(embeddings_list))
# print(len(documents_list))
# print("++++++++++++++++++++++++++++++++++++++++++++++")

def query_embeddings(collection_name: str, embeddings_list: list, documents_list: list, metadata_list: list, query_text: str):
    retriever = Retriever()
    retriever.list_collections_chroma()
    retriever.create_collection_chroma(collection_name)
    retriever.add_documents_chroma(
        collection_name=collection_name,
        embeddings_list=embeddings_list,
        documents_list=documents_list,
        metadata_list=metadata_list
    )

    # Embed the query
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_embeddings = embedder.embed_query(query_text)

    # Query the collection
    query_result = retriever.query_chroma(collection_name=collection_name, query_embeddings=[query_embeddings], n_results=5)
    
    # Extract results
    query_result_chunks = query_result["documents"][0]
    query_result_ids = query_result["ids"][0]
    # print(query_result_chunks)
    # return query_result_chunks, query_result_ids

    # context = '//\n'.join(["@" + query_result_ids[i] + "//" + query_result_chunks[i] for i in range(len(query_result_chunks))])
    context = '//\n'.join([query_result_chunks[i] for i in range(len(query_result_chunks))])
    # print(context)
    return context

# context = query_embeddings(os.path.basename("./Test2.pdf").split('.')[0], embeddings_list, documents_list, metadata_list, "What algorithmic methods are used in the paper?")
# print(context)

# Question format? If the questions follow only a few fixed formats, you can write the Hyde documents yourself.
# This can save time asking the LLM and provide more accurate results.

# GPT4o HyDE
# To answer this question, I need to know which specific paper you are referring to. Please provide the title or a brief description of the paper's main content so that I can give you a more precise answer. Typically, research papers might involve the following algorithmic methods:

# Machine Learning Algorithms: Such as linear regression, logistic regression, decision trees, random forests, support vector machines, k-nearest neighbors, neural networks, etc.
# Deep Learning Algorithms: Convolutional neural networks (CNNs), recurrent neural networks (RNNs), long short-term memory networks (LSTMs), generative adversarial networks (GANs), etc.
# Optimization Algorithms: Gradient descent, stochastic gradient descent, Newton's method, genetic algorithms, etc.
# Graph Algorithms: Depth-first search, breadth-first search, shortest path algorithms (like Dijkstra's algorithm), PageRank, etc.
# Statistical and Data Analysis Methods: Principal component analysis (PCA), t-distributed stochastic neighbor embedding (t-SNE), clustering analysis (such as k-means, hierarchical clustering), etc.
# If you can provide more details, I will be able to pinpoint the specific algorithmic methods used in the paper.
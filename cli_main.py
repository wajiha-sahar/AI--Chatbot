# package the semantic search into an API with front end
# set up a flask app below

import collections
import openai
import pandas as pd
import numpy as np
import sys
import requests

global FILE_NAME
FILE_NAME = 'tgg.parquet'
EMBEDDINGS_MODEL = "text-embedding-ada-002"
openai.api_key = "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"


def parse_dataset():
    """
    Parse a dataset of preprocessed text and embeddings.

    Returns:
    - An instance of the collections.namedtuple 'Engine', containing the following attributes:
        * page_text_corpus: a list of strings representing the preprocessed text data.
        * page_text_corpus_embeddings: a numpy array of shape (n, m) representing the precomputed embeddings for each text data.
        * page_numbers_list: a list of integers representing the page numbers for each text data.
        * chapter_number_list: a list of strings representing the chapter numbers for each text data.
        * top_k: an integer representing the maximum number of similar sections to return for a given query.

    Example:
    >>> engine = parse_dataset()
    """

    print("Loading embedding file...")
    tgg_file = pd.read_json('pages_with_embeddings.json')
    # tgg_file = pd.read_parquet(FILE_NAME)
    print("Converting to np array...")
    page_numbers_list = tgg_file['page_number'].tolist()
    chapter_number_list = tgg_file['chapter'].tolist()
    page_text_corpus = tgg_file['page'].tolist()
    page_text_corpus_embeddings = np.array(tgg_file['embedding'].tolist(), dtype=float)
    top_k = min(3, len(page_text_corpus))
    print("done converting to np array")
    return collections.namedtuple('Engine', 
    ['page_text_corpus', 
    'page_text_corpus_embeddings', 
    'page_numbers_list',
    'chapter_number_list',
    'top_k'])(
        page_text_corpus, 
        page_text_corpus_embeddings, 
        page_numbers_list,
        chapter_number_list,
        top_k)

def get_query_embedding_openai(prompt):
    model_id = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    hf_token = "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"
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    response = requests.post(api_url, headers=headers, json={"inputs": prompt, "options":{"wait_for_model":True}})
    return response.json()
    # response = openai.Embedding.create(
    #     model=EMBEDDINGS_MODEL,
    #     input=prompt
    # )
    # return response['data'][0]['embedding']

def prepare_contexts(dataset):
    """
    Create a dictionary of document section embeddings.

    Args:
    - dataset: contains preprocessed text data and their embeddings.

    Returns:
    - A dictionary where each key is a tuple representing a document section consisting of (page_text, page_number, chapter_number), 
    and each value is the corresponding embedding.
    """
    contexts = {}
    for page_text, page_number, chapter_number, embedding in zip(
        dataset.page_text_corpus, 
        dataset.page_numbers_list, 
        dataset.chapter_number_list, 
        dataset.page_text_corpus_embeddings
    ):
        contexts[(page_text, page_number, chapter_number)] = embedding
    return contexts

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query_embedding: str, contexts: dict[(str, int, int), np.array]) -> list[(float, (str, int, int))]:
    """
    Compare query embedding against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

def get_semantic_suggestions(prompt):
    """
    Generate a list of semantic suggestions based on a given prompt.

    Args:
    - prompt: a string representing the user's query.

    Returns:
    - A list of dictionaries containing the top-k most relevant document sections to the query.
        Each dictionary has the following keys:
            * page: a string representing the text of the document section.
            * chapter_number: a string representing the chapter number of the document section.
            * page_number: an integer representing the page number of the document section.

    Example:
    >>> suggestions = get_semantic_suggestions("What is the meaning of life?")
    """
    dataset_with_embeddings = parse_dataset()
    query_embedding = np.array(get_query_embedding_openai(prompt), dtype=float)
    relevant_sections = order_document_sections_by_query_similarity(
        query_embedding, 
        prepare_contexts(dataset=dataset_with_embeddings)
    )
    top_three = relevant_sections[:dataset_with_embeddings.top_k]
    final = []
    for _, (page_text, page_number, chapter_number) in top_three:
        final.append(
            {
                'page': page_text,
                'chapter_number': chapter_number,
                'page_number': page_number
            })
    return final 



if __name__ == '__main__':
    prompt = sys.argv[1]
    results = get_semantic_suggestions(prompt)
    for result in results:
        print("-"*80)
        print(f"Chapter: {result['chapter_number']}, Page: {result['page_number']}")
        print(result['page'])

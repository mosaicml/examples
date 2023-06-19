# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os

import gradio as gr
from composer.utils import get_file
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.embeddings import MosaicMLInstructorEmbeddings
from langchain.llms import MosaicML
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from tqdm import tqdm

DOCUMENT_PROMPT = PromptTemplate(input_variables=['page_content'],
                                 template='Context:\n{page_content}')

parser = argparse.ArgumentParser()
parser.add_argument('--llm_endpoint_url', type=str)
parser.add_argument('--embedding_endpoint_url', type=str)
parser.add_argument('--remote_folder_path', type=str)
parser.add_argument('--dataset_subset', type=str, default='small_full')

args, unknown = parser.parse_known_args()

# Load in the available tickers/years for the test set
path_to_current_file = os.path.realpath(__file__)
if args.dataset_subset == 'small_full':
    ticker_file_name = 'test_ticker_to_years_small.json'
elif args.dataset_subset == 'large_full':
    ticker_file_name = 'test_ticker_to_years_large.json'
else:
    raise ValueError(f'Unknown dataset subset {args.dataset_subset}')

with open(os.path.join(os.path.dirname(path_to_current_file), ticker_file_name),
          'r') as _json_file:
    ticker_to_years = {
        k: sorted(v, key=lambda x: int(x))
        for k, v in json.load(_json_file).items()
    }


def clean_response(input_text: str) -> str:
    """Clean the response from the model by stripping some bad answer prefixes,

    new lines, etc.

    Args:
        input_text (str): The response from the model.

    Returns:
        str: The cleaned response.
    """
    input_text = input_text.strip('\n')

    context_prefix = 'Context:'
    answer_prefix = 'Answer:'
    prefixes = [context_prefix, answer_prefix]
    while True:
        prefix_found = False
        for prefix in prefixes:
            if input_text.startswith(prefix):
                input_text = input_text[len(prefix):].strip()
                input_text = input_text.strip('\n')
                prefix_found = True
                break
        if not prefix_found:
            break

    input_text = input_text.lstrip('\n :')

    return input_text


def greet(
    ticker: str,
    year: str,
    query: str,
):
    """The main function that computes the answer and returns the.

    response to the gradio application.

    Args:
        ticker (str): The ticker of the company to query
        year (str): The year of the 10-K to query
        query (str): The query to ask the model

    Returns:
        answer: The answer produced by the model
        context: The chunks of source text retrieved and passed to the model
    """
    ticker = ticker.upper()
    remote_file_path = os.path.join(args.remote_folder_path, ticker,
                                    f'sec_{year}_txt.txt')
    local_file_path = os.path.join(os.getcwd(), 'local-data', ticker,
                                   f'sec_{year}_txt.txt')
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

    print(f'Downloading file from {remote_file_path} to {local_file_path}')

    if ticker not in ticker_to_years:
        return f'Invalid ticker {ticker} see test_ticker_to_years_{{large|small}}.json for all the tickers in the test set', '', ''

    if year not in ticker_to_years[ticker]:
        return f'Invalid year {year} for ticker {ticker}, available years are {ticker_to_years[ticker]}', '', ''

    if not os.path.exists(local_file_path):
        get_file(remote_file_path, local_file_path)

    with open(local_file_path, 'r') as f:
        doc = f.read()

    # Component for splitting the long document until chunks that fit in the embedding model
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        separators=[
            r'(?<=\.) ',
            r'(?<=\?) ',
            r'(?<=\!) ',
            r'\n',
        ],  # Split on periods, question marks, exclamation marks, new lines, spaces, and empty strings, in the order
    )
    split_doc = text_splitter.split_documents([Document(page_content=doc)])

    # Component for embedding the text, using the inference endpoint deployed using MosaicML
    embeddings = MosaicMLInstructorEmbeddings(
        endpoint_url=args.embedding_endpoint_url,
        embed_instruction='Represent the Financial statement for retrieval: ',
        query_instruction=
        'Represent the Financial question for retrieving supporting documents: '
    )

    # Embed the text in batches, to avoid exceeding the maximum payload size
    batches = []
    current_char_count = 0
    current_batch = []
    for page in split_doc:
        current_batch.append(page)
        current_char_count += len(page.page_content)
        if current_char_count > 4e5:
            batches.append(current_batch)
            current_batch = []
            current_char_count = 0

    if len(current_batch) > 0:
        batches.append(current_batch)

    text_embeddings = []

    for batch in tqdm(batches, desc='Embedding documents', total=len(batches)):
        batch_embeddings = embeddings.embed_documents(
            [d.page_content for d in batch])
        text_embeddings.extend(
            list(zip([d.page_content for d in batch], batch_embeddings)))

    # Component for storing the embeddings in a vector store, using FAISS
    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embeddings,
    )
    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={
            'k': 8
        },  # retrieve the top 8 most similar documents, this can be tweaked
    )

    # Component for generating the response, using the inference endpoint deployed using MosaicML
    llm = MosaicML(
        endpoint_url=args.llm_endpoint_url,
        inject_instruction_format=True,
        model_kwargs={
            'max_new_tokens':
                200,  # maximum number of response tokens to generate
            'do_sample': False,  # perform greedy decoding
            'use_cache': True
            # other HuggingFace generation parameters can be set as kwargs here to experiment with different decoding parameters
        },
    )

    # Prompt template for the query
    answer_question_string_template = (
        f'Use the following pieces of context to answer the question at the end in a single sentence. The context is from a {year} financial document about {ticker}.'
        '\n{context}'
        '\nQuestion: {question}')
    answer_question_prompt_template = PromptTemplate(
        template=answer_question_string_template,
        input_variables=['context', 'question'])

    # Component connecting the LLM with the prompt template
    llm_chain = LLMChain(
        llm=llm,
        prompt=answer_question_prompt_template,
    )

    # Component connecting the context documents with the LLM chain
    stuff_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name='context',
        document_prompt=DOCUMENT_PROMPT,
    )

    # Complete component for retrieval question answering
    answer_qa = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=stuff_documents_chain,
        return_source_documents=True,
    )

    # Get the response
    answer_response = answer_qa(query)

    # Clean the response
    answer = clean_response(answer_response['result'].lstrip('\n'))

    # Return the answer and the retrieved documents
    return answer, '\n\n'.join(
        [d.page_content for d in answer_response['source_documents']])


# Set default tickers, years, and queries for the gradio application
if args.dataset_subset == 'small_full':
    default_ticker = 'FRTG'
    default_year = '2020'
elif args.dataset_subset == 'large_full':
    default_ticker = 'UPST'
    default_year = '2020'
else:
    raise ValueError(
        f'Invalid dataset subset {args.dataset_subset}, must be one of small_full or large_full'
    )

# Simple gradio application for querying the model
with gr.Blocks() as demo:
    ticker = gr.Textbox(label='Ticker', value=default_ticker)
    year = gr.Textbox(label='Year', value=default_year)
    query = gr.Textbox(label='Query',
                       value='What was their revenue for the year?')
    answer = gr.Textbox(label='Answer')
    sources = gr.Textbox(label='Retrieved source documents')
    query_btn = gr.Button('Query')
    query_btn.click(fn=greet,
                    inputs=[ticker, year, query],
                    outputs=[answer, sources])

    demo.launch()

if __name__ == '__main__':
    demo.launch()

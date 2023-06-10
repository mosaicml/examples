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

DOCUMENT_PROMPT = PromptTemplate(input_variables=["page_content"],
                                 template="Context:\n{page_content}")

parser = argparse.ArgumentParser()
parser.add_argument('--llm_endpoint_url', type=str)
parser.add_argument('--embedding_endpoint_url', type=str)
parser.add_argument('--remote_folder_path', type=str)

args, unknown = parser.parse_known_args()

path_to_current_file = os.path.realpath(__file__)
with open(
        os.path.join(os.path.dirname(path_to_current_file), os.pardir,
                     'test_ticker_to_years.json'), 'r') as _json_file:
    ticker_to_years = {
        k: sorted(v, key=lambda x: int(x))
        for k, v in json.load(_json_file).items()
    }


def clean_response(input_text: str) -> str:
    input_text = input_text.strip('\n')

    context_prefix = "Context:"
    answer_prefix = "Answer:"
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
    ticker = ticker.upper()
    remote_file_path = os.path.join(args.remote_folder_path, ticker,
                                    f'sec_{year}_txt.txt')
    local_file_path = os.path.join(os.getcwd(), 'local-data', ticker,
                                   f'sec_{year}_txt.txt')
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

    print(f"Getting file {remote_file_path} to {local_file_path}")

    if ticker not in ticker_to_years:
        return f"Invalid ticker {ticker} see test_ticker_to_years.json for all the tickers in the test set", '', ''

    if year not in ticker_to_years[ticker]:
        return f"Invalid year {year} for ticker {ticker}, available years are {ticker_to_years[ticker]}", '', ''

    if not os.path.exists(local_file_path):
        get_file(remote_file_path, local_file_path)

    with open(local_file_path, 'r') as f:
        doc = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        separators=['\. ', '\? ', '! ', '\n', ' ', ''],
    )
    split_doc = text_splitter.split_documents([Document(page_content=doc)])

    embeddings = MosaicMLInstructorEmbeddings(
        endpoint_url=args.embedding_endpoint_url,
        embed_instruction='Represent the Financial statement for retrieval: ',
        query_instruction=
        'Represent the Financial question for retrieving supporting documents: ',
    )

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

    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embeddings,
    )
    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 8},
    )

    llm = MosaicML(
        endpoint_url=args.llm_endpoint_url,
        inject_instruction_format=True,
        model_kwargs={
            'max_new_tokens': 100,
            'do_sample': False,
            'use_cache': True
        },
    )

    answer_question_string_template = (
        f"Use the following pieces of context to answer the question at the end. The context is from a {year} financial document about {ticker}. If the question cannot be answered accurately from the provided context, say 'The question cannot be answered from the retrieved documents.'."
        "\n{context}"
        "\nQuestion: {question}"
        "\nHelpful answer with evidence from the context (remember to not answer if the question cannot be answered from the provided context):"
    )
    answer_question_prompt_template = PromptTemplate(
        template=answer_question_string_template,
        input_variables=["context", "question"])

    cite_answer_string_template = (
        f"Provide a full sentence direct quote from the context that contains the answer to the question. The context is from a {year} financial document about {ticker}."
        "\n{context}"
        "\nQuestion: {question}"
        "\nThe full sentence from the context that contains the answer to the question:"
    )
    cite_answer_prompt_template = PromptTemplate(
        template=cite_answer_string_template,
        input_variables=["context", "question"])

    llm_chain_cite = LLMChain(
        llm=llm,
        prompt=cite_answer_prompt_template,
    )

    stuff_documents_chain_cite = StuffDocumentsChain(
        llm_chain=llm_chain_cite,
        document_variable_name='context',
        document_prompt=DOCUMENT_PROMPT,
    )

    cite_qa = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=stuff_documents_chain_cite,
        return_source_documents=True,
    )

    llm_chain = LLMChain(
        llm=llm,
        prompt=answer_question_prompt_template,
    )

    stuff_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name='context',
        document_prompt=DOCUMENT_PROMPT,
    )

    answer_qa = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=stuff_documents_chain,
        return_source_documents=True,
    )

    answer_response = answer_qa(query)
    cite_response = cite_qa(query)

    answer = clean_response(answer_response['result'].lstrip('\n'))
    cite = clean_response(cite_response['result'].lstrip('\n'))

    return answer, cite, '\n\n'.join(
        [d.page_content for d in answer_response['source_documents']])


with gr.Blocks() as demo:
    ticker = gr.Textbox(label="Ticker", value='UPST')
    year = gr.Textbox(label="Year", value=ticker_to_years['UPST'][0])
    query = gr.Textbox(label="Query",
                       value="What was their revenue for the year?")
    answer = gr.Textbox(label="Answer")
    quote = gr.Textbox(label="Quote")
    sources = gr.Textbox(label="Retrieved source documents")
    query_btn = gr.Button("Query")
    query_btn.click(fn=greet,
                    inputs=[ticker, year, query],
                    outputs=[answer, quote, sources])

    demo.launch()

if __name__ == '__main__':
    demo.launch()

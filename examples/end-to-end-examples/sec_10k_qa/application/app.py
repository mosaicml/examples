import argparse
import gradio as gr
import os

from composer.utils import get_file

from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.embeddings import MosaicMLInstructorEmbeddings
from langchain.docstore.document import Document
from langchain.llms import MosaicML
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

folder_path = 'oci://mosaicml-internal-checkpoints/daniel/data/sec-filings-large/test/'

DOCUMENT_PROMPT = PromptTemplate(
    input_variables=["page_content"], template="Context:\n{page_content}"
)

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
    llm_endpoint_url: str,
    embedding_endpoint_url: str,
    remote_folder_path: str,
):
    ticker = ticker.upper()
    remote_file_path = f'{folder_path}/{ticker}/sec_{year}_txt.txt'
    local_file_path = f'./local-data/{ticker}/sec_{year}_txt.txt'
    os.makedirs(local_file_path, exist_ok=True)

    try:
        if not os.path.exists(local_file_path):
            get_file(remote_file_path, local_file_path)
    except FileNotFoundError:
        return f"Invalid ticker {ticker} or year {year}"
    
    with open(local_file_path, 'r') as f:
        doc = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        separators=['. ', '? ', '! ', '\n', ' '],
    )
    split_doc = text_splitter.split_documents([doc])

    embeddings = MosaicMLInstructorEmbeddings(
        endpoint_url=embedding_endpoint_url,
        embed_instruction='Represent the Financial statement for retrieval: ',
        query_instruction='Represent the Financial question for retrieving supporting documents: ',
    )
    vector_store = FAISS.from_documents(
        documents=[Document(page_content=doc) for doc in split_doc],
        embedding=embeddings
    )
    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 8},
    )

    llm = MosaicML(
        endpoint_url=llm_endpoint_url,
        inject_instruction_format=False,
        model_kwargs={'max_new_tokens': 100, 'do_sample': False, 'use_cache': True},
    )

    answer_question_string_template = (
        f"Use the following pieces of context to answer the question at the end. The context is from a {year} financial document about {ticker}. If the question cannot be answered accurately from the provided context, say 'The question cannot be answered from the retrieved documents.'."
        "\n{context}"
        "\nQuestion: {question}"
        "\nHelpful answer with evidence from the context (remember to not answer if the question cannot be answered from the provided context):"
    )
    answer_question_prompt_template = PromptTemplate(
        template=answer_question_string_template, input_variables=["context", "question"]
    )

    # cite_answer_string_template = (
    #     f"Extract the two sentences from the context that are most relevant for answering the question at the end. The context is from a {year} financial document about {ticker}."
    #     "\n{context}"
    #     "\nQuestion: {question}"
    #     "\nMost relevant two sentences from the context:"
    # )
    # cite_answer_prompt_template = PromptTemplate(
    #     template=cite_answer_string_template, input_variables=["context", "question"]
    # )

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
    return clean_response(answer_response['result'].lstrip('\n')), answer_response['source_documents']


def main(
    llm_endpoint_url: str,
    embedding_model_endpoint_url: str,
):
    with gr.Blocks() as demo:
        ticker = gr.Textbox(label="Ticker")
        year = gr.Textbox(label="Year")
        query = gr.Textbox(label="Query", value="What was their revenue for the year?")
        answer = gr.Textbox(label="Answer")
        quote = gr.Textbox(label="Quote")
        sources = gr.Textbox(label="Source documents")
        full_doc_text = gr.Textbox(label="Full document text")
        query_btn = gr.Button("Query")
        query_btn.click(fn=greet, inputs=[ticker, year, query, llm_endpoint_url, embedding_model_endpoint_url], outputs=[answer, quote, sources, full_doc_text])

        demo.launch()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_endpoint_url', type=str)
    parser.add_argument('--embedding_model_endpoint_url', type=str)

    args = parser.parse_args()

    main(args.llm_endpoint_url, args.embedding_model_endpoint_url)
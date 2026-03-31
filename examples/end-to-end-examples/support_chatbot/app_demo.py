# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser, Namespace

import gradio as gr
from chatbot import ChatBot
from langchain.embeddings import MosaicMLInstructorEmbeddings
from langchain.llms import MosaicML

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

EVAL_7B_TEMPLATE = (
    f'Answer the following question as one function, class, or object. If you do not know, just say "I do not know".'
    '\n{context}'
    '\nQuestion: {question}')

EVAL_30B_TEMPLATE = ("""<|im_start|>system
                     A conversation between a user and an LLM-based AI assistant about the codebase for the MosaicML library Composer.
                     Provide a helpful and simple answer given the following context to the question. If you do not know, just say "I
                     do not know".<|im_end|>
                     <|im_start|>context
                     {context}<|im_end|>
                     <|im_start|>user
                     {question}<|im_end|>
                     <|im_start|>assistant""")


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(description='Run a chatbot!')
    parser.add_argument(
        '--endpoint_url',
        type=str,
        default=
        'https://models.hosted-on.mosaicml.hosting/mpt-30b-chat/v1/predict',
        required=False,
        help='The endpoint of our MosaicML LLM Model')
    parser.add_argument(
        '--model_name',
        type=str,
        default='mpt-30b-chat',
        required=False,
        help=
        'The model name (mpt-30b-chat or mpt-7b) that determines which prompt template to use when evaluating'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=5000,
        required=False,
        help=
        'The maximum number tokens of both input and output of the model (it will cut off output if token exceeds this length)'
    )
    parser.add_argument('--chunk_size',
                        type=int,
                        default=800,
                        required=False,
                        help='The chunk size when splitting documents')
    parser.add_argument(
        '--chunk_overlap',
        type=int,
        default=400,
        required=False,
        help='The overlap between chunks when splitting documents')
    parser.add_argument(
        '--retrieval_k',
        type=int,
        default=5,
        required=False,
        help='The number of chunks to retrieve as context from vector store')
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        required=False,
        help=
        'The number of highest probability vocabulary tokens to keep for top-k-filtering'
    )
    parser.add_argument('--repository_urls',
                        type=str,
                        nargs='*',
                        default=[
                            'https://github.com/mosaicml/composer',
                            'https://github.com/mosaicml/streaming',
                            'https://github.com/mosaicml/examples',
                            'https://github.com/mosaicml/diffusion',
                            'https://github.com/mosaicml/llm-foundry'
                        ],
                        required=False,
                        help='The GitHub repository URLs to download')
    parser.add_argument('--complex_data_dir',
                        type=str,
                        required=False,
                        default=os.path.join(ROOT_DIR,
                                             'eval_data/complex_eval.jsonl'),
                        help='complex eval data for human eval')
    parser.add_argument('--simple_data_dir',
                        type=str,
                        required=False,
                        default=os.path.join(
                            ROOT_DIR, 'eval_data/composer_docstrings.jsonl'),
                        help='simple eval data for string comparison')
    parser.add_argument(
        '--complex_chat',
        type=int,
        default=0,
        required=False,
        help=
        'Which version of chatting to use (0 for normal, 1 for sub-query, 2 for relation sub-query) Each version is an improvement on the previous though increases response time.'
    )

    parsed = parser.parse_args()

    if parsed.repository_urls is not None:
        # Remove whitespace and turn URLs into a list
        parsed.repository_urls = ''.join(str(
            parsed.repository_urls).split()).split(',')

    return parsed


def main(endpoint_url: str, model_name: str, max_length: int, chunk_size: int,
         chunk_overlap: int, retrieval_k: int, top_k: int,
         repository_urls: list[str], complex_data_dir: str,
         simple_data_dir: str, chat_version: int) -> None:

    retrieval_dir = os.path.join(ROOT_DIR, 'retrieval_data')

    embeddings = MosaicMLInstructorEmbeddings()
    llm = MosaicML(
        inject_instruction_format=True,
        endpoint_url=endpoint_url,
        model_kwargs={
            'max_length': max_length,
            'top_k': top_k,
            'top_p': 0.95,
            'temperature': 0.1,
            # other HuggingFace generation parameters can be set as kwargs here to experiment with different decoding parameters
        },
    )

    chatbot = ChatBot(data_path=retrieval_dir,
                      embedding=embeddings,
                      model=llm,
                      k=retrieval_k,
                      chunk_size=chunk_size,
                      chunk_overlap=chunk_overlap)

    if not chatbot.vector_store:
        print("can't find vectors.pickle, loading it")
        if repository_urls is None:
            raise ValueError(
                'No repository URLs provided. Please provide a comma separated list of URLs to download'
            )
        chatbot.create_vector_store(repository_urls=repository_urls)

    def chat_wrapper(query: str) -> str:
        """Wrapper around chatbot.chat() for gradio
        Args:
            query (str): The query to chatbot

        Returns:
            str: The response from chatbot"""
        if query == '!eval_simple':
            if simple_data_dir is None:
                ValueError(
                    'No simple data directory provided. Please provide a directory with simple eval data'
                )
            if model_name == 'mpt-30b-chat':
                return chatbot.evaluate_simple(simple_data_dir,
                                               EVAL_30B_TEMPLATE)
            elif model_name == 'mpt-7b':
                return chatbot.evaluate_simple(simple_data_dir,
                                               EVAL_7B_TEMPLATE)

        elif query == '!eval_complex':
            if complex_data_dir is None:
                ValueError(
                    'No complex data directory provided. Please provide a directory with complex eval data'
                )
            if model_name == 'mpt-30b-chat':
                return chatbot.evaluate_complex(complex_data_dir,
                                                EVAL_30B_TEMPLATE)
            elif model_name == 'mpt-7b':
                return chatbot.evaluate_complex(complex_data_dir,
                                                EVAL_7B_TEMPLATE)

        if chat_version == 1:
            return chatbot.sub_query_chat(query)
        elif chat_version == 2:
            return chatbot.relation_sub_query_chat(query)
        else:
            return chatbot.chat(query)

    def gradio_chat():
        """Simple gradio application for querying the model."""
        with gr.Blocks() as demo:
            query = gr.Textbox(label='Query', value='What is AliBi?')
            answer = gr.Textbox(label='Answer')
            query_btn = gr.Button('Query')
            query_btn.click(fn=chat_wrapper, inputs=[query], outputs=[answer])
        demo.queue()
        demo.launch()

    gradio_chat()


if __name__ == '__main__':
    args = parse_args()
    main(endpoint_url=args.endpoint_url,
         model_name=args.model_name,
         max_length=args.max_length,
         chunk_size=args.chunk_size,
         chunk_overlap=args.chunk_overlap,
         retrieval_k=args.retrieval_k,
         top_k=args.top_k,
         repository_urls=args.repository_urls,
         complex_data_dir=args.complex_data_dir,
         simple_data_dir=args.simple_data_dir,
         chat_version=args.complex_chat)

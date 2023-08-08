from argparse import ArgumentParser, Namespace
import gradio as gr
from langchain.embeddings import MosaicMLInstructorEmbeddings
from langchain.llms import MosaicML
from chatbot import ChatBot
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Run a chatbot!'
    )
    parser.add_argument(
        '--endpoint_url',
        type=str,
        default='https://models.hosted-on.mosaicml.hosting/mpt-30b-chat/v1/predict',
        required=False,
        help='The endpoint of our MosaicML LLM Model')
    parser.add_argument(
        '--max_length',
        type=int,
        default=1000,
        required=False,
        help='The maximum size of context from LangChain')
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=1000,
        required=False,
        help='The chunk size when splitting documents')
    parser.add_argument(
        '--chunk_overlap',
        type=int,
        default=300,
        required=False,
        help='The overlap between chunks when splitting documents')
    parser.add_argument(
        '--retrieval_k',
        type=int,
        default=5,
        required=False,
        help='The number of chunks to retrieve as context from vector store')
    parser.add_argument(
        '--model_k',
        type=int,
        default=10,
        required=False,
        help='The number of outputs model should output')
    parser.add_argument(
        '--repository_urls',
        type=str,
        default='https://github.com/mosaicml/composer,https://github.com/mosaicml/streaming,https://github.com/mosaicml/examples,https://github.com/mosaicml/diffusion,https://github.com/mosaicml/llm-foundry',
        required=False,
        help='The GitHub repository URLs to download')
    
    parsed = parser.parse_args()
    
    if parsed.repository_urls is not None:
        # Remove whitespace and turn URLs into a list
        parsed.repository_urls = ''.join(str(parsed.repository_urls).split()).split(',')

    return parsed

def main(endpoint_url: str,
         max_length: int,
         chunk_size: int,
         chunk_overlap: int,
         retrieval_k: int,
         model_k: int,
         repository_urls: list[str]) -> None:
    
    retrieval_dir = os.path.join(ROOT_DIR, 'retrieval_data')

    embeddings = MosaicMLInstructorEmbeddings()
    llm = MosaicML(
        inject_instruction_format=True,
        endpoint_url= endpoint_url,
        model_kwargs={
            'output_len': max_length, 
            'top_k': model_k,
            'top_p': 0.95,
            'temperature': 0.1,
            # other HuggingFace generation parameters can be set as kwargs here to experiment with different decoding parameters
        },
    )

    chatbot = ChatBot(data_path= retrieval_dir,
                      embedding=embeddings,
                      model=llm,
                      k=retrieval_k,
                      chunk_size=chunk_size,
                      chunk_overlap=chunk_overlap,
                      max_length=max_length)
    
    if not os.path.isfile(os.path.join(retrieval_dir, 'vectors.pickle')):
        if repository_urls is None:
            raise ValueError('No repository URLs provided. Please provide a comma separated list of URLs to download')  
        chatbot.create_vector_store(repository_urls=repository_urls)
    
    def chat_wrapper(query: str) -> str:
        """Wrapper around chatbot.chat() for gradio
        Args:
            query (str): The query to chatbot

        Returns:
            str: The response from chatbot"""
        return chatbot.chat(query)

    def gradio_chat():
        """Simple gradio application for querying the model"""
        with gr.Blocks() as demo:
            query = gr.Textbox(label='Query',
                               value='What is AliBi?')
            answer = gr.Textbox(label='Answer')
            query_btn = gr.Button('Query')
            query_btn.click(fn=chat_wrapper,
                            inputs=[query],
                            outputs=[answer])
        demo.queue()
        demo.launch()

    gradio_chat()

if __name__ == "__main__":
    args = parse_args()
    main(
        endpoint_url=args.endpoint_url,
        max_length = args.max_length,
        chunk_size = args.chunk_size,
        chunk_overlap = args.chunk_overlap,
        retrieval_k = args.retrieval_k,
        model_k = args.model_k,
        repository_urls = args.repository_urls,
    )
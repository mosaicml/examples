import os
import json
import re
import string
import time
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from scripts.repo_downloader import RepoDownloader

import langchain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import MosaicMLInstructorEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain, LLMChain, RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.llms import MosaicML
from langchain.schema import Document

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = ['ChatBot']

def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Run a chatbot!'
    )
    parser.add_argument(
        '--endpoint_url',
        type=str,
        default='https://mpt-7b-support-bot-finetuned-7yiz5g.inf.hosted-on.mosaicml.hosting/predict',
        required=False,
        help='The endpoint of our MosaicML LLM Model')
    parser.add_argument(
        '--max_length',
        type=int,
        default=150,
        required=False,
        help='The maximum size of context from LangChain')
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=750,
        required=False,
        help='The chunk size when splitting documents')
    parser.add_argument(
        '--chunk_overlap',
        type=int,
        default=150,
        required=False,
        help='The overlap between chunks when splitting documents')
    parser.add_argument(
        '--retrieval_k',
        type=int,
        default=1,
        required=False,
        help='The number of chunks to retrieve as context from vector store')
    parser.add_argument(
        '--model_k',
        type=int,
        default=1,
        required=False,
        help='The number of outputs model should output')
    parser.add_argument(
        '--repository_urls',
        type=str,
        required=True,
        help='The GitHub repository URLs to download')
    
    parsed = parser.parse_args()
    
    if parsed.repository_urls is not None:
        # Remove whitespace and turn URLs into a list
        parsed.repository_urls = ''.join(str(parsed.repository_urls).split()).split(',')

    return parsed

class ChatBot:
    """Given a folder of .txt files from data_path, create a Chatbot object that can process the files into documents, split them
    into managable sizes, and store them in a vector store. The Chatbot can then be used to answer questions about the documents.

    Args:
        data_path (str): The path of the directory where the txt files of interest is located
        embedding (Any): The embedding to use for the vector store
        model (Any): The model to use for the LLMChain
        k (int): The number of similar documents to return from the vector store
        chunk_size (int): The size of the chunks to split the documents into
        chunk_overlap (int): The amount of overlap between chunks

    Warning:
        Be careful when setting k and chunk_size if using the MosaicML Model. There is an maximum input size and will throw an 
        error (ValueError: Error raised by inference API: Expecting value: line 1 column 1 (char 0).) if the input is too large.

    Example:
    .. testcode::


    from langchain.embeddings import MosaicMLInstructorEmbeddings
    from langchain.llms import MosaicML
    chatbot = ChatBot(data_path= "support_chatbot/retrieval_data",
                    embedding=MosaicMLInstructorEmbeddings(),
                    k=3,
                    model=MosaicML())
    chatbot.chat()


    """
    def __init__(self,
                    data_path: str,
                    embedding: langchain.embeddings.base.Embeddings,
                    model: langchain.llms.base.LLM,
                    chunk_size: int,
                    chunk_overlap: int,
                    k: int = 4,
                    ) -> None:
        
        self.data_path = data_path
        self.embedding = embedding
        self.model = model
        self.k = k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_data(self) -> list[Document]:
        """Given a directory find all .txt files and load them as documents into a list
    
        Returns:
            list[Document]: list of documents loaded from data_dir
        """
        data = []
        for dirpath, _, filenames in os.walk(self.data_path):
            for filename in filenames:
                if filename.endswith(".txt"):
                    file_path = os.path.join(dirpath, filename)
                    loaders = UnstructuredFileLoader(file_path, encoding='utf8')
                    data.append(loaders.load()[0])
        return data
    
    def split_pages(self,
                    pages: list[Document]) -> list[Document]:
        """Given a list of documents split them into smaller documents of size `self.chunk_size`
        
        Args:
            pages (list[Document]): list of pages (Documents) we want to split


        Returns:
            list[Document]: list of chunks (Documents) split from pages (Documents)
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                r'(?<=\.) ',
                r'(?<=\?) ',
                r'(?<=\!) ',
                r'\n',
            ],  # Split on periods, question marks, exclamation marks, new lines, spaces, and empty strings, in the order
        )
        return text_splitter.split_documents(pages)
    
    def documents_to_str(self,
                         documents: list[Document]) -> list[str]:
        return map(lambda doc: doc.page_content, documents)
    
    def clean_response(self, input_text: str) -> str:
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
    
    def store_vectors(self,
                      pages: list[Document]) -> None:
        content_batches = []
        content_current_batch = []


        current_char_count = 0   
        for page in pages:
            content_current_batch.append(page)
            current_char_count += len(page.page_content)

            if current_char_count > 1e4:
                content_batches.append(content_current_batch)
                content_current_batch = []
                current_char_count = 0


        if len(content_current_batch) > 0:
            content_batches.append(content_current_batch)


        txt_embeddings = []

        for batch in tqdm(content_batches, desc='Embedding documents', total=len(content_batches)):
            batch_embeddings = self.embedding.embed_documents([p.page_content for p in batch])
            txt_embeddings.extend(list(zip([p.page_content for p in batch], batch_embeddings)))

        # Component for storing the embeddings in a vector store, using FAISS
        vector_store = FAISS.from_embeddings(
            text_embeddings=txt_embeddings,
            metadatas=[p.metadata for p in pages],
            embedding=self.embedding
        )
        
        with open(os.path.join(ROOT_DIR, 'retrieval_data/vectors.pickle'), 'wb') as f:
            pickle.dump(vector_store, f)

    def create_chain(self,
                     prompt_template: str) -> RetrievalQAWithSourcesChain:
        pages = self.load_data()
        documents = self.split_pages(pages)

        if not os.path.isfile(os.path.join(self.data_path, 'vectors.pickle')):
            print("can't find vectors.pickle, loading it")
            self.store_vectors(documents)
        with open(os.path.join(self.data_path, 'vectors.pickle'), 'rb') as f:
            vector_store = pickle.load(f)


        retriever = vector_store.as_retriever(
            search_type='similarity',
            search_kwargs={
                'k': self.k
            }
        )

        answer_question_prompt_template = PromptTemplate(
            template=prompt_template,
            input_variables=['context', 'question'])

        # Component connecting the LLM with the prompt template
        llm_chain = LLMChain(
            llm=self.model,
            prompt=answer_question_prompt_template,
        )

        doc_prompt = PromptTemplate(input_variables=['page_content'],
                                    template='Context:\n{page_content}')

        # Component connecting the context documents with the LLM chain
        stuff_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name='context',
            document_prompt=doc_prompt,
        )

        # Complete component for retrieval question answering
        chain = RetrievalQA(
            retriever=retriever,
            combine_documents_chain=stuff_documents_chain,
            return_source_documents=True,
        )

        return chain
    
    def normalize_str(self, 
                      answer: str):
        """Lower text and remove punctuation, articles and extra whitespace.

        Copied from https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
        """

        def remove_articles(text: str) -> str:
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text: str) -> str:
            return ' '.join(text.split())

        def handle_punc(text: str) -> str:
            exclude = set(string.punctuation + ''.join([u'‘', u'’', u'´', u'`']))
            return ''.join(ch if ch not in exclude else ' ' for ch in text)

        def lower(text: str) -> str:
            return text.lower()
        
        def remove_parentheses(s):
            return re.sub(r'\(.*?\)', '', s)
        
        def replace_underscore(s):
            return re.sub('_', '-', s)

        return white_space_fix(remove_parentheses(remove_articles(handle_punc(lower(replace_underscore(answer)))))).strip()
    
    def evaluate(self, 
                data_path: str,
                chain: RetrievalQAWithSourcesChain) -> int:
        
        exact_match = 0
        close_match = 0
        total = 0
        total_lines = sum(1 for _ in open(data_path))

        with open(data_path, 'r') as file:
            for line in tqdm(file, total=total_lines, desc="Processing lines"):
                data = json.loads(line)
                question = data.get('context')
                continuation = data.get('continuation')
                response = chain(question)
                answer = self.clean_response(response['result'].lstrip('\n'))
                if self.normalize_str(answer) == self.normalize_str(continuation):
                    exact_match += 1
                elif self.normalize_str(continuation).replace(" ", "") in self.normalize_str(answer).replace(" ", ""):
                    close_match += 1
                else:
                    print('\n', self.normalize_str(answer), '||', self.normalize_str(continuation), '\n')
                    print(f'{exact_match} exact matches and {close_match} close matches out of {total} questions.')
                total += 1
                time.sleep(0.5)
        return f'Given Score: {(exact_match + 0.5*close_match)/ total} with {exact_match} exact matches and {close_match} close matches out of {total} questions.'

        
    def evaluate_mpt_7b(self, 
                        data_path: str) -> int:
        if not data_path.endswith('.jsonl'):
            raise ValueError('File is not a .jsonl file')

        # Change model endpoint (and save)
        save_prev_endpoint = self.model.endpoint_url
        self.model.endpoint_url = 'https://mpt-7b-support-bot-pypi-composer-dolly-rg1pfh.inf.hosted-on.mosaicml.hosting/predict'

        # Prompt template for the query
        answer_question_string_template = (
            f'Answer the following question as one function, class, or object. If you do not know, just say "I do not know".'
            '\n{context}'
            '\nQuestion: {question}')
        chain = self.create_chain(answer_question_string_template)
        score = self.evaluate(data_path, chain)
        self.model.endpoint_url = save_prev_endpoint
        return score

    def evaluate_mpt_30b_chat(self, 
                              data_path: str) -> int:
        if not data_path.endswith('.jsonl'):
            raise ValueError('File is not a .jsonl file')

        save_prev_endpoint = self.model.endpoint_url
        self.model.endpoint_url = 'https://mpt-7b-support-bot-pypi-composer-dolly-9jvpsp.inf.hosted-on.mosaicml.hosting/predict'
        # Prompt template for the query
        answer_question_string_template = """<|im_start|>system
            A conversation between a user and an LLM-based AI assistant about the codebase for the MosaicML library Composer. 
            Given some context, the assistant will provide only the function, class, or object name that answers the user's question, 
            or says "I don't know" if the answer isn't available.<|im_end|>
            <|im_start|>context
            {context}<|im_end|>
            <|im_start|>user
            {question}<|im_end|>
            <|im_start|>assistant
            """
        chain = self.create_chain(answer_question_string_template)
        score = self.evaluate(data_path, chain)
        self.model.endpoint_url = save_prev_endpoint
        return score


    def chat(self, 
             query: str,
             max_length: int) -> str:
        eval_root_dir = os.path.join(ROOT_DIR, 'train_data/pipeline_data/composer_docstrings.jsonl')

        if query == "!eval_7b":
            self.model.model_kwargs['output_len'] = 40
            score = self.evaluate_mpt_7b(eval_root_dir)
            self.model.model_kwargs['output_len'] = max_length
            print(score)
            return score
        elif query == "!eval_30b":
            self.model.model_kwargs['output_len'] = 40
            score = self.evaluate_mpt_30b_chat(eval_root_dir)
            self.model.model_kwargs['output_len'] = max_length
            print(score)
            return score
        else:
            answer_question_string_template = (
            # Prompt template for the query
            f'Provide a simple answer given the following context to the question. If you do not know, just say "I do not know".'
            '\n{context}'
            '\nQuestion: {question}')
            chain = self.create_chain(answer_question_string_template)
            response = chain(query)
            answer = self.clean_response(response['result'].lstrip('\n'))
            sources = ''.join([re.sub(r'[^\S ]+', '', d.page_content)+'\n' for d in response['source_documents']])
            return f"Answer: {answer} \nSources: \n{sources}"

def main(endpoint_url: str,
         max_length: int,
         chunk_size: int,
         chunk_overlap: int,
         retrieval_k: int,
         model_k: int,
         repository_urls: list[str]) -> None:
    
    retrieval_dir = os.path.join(ROOT_DIR, 'retrieval_data')
    for repo_url in repository_urls:
        downloader = RepoDownloader(retrieval_dir, "", repo_url)
        if os.path.exists(downloader.clone_dir):
            continue
        downloader.download_repo()


    embeddings = MosaicMLInstructorEmbeddings()
    llm = MosaicML(
        inject_instruction_format=True,
        endpoint_url= endpoint_url,
        model_kwargs={
            'output_len': max_length, 
            'top_k': model_k,
            # other HuggingFace generation parameters can be set as kwargs here to experiment with different decoding parameters
        },
    )

    chatbot = ChatBot(data_path= retrieval_dir,
                      embedding=embeddings,
                      model=llm,
                      k=retrieval_k,
                      chunk_size=chunk_size,
                      chunk_overlap=chunk_overlap)
    chatbot.chat(max_length=max_length, query="!eval_7b")


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
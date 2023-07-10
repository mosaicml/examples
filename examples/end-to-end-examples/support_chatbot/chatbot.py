import os
import sys
from scripts.repo_converter import RepoConverter
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
from tqdm import tqdm
from typing import Any
import json
import re
import string

MOSAICML_MAX_LENGTH = 150
DOCUMENT_PROMPT = PromptTemplate(input_variables=['page_content'],
                                 template='Context:\n{page_content}')

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
                    embedding: Any,
                    model: Any,
                    k: int = 4,
                    chunk_size: int = 1000,
                    chunk_overlap: int = 200,
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
        """Given a list of documents split them into smaller documents of size 1000
        
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
        
        with open('retrieval_data/vectors.pickle', 'wb') as f:
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

        # Component connecting the context documents with the LLM chain
        stuff_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name='context',
            document_prompt=DOCUMENT_PROMPT,
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

        return white_space_fix(remove_parentheses(remove_articles(handle_punc(lower(answer))))).strip()
    
    
        
    def evaluate(self, 
                data_path: str) -> int:
        if not data_path.endswith('.jsonl'):
            raise ValueError('File is not a .jsonl file')

        # Prompt template for the query
        answer_question_string_template = (
            f'Answer the following question as one function, class, or object. If you do not know, just say "I do not know".'
            '\n{context}'
            '\nQuestion: {question}')
        exact_match = 0
        close_match = 0
        total = 0
        total_lines = sum(1 for _ in open(data_path))
        chain = self.create_chain(answer_question_string_template)

        with open(data_path, 'r') as file:
            for line in tqdm(file, total=total_lines, desc="Processing lines"):
                data = json.loads(line)
                question = data.get('context')
                continuation = data.get('continuation')
                response = chain(question)
                answer = self.clean_response(response['result'].lstrip('\n'))
                if self.normalize_str(answer) == self.normalize_str(continuation):
                    exact_match += 1
                elif self.normalize_str(continuation) in self.normalize_str(answer):
                    close_match += 1
                else:
                    print('\n', answer, '||', continuation, '\n')
                total += 1
        return f'Given Score: {(exact_match + 0.5*close_match)/ total} with {exact_match} exact matches and {close_match} close matches out of {total} questions.'

    def chat(self):
        # Prompt template for the query
        answer_question_string_template = (
            f'Provide a robust answer given the following context to the question. If you do not know, just say "I do not know".'
            '\n{context}'
            '\nQuestion: {question}')
        chain = self.create_chain(answer_question_string_template)
        
        question = input("Ask a question: ")
        while question != "!exit":
            if question == "!eval":
                self.model.model_kwargs['max_new_tokens'] = 25
                print(self.evaluate("train_data/pipeline_data/composer_docstrings.jsonl"))
                self.model.model_kwargs['max_new_tokens'] = MOSAICML_MAX_LENGTH
                question = input("Ask a question: ")
                continue
            response = chain(question)
            answer = self.clean_response(response['result'].lstrip('\n'))
            print(answer)
            question = input("Ask a question: ")

def main():
    output_dir = 'retrieval_data'
    if len(sys.argv) < 2:
        raise ValueError("At least one repository URL must be provided as an argument.")
    
    for repo_url in sys.argv[1:]:
        converter = RepoConverter(output_dir, "", repo_url)
        if os.path.exists(converter.clone_dir):
            continue
        converter.convert_repo()


    embeddings = MosaicMLInstructorEmbeddings()
    llm = MosaicML(
        inject_instruction_format=True,
        model_kwargs={
            'max_new_tokens': MOSAICML_MAX_LENGTH, 
            'do_sample': True,  # perform greedy decoding
            'use_cache': True
            # other HuggingFace generation parameters can be set as kwargs here to experiment with different decoding parameters
        },
    )

    chatbot = ChatBot(data_path= "retrieval_data",
                      embedding=embeddings,
                      model=llm,
                      k=5,
                      chunk_size=1000,
                      chunk_overlap=100)
    chatbot.chat()


if __name__ == "__main__":
    main()
import os
import sys
from repo_converter import RepoConverter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import MosaicMLInstructorEmbeddings, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain, LLMChain
from langchain import OpenAI
from langchain.llms import MosaicML
from langchain.schema import Document
from tqdm import tqdm
from typing import Any
from getpass import getpass
MOSAICML_API_TOKEN = getpass()

class ChatBot:
    def __init__(self, 
                 data_path: str,
                 embedding: Any,
                 model: Any,
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 100, 
                 ) -> None:
        
        self.data_path = data_path
        self.embedding = embedding
        self.model = model
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
    '''
    def store_vectors(self, 
                      pages: list[Document]) -> None:
        batches = []
        current_char_count = 0
        current_batch = []
        for page in pages:
            current_batch.append(page)
            current_char_count += len(page.page_content)

            if current_char_count > 1e5:
                batches.append(current_batch)
                current_batch = []
                current_char_count = 0

        if len(current_batch) > 0:
            batches.append(current_batch)

        
        txt_embeddings = []
        for batch in tqdm(batches, desc='Embedding documents', total=len(batches)):
            batch_embeddings = self.embedding.embed_documents([p.page_content for p in batch])
            txt_embeddings.extend(list(zip([p.page_content for p in batch], batch_embeddings)))

        # Component for storing the embeddings in a vector store, using FAISS
        vector_store = FAISS.from_embeddings(
            text_embeddings=txt_embeddings,
            metadatas=[p.metadata for p in pages],
            embedding=self.embedding
        )
        
        with open('examples/end-to-end-examples/support_chatbot/data/vectors.pickle', 'wb') as f:
            pickle.dump(vector_store, f)
    '''
    def store_vectors(self, 
                      documents: list[Document]) -> None:
        vectoreStore_openAI = FAISS.from_documents(documents, self.embedding)
        with open('examples/end-to-end-examples/support_chatbot/data/vectors.pickle', 'wb') as f:
            pickle.dump(vectoreStore_openAI, f)

    def chat(self):
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
                'k': 8
            }
        )

        # Prompt template for the query
        answer_question_string_template = (
            f'Return a answer to the question  and provide an example if one would be helpful. If you do not know, just say "I do not know".'
            '\nQuestion: {summaries}')
        answer_question_prompt_template = PromptTemplate(
            template=answer_question_string_template,
            input_variables=['summaries'])

        chain = RetrievalQAWithSourcesChain.from_chain_type(llm=self.model, 
                                                            #retriever=retriever,
                                                            retriever=vector_store.as_retriever(),
                                                            return_source_documents=True,
                                                            chain_type_kwargs = {"prompt": answer_question_prompt_template})

        question = input("Ask a question: ")
        while question != "exit":
            response = chain({"question": question}, return_only_outputs=True)
            answer = response['answer']
            print(answer)
            question = input("Ask a question: ")
        

def main():
    output_dir = 'examples/end-to-end-examples/support_chatbot/data'
    current_dir = os.path.dirname('examples/end-to-end-examples/support_chatbot')
    os.environ["MOSAICML_API_TOKEN"] = MOSAICML_API_TOKEN
    if len(sys.argv) < 2:
        raise ValueError("At least one repository URL must be provided as an argument.")
    
    for repo_url in sys.argv[1:]:
        converter = RepoConverter(output_dir, current_dir, repo_url)
        converter.convert_repo()

    embeddings = MosaicMLInstructorEmbeddings()
    llm = MosaicML(
        inject_instruction_format=True,
        model_kwargs={
            'max_new_tokens':
                500,  # maximum number of response tokens to generate
            'do_sample': False,  # perform greedy decoding
            'use_cache': True
            # other HuggingFace generation parameters can be set as kwargs here to experiment with different decoding parameters
        },
    )

    api_key = input()
    os.environ["OPENAI_API_KEY"] = api_key

    chatbot = ChatBot(data_path= "examples/end-to-end-examples/support_chatbot/data", 
                      embedding=embeddings, 
                      #embedding=OpenAIEmbeddings(), 
                      model=llm)
    chatbot.chat()

if __name__ == "__main__":
    main()
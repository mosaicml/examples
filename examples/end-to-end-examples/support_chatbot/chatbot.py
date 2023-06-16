import os
import sys
from repo_converter import RepoConverter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import Any

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
            list[langchain.document_loaders.Document]: list of documents loaded from data_dir
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
    
    def store_vectors(self, 
                      documents: list[Document]) -> None:
        vectoreStore_openAI = FAISS.from_documents(documents, self.embeddings)
        with open('scripts/train/support_chatbot/data/vectors.pickle', 'wb') as f:
            pickle.dump(vectoreStore_openAI, f)

    def chat(self):
        pages = self.load_data()
        documents = self.split_pages(pages)

        if not os.path.isfile(os.path.join(self.data_path, 'vectors.pickle')):
            print("can't find vectors.pickle, loading it")
            self.store_vectors(documents)
        with open(os.path.join(self.data_path, 'vectors.pickle'), 'rb') as f:
            vector_store = pickle.load(f) 

        template = """
        Return a robust and in depth answer that is at least seven sentences to the question with examples: {summaries}. If you don't know, just say "I don't know".
        """

        prompt = PromptTemplate(
            input_variables=["summaries"],
            template=template,
        )

        chain = RetrievalQAWithSourcesChain.from_chain_type(llm=self.model, 
                                                            retriever=vector_store.as_retriever(),
                                                            return_source_documents=True,
                                                            chain_type_kwargs = {"prompt": prompt})
        summaries = input("Ask a question: ")
        while summaries != "exit":
            print(chain(summaries, return_only_outputs=True)['answer'])
            summaries = input("Ask a question: ")


def main():
    output_dir = 'examples/end-to-end-examples/support_chatbot/data'
    current_dir = os.path.dirname('examples/end-to-end-examples/support_chatbot')
    if len(sys.argv) < 2:
        raise ValueError("At least one repository URL must be provided as an argument.")
    
    openAI_key = input("Enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = openAI_key

    for repo_url in sys.argv[1:]:
        converter = RepoConverter(output_dir, current_dir, repo_url)
        converter.convert_repo()

    chatbot = ChatBot(data_path= "examples/end-to-end-examples/support_chatbot/data", 
                      embedding=OpenAIEmbeddings(), 
                      model=OpenAI(temperature=0.7, max_tokens=100))
    chatbot.chat()

if __name__ == "__main__":
    main()
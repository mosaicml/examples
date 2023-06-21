import os
import sys
from repo_converter import RepoConverter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import MosaicMLInstructorEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import MosaicML
from langchain.schema import Document
from tqdm import tqdm
from typing import Any
#from getpass import getpass

#MOSAICML_API_TOKEN = getpass()
MOSAICML_MAX_LENGTH = 2048

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
   chatbot = ChatBot(data_path= "examples/end-to-end-examples/support_chatbot/retrieval_data",
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
  
   def store_vectors(self,
                     pages: list[Document]) -> None:
       content_batches = []
       content_current_batch = []


       current_char_count = 0   
       for page in pages:
           content_current_batch.append(page)
           current_char_count += len(page.page_content)


           if current_char_count > 1e5:
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
      
       with open('examples/end-to-end-examples/support_chatbot/retrieval_data/vectors.pickle', 'wb') as f:
           pickle.dump(vector_store, f)


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
               'k': self.k
           }
       )

       # Prompt template for the query
       answer_question_string_template = (
           f'Return a robust answer to the question by first summarizing the object of the request and '
            'answer the prompt. If you do not know, just say "I do not know".'
            '\nQuestion: {summaries}')
       answer_question_prompt_template = PromptTemplate(
           template=answer_question_string_template,
           input_variables=['summaries'])

       chain = RetrievalQAWithSourcesChain.from_chain_type(llm=self.model,
                                                           retriever=retriever,
                                                           return_source_documents=True,
                                                           chain_type_kwargs = {"prompt": answer_question_prompt_template})
      
       question = input("Ask a question: ")
       while question != "exit":
           response = chain({"question": question}, return_only_outputs=True)
           answer = response['answer']
           print(answer)
           question = input("Ask a question: ")

def main():
   output_dir = 'examples/end-to-end-examples/support_chatbot/retrieval_data'
   current_dir = os.path.dirname('examples/end-to-end-examples/support_chatbot')
   #os.environ["MOSAICML_API_TOKEN"] = MOSAICML_API_TOKEN
   if len(sys.argv) < 2:
       raise ValueError("At least one repository URL must be provided as an argument.")
  
   for repo_url in sys.argv[1:]:
       converter = RepoConverter(output_dir, current_dir, repo_url)
       if os.path.exists(converter.clone_dir):
           continue
       converter.convert_repo()


   embeddings = MosaicMLInstructorEmbeddings()
   llm = MosaicML(
       inject_instruction_format=True,
       model_kwargs={
           'max_length': MOSAICML_MAX_LENGTH, 
           'do_sample': False,  # perform greedy decoding
           'use_cache': True
           # other HuggingFace generation parameters can be set as kwargs here to experiment with different decoding parameters
       },
   )

   chatbot = ChatBot(data_path= "examples/end-to-end-examples/support_chatbot/retrieval_data",
                     embedding=embeddings,
                     model=llm,
                     k=3,
                     chunk_size=2500)
   chatbot.chat()


if __name__ == "__main__":
   main()
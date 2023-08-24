import os
import json
import re
import string
import time
from tqdm import tqdm

import langchain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema import Document, BaseRetriever
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR) 
from repo_downloader import RepoDownloader 
from web_downloader import WebScraper
from oci_converser import OCIObjectStorageManager

class RetrieverWithScore(BaseRetriever):
    """Just a custom retriever to track distance between query and retrieval docs

    Args:
       search_type (str): How to measure similarity
       vector_store (FAISS): Retrieval Doc Embeddings
       k: How many chunks 

    """
    def __init__(self, 
                 search_type: str,
                 vector_store: FAISS,
                 k: int,
                 score_threshold: int):
        self.k = k
        self.vector_store=vector_store
        self.score_threshold = score_threshold
        self.search_type=search_type

    def _get_relevant_documents(self, query: str) -> list[Document]:
        # [NOTE] we removed the search type, only use search_type = "similarity"
        if self.search_type != "similarity":
            raise ValueError(f"Only search_type='similarity' is supported with scores")
        docs_and_scores = self.vector_store.similarity_search_with_score(query=query, 
                                                                         k=self.k, 
                                                                         score_threshold=self.score_threshold)
        for doc, score in docs_and_scores:
            doc.metadata = {**doc.metadata, **{"score": score}}
        return [doc for (doc, _) in docs_and_scores]

    def aget_relevant_documents(self, query):
        return self._get_relevant_documents(query)
    
    def get_relevant_documents(self, query: str) -> list[Document]:
        return self._get_relevant_documents(query)

__all__ = ['ChatBot']

EVAL_7B_TEMPLATE = (f'Answer the following question as one function, class, or object. If you do not know, just say "I do not know".'
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
SUBQUERY_INTENT_TEMPLATE = ("""<|im_start|>system
                               A conversation between a user and an LLM-based AI assistant about the codebase for MosaicML. 
                               Provide a helpful, short and simple answer given the following context to the question. Do not
                               attempt to explain any terms and do not go in depth.
                               If you do not know, just say "I do not know".<|im_end|>
                               <|im_start|>context
                               {context}<|im_end|>
                               <|im_start|>user
                               What is the user trying to learn from this question: {question}<|im_end|>
                               <|im_start|>assistant""")
SUBQUERY_RELATED_TEMPLATE = ("""<|im_start|>system
                               A conversation between a user and an LLM-based AI assistant about the codebase for MosaicML. 
                               Only output a "Yes" or "No" with no extra information given the following context to the question. 
                               If you are not sure, say "No"".<|im_end|>
                               <|im_start|>context
                               {context}<|im_end|>
                               <|im_start|>user
                               Can this question be answered by the given context: {question}<|im_end|>
                               <|im_start|>assistant""")
PARTIAL_SUBQA_TEMPLATE = ("""<|im_start|>system
                             A conversation between a user and an LLM-based AI assistant about the codebase for MosaicML. 
                             Given the context, the job of the assistant is to determine if the context is useful for answering the user's question.
                             If so, the assistant will break the question into smaller questions that can likely be answered by a single section of 
                             the relevant context. If the context is not directly related to the user's question, the assistant will just break the 
                             question into simpler questions not related to the context that may be helpful for answering the question.<|im_end|>
                             <|im_start|>context
                             {{context}}<|im_end|>
                             <|im_start|>user
                             {{question}} {} Can this question be answered with the context given alone? If so, break the question down into at most five
                             smaller questions that can likely be answered by a single section of the relevant documentation. If not, break the 
                             question down into at most five helpful questions.
                             Please only respond with a list of smaller questions without any extra information.<|im_end|>
                             <|im_start|>assistant""")
PARTIAL_COMBINE_TEMPLATE = ("""<|im_start|>system
                               A conversation between a user and an LLM-based AI assistant. 
                               Here are smaller questions regarding the user's question and their answers:
                               {}
                               Provide a helpful and in depth answer given the following context to the question. 
                               If you do not know, just say "I do not know".<|im_end|>
                               <|im_start|>context
                               {{context}}<|im_end|>
                               <|im_start|>user
                               {{question}}<|im_end|>
                               <|im_start|>assistant""")
PARTIAL_COMBINE_UR_TEMPLATE = ("""<|im_start|>system
                               A conversation between a user and an LLM-based AI assistant. 
                               Here are smaller questions regarding the user's question. If you don't know how to answer the question pretend like it doesn't exist:
                               {}
                               Provide a helpful and in depth answer given the following context to the question. 
                               If you do not know, just say "I do not know".<|im_end|>
                               <|im_start|>context
                               {{context}}<|im_end|>
                               <|im_start|>user
                               {{question}}<|im_end|>
                               <|im_start|>assistant""")


EVAL_SIMPLE_DIR = os.path.join(ROOT_DIR, 'train_data/pipeline_data/composer_docstrings.jsonl')
EVAL_COMPLEX_DIR = os.path.join(ROOT_DIR, 'train_data/pipeline_data/complex_eval.jsonl')

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
        Also, as of right now, there is a problem with inference where tokenizing will drop spaces before punctuation, as well 
        as dropping special characters required for running 30B prompt. This will cause an incorrect splicing of the answer from
        the question.

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
                 k: int,
                 slack_path: str = False,
                 ) -> None:
        
        self.data_path = data_path
        self.embedding = embedding
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.saved_state = {'k': k, 'chunk_size': chunk_size, 'chunk_overlap': chunk_overlap, 'model_k': model.model_kwargs['top_k'],
                            'endpoint_url': model.endpoint_url}
        self.chat_chain = None
        self.intent_chain = None
        self.subchain = None
        self.subsubchain = None
        self.slack_path = slack_path
        self.vector_store = None

        if os.path.isfile(os.path.join(data_path, 'vectors.pickle')):
            with open(os.path.join(self.data_path, 'vectors.pickle'), 'rb') as f:
                self.vector_store = pickle.load(f)

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
                    document = loaders.load()[0]
                    document.metadata = {**document.metadata, **{'file_name': filename.replace('{slash}', '/').replace('{dot}', '.').replace('{colon}', ':')[:-4]}}
                    data.append(document)
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
        """Clean the response from the model by stripping some bad answer prefixes, new lines, etc.

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
        """Given a list of documents, split them into chunks, and store them in a vector store.

        Args:
            pages (list[Document]): list of pages (Documents) we have splitted
        """
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
        
        if self.slack_path:
            with open(os.path.join(ROOT_DIR, 'retrieval_data_slack/vectors.pickle'), 'wb') as f:
                pickle.dump(vector_store, f)
                self.vector_store = vector_store
        else:
            with open(os.path.join(ROOT_DIR, 'retrieval_data_demo/vectors.pickle'), 'wb') as f:
                pickle.dump(vector_store, f)
                self.vector_store = vector_store

    def create_vector_store(self, repository_urls) -> None:
        """Download the repositories, load the data, split the data into chunks, and store the chunks in a vector store.

        Args:
            repository_urls (list[str]): list of repository urls to download
        """
        scraper = WebScraper(path=self.data_path)
        scraper.scrape()
        for repo_url in repository_urls:
            downloader = RepoDownloader(output_dir=self.data_path, current_dir="", repo_url=repo_url)
            if os.path.exists(downloader.clone_dir):
                continue
            downloader.download_repo()
        if self.slack_path:
            oci_manager = OCIObjectStorageManager(oci_uri=self.slack_path)
            if not os.path.exists(os.path.join(self.data_path, 'slack_data')):
                os.makedirs(os.path.join(self.data_path, 'slack_data'))
            oci_manager.download_directory(os.path.join(self.data_path, 'slack_data'))

        pages = self.load_data()
        documents = self.split_pages(pages)
        print("can't find vectors.pickle, loading it")
        self.store_vectors(documents)

    def create_chain(self,
                     prompt_template: str,
                     score_threshold: int= 0.4) -> RetrievalQA:
        """Create a RetrievalQAWithScores given a prompt template.
        
        Args:
            prompt_template (str): The prompt template to use for the chain
        """

        retriever = RetrieverWithScore(search_type='similarity',
                                       vector_store=self.vector_store,
                                       k=self.k,
                                       score_threshold=score_threshold)

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
    
    def set_eval_state(self, 
                       endpoint_url: str) -> None:
        """Set the state of the chatbot to the evaluation state. This is used to change the chunk size, chunk overlap, and k"""
        self.chunk_overlap = 150
        self.chunk_size = 750
        self.k = 1
        self.model.model_kwargs['output_len'] = 40
        self.model.endpoint_url = endpoint_url

    def reload_chat_state(self) -> None:
        """Reload the chatbot state to the saved state the user set when creating the chatbot"""
        self.chunk_overlap = self.saved_state['chunk_overlap']
        self.chunk_size = self.saved_state['chunk_size']
        self.k = self.saved_state['k']
        self.model.endpoint_url = self.saved_state['endpoint_url']
    
    def evaluate_simple(self, 
                        data_path: str,
                        answer_question_string_template: str) -> str:
        """Evaluate the chatbot on simple retrieval dataset given a data_path and a chain

        Args:
            data_path (str): The path to the dataset
            answer_question_string_template (str): The prompt to use for the chain

        Returns:
            str: The score of the chatbot on the dataset including number of exact matches, close matches, and total questions
        """
        chain = self.create_chain(answer_question_string_template)
        exact_match = 0
        close_match = 0
        total = 1
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

    def evaluate_complex(self, 
                         data_path: str,
                         answer_question_string_template: str) -> str:
        """Evaluate the chatbot on complex eval dataset given a data_path and a chain
        
        Args:
            data_path (str): The path to the dataset
            answer_question_string_template (str): The prompt to use for the chain

        Returns:
            A long string of all questions, answers, and responses
        """
        chain = self.create_chain(answer_question_string_template)
        total_lines = sum(1 for _ in open(data_path))
        with open(data_path, 'r') as file:
            save = ''
            for line in tqdm(file, total=total_lines, desc="Processing lines"):
                data = json.loads(line)
                question = data.get('context')
                continuation = data.get('continuation')
                response = chain(question)
                answer = self.clean_response(response['result'].lstrip('\n'))
                time.sleep(0.5)
                save += f'Question:\n{question}\nAnswer:\n{continuation}\nResponse:\n{answer}\n\n'
        return save
    
    def sub_query_chat(self,
                       query: str)-> str:
        if not self.intent_chain:
            save_k = self.k
            self.k = 5
            self.intent_chain = self.create_chain(SUBQUERY_INTENT_TEMPLATE)
            self.k = save_k
        intent_response = self.intent_chain(query)
        intent_answer = self.clean_response(intent_response['result'].lstrip('\n'))
        
        SUBQUERY_SUBQA_TEMPLATE = PARTIAL_SUBQA_TEMPLATE.format(intent_answer)
        subQA_chain = self.create_chain(SUBQUERY_SUBQA_TEMPLATE)
        subQA_response = subQA_chain(query)
        subQA_answer = self.clean_response(subQA_response['result'].lstrip('\n'))

        all_sub_QA = subQA_answer.split('\n')
        sub_QA_injection = ''
        # Don't create a new chain on every query
        if not self.subchain:
            self.subchain = self.create_chain(EVAL_30B_TEMPLATE)
        for sub_QA in all_sub_QA:
            response = self.subchain(sub_QA)
            if response['source_documents']:
                answer = self.clean_response(response['result'].lstrip('\n'))
                sub_QA_injection += f'Question: {sub_QA} \nAnswer: {answer}\n'

        SUBQUERY_COMBINE_TEMPLATE = PARTIAL_COMBINE_TEMPLATE.format(sub_QA_injection)
        combine_chain = self.create_chain(SUBQUERY_COMBINE_TEMPLATE)
        combine_response = combine_chain(query)
        combine_answer = self.clean_response(combine_response['result'].lstrip('\n'))
        combine_answer_sources = ''
        slack_deduplicate = True
        for d in combine_response['source_documents']:
            if d.metadata["score"] < 0.7:
                if 'message_from_slack' == combine_answer_sources[:18] and slack_deduplicate:
                    combine_answer_sources = combine_answer_sources + 'slack_data\n'
                    slack_deduplicate = False
                else:
                    combine_answer_sources = combine_answer_sources + f'{d.metadata["file_name"].replace("{slash}", "/")}\n'
        if not combine_answer_sources:
            return f'Answer: \n{combine_answer}\n\nIntent: \n{intent_answer}\n\nRelated Sub-questions: \n{sub_QA_injection}'
        else:
            return f'Answer: \n{combine_answer}\n\nIntent: \n{intent_answer}\n\nRelated Sub-questions: \n{sub_QA_injection}\nSources: \n{combine_answer_sources}'
          
    def relation_sub_query_chat(self,
                       query: str)-> str:
        if not self.intent_chain:
            save_k = self.k
            self.k = 3
            self.intent_chain = self.create_chain(SUBQUERY_INTENT_TEMPLATE)
            self.k = save_k
        intent_response = self.intent_chain(query)
        intent_answer = self.clean_response(intent_response['result'].lstrip('\n'))
        
        SUBQUERY_SUBQA_TEMPLATE = PARTIAL_SUBQA_TEMPLATE.format(intent_answer)
        subQA_chain = self.create_chain(SUBQUERY_SUBQA_TEMPLATE)
        subQA_response = subQA_chain(query)
        subQA_answer = self.clean_response(subQA_response['result'].lstrip('\n'))

        all_sub_QA = subQA_answer.split('\n')
        sub_QA_injection = ''
        sub_QA_UR_injection = ''
        # Don't create a new chain on every query
        if not self.subsubchain:
            self.subsubchain = self.create_chain(prompt_template=SUBQUERY_RELATED_TEMPLATE, score_threshold=0)
        for sub_QA in all_sub_QA:
            answerable = self.clean_response(self.subsubchain(sub_QA)['result'].lstrip('\n'))
            if "Yes" in answerable:
                if not self.subchain:
                    self.subchain = self.create_chain(EVAL_30B_TEMPLATE)
                response = self.subchain(sub_QA)
                answer = self.clean_response(response['result'].lstrip('\n'))
                sub_QA_injection += f'Question: {sub_QA} \nAnswer: {answer}\n'
            sub_QA_UR_injection += f'Question: {sub_QA}\n'

        if sub_QA_injection:
            SUBQUERY_COMBINE_TEMPLATE = PARTIAL_COMBINE_TEMPLATE.format(sub_QA_UR_injection)
            combine_chain = self.create_chain(SUBQUERY_COMBINE_TEMPLATE)
            combine_response = combine_chain(query)
            combine_answer = self.clean_response(combine_response['result'].lstrip('\n'))
            sources = ''
            slack_deduplicate = True
            for d in combine_response['source_documents']:
                if d.metadata["score"] < 0.7:
                    if 'message_from_slack' == sources[:18] and slack_deduplicate:
                        sources = sources + 'slack_data\n'
                        slack_deduplicate = False
                    else:
                        sources = sources + f'{d.metadata["file_name"].replace("{slash}", "/")}\n'
            if not sources:
                return f'Answer: \n{combine_answer}\n\nIntent: \n{intent_answer}\n\n Sub-questions: \n{sub_QA_injection}'
            else:
                return f'Answer: \n{combine_answer}\n\nIntent: \n{intent_answer}\n\n Sub-questions: \n{sub_QA_injection}\nSources: \n{sources}'
        #Still make the chatbot robust at answering non-related questions
        else:
            SUBQUERY_COMBINE_UR_TEMPLATE = PARTIAL_COMBINE_UR_TEMPLATE.format(sub_QA_injection)
            combine_chain = self.create_chain(SUBQUERY_COMBINE_UR_TEMPLATE)
            combine_response = combine_chain(query)
            combine_answer = self.clean_response(combine_response['result'].lstrip('\n'))
            
            return f'Answer: \n{combine_answer}\n\nIntent: \n{intent_answer}\n\n Sub-questions: \n{sub_QA_UR_injection}'

    def chat(self, 
             query: str) -> str:
        """Chat with the chatbot given a query
        
        Args:
            query (str): The query to ask the chatbot
        """
    
        if query == "!eval_7b":
            self.set_eval_state(endpoint_url='https://chatbot-7b-finetuned-rxyfc5.inf.hosted-on.mosaicml.hosting/predict')
            score = self.evaluate_simple(EVAL_SIMPLE_DIR, EVAL_7B_TEMPLATE)
            self.reload_chat_state()
            print(score)
            return score
        elif query == "!eval_7b_complex":
            self.model.endpoint_url = 'https://chatbot-7b-finetuned-rxyfc5.inf.hosted-on.mosaicml.hosting/predict'
            out = self.evaluate_complex(EVAL_COMPLEX_DIR, EVAL_7B_TEMPLATE)
            self.model.endpoint_url = self.saved_state['endpoint_url']
            return out
        elif query == "!eval_30b":
            score = self.evaluate_simple(EVAL_SIMPLE_DIR, EVAL_30B_TEMPLATE)
            print(score)
            return score
        elif query == "!eval_30b_complex":
            out = self.evaluate_complex(EVAL_30B_TEMPLATE, EVAL_30B_TEMPLATE)
            return out
        else:
            # Don't create a new chain on every query
            if not self.chat_chain:
                self.chat_chain = self.create_chain(EVAL_30B_TEMPLATE)
            response = self.chat_chain(query)
            answer = self.clean_response(response['result'].lstrip('\n'))
            sources = ''
            slack_deduplicate = True
            for d in response['source_documents']:
                if d.metadata["score"] < 0.6:
                    if 'message_from_slack' == sources[:18] and slack_deduplicate:
                        sources = sources + 'slack_data\n'
                        slack_deduplicate = False
                    else:
                        sources = sources + f'{d.metadata["file_name"].replace("{slash}", "/")}\n'
            if not sources:
                return f"Answer: {answer}"
            else:
                return f"Answer: {answer} \nSources: \n{sources}"
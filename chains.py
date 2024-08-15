from langchain_openai import ChatOpenAI
from decorators import error_traceback
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.globals import set_debug

class pdf_rag():
    
    @error_traceback
    def __init__(self, debug=False) -> None:
        self.llm = None
        self.prompt = None
        self.retreival = None
        self.rag_chain = None
        if debug==True:
             set_debug(True)
    
    @error_traceback
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    @error_traceback
    def setup(self, open_ai_key:str, 
              llm_model:str, 
              pdf_doc_path:str) -> None:

        # setup the LLM model
        self.llm = ChatOpenAI(model=llm_model,
                                    api_key=open_ai_key)

        # load the PDF document
        loader = PyPDFLoader(pdf_doc_path)
        main_docs = loader.load()

        # create the retriver
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        splits = text_splitter.split_documents(main_docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=open_ai_key))
    
        # Retrieve and generate using the relevant snippets of the blog.
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k":3}
        )
        self.retreival = {"context": retriever | self.format_docs, "question": RunnablePassthrough()}

        # create the prompt
        prompt_str = """You are an assistant for question-answering tasks. \
            Use the following pieces of retrieved context to answer the question. \
            If you don't know the answer, just say 'Information not present in document'. \
            Use three sentences maximum and keep the answer concise.\
            Context: {context} \
            Question: {question} \
            Answer: """
        self.prompt = ChatPromptTemplate.from_template(prompt_str)

        # create the chain
        #self.rag_chain =  self.retreival | self.prompt | self.llm | StrOutputParser()
        self.rag_chain =  self.retreival | self.prompt | self.llm

        print("rag object initialized")

    @error_traceback
    def run(self, input_message) -> str:
        out = self.rag_chain.invoke(input_message)
        return out.content, out.usage_metadata




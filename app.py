import os
import sys
from dotenv import load_dotenv, find_dotenv
import gradio as gr
import logging
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.globals import set_debug
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from decorators import error_traceback
from chains import pdf_rag


llm_model = "gpt-4o-mini"


@error_traceback
def text_splitting_tests():
    _ = load_dotenv(find_dotenv()) # read local .env file
    openai_api_key_str = os.environ['OPENAI_API_KEY']
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    langchain_api_key_str = os.environ["LANGCHAIN_API_KEY"]

    string_2_check = "What are the steps to \"Development with AWS Services\""
    string_2_check_v2 = "What is the temperature in Alaska"

    # run with rag class
    rag_obj = pdf_rag(debug=True)
    rag_obj.setup( open_ai_key=openai_api_key_str,
                    llm_model=llm_model, 
                    pdf_doc_path="documents/AWS_Certified_Developer_Associate_Updated_June_2018_Exam_Guide_v1.3pdf.pdf",
                  )
    out, usage_metadata = rag_obj.run(string_2_check)
    print("--------------OUT--------------")
    print(out)
    print(usage_metadata)

    

    '''
    set_debug(True)

    llm = ChatOpenAI(model="gpt-4o-mini")

    # load the PDF document
    loader = PyPDFLoader("documents/AWS_Certified_Developer_Associate_Updated_June_2018_Exam_Guide_v1.3pdf.pdf")
    pages = loader.load_and_split()
    print(f"num pages={len(pages)}")      
    

    loader = PyPDFLoader("documents/AWS_Certified_Developer_Associate_Updated_June_2018_Exam_Guide_v1.3pdf.pdf")
    main_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    splits = text_splitter.split_documents(main_docs)
    print(len(splits))
    print(len(splits[1].page_content))
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    
    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k":3}
    )
    docs = retriever.invoke(string_2_check_v3)
    for doc in docs:
        print("--------------PAGE:{}--------------".format(doc.metadata["page"]))
        print(doc.page_content)

    # make the RAG prompt template
    prompt = hub.pull("rlm/rag-prompt")
    
    # # self created template
    # prompt_str = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\
    #     Context: {context} \
    #     Question: {question} \
    #     Answer: """
    # prompt = ChatPromptTemplate.from_template(prompt_str)
    print(prompt)

    # make the RAG prompt example
    # example_messages = prompt.invoke(
    #     {"context": docs[0].page_content, "question": string_2_check}
    # ).to_messages()
    # print(example_messages)

    retreival = {"context": retriever | format_docs, "question": RunnablePassthrough()}
    print(retreival)

    #rag_chain =  retreival | prompt | llm | StrOutputParser()
    rag_chain =  retreival | prompt | llm 
    out = rag_chain.invoke(string_2_check_v2)
    print("--------------OUT--------------")
    print(out)


    # for chunk in rag_chain.stream(string_2_check):
    #     print(chunk, end="", flush=True)
    '''

if __name__ == "__main__":
    try:
        
        text_splitting_tests()

        '''
        # setup the gradio chatbot
        with gr.Blocks() as demo:
            gr.Markdown("## DALL-E Image Creation")
            with gr.Row():
                openai_api_key = gr.Textbox(label="OpenAI Key", type="password", value="", placeholder="Enter your OpenAI API key...")
                output_file_path = gr.Textbox(label="Save Log File Path (optional)", placeholder="Enter the file path to save the log file...", type="text")

            with gr.Row():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")

            generate_button = gr.Button("Generate")
            
            with gr.Row():
                result_text = gr.Textbox(label="Response output:", lines=10, interactive=False)
                images_output = gr.Image(label="Generated Image")

            # if the model changes to "dall-e-3", we need to change the resolution and n
            generate_button.click(
                generate_image_v2,
                inputs=[openai_api_key, output_file_path, prompt],
                outputs=[result_text, images_output]
            )
        
            # with gr.Row():
            #     inputs_x=gr.Textbox(label="image description"),
            #     outputs_x=gr.Image(label="DALL-E Image"),
            # btn2 = gr.Button("Run")
            
            # btn2.click(fn=agentObj.generate_image, inputs=inputs_x[0], outputs=outputs_x[0])

        demo.launch(share=True)
        '''

        # # Gradio interface components
        # with gr.Blocks() as demo:
        #     gr.Markdown("## DALL-E Image Creation")
        #     with gr.Row():
        #         openai_api_key = gr.Textbox(label="OpenAI Key", type="password", value=key, placeholder="Enter your OpenAI API key...")
        #         output_file_path = gr.Textbox(label="Save Log File Path (optional)", placeholder="Enter the file path to save the log file...", type="text")
        #     with gr.Row():
        #         prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
        #         n = gr.Slider(minimum=1, maximum=10, step=1, label="Images Requested", value=1)
        #         model = gr.Radio(choices=["dall-e-2", "dall-e-3"], label="Model", value="dall-e-2")
        #         resolution = gr.Radio(choices=["256x256", "512x512", "1024x1024"], label="Image Resolution", value="256x256")
        #     with gr.Row():
        #         quality = gr.Radio(choices=["standard", "hd"], label="Quality", value="standard")
        #         style = gr.Radio(choices=["natural", "vivid"], label="Style", value="natural")
                
        #     generate_button = gr.Button("Generate")
            
        #     with gr.Row():
        #         result_text = gr.Textbox(label="Response output:", lines=10, interactive=False)
        #         images_output = gr.Gallery(label="Generated Images")

        #     # if the model changes to "dall-e-3", we need to change the resolution and n
        #     model.change(match_res_n_model, [model,resolution,n],[resolution,n])
        #     generate_button.click(
        #         generate_images,
        #         inputs=[openai_api_key, output_file_path, prompt, n, resolution, model, style, quality],
        #         outputs=[result_text, images_output]
        #     )
        # demo.launch(share=True)

    except Exception as e:
            logging.error('file:{} line:{} type:{}, message:{}'.format(
                     os.path.basename(__file__), sys.exc_info()[-1].tb_lineno, type(e).__name__, str(e)))



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
rag_obj = pdf_rag(debug=True)

@error_traceback
def text_splitting_tests():
    # _ = load_dotenv(find_dotenv()) # read local .env file
    # openai_api_key_str = os.environ['OPENAI_API_KEY']
    # os.environ["LANGCHAIN_TRACING_V2"] = "true"
    # langchain_api_key_str = os.environ["LANGCHAIN_API_KEY"]

    # string_2_check = "What are the steps to \"Development with AWS Services\""
    # string_2_check_v2 = "What is the temperature in Alaska"

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

@error_traceback
def temp_func(openai_api_key, file_path, question):
    rag_obj.setup(open_ai_key=openai_api_key,
                    llm_model=llm_model, 
                    pdf_doc_path=file_path,
                  )
    out, usage_metadata = rag_obj.run(question)
    return out, usage_metadata

if __name__ == "__main__":
        
    #text_splitting_tests()

    # setup the gradio chatbot
    with gr.Blocks() as demo:
        gr.Markdown("## PDF RAG SEARCH")
        with gr.Row():
            file_path = gr.File(label="PDF file", file_types=[".pdf"], file_count="single")
            openai_api_key = gr.Textbox(label="OpenAI Key", type="password", value="", placeholder="Enter your OpenAI API key...")

        with gr.Row():
            question = gr.Textbox(label="Question", placeholder="Enter your question here...")

        generate_button = gr.Button("Generate")
        
        with gr.Row():
            result_text = gr.Textbox(label="Response output:", lines=10, interactive=False)
            images_output = gr.Textbox(label="Meta Data")
            

        # if the model changes to "dall-e-3", we need to change the resolution and n
        generate_button.click(
            temp_func,
            inputs=[openai_api_key, file_path, question],
            outputs=[result_text, images_output]
        )
    
        # with gr.Row():
        #     inputs_x=gr.Textbox(label="image description"),
        #     outputs_x=gr.Image(label="DALL-E Image"),
        # btn2 = gr.Button("Run")
        
        # btn2.click(fn=agentObj.generate_image, inputs=inputs_x[0], outputs=outputs_x[0])

    demo.launch(share=False)


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




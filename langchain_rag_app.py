import os.path
import warnings
from pydantic import PydanticDeprecatedSince20

# Ignore specific warning from logger
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import utils
import gradio as gr

# global variables
DATA_DIR = 'data/'
CHUNK_SIZE_IN_CHAR = 1000
VECTOR_STORE_PATH = "vector_store/chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
NUM_OF_RELEVANT_CHUNKS = 3
LLM_MODEL_ID = "gpt-4o-mini"
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 512


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_retrieved_resources(relevant_chunks):
    # join retrieved resources in one text
    retrieved_resources = ""
    for i, doc in enumerate(relevant_chunks):
        retrieved_resources += (f"Resource {i + 1}: \n"
                                f"File path: {doc.metadata['source']} \n"
                                f"Page: {doc.metadata['page']} \n\n")
    return retrieved_resources


def langchain_rag_answer(query):
    generated_text = ""
    retrieved_resources = ""
    for chunk in rag_chain.stream({"input": query}):
        if "context" in chunk:
            retrieved_resources = chunk["context"]
            retrieved_resources = format_retrieved_resources(retrieved_resources)
        if "answer" in chunk:
            generated_text += chunk["answer"]
        yield generated_text, retrieved_resources


if __name__ == "__main__":
    # load the documents
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()

    # Document Chunking, Create Embedding,  Build vector-store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE_IN_CHAR,chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # create and save to disk
    if not os.path.exists(VECTOR_STORE_PATH):
        vectorstore = Chroma.from_documents(documents=splits,
                                            embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
                                            persist_directory=VECTOR_STORE_PATH)
    # or load from disk
    else:
        vectorstore = Chroma(persist_directory=VECTOR_STORE_PATH,
                             embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL))
    retriever = vectorstore.as_retriever(search_type="similarity",
                                         search_kwargs={"k": NUM_OF_RELEVANT_CHUNKS})

    # load OpenAI LLM API
    llm_gpt4o_mini = ChatOpenAI(model_name=LLM_MODEL_ID,
                                max_tokens=MAX_NEW_TOKENS,
                                temperature=TEMPERATURE)

    system_prompt = """Based on the following context items, please answer the query.
     Don't return the thinking, only return the answer.
     Make sure your answers are as explanatory as possible.
     Use the following example as a reference for the ideal answer style.
     \nExample 1:
     Query: What is the role of backpropagation in neural networks?
     Answer: Backpropagation is a key algorithm used for training neural networks by minimizing the error between predicted and actual outputs. It involves a forward pass where the input data is propagated through the network to generate an output, and a backward pass where the error is propagated back through the network to update the weights. This is done using the gradient descent optimization method, which calculates the gradient of the loss function with respect to each weight and adjusts the weights to reduce the error. Backpropagation allows neural networks to learn complex patterns in data by iteratively improving the model's accuracy.
     \nNow use the following context items to answer the user query:
     {context}
     \nRelevant passages: <extract relevant passages from the context here>"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm_gpt4o_mini, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Launch the app
    #  gr.themes.Base() gr.themes.Default() gr.themes.Glass() gr.themes.Monochrome() gr.themes.Soft()
    theme = gr.themes.Monochrome()
    demo = utils.gradio_rag_blocks(title="Chat With Your Data! (LangChain)",
                                   description="Ask your documents using langchain (RAG) pipeline through " \
                                               "OpenAI's API.",
                                   submit_fun=langchain_rag_answer,
                                   theme=theme)
    free_port = utils.get_free_port()
    demo.launch(server_port=free_port)

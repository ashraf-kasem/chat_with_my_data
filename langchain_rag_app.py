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
import gradio_utils

# global variables
# VECTOR_STORE_PATH = ""
# EMBEDDING_MODEL = ""
# NUM_OF_RELEVANT_CHUNKS = 3
# LLM_MODEL_ID = ""
TEMPERATURE = 0.4
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
    directory_path = "data/"
    loader = PyPDFDirectoryLoader(directory_path)
    docs = loader.load()

    # Document Chunking, Create Embedding,  Build vector-store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # load OpenAI LLM API
    llm_gpt4o_mini = ChatOpenAI(model_name="gpt-4o-mini",
                                max_tokens=MAX_NEW_TOKENS,
                                temperature=TEMPERATURE)

    # create the prompt template
    # system_prompt = (
    #     "You are an assistant for question-answering tasks. "
    #     "Use the following pieces of retrieved context to answer "
    #     "the question. If you don't know the answer, say that you "
    #     "don't know. Use three sentences maximum and keep the "
    #     "answer concise."
    #     "\n\n"
    #     "{context}"
    # )

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
    demo = gradio_utils.gradio_rag_blocks(title="Chat With Your Data!",
                                          description="Ask your documents using langchain (RAG) pipeline through " \
                                                      "OpenAI's API.",
                                          submit_fun=langchain_rag_answer)
    demo.launch()

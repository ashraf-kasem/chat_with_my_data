import pandas as pd
import torch
import numpy as np
from sentence_transformers import util, SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TextIteratorStreamer
from threading import Thread
import gradio_utils

""" This script is to start RAG pipeline """

# global variables
VECTOR_STORE_PATH = "vector_store/embeddings.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL = "all-mpnet-base-v2"
NUM_OF_RELEVANT_CHUNKS = 5
SIMILARITY_THRESHOLD = 0.3
LLM_MODEL_ID = "google/gemma-2b-it"
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 512


def load_vector_store(vector_store_path, device):
    loaded_df = pd.read_csv(vector_store_path)
    # Convert embedding column back to np.array if they were string
    if isinstance(loaded_df["embedding"][0], str):
        loaded_df["embedding"] = loaded_df["embedding"].apply(
            lambda x: np.fromstring(x.strip("[]"), sep=" "))

    # Convert texts and embedding df to list of dicts (data index)
    data_index = loaded_df.to_dict(orient="records")

    # Convert embeddings to torch tensor and send to device
    embeddings = torch.tensor(np.array(loaded_df["embedding"].tolist()), dtype=torch.float32).to(device)
    return embeddings, data_index


def rag_retrieve(query, embedding_model, vectore_store, top_k):
    # embedd the query
    embedded_query = embedding_model.encode(query, convert_to_tensor=True)
    # dot product (cosine similarity because vectors are normalized)
    scores = util.dot_score(a=embedded_query, b=vectore_store)[0]
    # get the top k results
    scores, indices = torch.topk(input=scores, k=top_k)
    return scores, indices


def show_retrieval_results(data_dict, query, scores, indices):
    print(f"Query: {query}\n")
    print("Results:\n")
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        # Print file path the page number
        print(f"File path: {data_dict[index]['file_path']}")
        print(f"Page number: {data_dict[index]['page_number']}")
        # Print relevant sentence chunk
        print("Text:")
        print(data_dict[index]["sentence_chunk"])
        print("\n")


def load_llm(model_id):
    # My GPU is Nvidia RTX 3060 with 6GB memory
    # Loading 2 Billion  parameters model in full precision needs 2b * 4 ~ 8GB of GPU memory
    # I need to do quantization to int-8 or int-4
    # load in 4bit precision (boost the inference time significantly)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,
                                                     torch_dtype=torch.bfloat16,
                                                     quantization_config=quantization_config,
                                                     low_cpu_mem_usage=False)
    return tokenizer, llm_model


def prepare_augmented_prompt(query, relevant_chunks, tokenizer):
    """
    function to better format the prompt:
    - use few-shot prompting (in context learning)
    - add context from relevant chunks (augmentation)
    """

    # join relevant chunks in one context string
    chunks = [chunk["sentence_chunk"] for chunk in relevant_chunks]
    chunks = " -" + "\n -".join(chunks)

    # few-shot prompting
    base_prompt = """Based on the following context items, please answer the query.
     Don't return the thinking, only return the answer.
     Make sure your answers are as explanatory as possible.
     Use the following example as a reference for the ideal answer style.
     \nExample 1:
     Query: What is the role of backpropagation in neural networks?
     Answer: Backpropagation is a key algorithm used for training neural networks by minimizing the error between predicted and actual outputs. It involves a forward pass where the input data is propagated through the network to generate an output, and a backward pass where the error is propagated back through the network to update the weights. This is done using the gradient descent optimization method, which calculates the gradient of the loss function with respect to each weight and adjusts the weights to reduce the error. Backpropagation allows neural networks to learn complex patterns in data by iteratively improving the model's accuracy.
     \nNow use the following context items to answer the user query:
     {context}
     \nRelevant passages: <extract relevant passages from the context here>
     \nUser query: {query}
     Answer:"""

    # base_prompt = (
    #     "You are an assistant for question-answering tasks. "
    #     "Use the following pieces of retrieved context to answer "
    #     "the question. If you don't know the answer, say that you "
    #     "don't know. Use three sentences maximum and keep the "
    #     "answer concise."
    #     "\n\n"
    #     "{context}"
    #     "\nUser query: {query}"
    #     "Answer:"
    # )

    # Add relevant chunks
    base_prompt = base_prompt.format(context=chunks, query=query)
    # final prompt, suited for instruction-tuned models
    template = [{"role": "user", "content": base_prompt}]
    # add_generation_prompt argument tells the template to add tokens that indicate the start of a bot response
    prompt = tokenizer.apply_chat_template(conversation=template, tokenize=False, add_generation_prompt=True)

    return prompt


def augmented_generation(query, embedding_model, vector_store, data_index,
                         top_k, similarity_threshold, llm_model, tokenizer, temperature, max_new_tokens, device):
    # query your RAG to get relevant text
    scores, indices = rag_retrieve(query=query, embedding_model=embedding_model, vectore_store=vector_store,
                                   top_k=top_k)

    # only keep chunks with scores higher than the similarity threshold
    filtered_indices = [index for score, index in zip(scores, indices) if score > similarity_threshold]
    relevant_chunks = [data_index[i] for i in filtered_indices]

    # prepare the prompt
    prompt = prepare_augmented_prompt(query=query, relevant_chunks=relevant_chunks, tokenizer=tokenizer)

    # prompt the LLM
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    # for streaming the response
    response_streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Generate an output of tokens
    generation_kwargs = dict(**input_ids, streamer=response_streamer,
                             temperature=temperature,
                             do_sample=True,
                             max_new_tokens=max_new_tokens)

    # for streaming we run the generation in a different thread
    thread = Thread(target=llm_model.generate, kwargs=generation_kwargs)
    thread.start()

    # join retrieved resources in one text
    retrieved_resources = ""
    for i, source in enumerate(relevant_chunks):
        retrieved_resources += (f"Resource {i + 1}: \n"
                                f"File path: {source['file_path']} \n"
                                f"Page: {source['page_number']} \n"
                                f"Text: {source['sentence_chunk'][:200]} .... etc\n\n")
    if not retrieved_resources:
        retrieved_resources = "No resources found for your query!"
    return response_streamer, retrieved_resources


def rag_answer(query):
    # Clear GPU cache before generation
    torch.cuda.empty_cache()
    streamer, retrieved_resources = augmented_generation(query=query, embedding_model=embedding_model,
                                                         vector_store=embeddings, data_index=data_index,
                                                         top_k=NUM_OF_RELEVANT_CHUNKS,
                                                         similarity_threshold=SIMILARITY_THRESHOLD,
                                                         llm_model=llm_model,
                                                         tokenizer=tokenizer,
                                                         temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
                                                         device=DEVICE)
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text, retrieved_resources


if __name__ == "__main__":
    # load the vector-store
    embeddings, data_index = load_vector_store(VECTOR_STORE_PATH, DEVICE)

    # load the embedding model
    embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL,
                                          device=DEVICE)

    # load LLM locally
    tokenizer, llm_model = load_llm(model_id=LLM_MODEL_ID)

    # Launch the app
    demo = gradio_utils.gradio_rag_blocks(title="Chat With Your Data!",
                                          description="Ask your documents using my local " \
                                                      "Retrieval-Augmented Generation (RAG) pipeline.",
                                          submit_fun=rag_answer)
    demo.launch()

import os
import glob
import fitz
from tqdm import tqdm
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
import pandas as pd
from time import perf_counter as timer
import csv
import re
import random


""" The aim of this script is to create embeddings from your documents.
    We use an embedding model running locally on your GPU/CPU           """

# set the path to your data directory
DATA_DIR = 'data/'
# Add a sentencizer pipeline
NLP = English()
NLP.add_pipe("sentencizer")
# Define split size to turn groups of sentences into chunks
CHUNK_SIZE_IN_SENTENCES = 5
# define the min number of tokens in a chunk (the rest will be filtered)
MIN_TOKEN_LENGTH_PER_CHUNK = 30
# embedding model
EMBEDDING_MODEL = "all-mpnet-base-v2"
# device
DEVICE = "cuda"
# embedding output path
EMBEDDING_OUTPUT_PATH = "vector_store/embeddings.csv"

def get_sentences(txt):
    sentences = list(NLP(txt).sents)
    sentences = [str(sentence) for sentence in sentences]
    return sentences


def read_files(data_dir):
    # loop over your files
    print(f"Started processing files in directory: {data_dir}   ...")
    t1 = timer()
    extracted_data = []
    for file in glob.glob(os.path.join(data_dir, "*.pdf")):
        # open the doc
        document = fitz.open(file)
        # process
        # print("file path: " , file)
        for page_num, page in enumerate(document):
            # get the raw text of each page
            txt = page.get_text()
            # do some cleaning
            cleaned_text = txt.replace("\n", " ").strip()
            sentences = get_sentences(cleaned_text)
            entry = {"file_path": file,
                     "page_number": page_num,
                     "page_char_count": len(cleaned_text),
                     "page_word_count": len(cleaned_text.split(" ")),
                     "page_sentence_count": len(sentences),
                     "page_token_count": len(cleaned_text) / 4,
                     "text": cleaned_text,
                     "sentences": sentences}
            extracted_data.append(entry)
    t2 = timer()
    print(f"processing is finished, time needed: {t2 - t1:.5f} seconds")
    return extracted_data


def chunking(list_of_sentences, chunk_size):
    # We group sentences based on the chunk size (estimated in sentences)
    sentence_chunks = [list_of_sentences[i:i + chunk_size] for i in range(0, len(list_of_sentences), chunk_size)]
    return sentence_chunks


def convert_to_chunck_dict(text_dict):
    extracted_chunks = []
    for item in text_dict:
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["file_path"] = item["file_path"]
            chunk_dict["page_number"] = item["page_number"]
            # Join the sentences together into a paragraph-like structure
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
            chunk_dict["sentence_chunk"] = joined_sentence_chunk
            # Get stats about the chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token = ~4 characters
            extracted_chunks.append(chunk_dict)
    return extracted_chunks


def create_embeddings(chunks, embedding_model):
    # Embed each chunk one by one
    for item in tqdm(chunks):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])


def chunk_text(extracted_data):
    print("Chunking text ..")
    t1 = timer()
    for entry in extracted_data:
        entry["sentence_chunks"] = chunking(entry["sentences"], CHUNK_SIZE_IN_SENTENCES)
        entry["num_chunks"] = len(entry["sentence_chunks"])
    extracted_chunks = convert_to_chunck_dict(extracted_data)
    t2 = timer()
    print(f"Chunking is finished, time needed: {t2 - t1:.5f} seconds")
    return extracted_chunks


def create_embedding(chunks_df):
    # load embedding model
    print(f"Loading embedding model \"{EMBEDDING_MODEL}\" on {'GPU' if DEVICE=='cuda' else DEVICE}  ...")
    embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL, device=DEVICE)

    # start creating the embeddings
    print("Started creating the embeddings ...")
    t1 = timer()
    create_embeddings(chunks_df, embedding_model)
    t2 = timer()
    print(f"Creating the embeddings is finished, time needed: {t2 - t1:.5f} seconds")

    # save the embedding on disk
    embeddings_df = pd.DataFrame(chunks_df)
    os.makedirs(os.path.dirname(EMBEDDING_OUTPUT_PATH), exist_ok=True)
    embeddings_df.to_csv(EMBEDDING_OUTPUT_PATH, index=False, escapechar="\\")
    print(f"Embeddings have been saved on disk at:  {EMBEDDING_OUTPUT_PATH}")


if __name__ == "__main__":

    # load & process documents
    extracted_data = read_files(DATA_DIR)
    # chunking into groups of sentences
    extracted_chunks = chunk_text(extracted_data)

    # convert to dataframe & filter short chunks
    extracted_chunks_df = pd.DataFrame(extracted_chunks)
    extracted_chunks_df_filtered = extracted_chunks_df[extracted_chunks_df["chunk_token_count"]
                                                       > MIN_TOKEN_LENGTH_PER_CHUNK].to_dict(orient="records")
    # load embedding model & create embedding
    create_embedding(extracted_chunks_df_filtered)
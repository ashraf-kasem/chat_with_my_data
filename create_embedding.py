import os
import glob
import fitz
os.environ["TQDM_DISABLE"] = "1"
from tqdm import tqdm
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
import spacy
from spacy_cleaner import processing, Cleaner
from spacy_cleaner.processing import removers
import pandas as pd
from time import perf_counter as timer
import re

""" The aim of this script is to create embeddings from your documents.
    We use an embedding model running locally on your GPU/CPU           """

# set the path to your data directory
DATA_DIR = 'data/'
# Define split size to turn groups of sentences into chunks
CHUNK_SIZE_IN_SENTENCES = 6
# define the min number of tokens in a chunk (the rest will be filtered)
MIN_TOKEN_LENGTH_PER_CHUNK = 30
# embedding model
EMBEDDING_MODEL = "all-mpnet-base-v2"
# device
DEVICE = "cuda"
# embedding output path
EMBEDDING_OUTPUT_PATH = "vector_store/embeddings.csv"


# Add a sentencizer pipeline & cleaner
NLP = English()
NLP.add_pipe("sentencizer")
model = spacy.load("en_core_web_sm")
cleaner_pipeline = Cleaner(
    model,
    removers.remove_url_token,
    removers.remove_email_token)


def clean_text(text):
    cleaned_text = cleaner_pipeline.clean(text)
    return cleaned_text


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
            cleaned_text = clean_text([txt])[0]
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


def convert_to_chunk_dict(text_dict):
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


def chunk_text(data, chunk_size_in_sentences):
    print("Chunking text ..")
    t1 = timer()
    for entry in data:
        entry["sentence_chunks"] = chunking(entry["sentences"], chunk_size_in_sentences)
        entry["num_chunks"] = len(entry["sentence_chunks"])
    extracted_chunks = convert_to_chunk_dict(data)
    t2 = timer()
    print(f"Chunking is finished, time needed: {t2 - t1:.5f} seconds")
    return extracted_chunks


def create_embedding(chunks_df, embedding_model_name, device, embedding_output_path):
    # load embedding model
    print(f"Loading embedding model \"{embedding_model_name}\" on {'GPU' if device == 'cuda' else device}  ...")
    embedding_model = SentenceTransformer(model_name_or_path=embedding_model_name, device=device)

    # start creating the embeddings
    print("Started creating the embeddings ...")
    t1 = timer()
    create_embeddings(chunks_df, embedding_model)
    t2 = timer()
    print(f"Creating the embeddings is finished, time needed: {t2 - t1:.5f} seconds")

    # save the embedding on disk
    embeddings_df = pd.DataFrame(chunks_df)
    os.makedirs(os.path.dirname(embedding_output_path), exist_ok=True)
    embeddings_df.to_csv(embedding_output_path, index=False, escapechar="\\")
    print(f"Embeddings have been saved on disk at:  {embedding_output_path}")


def filter_chunks(chunks, min_token_length_per_chunk):
    chunks_df = pd.DataFrame(chunks)
    filtered_chunks = chunks_df[chunks_df["chunk_token_count"]
                                > min_token_length_per_chunk].to_dict(orient="records")
    return filtered_chunks


if __name__ == "__main__":
    # load & process documents
    extracted_data = read_files(DATA_DIR)

    # chunking into groups of sentences
    extracted_chunks = chunk_text(extracted_data, CHUNK_SIZE_IN_SENTENCES)

    # filter short chunks
    extracted_chunks_df_filtered = filter_chunks(extracted_chunks, MIN_TOKEN_LENGTH_PER_CHUNK)

    # load embedding model & create embedding & save on disk
    create_embedding(extracted_chunks_df_filtered, EMBEDDING_MODEL, DEVICE, EMBEDDING_OUTPUT_PATH)

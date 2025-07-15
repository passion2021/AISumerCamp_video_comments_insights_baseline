from sentence_transformers import SentenceTransformer
model = SentenceTransformer('./google-bert/bert-base-multilingual-cased',  local_files_only=True)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-zh")
model.save("./bge-small-zh")
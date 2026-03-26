from sentence_transformers import SentenceTransformer
import torch
import faiss

print("fine!")
print("CUDA:", torch.cuda.is_available())

model = SentenceTransformer("BAAI/bge-small-zh", device="cuda")
vec = model.encode("RAG system test")

print("vec length:", len(vec))
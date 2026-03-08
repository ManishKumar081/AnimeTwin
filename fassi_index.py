import faiss
import numpy as np

print("Loading embeddings...")

embeddings = np.load("embedding.npy").astype("float32")
ids = np.load("ids.npy").astype("int64")

dimension = embeddings.shape[1]

base_index = faiss.IndexFlatIP(dimension)

index = faiss.IndexIDMap(base_index)

index.add_with_ids(embeddings, ids)

print("Total vectors indexed:", index.ntotal)

faiss.write_index(index, "faiss_index.bin")

print("FAISS index saved.")
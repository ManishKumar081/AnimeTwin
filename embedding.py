import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from insightface.app import FaceAnalysis
import json

# ----------------------------
# Initialize InsightFace model
# ----------------------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

dataset_path = "Anime/facedataset"

all_embeddings = []
all_labels = []
all_image_paths = []
vector_ids = []

metadata = {}

print("Processing dataset with head expansion...")

# -----------------------------------
# LOOP THROUGH DATASET
# -----------------------------------
char_vectors = defaultdict(list)

current_id = 0

for label in sorted(os.listdir(dataset_path)):

    folder = os.path.join(dataset_path, label)

    if not os.path.isdir(folder):
        continue

    print(f"\nProcessing {label}")

    for img_name in tqdm(sorted(os.listdir(folder))):

        img_path = os.path.join(folder, img_name)

        try:

            img = np.array(Image.open(img_path).convert("RGB"))

            faces = app.get(img)

            if len(faces) == 0:
                continue

            face = faces[0]

            # -----------------------------------
            # EXPAND HEAD REGION
            # -----------------------------------
            x1, y1, x2, y2 = face.bbox.astype(int)

            h, w, _ = img.shape

            bw = x2 - x1
            bh = y2 - y1

            expand_x = int(0.5 * bw)
            expand_y = int(0.7 * bh)

            x1 = max(0, x1 - expand_x)
            y1 = max(0, y1 - expand_y)
            x2 = min(w, x2 + expand_x)
            y2 = min(h, y2 + int(0.3 * bh))

            head_crop = img[y1:y2, x1:x2]

            # -----------------------------------
            # TEXTURE EMBEDDING (HEAD REGION)
            # -----------------------------------
            head_faces = app.get(head_crop)

            if len(head_faces) > 0:
                texture_emb = head_faces[0].embedding
            else:
                texture_emb = face.embedding

            texture_emb = texture_emb / np.linalg.norm(texture_emb)

            # -----------------------------------
            # GEOMETRY EMBEDDING
            # -----------------------------------
            landmarks = face.landmark_2d_106

            center = np.mean(landmarks, axis=0)

            landmarks_norm = landmarks - center

            scale = np.max(np.linalg.norm(landmarks_norm, axis=1))

            landmarks_norm = landmarks_norm / scale

            geometry_emb = landmarks_norm.flatten()

            # -----------------------------------
            # HYBRID VECTOR
            # -----------------------------------
            hybrid_vector = np.concatenate([texture_emb, geometry_emb])

            hybrid_vector = hybrid_vector / np.linalg.norm(hybrid_vector)

            # SAVE EMBEDDING
            all_embeddings.append(hybrid_vector)
            all_labels.append(label)
            all_image_paths.append(img_path)

            char_vectors[label].append(hybrid_vector)

            vector_ids.append(current_id)

            metadata[current_id] = {
                "label": label,
                "image_path": img_path
            }

            current_id += 1

        except:
            continue


# -----------------------------------
# CONVERT TO NUMPY
# -----------------------------------
all_embeddings = np.vstack(all_embeddings).astype("float32")

np.save("embedding.npy", all_embeddings)
np.save("labels.npy", np.array(all_labels))
np.save("image_paths.npy", np.array(all_image_paths))
np.save("ids.npy", np.array(vector_ids))

with open("metadata.json", "w") as f:
    json.dump(metadata, f)

print("\nEmbeddings saved.")
print(len(all_embeddings))
print(len(all_labels))
print(len(all_image_paths))

# -----------------------------------
# BUILD CHARACTER CENTROIDS
# -----------------------------------
centroids = {}

for char in char_vectors:

    centroid = np.mean(char_vectors[char], axis=0)

    centroid = centroid / np.linalg.norm(centroid)

    centroids[char] = centroid

np.save("centroids.npy", centroids)

print("Character centroids saved.")
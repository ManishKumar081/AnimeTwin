import streamlit as st
import numpy as np
import faiss
import cv2
import math
import json
from PIL import Image
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Head-Aware Anime Face Finder",
    layout="wide"
)


# -----------------------------------
# LOAD MODEL
# -----------------------------------
@st.cache_resource
def load_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)
    return app


@st.cache_resource
def load_faiss():
    index = faiss.read_index("faiss_index.bin")
    return index


@st.cache_resource
def load_metadata():
    with open("metadata.json") as f:
        return json.load(f)


@st.cache_resource
def load_centroids():
    return np.load("centroids.npy", allow_pickle=True).item()


@st.cache_resource
def load_embeddings():
    emb = np.load("embedding.npy")
    lbl = np.load("labels.npy")
    return emb, lbl


face_app = load_model()
index = load_faiss()
metadata = load_metadata()
centroids = load_centroids()
embeddings, embed_labels = load_embeddings()


# -----------------------------------
# SIGMOID CALIBRATION FUNCTION
# -----------------------------------
def calibrated_similarity(raw_sim):

    MIN_SIM = 0.80
    MAX_SIM = 0.96

    s = (raw_sim - MIN_SIM) / (MAX_SIM - MIN_SIM)
    s = max(0, min(1, s))

    k = 9
    center = 0.60

    sigmoid = 1 / (1 + math.exp(-k * (s - center)))

    score = sigmoid * 75

    return score


# -----------------------------------
# TITLE
# -----------------------------------
st.title("AnimeTwin")

input_mode = st.radio("Image source", ["Camera", "Upload"], horizontal=True)

if input_mode == "Camera":
    uploaded_file = st.camera_input("Take a photo", width=300)
else:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    faces = face_app.get(img_np)

    if len(faces) == 0:
        st.error("No face detected!")
    else:

        face = faces[0]

        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w, _ = img_np.shape

        bw = x2 - x1
        bh = y2 - y1

        expand_x = int(0.5 * bw)
        expand_y = int(0.7 * bh)

        hx1 = max(0, x1 - expand_x)
        hy1 = max(0, y1 - expand_y)
        hx2 = min(w, x2 + expand_x)
        hy2 = min(h, y2 + int(0.3 * bh))

        head_crop = img_np[hy1:hy2, hx1:hx2]

        # -----------------------------------
        # TEXTURE EMBEDDING
        # -----------------------------------
        head_faces = face_app.get(head_crop)

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

        hybrid_vector = hybrid_vector.astype("float32").reshape(1, -1)

        # -----------------------------------
        # FAISS SEARCH
        # -----------------------------------
        k = 3
        distances, indices = index.search(hybrid_vector, k=k)

        # 🔹 IMPORTANT CHANGE HERE
        match_id = int(indices[0][0])
        match_data = metadata[str(match_id)]

        img_path = match_data["image_path"]
        best_character = match_data["label"]

        # -----------------------------------
        # CHARACTER SCORING (UNCHANGED)
        # -----------------------------------
        scores = []
        names = []

        for name, vec in centroids.items():

            sim = np.dot(hybrid_vector[0], vec) / (
                np.linalg.norm(hybrid_vector[0]) * np.linalg.norm(vec)
            )

            scores.append(sim)
            names.append(name)

        scores = np.array(scores)

        best_idx = np.argmax(scores)

        raw_similarity = scores[best_idx]
        similarity = calibrated_similarity(raw_similarity)

        # -----------------------------------
        # UI
        # -----------------------------------
        st.markdown("---")

        col1, col2 = st.columns([1,1])

        with col1:
            st.markdown("**Uploaded**")
            st.image(image, width=220)

        overlay = img_np.copy()

        cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(overlay,(hx1,hy1),(hx2,hy2),(255,0,0),2)

        for (lx, ly) in landmarks.astype(int):
            cv2.circle(overlay,(lx,ly),1,(241,235,156),-1)

        with col2:
            st.markdown("**Detection View**")
            st.image(overlay)

        st.markdown("---")

        # -----------------------------------
        # TOP MATCH
        # -----------------------------------
        st.markdown("### Top Match")

        st.image(img_path)

        st.markdown(f"**{best_character}**")

        st.progress(min(int(similarity),100))

        st.caption(f"Similarity: {similarity:.2f}% (sigmoid calibrated)")

        if similarity < 35:
            st.warning("Low confidence prediction.")

        # -----------------------------------
        # PCA VISUALIZATION
        # -----------------------------------
        st.markdown("---")
        st.markdown("### Character Embedding Views")

        pca = PCA(n_components=2)

        embeddings_2d = pca.fit_transform(embeddings)

        user_point = pca.transform(hybrid_vector)

        centroid_vectors = np.array(list(centroids.values()))
        centroid_names = list(centroids.keys())

        centroids_2d = pca.transform(centroid_vectors)

        unique_characters = np.unique(embed_labels)

        cols = st.columns(3)

        for i, character in enumerate(unique_characters):

            col = cols[i % 3]

            with col:

                mask = embed_labels == character

                fig, ax = plt.subplots(figsize=(4,4))

                ax.scatter(
                    embeddings_2d[mask,0],
                    embeddings_2d[mask,1],
                    alpha=0.6,
                    s=40
                )

                centroid_index = centroid_names.index(character)

                ax.scatter(
                    centroids_2d[centroid_index,0],
                    centroids_2d[centroid_index,1],
                    marker="X",
                    s=200,
                    color="black"
                )

                if character == best_character:

                    ax.scatter(
                        user_point[0,0],
                        user_point[0,1],
                        marker="*",
                        s=250,
                        color="red"
                    )

                ax.set_title(character)
                ax.set_xticks([])
                ax.set_yticks([])

                st.pyplot(fig)
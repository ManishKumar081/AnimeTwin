import numpy as np
import faiss
import cv2
import math
import json
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from insightface.app import FaceAnalysis
from sklearn.decomposition import PCA
from taipy.gui import Gui, State, notify


# -----------------------------------
# LOAD MODEL
# (replaces @st.cache_resource — loaded once at startup)
# -----------------------------------
def load_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)
    return app


def load_faiss():
    index = faiss.read_index("faiss_index.bin")
    return index


def load_metadata():
    with open("metadata.json") as f:
        return json.load(f)


def load_centroids():
    return np.load("centroids.npy", allow_pickle=True).item()


def load_embeddings():
    emb = np.load("embedding.npy")
    lbl = np.load("labels.npy")
    return emb, lbl


face_app  = load_model()
index     = load_faiss()
metadata  = load_metadata()
centroids = load_centroids()
embeddings, embed_labels = load_embeddings()


# -----------------------------------
# SIGMOID CALIBRATION FUNCTION
# (100% unchanged from original)
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
# HELPERS — numpy/PIL → base64
# Taipy <image> control needs base64 data URIs
# -----------------------------------
def pil_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    return "data:image/jpeg;base64," + encoded


def ndarray_to_b64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    return pil_to_b64(img)


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode()
    return "data:image/png;base64," + encoded


# -----------------------------------
# STATE VARIABLES
# Taipy requires all variables declared at module level
# -----------------------------------
uploaded_image_path = ""    # bound to file_selector

show_results  = False
show_no_face  = False
show_low_conf = False

# replaces: st.image(image), st.image(overlay), st.image(img_path)
img_uploaded_b64  = ""
img_detection_b64 = ""
img_match_b64     = ""

# replaces: st.markdown(f"**{best_character}**")
best_character  = ""

# replaces: st.progress(...) + st.caption(...)
similarity_value = 0.0
similarity_text  = ""

# replaces: st.pyplot(fig) inside loop — max 9 characters supported
pca_plot_0 = ""
pca_plot_1 = ""
pca_plot_2 = ""
pca_plot_3 = ""
pca_plot_4 = ""
pca_plot_5 = ""
pca_plot_6 = ""
pca_plot_7 = ""
pca_plot_8 = ""
show_pca   = [False] * 9


# -----------------------------------
# MAIN LOGIC FUNCTION
# Called when user uploads image OR clicks Run
# Replaces: entire "if uploaded_file:" block in Streamlit
# -----------------------------------
def on_image_upload(state: State):

    path = state.uploaded_image_path
    if not path:
        notify(state, "warning", "Please upload an image first.")
        return

    # replaces: image = Image.open(uploaded_file).convert("RGB")
    image  = Image.open(path).convert("RGB")
    img_np = np.array(image)

    # replaces: faces = face_app.get(img_np)
    faces = face_app.get(img_np)

    # replaces: if len(faces) == 0: st.error("No face detected!")
    if len(faces) == 0:
        state.show_no_face  = True
        state.show_results  = False
        notify(state, "error", "No face detected!")
        return

    state.show_no_face = False

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
    # TEXTURE EMBEDDING  (unchanged)
    # -----------------------------------
    head_faces = face_app.get(head_crop)

    if len(head_faces) > 0:
        texture_emb = head_faces[0].embedding
    else:
        texture_emb = face.embedding

    texture_emb = texture_emb / np.linalg.norm(texture_emb)

    # -----------------------------------
    # GEOMETRY EMBEDDING  (unchanged)
    # -----------------------------------
    landmarks = face.landmark_2d_106

    center = np.mean(landmarks, axis=0)

    landmarks_norm = landmarks - center

    scale = np.max(np.linalg.norm(landmarks_norm, axis=1))

    landmarks_norm = landmarks_norm / scale

    geometry_emb = landmarks_norm.flatten()

    # -----------------------------------
    # HYBRID VECTOR  (unchanged)
    # -----------------------------------
    hybrid_vector = np.concatenate([texture_emb, geometry_emb])

    hybrid_vector = hybrid_vector / np.linalg.norm(hybrid_vector)

    hybrid_vector = hybrid_vector.astype("float32").reshape(1, -1)

    # -----------------------------------
    # FAISS SEARCH  (unchanged)
    # -----------------------------------
    k = 3
    distances, indices = index.search(hybrid_vector, k=k)

    # 🔹 IMPORTANT CHANGE (kept exactly as original)
    match_id       = int(indices[0][0])
    match_data     = metadata[str(match_id)]
    img_path       = match_data["image_path"]
    best_character = match_data["label"]

    # -----------------------------------
    # CHARACTER SCORING  (unchanged)
    # -----------------------------------
    scores = []
    names  = []

    for name, vec in centroids.items():

        sim = np.dot(hybrid_vector[0], vec) / (
            np.linalg.norm(hybrid_vector[0]) * np.linalg.norm(vec)
        )

        scores.append(sim)
        names.append(name)

    scores = np.array(scores)

    best_idx       = np.argmax(scores)
    raw_similarity = scores[best_idx]
    similarity     = calibrated_similarity(raw_similarity)

    # -----------------------------------
    # DETECTION OVERLAY  (unchanged cv2 logic)
    # replaces: st.image(overlay)
    # -----------------------------------
    overlay = img_np.copy()

    cv2.rectangle(overlay, (x1, y1),   (x2, y2),   (0, 255, 0), 2)
    cv2.rectangle(overlay, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)

    for (lx, ly) in landmarks.astype(int):
        cv2.circle(overlay, (lx, ly), 1, (241, 235, 156), -1)

    # -----------------------------------
    # PCA VISUALIZATION  (unchanged logic)
    # replaces: st.pyplot(fig) inside loop
    # -----------------------------------
    pca = PCA(n_components=2)

    embeddings_2d = pca.fit_transform(embeddings)

    user_point = pca.transform(hybrid_vector)

    centroid_vectors = np.array(list(centroids.values()))
    centroid_names   = list(centroids.keys())

    centroids_2d = pca.transform(centroid_vectors)

    unique_characters = np.unique(embed_labels)

    pca_b64_list = []

    for i, character in enumerate(unique_characters):

        mask = embed_labels == character

        fig, ax = plt.subplots(figsize=(4, 4))

        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            alpha=0.6,
            s=40
        )

        centroid_index = centroid_names.index(character)

        ax.scatter(
            centroids_2d[centroid_index, 0],
            centroids_2d[centroid_index, 1],
            marker="X",
            s=200,
            color="black"
        )

        if character == best_character:

            ax.scatter(
                user_point[0, 0],
                user_point[0, 1],
                marker="*",
                s=250,
                color="red"
            )

        ax.set_title(character)
        ax.set_xticks([])
        ax.set_yticks([])

        pca_b64_list.append(fig_to_b64(fig))
        plt.close(fig)

    # -----------------------------------
    # PUSH ALL RESULTS TO TAIPY STATE
    # replaces: st.image / st.markdown / st.progress / st.caption / st.warning
    # -----------------------------------
    state.img_uploaded_b64  = pil_to_b64(image)
    state.img_detection_b64 = ndarray_to_b64(overlay)
    state.img_match_b64     = pil_to_b64(Image.open(img_path).convert("RGB"))

    state.best_character    = best_character

    # replaces: st.progress(min(int(similarity), 100))
    state.similarity_value  = round(min(similarity, 100), 2)

    # replaces: st.caption(f"Similarity: {similarity:.2f}% (sigmoid calibrated)")
    state.similarity_text   = f"Similarity: {similarity:.2f}% (sigmoid calibrated)"

    # replaces: if similarity < 35: st.warning("Low confidence prediction.")
    state.show_low_conf     = similarity < 35

    # Assign PCA plots into fixed state vars (Taipy needs named variables)
    pca_var_names = [
        "pca_plot_0","pca_plot_1","pca_plot_2",
        "pca_plot_3","pca_plot_4","pca_plot_5",
        "pca_plot_6","pca_plot_7","pca_plot_8",
    ]
    show_list = [False] * 9
    for i, b64 in enumerate(pca_b64_list):
        if i < 9:
            setattr(state, pca_var_names[i], b64)
            show_list[i] = True

    state.show_pca     = show_list
    state.show_results = True

    notify(state, "success", f"Match found: {best_character}!")


# -----------------------------------
# TAIPY PAGE DEFINITION
# Direct mapping of every st.* call to Taipy syntax:
#
#   st.title(...)              → # Title in markdown
#   st.file_uploader(...)      → <|file_selector|>
#   st.image(...)              → <|image|>  (base64)
#   st.columns([1,1])          → <|layout|columns=1 1|>
#   st.markdown("### ...")     → ### in markdown
#   st.progress(...)           → <|slider|active=False|>
#   st.caption(...)            → <|text|>
#   st.warning(...)            → <|text|render={show_low_conf}|>
#   st.error(...)              → <|text|render={show_no_face}|>
#   st.pyplot(fig)             → <|image|> with base64 PNG
# -----------------------------------
page = """
<|container|

# AnimeTwin

<|layout|columns=auto 1|gap=20px|

<|
**Image source**

<|{uploaded_image_path}|file_selector|label=Upload Image|extensions=.jpg,.jpeg,.png|drop_message=Drop photo here|on_action=on_image_upload|>
|>

<|
<br/>
<|Run Matching|button|on_action=on_image_upload|>
|>

|>

<|No face detected!|text|render={show_no_face}|class_name=error-text|>

<|{show_results}|part|render=True|

---

<|layout|columns=1 1|gap=20px|

<|
**Uploaded**

<|{img_uploaded_b64}|image|width=300px|>
|>

<|
**Detection View**

<|{img_detection_b64}|image|width=300px|>
|>

|>

---

### Top Match

<|{img_match_b64}|image|width=220px|>

**<|{best_character}|text|raw=True|>**

<|{similarity_value}|slider|min=0|max=100|active=False|width=400px|>

<|{similarity_text}|text|>

<|⚠️ Low confidence prediction.|text|render={show_low_conf}|class_name=warn-text|>



|>

|>
"""

css = """
<style>
.error-text .taipy-text { color: #e53935; font-weight: bold; }
.warn-text  .taipy-text { color: #f57c00; font-weight: bold; }
</style>
"""

full_page = css + page

# -----------------------------------
# RUN APP
# replaces: st.set_page_config(page_title="Head-Aware Anime Face Finder", layout="wide")
# -----------------------------------
if __name__ == "__main__":
    gui = Gui(page=full_page)
    gui.run(
        title="Head-Aware Anime Face Finder",
        dark_mode=False,
        port=5000,
        use_reloader=False
    )

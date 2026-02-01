from flask import Flask, request, jsonify
import torch
import numpy as np
from PIL import Image
import io
from facenet_pytorch import MTCNN, InceptionResnetV1
import json

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Face detector + embedding model (all open source)
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def get_embedding(image_bytes):
    """Return 1D face embedding list, or None if no face."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    face = mtcnn(img)

    if face is None:
      # No face detected
      return None

    with torch.no_grad():
        emb = model(face.unsqueeze(0)).cpu().numpy()[0]  # shape (512,)

    return emb.tolist()


@app.route("/register", methods=["POST"])
def register():
    """
    Input:
      - multipart/form-data with field 'image' (file)
    Output:
      - { success: true, embedding: [...] } or error
    """
    if "image" not in request.files:
        return jsonify({"success": False, "msg": "No image file sent"}), 400

    file = request.files["image"]
    emb = get_embedding(file.read())

    if emb is None:
        return jsonify({"success": False, "msg": "No face detected"}), 400

    return jsonify({"success": True, "embedding": emb})


@app.route("/verify", methods=["POST"])
def verify():
    """
    Input:
      - multipart/form-data
        - 'image': file
        - 'stored_embedding': JSON string of [512 floats]
    Output:
      - { match: true/false, similarity: float }
    """
    if "image" not in request.files or "stored_embedding" not in request.form:
        return jsonify({"match": False, "msg": "Image or embedding missing"}), 400

    file = request.files["image"]
    stored_list = json.loads(request.form["stored_embedding"])

    current_emb = get_embedding(file.read())
    if current_emb is None:
        return jsonify({"match": False, "msg": "No face detected"}), 400

    stored_vec = np.array(stored_list, dtype="float32")
    current_vec = np.array(current_emb, dtype="float32")

    # Cosine similarity
    num = np.dot(stored_vec, current_vec)
    den = np.linalg.norm(stored_vec) * np.linalg.norm(current_vec)
    similarity = float(num / den)

    # Threshold: you can tweak (0.7 ~ 0.8 typical)
    is_match = similarity > 0.7

    return jsonify({"match": bool(is_match), "similarity": similarity})


if __name__ == "__main__":
    # Runs on http://localhost:6000
    app.run(host="0.0.0.0", port=6000)

from flask import Flask, request, jsonify
import torch
import numpy as np
from PIL import Image
import io
from facenet_pytorch import MTCNN, InceptionResnetV1
import json

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Face detector + embedding model
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "face-service running"})


def get_embedding(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    face = mtcnn(img)

    if face is None:
        return None

    with torch.no_grad():
        emb = model(face.unsqueeze(0)).cpu().numpy()[0]

    return emb.tolist()


@app.route("/register", methods=["POST"])
def register():
    if "image" not in request.files:
        return jsonify({"success": False, "msg": "No image file sent"}), 400

    emb = get_embedding(request.files["image"].read())
    if emb is None:
        return jsonify({"success": False, "msg": "No face detected"}), 400

    return jsonify({"success": True, "embedding": emb})


@app.route("/verify", methods=["POST"])
def verify():
    if "image" not in request.files or "stored_embedding" not in request.form:
        return jsonify({"match": False, "msg": "Image or embedding missing"}), 400

    stored_vec = np.array(json.loads(request.form["stored_embedding"]), dtype="float32")
    current_emb = get_embedding(request.files["image"].read())

    if current_emb is None:
        return jsonify({"match": False, "msg": "No face detected"}), 400

    current_vec = np.array(current_emb, dtype="float32")

    similarity = float(
        np.dot(stored_vec, current_vec)
        / (np.linalg.norm(stored_vec) * np.linalg.norm(current_vec))
    )

    return jsonify({
        "match": similarity > 0.7,
        "similarity": similarity
    })


if __name__ == "__main__":
    app.run(debug=True)

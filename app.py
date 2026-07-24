import os
import io
import json
from datetime import datetime
from flask import Flask, render_template, request
from PIL import Image
import numpy as np

import cloudinary
import cloudinary.uploader
from supabase import create_client, Client

app = Flask(__name__)
app.secret_key = "smart_textile_design_finder_secret_key"

# Configurations
SUPABASE_URL = os.getenv("SUPABASE_URL", "YOUR_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "YOUR_SUPABASE_KEY")

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "YOUR_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "YOUR_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "YOUR_API_SECRET")

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

def extract_simple_features(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((16, 16))
        arr = np.array(image, dtype=np.float32).flatten()
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()
    except Exception as e:
        print(f"Feature Extraction Error: {e}")
        return None

def cosine_similarity(vec1, vec2):
    try:
        a = np.array(vec1, dtype=np.float32)
        b = np.array(vec2, dtype=np.float32)
        if len(a) != len(b):
            return 0.0
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))
    except Exception:
        return 0.0

@app.route("/", methods=["GET", "POST"])
def index():
    message = None
    similar_images = []

    if request.method == "POST":
        action = request.form.get("action")

        cam_file = request.files.get("camera_image")
        gal_file = request.files.get("gallery_image")
        
        selected_file = None
        if cam_file and cam_file.filename != '':
            selected_file = cam_file
        elif gal_file and gal_file.filename != '':
            selected_file = gal_file

        # --- FIND SIMILAR DESIGN ---
        if action == "search":
            if not selected_file:
                message = "Kripya search karne ke liye camera ya gallery se photo select karein!"
            else:
                try:
                    img_bytes = selected_file.read()
                    query_features = extract_simple_features(img_bytes)

                    if query_features is None:
                        message = "Image read nahi ho saki, kripya doosri photo try karein."
                    else:
                        response = supabase.table("designs").select("*").execute()
                        db_records = response.data if response and hasattr(response, 'data') else []

                        results = []
                        for record in db_records:
                            raw_emb = record.get("embedding")
                            if raw_emb:
                                if isinstance(raw_emb, str):
                                    try:
                                        db_vec = json.loads(raw_emb)
                                    except Exception:
                                        db_vec = []
                                else:
                                    db_vec = raw_emb

                                if isinstance(db_vec, list) and len(db_vec) > 0:
                                    score = cosine_similarity(query_features, db_vec)
                                    match_percentage = round(score * 100, 2)
                                    record['score'] = match_percentage
                                    results.append(record)

                        # Match Percentage ke mutabiq Sort karein
                        results = sorted(results, key=lambda x: x['score'], reverse=True)
                        similar_images = results[:12]

                        if not similar_images:
                            message = "Database me koi bhi design feature nahi mila. Pehle design add karein!"
                        else:
                            message = f"{len(similar_images)} designs me se best match results neeche dikh rahe hain!"

                except Exception as e:
                    message = f"Search Error: {str(e)}"

        # --- ADD TO DATABASE ---
        elif action == "add":
            if not selected_file:
                message = "Database me add karne ke liye photo zaroori hai!"
            else:
                try:
                    img_bytes = selected_file.read()
                    features = extract_simple_features(img_bytes)

                    upload_result = cloudinary.uploader.upload(img_bytes, folder="design_finder_db")
                    image_url = upload_result.get("secure_url")

                    custom_design_id = request.form.get("design_id", "").strip()
                    if not custom_design_id:
                        custom_design_id = f"DES-{datetime.now().strftime('%Y%m%d%H%M%S')}"

                    db_payload = {
                        "design_id": custom_design_id,
                        "image_url": image_url,
                        "fabric": request.form.get("fabric", "").strip(),
                        "work_type": request.form.get("work_type", "").strip(),
                        "stitch": request.form.get("stitch", "").strip(),
                        "color": request.form.get("color", "").strip(),
                        "occasion": request.form.get("occasion", "").strip(),
                        "notes": request.form.get("notes", "").strip(),
                        "embedding": json.dumps(features),
                        "created_at": datetime.utcnow().isoformat()
                    }

                    supabase.table("designs").insert(db_payload).execute()
                    message = f"Design '{custom_design_id}' safaltapoorvak save ho gaya!"

                except Exception as e:
                    message = f"Save Error: {str(e)}"

    return render_template("index.html", message=message, similar_images=similar_images)

@app.route("/gallery")
def gallery():
    try:
        response = supabase.table("designs").select("*").order("created_at", desc=True).execute()
        designs = response.data if response and hasattr(response, 'data') else []
    except Exception:
        designs = []
    return render_template("gallery.html", designs=designs)

@app.route("/dashboard")
def dashboard():
    try:
        response = supabase.table("designs").select("*").execute()
        designs = response.data if response and hasattr(response, 'data') else []
        total_count = len(designs)
    except Exception:
        designs = []
        total_count = 0
    return render_template("dashboard.html", total_count=total_count, designs=designs)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

import os
import io
import json
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from PIL import Image

# ML / Feature Extraction Libraries (e.g. PyTorch / torchvision)
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# Database / Cloud Storage Integrations (Supabase & Cloudinary)
import cloudinary
import cloudinary.uploader
from supabase import create_client, Client

app = Flask(__name__)
app.secret_key = "smart_textile_design_finder_secret_key"

# ---------------------------------------------------------
# 1. Configuration (Cloudinary & Supabase Setup)
# ---------------------------------------------------------
# Apne Environment Variables ya Direct Keys yahan verify karein
SUPABASE_URL = os.getenv("SUPABASE_URL", "YOUR_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "YOUR_SUPABASE_KEY")

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "YOUR_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "YOUR_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "YOUR_API_SECRET")

# Cloudinary Init
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

# Supabase Client Init
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# ---------------------------------------------------------
# 2. AI Model for Feature Extraction (ResNet18)
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.ResNet18_Weights.DEFAULT
resnet_model = models.resnet18(weights=weights)
# Remove the final classification layer to get feature vector (512-dim)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model.eval()
resnet_model.to(device)

# Image Transformations
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image_bytes):
    """Image bytes se feature vector (embedding) extract karta hai"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor = img_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet_model(tensor).squeeze().cpu().numpy()
        # Normalize vector for cosine similarity
        norm_embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        return norm_embedding.tolist()
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

def cosine_similarity(v1, v2):
    """Do embeddings ke beech cosine similarity calculate karta hai"""
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# ---------------------------------------------------------
# 3. Main Route (Index Page) - Handles Search & Add
# ---------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    message = None
    similar_images = []

    if request.method == "POST":
        action = request.form.get("action")

        # Camera ya Gallery dono inputs me se image retrieve karein
        cam_file = request.files.get("camera_image")
        gal_file = request.files.get("gallery_image")
        
        selected_file = None
        if cam_file and cam_file.filename != '':
            selected_file = cam_file
        elif gal_file and gal_file.filename != '':
            selected_file = gal_file

        # =========================================================
        # SEARCH SIMILAR DESIGN LOGIC
        # =========================================================
        if action == "search":
            if not selected_file:
                message = "Kripya search karne ke liye pehle camera ya gallery se image select karein!"
            else:
                try:
                    img_bytes = selected_file.read()
                    query_embedding = extract_features(img_bytes)

                    if query_embedding is None:
                        message = "Image process nahi ho saki. Kripya doosri image try karein."
                    else:
                        # Supabase se saare designs fetch karein
                        response = supabase.table("designs").select("*").execute()
                        db_records = response.data if response and hasattr(response, 'data') else []

                        results = []
                        for record in db_records:
                            # Verify if embedding exists in DB record
                            raw_emb = record.get("embedding")
                            if raw_emb:
                                if isinstance(raw_emb, str):
                                    db_emb = json.loads(raw_emb)
                                else:
                                    db_emb = raw_emb
                                
                                # Cosine similarity percentage
                                score = cosine_similarity(query_embedding, db_emb)
                                match_percentage = round(score * 100, 2)

                                # Score threshold (e.g. 40% se zyada matching wale dikhao)
                                if match_percentage >= 40.0:
                                    record['score'] = match_percentage
                                    results.append(record)

                        # Sort results by similarity score (Highest match first)
                        results = sorted(results, key=lambda x: x['score'], reverse=True)
                        similar_images = results[:12] # Top 12 matches

                        if not similar_images:
                            message = "Koyi milta-julta design nahi mila."
                        else:
                            message = f"{len(similar_images)} matching designs mile!"

                except Exception as e:
                    print(f"Search Execution Error: {e}")
                    message = f"Search karne me error aaya: {str(e)}"

        # =========================================================
        # ADD TO DATABASE LOGIC
        # =========================================================
        elif action == "add":
            if not selected_file:
                message = "Database me add karne ke liye photo select/capture karna zaroori hai!"
            else:
                try:
                    img_bytes = selected_file.read()
                    
                    # 1. Extract Embeddings
                    embedding = extract_features(img_bytes)

                    # 2. Upload Image to Cloudinary
                    upload_result = cloudinary.uploader.upload(
                        img_bytes, 
                        folder="design_finder_db"
                    )
                    image_url = upload_result.get("secure_url")

                    # 3. Collect Form Data
                    # Custom Design ID (e.g., "123 shubham")
                    custom_design_id = request.form.get("design_id", "").strip()
                    if not custom_design_id:
                        custom_design_id = f"DES-{datetime.now().strftime('%Y%m%d%H%M%S')}"

                    fabric = request.form.get("fabric", "").strip()
                    work_type = request.form.get("work_type", "").strip()
                    
                    # Stitch Count (e.g. 100000)
                    stitch_input = request.form.get("stitch", "").strip()
                    stitch_count = int(stitch_input) if stitch_input.isdigit() else stitch_input

                    color = request.form.get("color", "").strip()
                    occasion = request.form.get("occasion", "").strip()
                    notes = request.form.get("notes", "").strip()

                    # 4. Save Record in Supabase
                    db_payload = {
                        "design_id": custom_design_id,
                        "image_url": image_url,
                        "fabric": fabric,
                        "work_type": work_type,
                        "stitch": str(stitch_count),
                        "color": color,
                        "occasion": occasion,
                        "notes": notes,
                        "embedding": json.dumps(embedding),
                        "created_at": datetime.utcnow().isoformat()
                    }

                    supabase.table("designs").insert(db_payload).execute()
                    message = f"Design '{custom_design_id}' safaltapoorvak database me add ho gaya hai!"

                except Exception as e:
                    print(f"Database Insert Error: {e}")
                    message = f"Database me save nahi ho saka: {str(e)}"

    return render_template("index.html", message=message, similar_images=similar_images)

# ---------------------------------------------------------
# 4. Other Routes (Gallery, Dashboard, Design View)
# ---------------------------------------------------------
@app.route("/gallery")
def gallery():
    try:
        response = supabase.table("designs").select("*").order("created_at", desc=True).execute()
        designs = response.data if response and hasattr(response, 'data') else []
    except Exception as e:
        designs = []
    return render_template("gallery.html", designs=designs)

@app.route("/dashboard")
def dashboard():
    try:
        response = supabase.table("designs").select("*").execute()
        designs = response.data if response and hasattr(response, 'data') else []
        total_count = len(designs)
    except Exception as e:
        designs = []
        total_count = 0
    return render_template("dashboard.html", total_count=total_count, designs=designs)

@app.route("/design/<filename_or_id>")
def design_detail(filename_or_id):
    try:
        response = supabase.table("designs").select("*").eq("design_id", filename_or_id).execute()
        data = response.data if response and hasattr(response, 'data') else []
        design = data[0] if data else None
    except Exception as e:
        design = None
    return render_template("detail.html", design=design)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

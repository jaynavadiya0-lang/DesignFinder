import os
import io
import json
from datetime import datetime
from flask import Flask, render_template, request
from PIL import Image
import imagehash

import cloudinary
import cloudinary.uploader
from supabase import create_client, Client

app = Flask(__name__)
app.secret_key = "smart_textile_design_finder_secret_key"

# --- Configurations ---
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

def get_image_hash(image_bytes):
    """Calculates perceptual hash for fast image similarity comparison"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        phash = imagehash.phash(image)
        return str(phash)
    except Exception as e:
        print(f"Hash Generation Error: {e}")
        return None

def calculate_similarity(hash1, hash2):
    """Calculates percentage match between two image hashes safely"""
    try:
        # String cleanup (agar purana data string formatted na ho)
        str_h1 = str(hash1).strip()
        str_h2 = str(hash2).strip()

        # Check for valid hex string length for phash
        if len(str_h1) != 16 or len(str_h2) != 16:
            return 0.0

        h1 = imagehash.hex_to_hash(str_h1)
        h2 = imagehash.hex_to_hash(str_h2)
        difference = h1 - h2  # Hamming distance
        match_percentage = max(0, 100 - (difference * 1.5625))
        return round(match_percentage, 2)
    except Exception as e:
        print(f"Match calculation error: {e}")
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

        # --- FIND SIMILAR DESIGN LOGIC ---
        if action == "search":
            if not selected_file:
                message = "Kripya search karne ke liye pehle image select karein!"
            else:
                try:
                    img_bytes = selected_file.read()
                    query_hash = get_image_hash(img_bytes)

                    if not query_hash:
                        message = "Image process nahi ho saki. Kripya dusri image try karein."
                    else:
                        response = supabase.table("designs").select("*").execute()
                        db_records = response.data if response and hasattr(response, 'data') else []

                        results = []
                        for record in db_records:
                            db_hash = record.get("embedding")
                            if db_hash:
                                score = calculate_similarity(query_hash, db_hash)
                                # Unko hi filter karein jinme thodi bhi similarity ho
                                if score >= 10.0:
                                    record['score'] = score
                                    results.append(record)

                        # Match score ke mutabiq order karein
                        results = sorted(results, key=lambda x: x['score'], reverse=True)
                        similar_images = results[:12]

                        if not similar_images:
                            message = "Koyi milta-julta design nahi mila."
                        else:
                            message = f"{len(similar_images)} matching designs dekhiye!"

                except Exception as e:
                    print(f"Search Route Error: {e}")
                    message = f"Search Error: {str(e)}"

        # --- ADD TO DATABASE LOGIC ---
        elif action == "add":
            if not selected_file:
                message = "Database me add karne ke liye photo select/capture karein!"
            else:
                try:
                    img_bytes = selected_file.read()
                    img_hash = get_image_hash(img_bytes)

                    # Upload to Cloudinary
                    upload_result = cloudinary.uploader.upload(img_bytes, folder="design_finder_db")
                    image_url = upload_result.get("secure_url")

                    # Custom Design ID (Name + Number support)
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
                        "embedding": img_hash,
                        "created_at": datetime.utcnow().isoformat()
                    }

                    supabase.table("designs").insert(db_payload).execute()
                    message = f"Design '{custom_design_id}' safaltapoorvak save ho gaya!"

                except Exception as e:
                    print(f"Database Save Error: {e}")
                    message = f"Database me save nahi ho saka: {str(e)}"

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

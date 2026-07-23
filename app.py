from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from PIL import Image
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import shutil
from datetime import datetime
import requests
import cloudinary
import cloudinary.uploader

from supabase import create_client

app = Flask(__name__)

# -------------------------------------------------
# CONFIGURATIONS & ENVIRONMENT VARIABLES
# -------------------------------------------------
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME", "").strip(),
    api_key=os.environ.get("CLOUDINARY_API_KEY", "").strip(),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET", "").strip(),
    secure=True
)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_KEY:
    print("WARNING: SUPABASE credentials missing in environment variables!")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

UPLOAD_FOLDER = "uploads"
DATABASE_FOLDER = "design_database"
TEMPLATES_FOLDER = "templates"
STATIC_FOLDER = "static"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)
os.makedirs(TEMPLATES_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# -------------------------------------------------
# HIGH-ACCURACY DESIGN FEATURE DETECTOR
# -------------------------------------------------
orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

FABRIC_OPTIONS = [
    "Viscose", "Cotton", "Georgette", "Chiffon", "Silk",
    "Linen", "Rayon", "Crepe", "Organza", "Net"
]

WORK_TYPE_OPTIONS = [
    "Floral", "Butta", "Jaal", "Border", "Pallu",
    "Allover", "Embroidery", "Printed", "Weaving", "Traditional"
]

STITCH_OPTIONS = ["Unstitched", "Semi-Stitched", "Fully Stitched", "Dress Material", "Saree", "Lehenga"]

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def save_stream_compressed(file_storage, dest_path, max_dim=600):
    img = Image.open(file_storage.stream)
    img = img.convert("RGB")
    
    w, h = img.size
    if max(w, h) > max_dim:
        if w > h:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        else:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    img.save(dest_path, "JPEG", quality=85, optimize=True)

def get_unique_filename(original_name):
    base_name = secure_filename(original_name) or "image.jpg"
    name, ext = os.path.splitext(base_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{timestamp}.jpg"

def get_uploaded_file_from_request():
    possible_keys = ["image", "camera_image", "gallery_image"]
    for key in possible_keys:
        f = request.files.get(key)
        if f and f.filename:
            return f
    return None

def process_single_dropdown_field(field_name):
    """Processes dynamic single-dropdown value with optional custom entry."""
    selected_val = request.form.get(f"{field_name}_select", "").strip()
    custom_val = request.form.get(f"{field_name}_custom", "").strip()
    
    if selected_val == "Other":
        return custom_val if custom_val else "Other"
    return selected_val if selected_val else custom_val

# -------------------------------------------------
# MATCHING ENGINE
# -------------------------------------------------
def extract_advanced_features(img):
    h, w = img.shape[:2]
    if max(h, w) > 300:
        scale = 300 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    kp, des = orb.detectAndCompute(gray_enhanced, None)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist_hsv, hist_hsv)

    edges = cv2.Canny(gray_enhanced, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    return {"kp": kp, "des": des, "hist": hist_hsv, "edge_density": edge_density}

def get_image_match_score(upload_path, db_path):
    try:
        img1 = cv2.imread(upload_path)
        img2 = cv2.imread(db_path)

        if img1 is None or img2 is None:
            return 0

        f1 = extract_advanced_features(img1)
        f2 = extract_advanced_features(img2)

        feat_score = 0
        if f1["des"] is not None and f2["des"] is not None and len(f1["des"]) >= 4 and len(f2["des"]) >= 4:
            try:
                matches = bf_orb.knnMatch(f1["des"], f2["des"], k=2)
                good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
                base_kp = min(len(f1["kp"]), len(f2["kp"]))
                feat_score = (len(good) / base_kp) * 100 if base_kp > 0 else 0
            except Exception:
                feat_score = 0

        try:
            color_corr = cv2.compareHist(f1["hist"], f2["hist"], cv2.HISTCMP_CORREL)
            color_score = max(0, min(100, ((color_corr + 1) / 2) * 100))
        except Exception:
            color_score = 0

        try:
            density_diff = abs(f1["edge_density"] - f2["edge_density"])
            texture_score = max(0, 100 - (density_diff * 400))
        except Exception:
            texture_score = 0

        composite_score = (feat_score * 0.50) + (color_score * 0.30) + (texture_score * 0.20)
        final_score = min(100, round(composite_score * 1.5, 1))
        return final_score if final_score >= 12 else 0
        
    except Exception as e:
        print("Matching Error:", e)
        return 0

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    message = None
    similar_images = []
    image_name = None

    if request.method == "POST":
        action = request.form.get("action", "").strip()
        file = get_uploaded_file_from_request()

        if not file or not file.filename:
            message = "⚠️ Please capture or select an image first."
        else:
            original_filename = secure_filename(file.filename) or "image.jpg"
            saved_upload_name = get_unique_filename(original_filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], saved_upload_name)

            try:
                save_stream_compressed(file, filepath, max_dim=600)
                image_name = saved_upload_name

                if action == "add":
                    fabric = process_single_dropdown_field("fabric")
                    work_type = process_single_dropdown_field("work_type")
                    stitch = request.form.get("stitch", "").strip()
                    manual_design_id = request.form.get("design_id", "").strip()
                    color = request.form.get("color", "").strip()
                    occasion = request.form.get("occasion", "").strip()
                    notes = request.form.get("notes", "").strip()

                    # Auto Generate Design ID if left blank
                    design_id = manual_design_id or f"DES-{datetime.now().strftime('%Y%m%d%H%M%S')}"

                    upload_result = cloudinary.uploader.upload(
                        filepath,
                        folder="DesignFinder",
                        quality="85"
                    )
                    
                    db_local_path = os.path.join(DATABASE_FOLDER, saved_upload_name)
                    shutil.copy(filepath, db_local_path)

                    supabase.table("designs").insert({
                        "design_id": design_id,
                        "filename": saved_upload_name,
                        "image_url": upload_result.get("secure_url", ""),
                        "public_id": upload_result.get("public_id", ""),
                        "fabric": fabric,
                        "work_type": work_type,
                        "stitch": stitch,
                        "color": color,
                        "occasion": occasion,
                        "notes": notes
                    }).execute()

                    message = f"✅ Design '{design_id}' registered successfully in Database!"

                elif action == "search":
                    all_meta_res = supabase.table("designs").select("*").execute()
                    meta_list = all_meta_res.data or []
                    
                    # Auto restore missing files from Cloudinary
                    for meta in meta_list:
                        fn = meta.get("filename")
                        img_url = meta.get("image_url")
                        if fn and img_url:
                            local_path = os.path.join(DATABASE_FOLDER, fn)
                            if not os.path.exists(local_path):
                                try:
                                    img_data = requests.get(img_url, timeout=5).content
                                    with open(local_path, "wb") as handler:
                                        handler.write(img_data)
                                except Exception as e:
                                    print(f"Error restoring image {fn}:", e)

                    if os.path.exists(DATABASE_FOLDER):
                        valid_files = [f for f in os.listdir(DATABASE_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png'))]
                        meta_dict = {item["filename"]: item for item in meta_list}
                        
                        for db_file in valid_files:
                            db_path = os.path.join(DATABASE_FOLDER, db_file)
                            if os.path.isfile(db_path):
                                score = get_image_match_score(filepath, db_path)
                                if score > 12:
                                    meta = meta_dict.get(db_file, {})
                                    similar_images.append({
                                        "filename": db_file,
                                        "design_id": meta.get("design_id", "N/A"),
                                        "score": score,
                                        "fabric": meta.get("fabric", "N/A"),
                                        "work_type": meta.get("work_type", "N/A"),
                                        "stitch": meta.get("stitch", "N/A"),
                                        "color": meta.get("color", "N/A"),
                                        "occasion": meta.get("occasion", "N/A"),
                                        "notes": meta.get("notes", ""),
                                        "image_url": meta.get("image_url", "")
                                    })
                        similar_images.sort(key=lambda x: x["score"], reverse=True)
                        message = f"🔍 Search Completed: Found {len(similar_images)} matching designs."

            except Exception as e:
                message = f"❌ Error: {str(e)}"

    return render_template(
        "index.html",
        image_name=image_name,
        similar_images=similar_images,
        message=message,
        fabric_options=FABRIC_OPTIONS,
        work_type_options=WORK_TYPE_OPTIONS,
        stitch_options=STITCH_OPTIONS
    )

@app.route("/gallery")
def gallery():
    try:
        response = supabase.table("designs").select("*").order("id", desc=True).execute()
        designs = response.data if response.data else []
    except Exception:
        designs = []
    return render_template("gallery.html", designs=designs, fabric_options=FABRIC_OPTIONS, work_type_options=WORK_TYPE_OPTIONS)

@app.route("/dashboard")
def dashboard():
    try:
        response = supabase.table("designs").select("*").order("id", desc=True).execute()
        designs = response.data if response.data else []
    except Exception:
        designs = []

    return render_template(
        "dashboard.html",
        total_designs=len(designs),
        total_fabric=len(set(d["fabric"] for d in designs if d.get("fabric"))),
        total_work_type=len(set(d["work_type"] for d in designs if d.get("work_type"))),
        recent_designs=designs[:10]
    )

@app.route("/compare", methods=["GET", "POST"])
def compare():
    try:
        res = supabase.table("designs").select("*").execute()
        designs = res.data or []
    except Exception:
        designs = []

    result, design1, design2 = None, None, None

    if request.method == "POST":
        d1_fn = request.form.get("design1")
        d2_fn = request.form.get("design2")

        p1 = os.path.join(DATABASE_FOLDER, d1_fn) if d1_fn else ""
        p2 = os.path.join(DATABASE_FOLDER, d2_fn) if d2_fn else ""

        if os.path.exists(p1) and os.path.exists(p2):
            result = get_image_match_score(p1, p2)

        design1 = next((d for d in designs if d["filename"] == d1_fn), None)
        design2 = next((d for d in designs if d["filename"] == d2_fn), None)

    return render_template("compare.html", designs=designs, result=result, design1=design1, design2=design2)

@app.route("/design/<filename>")
def design_detail(filename):
    try:
        res = supabase.table("designs").select("*").eq("filename", filename).execute()
        design = res.data[0] if res.data else None
    except Exception:
        design = None

    if not design:
        return redirect(url_for("gallery"))

    return render_template("design_detail.html", design=design)

@app.route("/edit/<filename>", methods=["GET", "POST"])
def edit_design(filename):
    try:
        res = supabase.table("designs").select("*").eq("filename", filename).execute()
        details = res.data[0] if res.data else {}
    except Exception:
        details = {}

    if request.method == "POST":
        fabric = process_single_dropdown_field("fabric")
        work_type = process_single_dropdown_field("work_type")
        stitch = request.form.get("stitch", "").strip()
        color = request.form.get("color", "").strip()
        occasion = request.form.get("occasion", "").strip()
        notes = request.form.get("notes", "").strip()

        supabase.table("designs").update({
            "fabric": fabric,
            "work_type": work_type,
            "stitch": stitch,
            "color": color,
            "occasion": occasion,
            "notes": notes
        }).eq("filename", filename).execute()

        return redirect(url_for("gallery"))

    return render_template(
        "edit_design.html",
        filename=filename,
        details=details,
        fabric_options=FABRIC_OPTIONS,
        work_type_options=WORK_TYPE_OPTIONS,
        stitch_options=STITCH_OPTIONS
    )

@app.route("/delete/<filename>")
def delete_design(filename):
    try:
        res = supabase.table("designs").select("public_id").eq("filename", filename).execute()
        if res.data and res.data[0].get("public_id"):
            cloudinary.uploader.destroy(res.data[0]["public_id"])
        
        supabase.table("designs").delete().eq("filename", filename).execute()
    except Exception as e:
        print("Delete error:", e)

    file_path = os.path.join(DATABASE_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    return redirect(url_for("gallery"))

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/database/<filename>")
def database_file(filename):
    return send_from_directory(DATABASE_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)

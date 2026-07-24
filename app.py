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
# Increased keypoints to capture detailed embroidery/prints
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

# -------------------------------------------------
# HELPERS & COMPRESSION
# -------------------------------------------------
def save_stream_compressed(file_storage, dest_path, max_dim=600):
    """Resizes and normalizes images for high-accuracy feature extraction."""
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

def get_final_fabric_from_form():
    fabric_select = request.form.get("fabric_select", "").strip()
    fabric_custom = request.form.get("fabric_custom", "").strip()
    return fabric_custom if fabric_select == "Other" else (fabric_select or fabric_custom)

def get_final_work_type_from_form():
    work_type_select = request.form.get("work_type_select", "").strip()
    work_type_custom = request.form.get("work_type_custom", "").strip()
    return work_type_custom if work_type_select == "Other" else (work_type_select or work_type_custom)

def get_select_and_custom(existing_value, options):
    if not existing_value:
        return "", ""
    if existing_value in options:
        return existing_value, ""
    return "Other", existing_value

# -------------------------------------------------
# ADVANCED MULTI-FEATURE MATCHING ENGINE
# -------------------------------------------------
def extract_advanced_features(img):
    """Extracts Scale-Invariant ORB Keypoints, Multi-channel HSV Histograms, & Texture Canny Gradients."""
    h, w = img.shape[:2]
    if max(h, w) > 300:
        scale = 300 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 1. ORB Pattern Keypoints
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better lighting resilience
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    kp, des = orb.detectAndCompute(gray_enhanced, None)
    
    # 2. Multi-channel HSV Color Analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist_hsv, hist_hsv)

    # 3. Structural Texture Density (Canny Edges)
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

        # A. Pattern Feature Match (50% Weight)
        feat_score = 0
        if f1["des"] is not None and f2["des"] is not None and len(f1["des"]) >= 4 and len(f2["des"]) >= 4:
            try:
                matches = bf_orb.knnMatch(f1["des"], f2["des"], k=2)
                good = []
                for m_pair in matches:
                    if len(m_pair) == 2:
                        m, n = m_pair
                        if m.distance < 0.75 * n.distance:  # Strict Lowe's ratio test
                            good.append(m)
                            
                base_kp = min(len(f1["kp"]), len(f2["kp"]))
                feat_score = (len(good) / base_kp) * 100 if base_kp > 0 else 0
            except Exception:
                feat_score = 0

        # B. Color Similarity Score (30% Weight)
        try:
            color_corr = cv2.compareHist(f1["hist"], f2["hist"], cv2.HISTCMP_CORREL)
            color_score = max(0, min(100, ((color_corr + 1) / 2) * 100))
        except Exception:
            color_score = 0

        # C. Texture Density Match (20% Weight)
        try:
            density_diff = abs(f1["edge_density"] - f2["edge_density"])
            texture_score = max(0, 100 - (density_diff * 400))
        except Exception:
            texture_score = 0

        # Composite Match Score Calculation
        composite_score = (feat_score * 0.50) + (color_score * 0.30) + (texture_score * 0.20)
        
        # Scaling multiplier for user clarity
        final_score = min(100, round(composite_score * 1.5, 1))
        return final_score if final_score >= 10 else 0
        
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
            message = "Please choose image from camera or gallery."
        else:
            original_filename = secure_filename(file.filename) or "image.jpg"
            saved_upload_name = get_unique_filename(original_filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], saved_upload_name)

            try:
                save_stream_compressed(file, filepath, max_dim=600)
                image_name = saved_upload_name

                if action == "add":
                    fabric = get_final_fabric_from_form()
                    work_type = get_final_work_type_from_form()
                    stitch = request.form.get("stitch", "").strip()
                    manual_design_id = request.form.get("design_id", "").strip()
                    color = request.form.get("color", "").strip()
                    occasion = request.form.get("occasion", "").strip()
                    notes = request.form.get("notes", "").strip()

                    design_id = manual_design_id or f"D{int(datetime.now().timestamp())}"

                    upload_result = cloudinary.uploader.upload(
                        filepath,
                        folder="DesignFinder",
                        quality="80"
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

                    message = f"Design {design_id} added successfully!"

                elif action == "search":
                    # 1. Fetch DB Records Batch
                    all_meta_res = supabase.table("designs").select("*").execute()
                    meta_list = all_meta_res.data or []
                    
                    # 2. Auto-restore images locally if Render ephemeral disk wiped them
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

                    # 3. High-Accuracy Match Engine Execution
                    if os.path.exists(DATABASE_FOLDER):
                        valid_files = [f for f in os.listdir(DATABASE_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png'))]
                        meta_dict = {item["filename"]: item for item in meta_list}
                        
                        for db_file in valid_files:
                            db_path = os.path.join(DATABASE_FOLDER, db_file)
                            if os.path.isfile(db_path):
                                score = get_image_match_score(filepath, db_path)
                                if score > 12:  # Filter weak matches
                                    meta = meta_dict.get(db_file, {})
                                    similar_images.append({
                                        "filename": db_file,
                                        "design_id": meta.get("design_id", "N/A"),
                                        "score": score,
                                        "fabric": meta.get("fabric", ""),
                                        "work_type": meta.get("work_type", ""),
                                        "color": meta.get("color", ""),
                                        "occasion": meta.get("occasion", ""),
                                        "image_url": meta.get("image_url", "")
                                    })
                        similar_images.sort(key=lambda x: x["score"], reverse=True)
                        message = f"Found {len(similar_images)} matching designs."

            except Exception as e:
                message = f"Operation failed: {str(e)}"

    return render_template(
        "index.html",
        image_name=image_name,
        similar_images=similar_images,
        message=message,
        fabric_options=FABRIC_OPTIONS,
        work_type_options=WORK_TYPE_OPTIONS
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

    return render_template("design_detail.html", design=design, previous_design=None, next_design=None, similar_designs=[])

@app.route("/edit/<filename>", methods=["GET", "POST"])
def edit_design(filename):
    try:
        res = supabase.table("designs").select("*").eq("filename", filename).execute()
        details = res.data[0] if res.data else {}
    except Exception:
        details = {}

    if request.method == "POST":
        fabric = get_final_fabric_from_form()
        work_type = get_final_work_type_from_form()
        color = request.form.get("color", "").strip()
        occasion = request.form.get("occasion", "").strip()
        notes = request.form.get("notes", "").strip()

        supabase.table("designs").update({
            "fabric": fabric,
            "work_type": work_type,
            "color": color,
            "occasion": occasion,
            "notes": notes
        }).eq("filename", filename).execute()

        return redirect(url_for("gallery"))

    selected_fabric, custom_fabric = get_select_and_custom(details.get("fabric", ""), FABRIC_OPTIONS)
    selected_work_type, custom_work_type = get_select_and_custom(details.get("work_type", ""), WORK_TYPE_OPTIONS)

    return render_template(
        "edit_design.html",
        filename=filename,
        details=details,
        fabric_options=FABRIC_OPTIONS,
        work_type_options=WORK_TYPE_OPTIONS,
        selected_fabric=selected_fabric,
        custom_fabric=custom_fabric,
        selected_work_type=selected_work_type,
        custom_work_type=custom_work_type
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

@app.route("/manifest.json")
def manifest():
    return send_from_directory("static", "manifest.json")

@app.route("/service-worker.js")
def service_worker():
    return send_from_directory("static", "service-worker.js")

if __name__ == "__main__":
    app.run(debug=True)

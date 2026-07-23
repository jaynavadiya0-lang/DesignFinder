from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from PIL import Image
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import json
from collections import Counter
from datetime import datetime
import cloudinary
import cloudinary.uploader

from supabase import create_client

app = Flask(__name__)

# -------------------------------------------------
# CONFIGURATIONS
# -------------------------------------------------
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
    secure=True
)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase = create_client(
    SUPABASE_URL if SUPABASE_URL else "",
    SUPABASE_KEY if SUPABASE_KEY else ""
)

UPLOAD_FOLDER = "uploads"
DATABASE_FOLDER = "design_database"
TEMPLATES_FOLDER = "templates"
STATIC_FOLDER = "static"
METADATA_FILE = "design_metadata.json"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)
os.makedirs(TEMPLATES_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# -------------------------------------------------
# FEATURE DETECTORS
# -------------------------------------------------
orb = cv2.ORB_create(nfeatures=1200)
akaze = cv2.AKAZE_create()

bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
bf_akaze = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# -------------------------------------------------
# OPTIONS
# -------------------------------------------------
FABRIC_OPTIONS = [
    "Viscose", "Cotton", "Georgette", "Chiffon", "Silk",
    "Linen", "Rayon", "Crepe", "Organza", "Net"
]

WORK_TYPE_OPTIONS = [
    "Floral", "Butta", "Jaal", "Border", "Pallu",
    "Allover", "Embroidery", "Printed", "Weaving", "Traditional"
]

# -------------------------------------------------
# JSON & FILE HELPERS
# -------------------------------------------------
def load_json_file(path, default_value):
    if not os.path.exists(path):
        return default_value
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_value

def save_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_metadata():
    return load_json_file(METADATA_FILE, {})

def save_metadata(data):
    save_json_file(METADATA_FILE, data)

def get_unique_filename(original_name):
    base_name = secure_filename(original_name)
    if not base_name:
        base_name = "image.jpg"
    name, ext = os.path.splitext(base_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if not ext:
        ext = ".jpg"
    return f"{name}_{timestamp}{ext}"

def get_uploaded_file_from_request():
    possible_keys = ["image", "camera_image", "gallery_image"]
    for key in possible_keys:
        f = request.files.get(key)
        if f and f.filename:
            return f
    return None

# -------------------------------------------------
# FORM HELPERS
# -------------------------------------------------
def get_final_fabric_from_form():
    fabric_select = request.form.get("fabric_select", "").strip()
    fabric_custom = request.form.get("fabric_custom", "").strip()
    if fabric_select == "Other":
        return fabric_custom
    elif fabric_select:
        return fabric_select
    return fabric_custom

def get_final_work_type_from_form():
    work_type_select = request.form.get("work_type_select", "").strip()
    work_type_custom = request.form.get("work_type_custom", "").strip()
    if work_type_select == "Other":
        return work_type_custom
    elif work_type_select:
        return work_type_select
    return work_type_custom

def get_select_and_custom(existing_value, options):
    if not existing_value:
        return "", ""
    if existing_value in options:
        return existing_value, ""
    return "Other", existing_value

# -------------------------------------------------
# OPENCV MATCHING LOGIC
# -------------------------------------------------
def load_cv_image(image_path):
    return cv2.imread(image_path)

def resize_for_matching(img, max_side=900):
    h, w = img.shape[:2]
    largest = max(h, w)
    if largest > max_side:
        scale = max_side / largest
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def enhance_image(img):
    img = resize_for_matching(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def get_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def get_edges(gray):
    return cv2.Canny(gray, 80, 180)

def get_main_patches(img):
    h, w = img.shape[:2]
    patches = [img]
    center = img[int(h * 0.12):int(h * 0.88), int(w * 0.12):int(w * 0.88)]
    upper = img[int(h * 0.05):int(h * 0.58), int(w * 0.12):int(w * 0.88)]
    left_mid = img[int(h * 0.20):int(h * 0.88), int(w * 0.02):int(w * 0.46)]
    right_mid = img[int(h * 0.20):int(h * 0.88), int(w * 0.54):int(w * 0.98)]
    for p in [center, upper, left_mid, right_mid]:
        if p is not None and p.size > 0:
            patches.append(p)
    return patches

def get_tiles(img, rows=2, cols=2):
    h, w = img.shape[:2]
    tiles = []
    tile_h, tile_w = h // rows, w // cols
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * tile_h, (r + 1) * tile_h if r < rows - 1 else h
            x1, x2 = c * tile_w, (c + 1) * tile_w if c < cols - 1 else w
            tile = img[y1:y2, x1:x2]
            if tile is not None and tile.size > 0:
                tiles.append(tile)
    return tiles

def build_patch_features(img):
    img = enhance_image(img)
    patches = get_main_patches(img)
    h, w = img.shape[:2]
    center = img[int(h * 0.12):int(h * 0.88), int(w * 0.12):int(w * 0.88)]
    if center is not None and center.size > 0:
        patches.extend(get_tiles(center, 2, 2))

    features = []
    for patch in patches:
        if patch is None or patch.size == 0:
            continue
        patch = resize_for_matching(patch, max_side=320)
        gray = get_gray(patch)
        edges = get_edges(gray)
        orb_kp, orb_des = orb.detectAndCompute(gray, None)
        akaze_kp, akaze_des = akaze.detectAndCompute(gray, None)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        features.append({
            "orb_kp": orb_kp, "orb_des": orb_des,
            "akaze_kp": akaze_kp, "akaze_des": akaze_des,
            "hist": hist, "edges": edges
        })
    return features

def knn_good_match_score(des1, kp1, des2, kp2, matcher, ratio=0.78):
    if des1 is None or des2 is None or kp1 is None or kp2 is None:
        return 0
    if len(kp1) == 0 or len(kp2) == 0 or len(des1) < 2 or len(des2) < 2:
        return 0
    des1 = np.ascontiguousarray(des1, dtype=np.uint8)
    des2 = np.ascontiguousarray(des2, dtype=np.uint8)
    if des1.ndim != 2 or des2.ndim != 2 or des1.shape[1] != des2.shape[1]:
        return 0
    try:
        matches = matcher.knnMatch(des1, des2, k=2)
    except Exception:
        return 0
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    base = min(len(kp1), len(kp2))
    return (len(good) / base) * 100 if base > 0 else 0

def edge_similarity_score(edge1, edge2):
    try:
        diff = np.mean(cv2.absdiff(cv2.resize(edge1, (220, 220)), cv2.resize(edge2, (220, 220))))
        return max(0, 100 - (diff / 255.0) * 100)
    except Exception:
        return 0

def hist_similarity_score(hist1, hist2):
    try:
        corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0, min(100, ((corr + 1) / 2) * 100))
    except Exception:
        return 0

def compare_single_feature_patch(f1, f2):
    orb_score = knn_good_match_score(f1["orb_des"], f1["orb_kp"], f2["orb_des"], f2["orb_kp"], bf_orb)
    akaze_score = knn_good_match_score(f1["akaze_des"], f1["akaze_kp"], f2["akaze_des"], f2["akaze_kp"], bf_akaze) if f1["akaze_des"] is not None and f2["akaze_des"] is not None else 0
    hist_score = hist_similarity_score(f1["hist"], f2["hist"])
    edge_score = edge_similarity_score(f1["edges"], f2["edges"])
    return (orb_score * 0.38 + akaze_score * 0.32 + hist_score * 0.15 + edge_score * 0.15)

def compare_feature_sets(features1, features2):
    best_patch_scores = []
    for f1 in features1:
        best_score = max([compare_single_feature_patch(f1, f2) for f2 in features2], default=0)
        if best_score > 0:
            best_patch_scores.append(best_score)
    if not best_patch_scores:
        return 0
    best_patch_scores.sort(reverse=True)
    n = len(best_patch_scores)
    if n >= 6:
        final_score = sum(best_patch_scores[i] * w for i, w in enumerate([0.22, 0.20, 0.18, 0.15, 0.13, 0.12]))
    elif n >= 4:
        final_score = sum(best_patch_scores[i] * w for i, w in enumerate([0.32, 0.28, 0.22, 0.18]))
    elif n >= 2:
        final_score = best_patch_scores[0] * 0.60 + best_patch_scores[1] * 0.40
    else:
        final_score = best_patch_scores[0]
    return min(100, round(final_score * 2.2, 2))

def get_image_match_score(upload_path, db_path):
    upload_img = load_cv_image(upload_path)
    db_img = load_cv_image(db_path)
    if upload_img is None or db_img is None:
        return 0
    return compare_feature_sets(build_patch_features(upload_img), build_patch_features(db_img))

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
                file.save(filepath)
                image_name = saved_upload_name

                if action == "add":
                    fabric = get_final_fabric_from_form()
                    work_type = get_final_work_type_from_form()
                    stitch = request.form.get("stitch", "").strip()
                    manual_design_id = request.form.get("design_id", "").strip()
                    color = request.form.get("color", "").strip()
                    occasion = request.form.get("occasion", "").strip()
                    notes = request.form.get("notes", "").strip()

                    design_id = manual_design_id
                    if not design_id:
                        result = supabase.table("designs").select("id", count="exact").execute()
                        total = result.count if result.count else 0
                        design_id = f"D{total + 1:05d}"

                    duplicate = supabase.table("designs").select("id").eq("design_id", design_id).execute()
                    if duplicate.data:
                        message = f"Design ID {design_id} already exists."
                    else:
                        # 1. Cloudinary upload
                        upload_result = cloudinary.uploader.upload(filepath, folder="DesignFinder")
                        
                        # 2. Local database cache save (for OpenCV matching)
                        db_local_path = os.path.join(DATABASE_FOLDER, saved_upload_name)
                        cv2.imwrite(db_local_path, cv2.imread(filepath))

                        # 3. Save to Supabase
                        supabase.table("designs").insert({
                            "design_id": design_id,
                            "filename": saved_upload_name,
                            "image_url": upload_result["secure_url"],
                            "public_id": upload_result["public_id"],
                            "fabric": fabric,
                            "work_type": work_type,
                            "stitch": stitch,
                            "color": color,
                            "occasion": occasion,
                            "notes": notes
                        }).execute()
                        message = f"Design {design_id} added successfully."

                elif action == "search":
                    # Image Match Engine Logic
                    if os.path.exists(DATABASE_FOLDER):
                        for db_file in os.listdir(DATABASE_FOLDER):
                            db_path = os.path.join(DATABASE_FOLDER, db_file)
                            if os.path.isfile(db_path):
                                score = get_image_match_score(filepath, db_path)
                                if score > 15:  # threshold %
                                    res = supabase.table("designs").select("*").eq("filename", db_file).execute()
                                    meta = res.data[0] if res.data else {}
                                    similar_images.append({
                                        "filename": db_file,
                                        "design_id": meta.get("design_id", "N/A"),
                                        "score": score,
                                        "fabric": meta.get("fabric", ""),
                                        "work_type": meta.get("work_type", ""),
                                        "color": meta.get("color", ""),
                                        "occasion": meta.get("occasion", "")
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
    except Exception as e:
        print(e)
        designs = []
    return render_template("gallery.html", designs=designs, fabric_options=FABRIC_OPTIONS, work_type_options=WORK_TYPE_OPTIONS)

@app.route("/dashboard")
def dashboard():
    try:
        response = supabase.table("designs").select("*").order("id", desc=True).execute()
        designs = response.data if response.data else []
    except Exception as e:
        print(e)
        designs = []

    total_designs = len(designs)
    fabric_count = len(set(d["fabric"] for d in designs if d.get("fabric")))
    work_type_count = len(set(d["work_type"] for d in designs if d.get("work_type")))

    return render_template(
        "dashboard.html",
        total_designs=total_designs,
        total_fabric=fabric_count,
        total_work_type=work_type_count,
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

    return render_template(
        "design_detail.html",
        design=design,
        previous_design=None,
        next_design=None,
        similar_designs=[]
    )

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

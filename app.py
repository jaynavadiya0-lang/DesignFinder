from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from PIL import Image
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import json
import shutil
from collections import Counter
from datetime import datetime
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
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)
os.makedirs(TEMPLATES_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# -------------------------------------------------
# FEATURE DETECTORS
# -------------------------------------------------
orb = cv2.ORB_create(nfeatures=500)
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

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
# FILE & FORM HELPERS
# -------------------------------------------------
def get_unique_filename(original_name):
    base_name = secure_filename(original_name) or "image.jpg"
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
# LIGHTWEIGHT & FAST OPENCV MATCHING (FOR RENDER)
# -------------------------------------------------
def build_patch_features(img):
    h, w = img.shape[:2]
    if max(h, w) > 400:
        scale = 400 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)

    return {"kp": kp, "des": des, "hist": hist}

def get_image_match_score(upload_path, db_path):
    try:
        img1 = cv2.imread(upload_path)
        img2 = cv2.imread(db_path)

        if img1 is None or img2 is None:
            return 0

        f1 = build_patch_features(img1)
        f2 = build_patch_features(img2)

        feat_score = 0
        if f1["des"] is not None and f2["des"] is not None and len(f1["des"]) >= 2 and len(f2["des"]) >= 2:
            try:
                matches = bf_orb.knnMatch(f1["des"], f2["des"], k=2)
                good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.78 * n.distance]
                base = min(len(f1["kp"]), len(f2["kp"]))
                feat_score = (len(good) / base) * 100 if base > 0 else 0
            except Exception:
                feat_score = 0

        try:
            corr = cv2.compareHist(f1["hist"], f2["hist"], cv2.HISTCMP_CORREL)
            hist_score = max(0, min(100, ((corr + 1) / 2) * 100))
        except Exception:
            hist_score = 0

        final_score = (feat_score * 0.7) + (hist_score * 0.3)
        return min(100, round(final_score * 1.8, 2))
    except Exception as e:
        print("Match score calculation error:", e)
        return 0

# -------------------------------------------------
# MAIN ROUTES
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
                        try:
                            result = supabase.table("designs").select("id", count="exact").execute()
                            total = result.count if result.count is not None else 0
                            design_id = f"D{total + 1:05d}"
                        except Exception:
                            design_id = f"D{int(datetime.now().timestamp())}"

                    # 1. Upload to Cloudinary
                    upload_result = cloudinary.uploader.upload(filepath, folder="DesignFinder")
                    
                    # 2. Local File Copy for OpenCV Cache
                    db_local_path = os.path.join(DATABASE_FOLDER, saved_upload_name)
                    shutil.copy(filepath, db_local_path)

                    # 3. Save to Supabase Database
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
                    if os.path.exists(DATABASE_FOLDER):
                        for db_file in os.listdir(DATABASE_FOLDER):
                            db_path = os.path.join(DATABASE_FOLDER, db_file)
                            if os.path.isfile(db_path):
                                score = get_image_match_score(filepath, db_path)
                                if score > 5:
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
                # Catch exact error message on UI screen
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
        print("Gallery Fetch Error:", e)
        designs = []
    return render_template("gallery.html", designs=designs, fabric_options=FABRIC_OPTIONS, work_type_options=WORK_TYPE_OPTIONS)

@app.route("/dashboard")
def dashboard():
    try:
        response = supabase.table("designs").select("*").order("id", desc=True).execute()
        designs = response.data if response.data else []
    except Exception as e:
        print("Dashboard Fetch Error:", e)
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

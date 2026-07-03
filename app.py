from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from PIL import Image
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import json
from collections import Counter
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
DATABASE_FOLDER = "design_database"
TEMPLATES_FOLDER = "templates"
STATIC_FOLDER = "static"

METADATA_FILE = "design_metadata.json"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB

# -------------------------------------------------
# FOLDERS
# -------------------------------------------------
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)
os.makedirs(TEMPLATES_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# -------------------------------------------------
# FEATURE DETECTORS
# -------------------------------------------------
orb = cv2.ORB_create(nfeatures=2500)
akaze = cv2.AKAZE_create()

bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
bf_akaze = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# -------------------------------------------------
# OPTIONS
# -------------------------------------------------
FABRIC_OPTIONS = [
    "Viscose",
    "Cotton",
    "Georgette",
    "Chiffon",
    "Silk",
    "Linen",
    "Rayon",
    "Crepe",
    "Organza",
    "Net"
]

WORK_TYPE_OPTIONS = [
    "Floral",
    "Butta",
    "Jaal",
    "Border",
    "Pallu",
    "Allover",
    "Embroidery",
    "Printed",
    "Weaving",
    "Traditional"
]

# -------------------------------------------------
# JSON HELPERS
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


# -------------------------------------------------
# METADATA HELPERS
# -------------------------------------------------
def load_metadata():
    return load_json_file(METADATA_FILE, {})


def save_metadata(data):
    save_json_file(METADATA_FILE, data)


def get_next_design_id(metadata):
    max_num = 0
    for _, details in metadata.items():
        design_id = details.get("design_id", "")
        if isinstance(design_id, str) and design_id.startswith("D"):
            try:
                num = int(design_id[1:])
                max_num = max(max_num, num)
            except Exception:
                pass
    return f"D{max_num + 1:03d}"


def assign_ids_to_old_images():
    metadata = load_metadata()
    updated = False

    max_num = 0
    for _, details in metadata.items():
        design_id = details.get("design_id", "")
        if isinstance(design_id, str) and design_id.startswith("D"):
            try:
                num = int(design_id[1:])
                max_num = max(max_num, num)
            except Exception:
                pass

    if os.path.exists(DATABASE_FOLDER):
        for filename in sorted(os.listdir(DATABASE_FOLDER)):
            file_path = os.path.join(DATABASE_FOLDER, filename)
            if not os.path.isfile(file_path):
                continue

            if filename not in metadata:
                max_num += 1
                metadata[filename] = {
                    "design_id": f"D{max_num:03d}",
                    "fabric": "",
                    "work_type": "",
                    "color": "",
                    "occasion": "",
                    "notes": ""
                }
                updated = True

    if updated:
        save_metadata(metadata)


assign_ids_to_old_images()

# -------------------------------------------------
# SEARCH HISTORY HELPERS
# -------------------------------------------------
    item = {
        "timestamp": datetime.now().strftime("%d-%m-%Y %I:%M %p"),
        "query_image": query_image,
        "results": results[:3]
    }

    history.insert(0, item)
    history = history[:100]
    save_search_history(history)


# -------------------------------------------------
# FILE / UPLOAD HELPERS
# -------------------------------------------------
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
    """
    Support all possible mobile/public-link cases:
    - input name="image"
    - input name="camera_image"
    - input name="gallery_image"
    """
    possible_keys = ["image", "camera_image", "gallery_image"]

    for key in possible_keys:
        f = request.files.get(key)
        if f and f.filename:
            return f

    return None


# -------------------------------------------------
# COLOR HELPERS
# -------------------------------------------------
def get_color_name(rgb):
    r, g, b = rgb

    if r > 180 and g < 100 and b < 100:
        return "Red"
    elif r > 150 and b > 150:
        return "Pink"
    elif b > r and b > g:
        return "Blue"
    elif g > r and g > b:
        return "Green"
    elif r > 150 and g > 100 and b < 100:
        return "Orange"
    elif r > 180 and g > 180 and b < 120:
        return "Yellow"
    elif r < 80 and g < 80 and b < 80:
        return "Black"
    else:
        return "Mixed Color"


def suggest_combination(color_name):
    suggestions = {
        "Red": "Gold, Cream, Black",
        "Blue": "Silver, White, Pink",
        "Green": "Gold, Maroon, Beige",
        "Pink": "Silver, White, Navy",
        "Orange": "Green, Cream, Gold",
        "Yellow": "Green, Maroon, Black",
        "Black": "Gold, Red, Silver"
    }
    return suggestions.get(color_name, "Gold, Silver, Cream")


# -------------------------------------------------
# IMAGE PREPROCESSING
# -------------------------------------------------
def load_cv_image(image_path):
    return cv2.imread(image_path)


def resize_for_matching(img, max_side=900):
    h, w = img.shape[:2]
    largest = max(h, w)
    if largest > max_side:
        scale = max_side / largest
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
    return img


def enhance_image(img):
    img = resize_for_matching(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img


def get_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def get_edges(gray):
    return cv2.Canny(gray, 80, 180)


# -------------------------------------------------
# PATCHES / TILES
# -------------------------------------------------
def get_main_patches(img):
    h, w = img.shape[:2]
    patches = []

    patches.append(img)

    center = img[int(h * 0.12):int(h * 0.88), int(w * 0.12):int(w * 0.88)]
    upper = img[int(h * 0.05):int(h * 0.58), int(w * 0.12):int(w * 0.88)]
    left_mid = img[int(h * 0.20):int(h * 0.88), int(w * 0.02):int(w * 0.46)]
    right_mid = img[int(h * 0.20):int(h * 0.88), int(w * 0.54):int(w * 0.98)]

    for p in [center, upper, left_mid, right_mid]:
        if p is not None and p.size > 0:
            patches.append(p)

    return patches


def get_tiles(img, rows=3, cols=3):
    h, w = img.shape[:2]
    tiles = []
    tile_h = h // rows
    tile_w = w // cols

    for r in range(rows):
        for c in range(cols):
            y1 = r * tile_h
            y2 = (r + 1) * tile_h if r < rows - 1 else h
            x1 = c * tile_w
            x2 = (c + 1) * tile_w if c < cols - 1 else w
            tile = img[y1:y2, x1:x2]
            if tile is not None and tile.size > 0:
                tiles.append(tile)

    return tiles


# -------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------
def get_orb_desc(gray):
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des


def get_akaze_desc(gray):
    kp, des = akaze.detectAndCompute(gray, None)
    return kp, des


def build_patch_features(img):
    img = enhance_image(img)
    patches = get_main_patches(img)

    h, w = img.shape[:2]
    center = img[int(h * 0.12):int(h * 0.88), int(w * 0.12):int(w * 0.88)]
    if center is not None and center.size > 0:
        patches.extend(get_tiles(center, 3, 3))

    features = []

    for patch in patches:
        if patch is None or patch.size == 0:
            continue

        patch = resize_for_matching(patch, max_side=450)
        gray = get_gray(patch)
        edges = get_edges(gray)

        orb_kp, orb_des = get_orb_desc(gray)
        akaze_kp, akaze_des = get_akaze_desc(gray)

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        features.append({
            "orb_kp": orb_kp,
            "orb_des": orb_des,
            "akaze_kp": akaze_kp,
            "akaze_des": akaze_des,
            "hist": hist,
            "edges": edges
        })

    return features


# -------------------------------------------------
# MATCH SCORING
# -------------------------------------------------
def knn_good_match_score(des1, kp1, des2, kp2, matcher, ratio=0.75):
    if des1 is None or des2 is None:
        return 0
    if len(des1) < 2 or len(des2) < 2:
        return 0

    try:
        if des1 is None or des2 is None:
    return 0

if len(des1) < 2 or len(des2) < 2:
    return 0
        matches = matcher.knnMatch(des1, des2, k=2)
    except Exception:
        return 0

    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)

    base = min(len(kp1), len(kp2))
    if base == 0:
        return 0

    score = (len(good) / base) * 100
    return score


def edge_similarity_score(edge1, edge2):
    try:
        e1 = cv2.resize(edge1, (220, 220))
        e2 = cv2.resize(edge2, (220, 220))
        diff = np.mean(cv2.absdiff(e1, e2))
        score = max(0, 100 - (diff / 255.0) * 100)
        return score
    except Exception:
        return 0


def hist_similarity_score(hist1, hist2):
    try:
        corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        score = max(0, min(100, ((corr + 1) / 2) * 100))
        return score
    except Exception:
        return 0


def compare_single_feature_patch(f1, f2):
    orb_score = knn_good_match_score(
        f1["orb_des"], f1["orb_kp"],
        f2["orb_des"], f2["orb_kp"],
        bf_orb,
        ratio=0.78
    )

    if (
        f1["akaze_des"] is None or
        f2["akaze_des"] is None
    ):
        akaze_score = 0
    else:
        if (
    f1["akaze_des"] is None or
    f2["akaze_des"] is None
):
    akaze_score = 0
else:
        akaze_score = knn_good_match_score(
            f1["akaze_des"], f1["akaze_kp"],
            f2["akaze_des"], f2["akaze_kp"],
            bf_akaze,
            ratio=0.78
        )

    hist_score = hist_similarity_score(
        f1["hist"],
        f2["hist"]
    )

    edge_score = edge_similarity_score(
        f1["edges"],
        f2["edges"]
    )

    final = (
        orb_score * 0.38 +
        akaze_score * 0.32 +
        hist_score * 0.15 +
        edge_score * 0.15
    )

    return final

def compare_feature_sets(features1, features2):
    best_patch_scores = []

    for f1 in features1:
        best_score = 0
        for f2 in features2:
            score = compare_single_feature_patch(f1, f2)
            if score > best_score:
                best_score = score
        if best_score > 0:
            best_patch_scores.append(best_score)

    if not best_patch_scores:
        return 0

    best_patch_scores.sort(reverse=True)

    if len(best_patch_scores) >= 6:
        final_score = (
            best_patch_scores[0] * 0.22 +
            best_patch_scores[1] * 0.20 +
            best_patch_scores[2] * 0.18 +
            best_patch_scores[3] * 0.15 +
            best_patch_scores[4] * 0.13 +
            best_patch_scores[5] * 0.12
        )
    elif len(best_patch_scores) >= 4:
        final_score = (
            best_patch_scores[0] * 0.32 +
            best_patch_scores[1] * 0.28 +
            best_patch_scores[2] * 0.22 +
            best_patch_scores[3] * 0.18
        )
    elif len(best_patch_scores) >= 2:
        final_score = (
            best_patch_scores[0] * 0.60 +
            best_patch_scores[1] * 0.40
        )
    else:
        final_score = best_patch_scores[0]

    final_score = min(100, round(final_score * 2.2, 2))
    return final_score


def get_image_match_score(upload_path, db_path):
    upload_img = load_cv_image(upload_path)
    db_img = load_cv_image(db_path)

    if upload_img is None or db_img is None:
        return 0

    upload_features = build_patch_features(upload_img)
    db_features = build_patch_features(db_img)

    return compare_feature_sets(upload_features, db_features)


def is_duplicate_design(upload_path, db_path):
    score = get_image_match_score(upload_path, db_path)
    return score >= 88, score


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
    else:
        return fabric_custom


def get_final_work_type_from_form():
    work_type_select = request.form.get("work_type_select", "").strip()
    work_type_custom = request.form.get("work_type_custom", "").strip()

    if work_type_select == "Other":
        return work_type_custom
    elif work_type_select:
        return work_type_select
    else:
        return work_type_custom


def get_select_and_custom(existing_value, options):
    if not existing_value:
        return "", ""
    if existing_value in options:
        return existing_value, ""
    return "Other", existing_value


# -------------------------------------------------
# DASHBOARD HELPER
# -------------------------------------------------
def build_dashboard_data():
    metadata = load_metadata()
    designs = []

    if os.path.exists(DATABASE_FOLDER):
        for filename in os.listdir(DATABASE_FOLDER):
            file_path = os.path.join(DATABASE_FOLDER, filename)
            if not os.path.isfile(file_path):
                continue

            details = metadata.get(filename, {})
            designs.append({
                "filename": filename,
                "design_id": details.get("design_id", "N/A"),
                "fabric": details.get("fabric", "").strip(),
                "work_type": details.get("work_type", "").strip(),
                "color": details.get("color", "").strip(),
                "occasion": details.get("occasion", "").strip(),
                "notes": details.get("notes", "").strip(),
                "created_time": os.path.getmtime(file_path)
            })

    designs.sort(key=lambda x: x["created_time"], reverse=True)

    work_type_counter = Counter()
    for d in designs:
        if d["work_type"]:
            work_type_counter[d["work_type"]] += 1

    return {
        "total_designs": len(designs),
        "total_work_types": len(work_type_counter),
        "recent_designs": designs[:12]
    }


# -------------------------------------------------
# DESIGN HELPERS
# -------------------------------------------------
def get_all_design_items():
    metadata = load_metadata()
    designs = []

    if os.path.exists(DATABASE_FOLDER):
        for filename in os.listdir(DATABASE_FOLDER):
            file_path = os.path.join(DATABASE_FOLDER, filename)
            if not os.path.isfile(file_path):
                continue

            details = metadata.get(filename, {})
            designs.append({
                "filename": filename,
                "design_id": details.get("design_id", "N/A"),
                "fabric": details.get("fabric", "").strip(),
                "work_type": details.get("work_type", "").strip(),
                "color": details.get("color", "").strip(),
                "occasion": details.get("occasion", "").strip(),
                "notes": details.get("notes", "").strip(),
                "created_time": os.path.getmtime(file_path)
            })

    designs.sort(key=lambda x: x["created_time"], reverse=True)
    return designs


def get_similar_designs_for_detail(target_filename, top_n=6):
    target_path = os.path.join(DATABASE_FOLDER, target_filename)
    if not os.path.exists(target_path):
        return []

    metadata = load_metadata()
    similar = []

    for db_image in os.listdir(DATABASE_FOLDER):
        db_path = os.path.join(DATABASE_FOLDER, db_image)

        if not os.path.isfile(db_path):
            continue
        if db_image == target_filename:
            continue

        try:
            score = get_image_match_score(target_path, db_path)
            details = metadata.get(db_image, {})
            similar.append({
                "filename": db_image,
                "score": round(score, 2),
                "design_id": details.get("design_id", "N/A"),
                "fabric": details.get("fabric", ""),
                "work_type": details.get("work_type", ""),
                "color": details.get("color", ""),
                "occasion": details.get("occasion", ""),
                "notes": details.get("notes", "")
            })
        except Exception:
            pass

    similar.sort(key=lambda x: x["score"], reverse=True)
    return similar[:top_n]


# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    image_name = None
    image_size = None
    dominant_color = None
    color_name = None
    combination = None
    similar_images = []
    message = None

    if request.method == "POST":
        action = request.form.get("action", "").strip()
        file = get_uploaded_file_from_request()

        if not file or not file.filename:
            message = "Please choose image from camera or gallery."
            return render_template(
                "index.html",
                image_name=None,
                image_size=None,
                dominant_color=None,
                color_name=None,
                combination=None,
                similar_images=[],
                message=message,
                fabric_options=FABRIC_OPTIONS,
                work_type_options=WORK_TYPE_OPTIONS
            )

        original_filename = secure_filename(file.filename)
        if not original_filename:
            original_filename = "image.jpg"

        saved_upload_name = get_unique_filename(original_filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], saved_upload_name)

        try:
            file.save(filepath)
        except Exception:
            message = "Image upload failed. Please try again."
            return render_template(
                "index.html",
                image_name=None,
                image_size=None,
                dominant_color=None,
                color_name=None,
                combination=None,
                similar_images=[],
                message=message,
                fabric_options=FABRIC_OPTIONS,
                work_type_options=WORK_TYPE_OPTIONS
            )

        # ---------------- ADD TO DATABASE ----------------
        if action == "add":
            duplicate_found = False
            duplicate_score = 0

            if os.path.exists(DATABASE_FOLDER):
                for db_image in os.listdir(DATABASE_FOLDER):
                    db_path = os.path.join(DATABASE_FOLDER, db_image)
                    if not os.path.isfile(db_path):
                        continue
                    try:
                        is_dup, score = is_duplicate_design(filepath, db_path)
                        if is_dup:
                            duplicate_found = True
                            duplicate_score = round(score, 2)
                            break
                    except Exception:
                        pass

            if duplicate_found:
                message = f"This design already exists in database. Match: {duplicate_score}%"
            else:
                metadata = load_metadata()
                design_id = get_next_design_id(metadata)

                fabric = get_final_fabric_from_form()
                work_type = get_final_work_type_from_form()
                color = request.form.get("color", "").strip()
                occasion = request.form.get("occasion", "").strip()
                notes = request.form.get("notes", "").strip()

                db_save_name = saved_upload_name
                db_save_path = os.path.join(DATABASE_FOLDER, db_save_name)

                try:
                    with open(filepath, "rb") as src, open(db_save_path, "wb") as dst:
                        dst.write(src.read())
                except Exception:
                    message = "Unable to save design in database."
                    return render_template(
                        "index.html",
                        image_name=None,
                        image_size=None,
                        dominant_color=None,
                        color_name=None,
                        combination=None,
                        similar_images=[],
                        message=message,
                        fabric_options=FABRIC_OPTIONS,
                        work_type_options=WORK_TYPE_OPTIONS
                    )

                metadata[db_save_name] = {
                    "design_id": design_id,
                    "fabric": fabric,
                    "work_type": work_type,
                    "color": color,
                    "occasion": occasion,
                    "notes": notes
                }
                save_metadata(metadata)

                message = f"Design added successfully with Design ID: {design_id}"

            return render_template(
                "index.html",
                message=message,
                image_name=None,
                image_size=None,
                dominant_color=None,
                color_name=None,
                combination=None,
                similar_images=[],
                fabric_options=FABRIC_OPTIONS,
                work_type_options=WORK_TYPE_OPTIONS
            )

        # ---------------- SEARCH SIMILAR ----------------
        image_name = saved_upload_name

        try:
            pil_img = Image.open(filepath).convert("RGB")
            image_size = pil_img.size

            small_img = pil_img.resize((100, 100))
            colors = small_img.getcolors(10000)

            if colors:
                colors = sorted(colors, reverse=True)
                dominant_color = colors[0][1]
                color_name = get_color_name(dominant_color)
                combination = suggest_combination(color_name)
        except Exception:
            image_size = None
            dominant_color = None
            color_name = None
            combination = None

        metadata = load_metadata()

        if os.path.exists(DATABASE_FOLDER):
            for db_image in os.listdir(DATABASE_FOLDER):
                db_path = os.path.join(DATABASE_FOLDER, db_image)

                if not os.path.isfile(db_path):
                    continue

                try:
                    score = get_image_match_score(filepath, db_path)
                    details = metadata.get(db_image, {})

                    similar_images.append({
                        "filename": db_image,
                        "score": round(score, 2),
                        "design_id": details.get("design_id", "N/A"),
                        "fabric": details.get("fabric", ""),
                        "work_type": details.get("work_type", ""),
                        "color": details.get("color", ""),
                        "occasion": details.get("occasion", ""),
                        "notes": details.get("notes", "")
                    })
                except Exception:
                    pass

            similar_images.sort(key=lambda x: x["score"], reverse=True)
            similar_images = similar_images[:3]

        if not similar_images:
            message = "No similar design found."

    return render_template(
        "index.html",
        image_name=image_name,
        image_size=image_size,
        dominant_color=dominant_color,
        color_name=color_name,
        combination=combination,
        similar_images=similar_images,
        message=message,
        fabric_options=FABRIC_OPTIONS,
        work_type_options=WORK_TYPE_OPTIONS
    )


@app.route("/gallery")
def gallery():
    assign_ids_to_old_images()
    metadata = load_metadata()
    designs = []

    filter_design_id = request.args.get("design_id", "").strip()
    filter_fabric = request.args.get("fabric", "").strip()
    filter_work_type = request.args.get("work_type", "").strip()
    filter_color = request.args.get("color", "").strip()
    filter_occasion = request.args.get("occasion", "").strip()
    sort_by = request.args.get("sort_by", "newest").strip()

    if os.path.exists(DATABASE_FOLDER):
        for filename in os.listdir(DATABASE_FOLDER):
            file_path = os.path.join(DATABASE_FOLDER, filename)

            if os.path.isfile(file_path):
                details = metadata.get(filename, {})
                item = {
                    "filename": filename,
                    "design_id": details.get("design_id", "N/A"),
                    "fabric": details.get("fabric", ""),
                    "work_type": details.get("work_type", ""),
                    "color": details.get("color", ""),
                    "occasion": details.get("occasion", ""),
                    "notes": details.get("notes", ""),
                    "created_time": os.path.getmtime(file_path)
                }

                if filter_design_id and filter_design_id.lower() not in item["design_id"].lower():
                    continue
                if filter_fabric and filter_fabric.lower() != item["fabric"].lower():
                    continue
                if filter_work_type and filter_work_type.lower() not in item["work_type"].lower():
                    continue
                if filter_color and filter_color.lower() not in item["color"].lower():
                    continue
                if filter_occasion and filter_occasion.lower() not in item["occasion"].lower():
                    continue

                designs.append(item)

    if sort_by == "newest":
        designs.sort(key=lambda x: x["created_time"], reverse=True)
    elif sort_by == "oldest":
        designs.sort(key=lambda x: x["created_time"])
    elif sort_by == "id_asc":
        designs.sort(key=lambda x: x["design_id"])
    elif sort_by == "id_desc":
        designs.sort(key=lambda x: x["design_id"], reverse=True)
    elif sort_by == "fabric_asc":
        designs.sort(key=lambda x: (x["fabric"] or "").lower())
    elif sort_by == "color_asc":
        designs.sort(key=lambda x: (x["color"] or "").lower())

    return render_template(
        "gallery.html",
        designs=designs,
        fabric_options=FABRIC_OPTIONS,
        work_type_options=WORK_TYPE_OPTIONS,
        filter_design_id=filter_design_id,
        filter_fabric=filter_fabric,
        filter_work_type=filter_work_type,
        filter_color=filter_color,
        filter_occasion=filter_occasion,
        sort_by=sort_by
    )


@app.route("/dashboard")
def dashboard():
    data = build_dashboard_data()
    return render_template(
        "dashboard.html",
        total_designs=data["total_designs"],
        total_work_types=data["total_work_types"],
        recent_designs=data["recent_designs"]
    )

@app.route("/design/<filename>")
def design_detail(filename):
    assign_ids_to_old_images()
    metadata = load_metadata()
    file_path = os.path.join(DATABASE_FOLDER, filename)

    if not os.path.exists(file_path):
        return redirect(url_for("gallery"))

    details = metadata.get(filename, {})
    all_designs = get_all_design_items()

    current_design = None
    current_index = -1

    for i, item in enumerate(all_designs):
        if item["filename"] == filename:
            current_design = item
            current_index = i
            break

    if current_design is None:
        current_design = {
            "filename": filename,
            "design_id": details.get("design_id", "N/A"),
            "fabric": details.get("fabric", ""),
            "work_type": details.get("work_type", ""),
            "color": details.get("color", ""),
            "occasion": details.get("occasion", ""),
            "notes": details.get("notes", "")
        }

    previous_design = all_designs[current_index - 1] if current_index > 0 else None
    next_design = all_designs[current_index + 1] if current_index < len(all_designs) - 1 and current_index != -1 else None
    similar_designs = get_similar_designs_for_detail(filename, top_n=6)

    return render_template(
        "design_detail.html",
        design=current_design,
        previous_design=previous_design,
        next_design=next_design,
        similar_designs=similar_designs
    )


@app.route("/edit/<filename>", methods=["GET", "POST"])
def edit_design(filename):
    metadata = load_metadata()

    if filename not in metadata:
        metadata[filename] = {
            "design_id": "N/A",
            "fabric": "",
            "work_type": "",
            "color": "",
            "occasion": "",
            "notes": ""
        }

    if request.method == "POST":
        metadata[filename]["fabric"] = get_final_fabric_from_form()
        metadata[filename]["work_type"] = get_final_work_type_from_form()
        metadata[filename]["color"] = request.form.get("color", "").strip()
        metadata[filename]["occasion"] = request.form.get("occasion", "").strip()
        metadata[filename]["notes"] = request.form.get("notes", "").strip()

        save_metadata(metadata)
        return redirect(url_for("gallery"))

    details = metadata.get(filename, {})

    selected_fabric, custom_fabric = get_select_and_custom(
        details.get("fabric", ""),
        FABRIC_OPTIONS
    )
    selected_work_type, custom_work_type = get_select_and_custom(
        details.get("work_type", ""),
        WORK_TYPE_OPTIONS
    )

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
    file_path = os.path.join(DATABASE_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    metadata = load_metadata()
    if filename in metadata:
        del metadata[filename]
        save_metadata(metadata)

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

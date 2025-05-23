import os
from deepface import DeepFace
import numpy as np

def cosine_distance(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
from PIL import Image
from tqdm import tqdm
import shutil

# --- CONFIGURATION ---
PHOTO_DIR = "photos_face"
QUERY_IMAGE = "selfie.png"
RESULTS_DIR = "photos_match"
SEUIL_COSINE = 0.35  # Plus petit = plus strict

os.makedirs(RESULTS_DIR, exist_ok=True)

# --- EMBEDDING DU SELFIE ---
print("🔍 Analyse du selfie...")
query_rep = DeepFace.represent(img_path=QUERY_IMAGE, model_name="Facenet", detector_backend="opencv")[0]
query_emb = query_rep["embedding"]

# --- PARCOURS DES PHOTOS ---
print("\n📂 Recherche de visages similaires dans le dossier...")
matches = []

for file in tqdm(os.listdir(PHOTO_DIR)):
    path = os.path.join(PHOTO_DIR, file)
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    try:
        rep = DeepFace.represent(img_path=path, model_name="Facenet", detector_backend="opencv", enforce_detection=False)
        if isinstance(rep, list) and len(rep) > 0:
            emb = rep[0]["embedding"]
            distance = cosine_distance(query_emb, emb)
            if distance < SEUIL_COSINE:
                matches.append((path, distance))
                shutil.copy(path, os.path.join(RESULTS_DIR, file))
    except Exception as e:
        print(f"Erreur pour {file} : {e}")

# --- AFFICHAGE DES RÉSULTATS ---
print(f"\n✅ {len(matches)} photo(s) similaire(s) trouvée(s) :")
for path, score in sorted(matches, key=lambda x: x[1]):
    print(f"{path} - Similarité (cosine) : {score:.4f}")

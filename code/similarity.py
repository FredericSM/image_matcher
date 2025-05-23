import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
PHOTO_DIR = "ness_photos"             # Dossier contenant les photos
QUERY_IMAGE = "user.jpg"       # Photo de référence
SEUIL_SIMILARITE = 0.90              # Seuil de similarité
MAX_DISPLAY = 5                      # Nombre max d’images à afficher

# --- CHARGER DINOv2 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
model.eval()

# --- TRANSFORMATIONS POUR DINO ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@torch.no_grad()
def extract_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    embedding = model(tensor)
    return embedding.cpu().numpy().flatten()

# --- INDEXATION ---
print("Indexation des photos...")
image_paths = [os.path.join(PHOTO_DIR, f) for f in os.listdir(PHOTO_DIR)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]
image_embeddings = []
for path in tqdm(image_paths):
    try:
        emb = extract_embedding(path)
        image_embeddings.append(emb)
    except Exception as e:
        print(f"Erreur pour {path} : {e}")
image_embeddings = np.array(image_embeddings)

# --- RECHERCHE ---
print("Recherche des photos similaires à la référence...")
query_emb = extract_embedding(QUERY_IMAGE)
similarities = cosine_similarity([query_emb], image_embeddings)[0]
matching_indices = [i for i, score in enumerate(similarities) if score > SEUIL_SIMILARITE]

# --- RÉSULTATS ---
print(f"\nPhotos similaires trouvées (similarité > {SEUIL_SIMILARITE}): {len(matching_indices)}")
for idx in matching_indices:
    print(f"{image_paths[idx]} - Similarité : {similarities[idx]:.4f}")

# --- GÉNÉRATION DE LA PAGE HTML ---
html_path = "resultats_similaires.html"

with open(html_path, "w", encoding="utf-8") as f:
    f.write("<html><head><title>Photos similaires</title></head><body>\n")
    f.write("<h2>Photos similaires à la référence</h2>\n")
    for idx in matching_indices:
        path = os.path.abspath(image_paths[idx])
        score = similarities[idx]
        f.write(f'<div style="margin-bottom:20px;">')
        f.write(f'<img src="file://{path}" width="300"><br>')
        f.write(f'<span>Similarité : {score:.4f}</span></div><hr>\n')
    f.write("</body></html>")

# --- OUVERTURE DANS LE NAVIGATEUR ---
import webbrowser
webbrowser.open(f"file://{os.path.abspath(html_path)}")

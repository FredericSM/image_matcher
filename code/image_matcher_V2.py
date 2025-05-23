import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
PHOTO_DIR = "ness_photos"            # Dossier contenant les photos
QUERY_IMAGE = "image.png"      # Image de comparaison
TOP_K = 5                           # Nombre de résultats à afficher

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

# --- INDEXATION DES PHOTOS ---
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

# --- RECHERCHE DE SIMILARITÉ ---
print("Recherche de la photo utilisateur...")
query_emb = extract_embedding(QUERY_IMAGE)
similarities = cosine_similarity([query_emb], image_embeddings)[0]
top_indices = similarities.argsort()[-TOP_K:][::-1]

# --- AFFICHAGE DES RÉSULTATS ---
plt.figure(figsize=(15, 5))
for i, idx in enumerate(top_indices):
    img_path = image_paths[idx]
    similarity = similarities[idx]
    img = Image.open(img_path)
    plt.subplot(1, TOP_K, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Sim: {similarity:.2f}")
plt.suptitle("Photos les plus similaires (DINOv2)", fontsize=16)
plt.tight_layout()
html_file = "resultats_similaires.html"
with open(html_file, "w") as f:
    f.write("<html><body><h2>Résultats les plus similaires</h2>\n")
    for i, idx in enumerate(top_indices):
        path = os.path.abspath(image_paths[idx])
        similarity = similarities[idx]
        f.write(f'<div><img src="file://{path}" width="300"><br>Sim: {similarity:.2f}</div><hr>\n')
    f.write("</body></html>")

import webbrowser
webbrowser.open(f"file://{os.path.abspath(html_file)}")


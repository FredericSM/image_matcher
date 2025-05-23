import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from PIL import Image

# --- CONFIGURATION ---
PHOTO_DIR = "ness_photos"  # Dossier avec les photos du photographe
QUERY_IMAGE = "image.png"  # Image à comparer
TOP_K = 10  # Nombre de résultats à retourner

# --- MODELE PRETENTRAINE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained=True).features.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor).squeeze().cpu().numpy().flatten()
    return embedding / np.linalg.norm(embedding)

# --- INDEXATION ---
print("Indexation des photos...")
image_paths = [os.path.join(PHOTO_DIR, f) for f in os.listdir(PHOTO_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
image_embeddings = []
for path in tqdm(image_paths):
    try:
        emb = extract_embedding(path)
        image_embeddings.append(emb)
    except Exception as e:
        print(f"Erreur pour {path} : {e}")
image_embeddings = np.array(image_embeddings)

# --- RECHERCHE ---
print("Recherche de la photo utilisateur...")
query_emb = extract_embedding(QUERY_IMAGE)
similarities = cosine_similarity([query_emb], image_embeddings)[0]
top_indices = similarities.argsort()[-TOP_K:][::-1]

print("\nPhotos les plus similaires :")
for idx in top_indices:
    print(f"{image_paths[idx]} - Similarité : {similarities[idx]:.4f}")


# Affichage des images similaires dans une grille
plt.figure(figsize=(15, 5))
for i, idx in enumerate(top_indices):
    img_path = image_paths[idx]
    similarity = similarities[idx]

    # Chargement et affichage de l'image
    img = Image.open(img_path)
    plt.subplot(1, TOP_K, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Sim: {similarity:.2f}")

plt.suptitle("Photos les plus similaires", fontsize=16)
plt.tight_layout()
plt.show()


import webbrowser
import os

for idx in top_indices:
    img_path = os.path.abspath(image_paths[idx])  # Chemin absolu
    file_url = f"file://{img_path}"               # Convertir en URL locale
    print(f"Ouverture de : {file_url}")
    webbrowser.open(file_url)    
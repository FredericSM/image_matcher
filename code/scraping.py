import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# URL de la page à scraper
url = "https://ness-photo.com/folio/451/samedi-10-mai-2025.html"

# Créer un dossier pour enregistrer les images
os.makedirs("ness_photos", exist_ok=True)

# Envoyer une requête HTTP à la page
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Trouver toutes les balises <img> sur la page
img_tags = soup.find_all("img")

# Télécharger chaque image
for img in img_tags:
    img_url = img.get("src")
    if img_url:
        # Construire l'URL complète de l'image
        full_url = urljoin(url, img_url)
        # Extraire le nom du fichier
        filename = os.path.basename(full_url)
        # Télécharger l'image
        img_data = requests.get(full_url).content
        # Enregistrer l'image dans le dossier
        with open(os.path.join("ness_photos", filename), "wb") as f:
            f.write(img_data)
        print(f"Téléchargé : {filename}")

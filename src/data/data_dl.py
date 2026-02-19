import os
import zipfile
from pathlib import Path
import subprocess

# =========================
# PARAMÈTRES
# =========================
DATASET_ID = "altruistdelhite04/loan-prediction-problem-dataset"
DATASET_NAME = "loan_data"

# =========================
# CHEMINS (NOTEBOOK)
# =========================
NOTEBOOK_DIR = Path.cwd()          # dossier du notebook
PROJECT_ROOT = NOTEBOOK_DIR.parent.parent # répertoire parent

BASE_DIR = PROJECT_ROOT / "data" / "raw"
EXTRACT_PATH = BASE_DIR / DATASET_NAME

# =========================
# DOSSIERS
# =========================
BASE_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_PATH.mkdir(exist_ok=True)

# =========================
# TÉLÉCHARGEMENT (KAGGLE API)
# =========================
subprocess.run(
    [
        "kaggle",
        "datasets",
        "download",
        "-d", DATASET_ID,
        "-p", str(BASE_DIR),
        "--force"
    ],
    check=True
)

# =========================
# ZIP
# =========================
zip_files = list(BASE_DIR.glob("*.zip"))
if not zip_files:
    raise FileNotFoundError("Aucun fichier zip trouvé")

zip_path = zip_files[0]

# =========================
# EXTRACTION
# =========================
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_PATH)

print("✅ Dataset téléchargé dans le répertoire parent")
print(f"📂 Emplacement : {EXTRACT_PATH}")
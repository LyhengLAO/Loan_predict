# Loan_predict

## Description
**Loan_predict** est un projet de machine learning permettant de prédire l’acceptation ou le refus d’un prêt bancaire.  
Le projet inclut :
- le téléchargement des données,
- le nettoyage et la préparation,
- l’entraînement des modèles,
- une application web interactive avec Streamlit.

---

## Prérequis

### 1. Python
Assurez-vous d’avoir **Python 3.8+** installé.

### 2. Bibliothèques
Installez les dépendances (exemple) :
```bash
pip install -r requirements.txt
````

### 3. Kaggle
Pour continuer, il faut **avoir un token API Kaggle**.

1. Crée / récupère ton token :
   - Kaggle → **Account** → section **API** → **Create New API Token**
   - Tu obtiens un fichier `kaggle.json`

2. Place le token :
   - **Linux / macOS** : `~/.kaggle/kaggle.json`
   - **Windows** : `C:\Users\<TonNom>\.kaggle\kaggle.json`

3. Donne les permissions (Linux/macOS) :
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

## Utilisation

### 1. les données

se placer dans le fichier src/data

```bash
python data_dl.py
```

Télécharge les données depuis Kaggle.

```bash
python data_processing.py
```

Nettoie les données et les prépare pour l’entraînement.

### 2.Entraînement

se place dans src/model

```bash
python model_creation.py
```

Entraîne les modèles de machine learning et sauvegarde les résultats.

### 3.Lancer l’application web (Streamlit)
dans le fichier app

```bash
streamlit run app.py
```

## Résultat

Une interface web interactive s’ouvre dans le navigateur, permettant :

- de saisir les informations d’un client,

- d’obtenir une prédiction d’acceptation ou de refus du prêt.

import pandas as pd
from lightgbm import Booster

# Les features à fournir manuellement
features_to_input = ["HC_HB", "HC_HTE", "HC_VGM"]

# Toutes les features du modèle, avec les autres à None par défaut
all_features = [
    "HC_LEUCO", "HC_ERYTH", "HC_HB", "HC_HTE", "HC_VGM", "HC_TCMH",
    "HC_CCMH", "HC_IDR", "HC_NP", "HC_VPM", "HC_PN", "HC_IG", "HC_PE",
    "HC_PB", "HC_LY", "HC_MONO", "HC_RETIA", "HC_IRF", "HC_BLS",
    "HC_METAM", "HC_MYELON"
]

def demander_valeur(feature_name):
    while True:
        val = input(f"Entrez la valeur pour {feature_name} (ou rien pour None) : ").strip()
        if val == "":
            return None
        try:
            return float(val)
        except ValueError:
            print("Valeur invalide, veuillez entrer un nombre.")

print("=== Saisie des données patient ===")
data_input = {}

# Saisir uniquement les 3 features
for feat in all_features:
    if feat in features_to_input:
        data_input[feat] = demander_valeur(feat)
    else:
        data_input[feat] = None

X_input = pd.DataFrame([data_input])

# Charger le modèle LightGBM (Booster)
# Assurez-vous que le fichier lgbm_model_Anemie.txt est dans le même répertoire
model = Booster(model_file="lgbm_model_Anemie.txt")

# Prédiction de la probabilité
proba = model.predict(X_input)[0]

# Seuil à 0.5 pour la classe binaire
prediction = 1 if proba >= 0.5 else 0

if prediction == 1:
    print("\nDiagnostic : Anémie détectée.")
else:
    print("\nDiagnostic : Pas d'anémie détectée.")


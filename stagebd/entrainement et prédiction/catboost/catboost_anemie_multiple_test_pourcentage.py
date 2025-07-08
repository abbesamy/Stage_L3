import pandas as pd
from catboost import CatBoostClassifier

#Liste des colonnes à saisir
features = [
    "HC_LEUCO", "HC_ERYTH", "HC_HB", "HC_HTE", "HC_VGM", "HC_TCMH",
    "HC_CCMH", "HC_IDR", "HC_NP", "HC_VPM", "HC_PN", "HC_IG", "HC_PE",
    "HC_PB", "HC_LY", "HC_MONO", "HC_RETIA", "HC_IRF", "HC_BLS",
    "HC_METAM", "HC_MYELON"
]

#Liste des labels
labels = [
    "carence_fer", "carence_B12", "carence_folates", "anemie_G6PD",
    "anemie_metabolisme_nucleotides", "thalassemie", "drepanocytose",
    "trait_drepanocytaire", "anemie_hemolytique", "aplasie_medullaire",
    "anemie_maladie_chronique", "anemie_sideroblastique_secondaire",
    "autre_anemie"
]

#Fonction de saisie utilisateur
def demander_valeur(feature_name):
    while True:
        val = input(f"{feature_name} : ").strip()
        if val == "":
            return None
        try:
            return float(val)
        except ValueError:
            print("Veuillez entrer un nombre valide ou rien pour None.")

#Saisie interactive
print("=== Entrez les données biologiques du patient ===")
data_input = {feat: demander_valeur(feat) for feat in features}
X_input = pd.DataFrame([data_input])

#Chargement et prédiction avec chaque modèle
print("\n=== Prédiction (avec pourcentages) ===")
results = {}
for label in labels:
    model = CatBoostClassifier()
    model.load_model(f"catboost_model_{label}.cbm")
    probas = model.predict_proba(X_input)[0][1]
    results[label] = probas

#Affichage du diagnostic avec pourcentages
print("\n=== Résultats du diagnostic ===")
aucun = True
for label, proba in results.items():
    pourcentage = round(proba * 100, 2)
    if proba >= 0.5:
        print(f"{label.replace('_', ' ').capitalize()} détectée ({pourcentage}%).")
        aucun = False
    else:
        print(f"{label.replace('_', ' ').capitalize()} peu probable ({pourcentage}%).")

if aucun:
    print("\nAucune anémie ni carence détectée avec forte probabilité.")

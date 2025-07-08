import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# Liste des colonnes de resultats d'analyse
X_columns = [
    "HC_LEUCO", "HC_ERYTH", "HC_HB", "HC_HTE", "HC_VGM", "HC_TCMH", "HC_CCMH", "HC_IDR",
    "HC_NP", "HC_VPM", "HC_PN", "HC_IG", "HC_PE", "HC_PB", "HC_LY", "HC_MONO", 
    "HC_RETIA", "HC_IRF", "HC_BLS", "HC_METAM", "HC_MYELON"
]

# Remplissage manuel des résultats 
print("Veuillez entrer les valeurs des resultats biologiques ou appuyez sur Entree pour ignorer une valeur :")
input_values = []
for col in X_columns:
    val = input(f"{col} : ")
    if val.strip() == "":
        input_values.append(np.nan)  # Valeur manquante
    else:
        try:
            input_values.append(float(val.replace(",", ".")))
        except ValueError:
            print("==> Entree invalide")
            input_values.append(np.nan)

# Transformation en DataFrame
X_input = pd.DataFrame([input_values], columns=X_columns)

# === Chargement des modeles sauvegardés
pathologies = ["Anemie", "Leucemie", "B12", "Sane"]
models = {}
for path in pathologies:
    model = XGBClassifier()
    model.load_model(f"xgboost_model2_{path}.json")
    models[path] = model

# Prédiction de chaque pathologie avec probabilité
probabilites = {}
for path in pathologies:
    proba = models[path].predict_proba(X_input)[0][1]  # Proba de classe POSITIF
    probabilites[path] = proba

# recherche de la pathologie avec la plus haute probabilité
pathologie_max = max(probabilites, key=probabilites.get)
proba_max = probabilites[pathologie_max]

# Affichage
print("\n** Résultat principal prédictif **")
print(f"{pathologie_max} : POSITIF probable ({round(proba_max * 100, 2)}% de confiance)")

print("\n**** Détail des probabilités ****")
for path in pathologies:
    print(f"{path} : {round(probabilites[path] * 100, 2)}%")

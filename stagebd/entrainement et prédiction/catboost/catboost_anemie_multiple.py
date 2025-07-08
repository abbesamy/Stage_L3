import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from catboost import CatBoostClassifier

# 1. Charger les données
df = pd.read_csv("final_dataset1_ano_clean.csv")

# 2. Définir les groupes de codes CIM
cim_groups = {
    "carence_fer": {"D50.0", "D50.8", "D50.9"},
    "carence_B12": {"D51.0", "D51.3", "D51.8", "D51.9"},
    "carence_folates": {"D52.0", "D52.1", "D52.8", "D52.9"},
    "anemie_G6PD": {"D55.0"},
    "anemie_metabolisme_nucleotides": {"D55.9"},
    "thalassemie": {"D56.0", "D56.1", "D56.9"},
    "drepanocytose": {"D57.0", "D57.1", "D57.2"},
    "trait_drepanocytaire": {"D57.3"},
    "anemie_hemolytique": {"D59.0", "D59.1", "D59.5", "D59.8", "D59.9"},
    "aplasie_medullaire": {"D60.0", "D60.9", "D61.0", "D61.1", "D61.2", "D61.3"},
    "anemie_maladie_chronique": {"D63.0", "D63.8"},
    "anemie_sideroblastique_secondaire": {"D64.1", "D64.3"},
    "autre_anemie": {"D64.9"}
}

# 3. Ajouter une colonne pour chaque groupe de codes CIM
def has_code(cim_str, code_set):
    return int(any(code in str(cim_str).replace(",", " ").split() for code in code_set))

for label, code_set in cim_groups.items():
    df[label] = df["CIM_Codes"].apply(lambda x: has_code(x, code_set))

# 4. Définir les features et les cibles multilabel
target_labels = list(cim_groups.keys())
X = df.drop(columns=["IEP", "CIM_Codes", "CIM_Libelles"] + target_labels)
y = df[target_labels]

# 5. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# 6. Initialiser CatBoost
catboost_model = CatBoostClassifier(verbose=0, random_state=42)

# 7. Entraîner MultiOutputClassifier
multi_target_model = MultiOutputClassifier(catboost_model)
multi_target_model.fit(X_train, y_train)

# 8. Prédictions
y_pred = multi_target_model.predict(X_test)

# 9. Rapport classification par label
for i, label in enumerate(target_labels):
    print(f"=== Rapport classification pour {label} ===")
    print(classification_report(y_test[label], y_pred[:, i]))

# 10. Sauvegarde des modèles individuels
for i, col in enumerate(target_labels):
    multi_target_model.estimators_[i].save_model(f"catboost_model_{col}.cbm")

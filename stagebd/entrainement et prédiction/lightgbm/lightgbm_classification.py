import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier

def main(csv_path):
    df = pd.read_csv(csv_path)

    # 1. Définition des codes CIM
    anemia_codes = {"D50.0", "D50.8", "D50.9", "D63.0", "D64.9", "D52.0", "D52.1", "D52.8", "D52.9"}
    leukemia_codes = {"C91.0", "C91.1", "C91.2"}
    B12_codes      = {"D51.0", "D51.3", "D51.8", "D51.9"}
    sane_codes     = {"RAS"}

    # 2. Création des colonnes cibles
    df["Anemie"]   = df["CIM_Codes"].apply(lambda x: 1 if any(code in str(x).split(". ") for code in anemia_codes) else 0)
    df["Leucemie"] = df["CIM_Codes"].apply(lambda x: 1 if any(code in str(x).split(". ") for code in leukemia_codes) else 0)
    df["B12"]      = df["CIM_Codes"].apply(lambda x: 1 if any(code in str(x).split(". ") for code in B12_codes) else 0)
    df["Sane"]     = df["CIM_Codes"].apply(lambda x: 1 if any(code in str(x).split(". ") for code in sane_codes) else 0)

    # 3. Préparation des features et de la cible multilabel
    X = df.drop(columns=["IEP", "CIM_Codes", "CIM_Libelles", "Anemie", "Leucemie", "B12", "Sane"])
    y = df[["Anemie", "Leucemie", "B12", "Sane"]]

    # 4. Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y["Anemie"]
    )

    # 5. Initialisation et entraînement du modèle
    lgbm = LGBMClassifier(random_state=42, verbose=-1)
    model = MultiOutputClassifier(lgbm)
    model.fit(X_train, y_train)

    # 6. Prédictions et rapport
    y_pred = model.predict(X_test)
    for i, col in enumerate(y.columns):
        print(f"=== Rapport pour {col} ===")
        print(classification_report(y_test[col], y_pred[:, i]))

    # 7. Sauvegarde des modèles
    for i, col in enumerate(y.columns):
        model.estimators_[i].booster_.save_model(f"lgbm_model_{col}.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification multilabel avec LightGBM")
    parser.add_argument("--csv", required=True, help="chemin vers le fichier CSV anonymisé")
    args = parser.parse_args()
    main(args.csv)


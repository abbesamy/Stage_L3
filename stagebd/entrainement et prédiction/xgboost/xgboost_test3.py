import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

# Chargement les données
df = pd.read_csv("final_dataset1_ano_clean.csv")

# Definition des codes CIM
anemia_codes = {"D50.0", "D50.8", "D50.9", "D63.0", "D64.9", "D52.0", "D52.1", "D52.8", "D52.9"}
leukemia_codes = {"C91.0", "C91.1", "C91.2"}
B12_codes = {"D51.0", "D51.3", "D51.8", "D51.9"}

#  Creation les colonnes cibles
df["Anemie"] = df["CIM_Codes"].apply(lambda x: 1 if any(code in str(x).split(". ") for code in anemia_codes) else 0)
df["Leucemie"] = df["CIM_Codes"].apply(lambda x: 1 if any(code in str(x).split(". ") for code in leukemia_codes) else 0)
df["B12"] = df["CIM_Codes"].apply(lambda x: 1 if any(code in str(x).split(". ") for code in B12_codes) else 0)
df["Sane"] = df[["Anemie", "Leucemie", "B12"]].sum(axis=1).apply(lambda x: 1 if x == 0 else 0)


X = df.drop(columns=["IEP", "CIM_Codes", "CIM_Libelles", "Anemie", "Leucemie", "B12", "Sane"])
y = df[["Anemie", "Leucemie", "B12", "Sane"]]

# Reequilibrage des données
combined = pd.concat([X, y], axis=1)
classes = []
for label in y.columns:
    maj = combined[combined[label] == 0]
    mino = combined[combined[label] == 1]
    if len(mino) > 0 and len(mino) < len(maj):
        mino_upsampled = resample(mino, 
                                   replace=True,
                                   n_samples=len(maj),
                                   random_state=42)
        combined = pd.concat([maj, mino_upsampled], ignore_index=True)

# Split
X_augmented = combined[X.columns]
y_augmented = combined[y.columns]

X_train, X_test, y_train, y_test = train_test_split(
    X_augmented, y_augmented, test_size=0.2, random_state=42, stratify=y_augmented["Anemie"]
)

# Entrainement
estimators = []
for label in y.columns:
    ratio = (y_train[label] == 0).sum() / max((y_train[label] == 1).sum(), 1)
    model = XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=ratio)
    estimators.append(model)

multi_target_model = MultiOutputClassifier(estimator=estimators[0])
multi_target_model.estimators_ = estimators
multi_target_model.fit(X_train, y_train)

# Prediction et evaluation
y_pred = multi_target_model.predict(X_test)
for i, label in enumerate(y.columns):
    print(f"\n***Le Rapport classification pour {label} ***")
    print(classification_report(y_test[label], y_pred[:, i]))

# Sauvegarde
for i, col in enumerate(y.columns):
    multi_target_model.estimators_[i].save_model(f"xgboost_model2_{col}.json")

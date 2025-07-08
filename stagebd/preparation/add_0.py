import pandas as pd
import numpy as np

def remplir_zeros_based_on_norm():
    # Charger le fichier
    df = pd.read_excel("DONNEES_FUSIONNEES_AVEC_CIM.xlsx")
    
    # Identifier les colonnes avec leurs normes correspondantes
    colonnes_normes = {}
    for col in df.columns:
        if '_NORME' in col:
            base_col = col.replace('_NORME', '')
            if base_col in df.columns:
                colonnes_normes[base_col] = col
    
    # Parcourir chaque paire colonne/norme
    for col, norm_col in colonnes_normes.items():
        # Vérifier si la norme est exactement 0 (et pas juste qui contient un 0)
        if (df[norm_col].astype(str) == '0').any():
            # Remplacer UNIQUEMENT les cellules vraiment vides (NaN ou '')
            df[col] = df[col].replace('', np.nan).fillna(0)
    
    # Sauvegarder le résultat
    df.to_excel("DONNEES_FUSIONNEES_AVEC_CIM_ET_ZEROS.xlsx", index=False)
    print("Fichier mis à jour avec les zéros ajoutés uniquement où la norme = 0")

if __name__ == "__main__":
    remplir_zeros_based_on_norm()
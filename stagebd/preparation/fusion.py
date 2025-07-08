import pandas as pd
import sys

def main():
    if len(sys.argv) != 4:
        print("Utilisation : python fusion_xlsx.py fichier1.xlsx fichier2.xlsx fichier_sortie.xlsx")
        return

    fichier1 = sys.argv[1]
    fichier2 = sys.argv[2]
    fichier_sortie = sys.argv[3]

    # Chargement des fichiers tableurs
    df1 = pd.read_excel(fichier1)
    df2 = pd.read_excel(fichier2)

    # Fusion des deux
    df_total = pd.concat([df1, df2], ignore_index=True)

    # Export
    df_total.to_excel(fichier_sortie, index=False)
    print(f"Fichier fusionné sauvegardé sous : {fichier_sortie}")

if __name__ == "__main__":
    main()

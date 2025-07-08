#!/usr/bin/env python3
import pandas as pd
import csv
import re
import sys
import os
import tempfile


def convert_xlsx_to_csv(input_file, tmp_file):
    if not os.path.isfile(input_file):
        print("Error fichier introuvable : " + str(input_file))
        sys.exit(1)
    df = pd.read_excel(input_file)
    df.to_csv(tmp_file, index=False)
    print("\n\n\nConversion Excel en CSV : " + str(tmp_file))


def quote_all_cells(input_file, tmp_file):
    df = pd.read_csv(input_file, dtype=str)
    df = df.fillna("")
    df.to_csv(tmp_file, index=False, quoting=csv.QUOTE_ALL, quotechar='"')
    print("Transformation des champs entre quote : " + str(tmp_file))


def remove_norme_columns(input_file, tmp_file):
    df = pd.read_csv(input_file)
    filtered = [c for c in df.columns if "NORME" not in c.upper()]
    df[filtered].to_csv(tmp_file, index=False)
    print("Colonnes des normes supprimées : " + str(tmp_file))


def replace_commas_in_quotes(input_file, tmp_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    result = []
    inside = False
    i = 0
    while i < len(text):
        char = text[i]
        if char == '"':
            if inside and i+1 < len(text) and text[i+1] == '"':
                result.append('""')
                i += 2
                continue
            inside = not inside
            result.append('"')
            i += 1
            continue
        if inside and char == ',':
            result.append('.')
            i += 1
            continue
        result.append(char)
        i += 1
    with open(tmp_file, 'w', encoding='utf-8', newline='') as f:
        f.write(''.join(result))
    print("Virgules remplacées par des points : " + str(tmp_file))


def fix_leading_dot_values(input_file, tmp_file):
    pattern = re.compile(r'^\.(\d+)$')
    with open(input_file, 'r', encoding='utf-8', newline='') as fin, \
         open(tmp_file, 'w', encoding='utf-8', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for row in reader:
            newr = []
            for cell in row:
                m = pattern.match(cell)
                if m:
                    newr.append('0.' + m.group(1))
                else:
                    newr.append(cell)
            writer.writerow(newr)
    print("Ajout du 0 devant le . (.23 devient 0.23) : " + str(tmp_file))


def clean_csv(input_file, output_file, cols_to_check):
    df = pd.read_csv(input_file)
    initial = len(df)
    df_clean = df.dropna(subset=cols_to_check)
    final = len(df_clean)
    df_clean.to_csv(output_file, index=False)
    print("\nNettoyage effectué : " + str(initial) + " -> " + str(final) + " lignes dans " + str(output_file))



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage : {sys.argv[0]} fichier_entree.xlsx fichier_sortie.csv")
        sys.exit(1)

    input_excel = sys.argv[1]
    output_csv = sys.argv[2]
    cols_to_check = ['HC_LEUCO', 'HC_ERYTH', 'HC_HB', 'HC_HTE', 'HC_VGM', 'HC_NP']

    # Création de fichiers temporaires pour chaque étape
    steps = [convert_xlsx_to_csv, quote_all_cells, remove_norme_columns,
             replace_commas_in_quotes, fix_leading_dot_values]
    prev = input_excel
    temps = []
    for step in steps:
        fd, path = tempfile.mkstemp(suffix='.csv')
        os.close(fd)
        step(prev, path)
        temps.append(path)
        prev = path

    # Étape finale : nettoyage et export
    clean_csv(prev, output_csv, cols_to_check)

    # Suppression des fichiers temporaires
    for f in temps:
        os.remove(f)

NB : certains fichiers n'ont pas été ajoutés au depot distant pour des raisons de confidentialité


Le fichier DONNEES_FUSIONNEES.xlsx (confidentiel) correspond au fichier tableur des patients (jeu de données), le fichier NUM.xlsx (confidentiel) correspond au diagnostic des mêmes patients représentés par les codes cim des maladies ayant été diagnostiquées (pour mieux comprendre les codes cim : https://www.aideaucodage.fr/cim).

dossier preparation : 

    Pour le transformer en un jeu de donnée utilisable par les programmes de gradient boosting, il faut dans un premier temps utiliser le fichier fusion.py qui permet de fusionner le fichier tableur avec les données d'analyses de sang avec le fichier tableur des codes CIM. Il faut lui passer en paramètre le nom des deux fichiers tableurs (NUM.xlsx et DONNES_FUSIONNES.xlsx).
    Puis il faut utiliser le programme add_0.py qui va mettre la valeur 0 toutes les cellules vides n'ayant pas de norme (ce qui correspond à une caractéristique dans les analyses de sang où la norme est à 0).
    Ces deux étapes permettent d'obtenir l'équivalent du fichier DONNEES_FUSIONNES_AVEC_CIM_ET_ZEROS.xlsx. (confidentiel) On supprime également les NFS qui n'ont pas de valeur dans ces 6 colonnes :
    - HC_LEUCO
    - HC_ERYTH
    - HC_HB*
    - HC_HTE*
    - HC_VGM*
    - HC_NP

    *ces 3 colonnes permettent de travailler sur l'anémie


    Il faut ensuite utiliser le script python creation_dataset.py en mettant en paramètre le fichier tableur obtenu suite aux étapes précédentes. Ce script permet de passer d'un fichier tableur à un fichier CSV utilisable directement par les modèles de boosting, cette étape permet d'obtenir l'équivalent du fichier dataset.csv.

dossier entrainement et prediction : 

    dossier catboost :
    - catboost_anemie_multiple.py : génère les modèles pour détecter les différents type d'anémie
    - catboost_anemie_multiple_test_pourcentage.py : utilise les modèles et des entrées utilisateurs afin de détecter différents type d'anémie

    dossier lightgbm :
    - lightgbm_classification.py : génère les modèles pour détecter certaines anémie, la carence en B12, la leucémie et les patients sain
    - lightgbm_anemie_test.py : utilise les modèles pour prédire les différentes pathologies

    dossier xgboost : 
    - xgboost_test3.py : fichier de création du modèle utilisant xgboost avec entrainement, test et evaluation.
    - test_model3.py : fichier de test du modèle avec des résultats concrets.
    - xgboost_model2_Anemie.json / xgboost_model2_B12.json / xgboost_model2_Leucemie.json / xgboost_model2_Sane.json : fichiers de sauvegarde de l'apprentissage pour la détéction de l'anémie / B12 / Leucemie / patients saints.

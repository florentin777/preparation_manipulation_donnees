
import pandas as pd


def profile_dataframe(df, name='jeu de données', max_cardinality=25):
    """Affiche un rapport d'exploration standard."""

    # 1. Nom, dimensions et mémoire
    print(f"=== Profil du jeu de données : {name} ===")
    print(f"Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    memoire = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Empreinte mémoire : {memoire:.2f} Mo")

    # 2. Lignes dupliquées
    print(f"\nNombre de lignes dupliquées : {df.duplicated().sum()}")

    # 3. Profil des colonnes
    profil = pd.DataFrame({
        'type': df.dtypes.astype(str),
        'valeurs_manquantes': df.isna().sum(),
        'taux_manquants_%': (df.isna().mean() * 100).round(2),
        'valeurs_distinctes': df.nunique()
    })

    print("\n=== Profil des colonnes ===")
    display(profil)

    # 4. Statistiques numériques
    print("\n=== Statistiques des colonnes numériques ===")
    numeriques = df.select_dtypes(include='number')

    if not numeriques.empty:
        display(numeriques.describe())
    else:
        print("Aucune colonne numérique.")

    # 5. Répartition des modalités des colonnes texte
    print("\n=== Répartition des modalités des colonnes texte ===")
    colonnes_texte = df.select_dtypes(include=['object', 'string'])

    for colonne in colonnes_texte.columns:
        cardinalite = df[colonne].nunique()

        if cardinalite < max_cardinality:
            print(f"\nColonne : {colonne} ({cardinalite} modalités)")
            display(df[colonne].value_counts(dropna=False))


import json
from pathlib import Path

import pandas as pd


TEXT_COLUMNS = ['canal', 'statut', 'ville_livraison']
KEY_COLUMNS = ['id_commande', 'id_client', 'id_produit', 'id_magasin']


def parse_price(series):
    """Convertit des prix numériques ou textuels ('123,45 EUR') en nombres."""
    text = series.astype(str).str.replace(' EUR', '', regex=False)
    return pd.to_numeric(
        text.str.replace(',', '.', regex=False).str.strip(),
        errors='coerce'
    )


def normalize_text(series):
    """Uniformise la casse, les accents et les espaces des textes descriptifs."""
    text = series.astype(str).str.strip().str.lower()
    text = text.str.normalize('NFKD').str.encode('ascii', 'ignore').str.decode('utf-8')
    return text.str.replace(r'\s+', ' ', regex=True)


def clean_sales(df, drop_negative_quantities=True, max_quantity=None, verbose=True):
    """Nettoie les ventes et renvoie (DataFrame nettoyé, journal des traitements)."""
    log = {'lignes_initiales': len(df)}

    # 1. Ne pas modifier le DataFrame d'origine.
    df = df.copy()

    # 2. Convertir les prix textuels en nombres.
    before = df['prix_unitaire'].isna().sum()
    df['prix_unitaire'] = parse_price(df['prix_unitaire'])
    log['prix_invalides'] = int(df['prix_unitaire'].isna().sum() - before)

    # 3. Normaliser uniquement les colonnes descriptives, jamais les identifiants.
    for column in TEXT_COLUMNS:
        mask = df[column].notna()
        df.loc[mask, column] = normalize_text(df.loc[mask, column])
    log['colonnes_texte_normalisees'] = len(TEXT_COLUMNS)

    # 4. Supprimer les doublons stricts.
    before = len(df)
    df = df.drop_duplicates().copy()
    log['doublons_supprimes'] = before - len(df)

    # 5. Convertir les dates et supprimer celles qui sont invalides.
    df['date_commande'] = pd.to_datetime(
        df['date_commande'], format='mixed', dayfirst=True, errors='coerce'
    )
    before = len(df)
    df = df.dropna(subset=['date_commande']).copy()
    log['dates_invalides_supprimees'] = before - len(df)

    # 6. Remplacer les remises manquantes par zéro.
    df['remise_pct'] = pd.to_numeric(df['remise_pct'], errors='coerce')
    log['remises_remplacees'] = int(df['remise_pct'].isna().sum())
    df['remise_pct'] = df['remise_pct'].fillna(0)

    # 7. Traiter les quantités selon les paramètres.
    df['quantite'] = pd.to_numeric(df['quantite'], errors='coerce')
    if drop_negative_quantities:
        before = len(df)
        df = df[df['quantite'] >= 0].copy()
        log['quantites_negatives_supprimees'] = before - len(df)

    if max_quantity is not None:
        before = len(df)
        df = df[df['quantite'] <= max_quantity].copy()
        log['quantites_trop_grandes_supprimees'] = before - len(df)

    # 8. Calculer le montant après remise.
    df['montant'] = (
        df['quantite'] * df['prix_unitaire'] * (1 - df['remise_pct'] / 100)
    ).round(2)

    log['lignes_finales'] = len(df)
    if verbose:
        for key, value in log.items():
            print(f'  {key:<28} {value}')
    return df, log


def main():
    """Nettoie le CSV brut, teste l'idempotence et exporte les livrables."""
    root = Path(__file__).resolve().parent.parent
    raw_path = root / 'data' / 'raw' / 'ventes_brutes.csv'
    processed_dir = root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_sales = pd.read_csv(raw_path)
    clean_df, log = clean_sales(raw_sales)

    second_pass, _ = clean_sales(clean_df, verbose=False)
    assert second_pass.equals(clean_df), 'La fonction de nettoyage n’est pas idempotente.'
    print('OK — fonction idempotente')

    output_path = processed_dir / 'ventes_nettoyees.parquet'
    clean_df.to_parquet(output_path, index=False)

    log_path = processed_dir / 'journal_nettoyage.json'
    with log_path.open('w', encoding='utf-8') as file:
        json.dump(log, file, indent=2, ensure_ascii=False, default=str)

    print(f'Export Parquet : {output_path}')
    print(f'Journal : {log_path}')


if __name__ == '__main__':
    main()

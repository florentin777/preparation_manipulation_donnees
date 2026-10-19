import pandas as pd


def scale_minmax(series):
    """Ramène une série numérique sur l'intervalle [0, 1]."""
    return (series - series.min()) / (series.max() - series.min())


def scale_zscore(series):
    """Centre et réduit une série numérique."""
    return (series - series.mean()) / series.std()


def group_rare_categories(series, min_frequency=0.01, other_label='autre'):
    """Regroupe les catégories moins fréquentes que le seuil donné."""
    frequencies = series.value_counts(normalize=True)
    rare_labels = frequencies[frequencies < min_frequency].index
    return series.where(~series.isin(rare_labels), other_label)


def prepare_features(df, products):
    """Joint les produits, crée les indicateurs et encode les ventes."""

    # 1. Copier l'entrée et joindre le référentiel produits.
    result = df.copy()
    result = result.merge(products, on='id_produit', how='left')

    # 2. Créer les quatre variables métier de la Q5.
    # Quel est le montant réellement payé après remise ?
    result['montant_apres_remise'] = (
        result['quantite']
        * result['prix_unitaire']
        * (1 - result['remise_pct'].fillna(0) / 100)
    ).round(2)

    # Combien dépense en moyenne chaque client par vente ?
    result['panier_moyen_client'] = (
        result.groupby('id_client')['montant_apres_remise'].transform('mean')
    )

    # Quelles commandes ont été annulées ou retournées ?
    result['commande_non_aboutie'] = (
        result['statut'].isin(['annule', 'retourne']).astype(int)
    )

    # Le prix est-il supérieur ou inférieur au prix médian de la catégorie ?
    result['ecart_prix_median'] = (
        result['prix_unitaire']
        - result.groupby('categorie')['prix_unitaire'].transform('median')
    )

    # 3. Encoder les modalités avec one-hot encoding.
    one_hot = pd.get_dummies(
        result[['canal', 'statut', 'categorie']],
        dtype=int,
    )

    sous_cat_grouped = group_rare_categories(
        result['sous_categorie'], min_frequency=0.02
    )
    one_hot_sous_cat = pd.get_dummies(
        sous_cat_grouped, prefix='sous_categorie', dtype=int
    )

    result = pd.concat(
        [
            result.drop(columns=['canal', 'statut', 'categorie', 'sous_categorie']),
            one_hot,
            one_hot_sous_cat,
        ],
        axis=1,
    )

    # 4. Centrer et réduire le montant avec la fonction de la Q1.
    result['montant_zscore'] = scale_zscore(result['montant'])

    return result

# %% [markdown]
# # 5-2 - Topic modeling avec BERTopic (analyse thématique)

# %%
# TODO : modifier le chapo si besoin
# TODO : renvoyer le 85 quelque part !
# TODO : décider de la réduction des outliers
# TODO : décider de la stratégie de réduction (sur quel niveau)
# TODO : + cohérence (ie pas le faire aléatoirement sur chaque niveau et que aboutisse a des regroupements différents)
# TODO : voudrait dire possible le faire à bas niveau puis décider de mapper de la vers au dessus
# n= ne plus partir pour les n1/n2 du topic de base mais des modifs topic dans n3, etc.

# TODO: revoir regroupement des topics
# TODO: revoir Topic distribution
# TODO: aviser alternatives representation sur le nom des topics ?

# TODO: check ça : "2026-11-27 13:59:28,732 - BERTopic - WARNING: Using a custom list of topic assignments may lead to errors if topic reduction techniques are used afterwards. Make sure that manually assigning topics is the last step in the pipeline.Note that topic embeddings will also be created through weightedc-TF-IDF embeddings instead of centroid embeddings."

# %% [markdown]
# Lit les embeddings issus de 5_1 et construit un topic model BERTopic à partir
# des embeddings pré-calculés et des paramètres fixés (UMAP, HDBSCAN et c-TF-IDF
# standards, stopwords français). Aucune réduction automatique du nombre de topics
# n'est appliquée, afin de privilégier une lecture manuelle fine.
#
# Explore les topics (mots-clés, hiérarchie, similarités), puis les fusionne sur la base
# des similarités et de la lecture des textes pour obtenir 3 niveaux de classification.
# Analyse les résultats (topics par affiliation, dynamique temporelle).
#
# Écrit le modèle BERTopic, les figures HTML et l'export final des documents avec topic.
#
# NOTE : les calculs UMAP et BERTopic n'étant pas strictement reproductibles
# entre environnements (uv/conda car BLAS/LAPACK diffèrent), l'export complet
# du modèle BERTopic est fourni, de même qu'une réduction UMAP pré-calculée.
# Le calcul initial a été réalisé dans un environnement conda
# (cf. bertopic_repu_env.yml).

# %%
# =============================================================
# Imports et définition des constantes / paths
# =============================================================

import hashlib
import runpy
import copy
import json
from pathlib import Path
import pandas as pd
import numpy as np
from umap import UMAP  # attention umap-learn pour l'installation
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from stopwordsiso import stopwords
# from bertopic.dimensionality import BaseDimensionalityReduction
# from bertopic.vectorizers import ClassTfidfTransformer

DATASET_DIR = Path(
    "../models/embeddings/"
)  # Chemin de la copie fournie des embeddings pour reproductibilité. Remplacer au besoin si recalcul.

MODEL_DIR = Path("../models/bertopic/qwen3-emb-8B-bertopic-base")

MAPPING_PATH = Path(
    "../models/bertopic/regroupements_topics.py"
)  # path liste des regroupements de topics

# metadonnées des embeddings pour vérifier correspondance avec 5-1
META_PATH = DATASET_DIR / "embeddings_meta.json"

# hash du modèle bertopic pour vérifier correspondance
MODEL_HASH_PATH = Path("../models/bertopic/qwen3-emb-8B-bertopic-base_texts_sha256.txt")

# pré-calcul du UMAP si besoin
# UMAP_PATH = Path("../models/bertopic/umap_nn15_nc5_md0.0_cosine_seed42.npy")

RANDOM_SEED = 42


# =============================================================
# Création du modèle BERTopic avec les embeddings pré-calculés
# ou chargement du modèle pré-entraîné si disponible
# =============================================================

# ----- Charger docs et precomputed embeddings -----

df = pd.read_csv(DATASET_DIR / "dataset_repu.csv")
docs = df["texte"].tolist()

precomputed_embeddings = np.load(DATASET_DIR / "embeddings.npy")  # Shape : N * 4096

# reduced = np.load(UMAP_PATH)  # si besoin utiliser le UMAP pré-calculé

# ----- Préparer les colonnes utiles pour l'analyse -----
df["dateSeance_ts"] = pd.to_datetime(
    df["dateSeance"].astype(str),
    format="%Y%m%d%H%M%S%f",
)

df["dateSeance_day"] = df["dateSeance_ts"].dt.normalize()

df["affiliation_et_gouv"] = df["affiliation_et_gouv"].fillna("Inconnu")

timestamps = df["dateSeance_ts"].tolist()
affiliations = df["affiliation_et_gouv"].tolist()


# ----- Calculer le hash des textes pour vérifier correspondance -----
def hash_texts(texts) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(str(t).encode("utf-8"))
        h.update(b"\x00")  # séparateur, évite que ["ab","c"] == ["a","bc"]
    return h.hexdigest()


current_hash = hash_texts(df["texte"])

# ----- Vérification de la correspondance avec 5-1 -----

if not META_PATH.exists():
    raise FileNotFoundError(f"Métadonnées des embeddings introuvables : {META_PATH}")

with META_PATH.open("r", encoding="utf-8") as f:
    meta = json.load(f)

# Vérification du nombre de textes et d'embeddings
assert len(df) == len(precomputed_embeddings), (
    "Le dataset et les embeddings n'ont pas le même nombre de lignes."
)
# Vérification de la forme des embeddings par rapport au métadonnées
assert list(precomputed_embeddings.shape) == meta["embedding_shape"]

# Vérification que les textes et leur ordre correspondent à 5-1
assert current_hash == meta["texts_sha256"], (
    "Les textes ou leur ordre ne correspondent pas aux embeddings."
)

# ----- Modèle BERTopic de base : chargement ou création -----

if MODEL_DIR.exists():
    print(f"Chargement du modèle BERTopic de base : {MODEL_DIR}")

    topic_model = BERTopic.load(MODEL_DIR)

    assert len(topic_model.topics_) == len(df), (
        "Le modèle BERTopic ne correspond pas au dataset (nombre de lignes). "
        "Supprimer MODEL_DIR pour le recalculer."
    )

    if not MODEL_HASH_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_HASH_PATH} introuvable : alignement non vérifiable."
        )
    assert MODEL_HASH_PATH.read_text() == current_hash, (
        "Le modèle BERTopic a été entraîné sur d'autres textes. "
        "Supprimer MODEL_DIR pour le recalculer."
    )

else:
    print("Modèle de base absent : création en cours...")
    # ----- Paramètres BERTopic -----

    # UMAP
    # (aussi possible utiliser umap pré-calculé si besoin)

    umap_model = UMAP(
        n_neighbors=15,  # default=15
        n_components=5,  # default=5
        min_dist=0.0,  # default=0.0
        metric="cosine",  # default="cosine"
        random_state=RANDOM_SEED,  # pour reproductibilité
    )

    # HDBSCAN
    # NOTE : version standard car choix fait de ne pas chercher
    # à réduire automatiquement le nombre de topics et privilégier
    # une lecture manuelle fine pour réaliser les choix de fusion.

    # VECTORIZER
    french_stopwords = list(stopwords("fr"))
    vectorizer_model = CountVectorizer(stop_words=french_stopwords)

    # CTFIDF
    # NOTE : version standard car choix fait de privilégier une
    # lecture fine des topics pour déterminer leur nom et contenu.

    # ----- Créer et ajuster le modèle -----

    # initialiser le modèle avec paramètres
    topic_model = BERTopic(
        language="french",  # affecte l'affichage des caractères
        embedding_model=None,  # pas d'embedding_model puisque pré-calculés
        umap_model=umap_model,  # ou si besoin utiliser le pré-calcul umap (reduced)
        vectorizer_model=vectorizer_model,  # enlever les stopwords après embbedings/clustering
    )

    # Fit + transform en utilisant les precomputed_embeddings
    topics, probs = topic_model.fit_transform(
        documents=docs, embeddings=precomputed_embeddings
    )

    # Sauvegarde du modèle de base
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

    topic_model.save(
        MODEL_DIR,
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=False,
    )

    MODEL_HASH_PATH.write_text(current_hash)

    print(f"Modèle sauvegardé : {MODEL_DIR}")


# %%
# # Les principales fonctions à tester pour avoir un aperçu simple :

# topic_model.get_topic_info()
# topic_model.visualize_barchart()
# topic_model.visualize_topics()
# topic_model.visualize_hierarchy()
# topic_model.visualize_documents(df["Texte_clean"].to_list())

# Si modification du libellé et qu'on le veut passer l'argument :
# custom_labels=True


# %%
# ===============================================
# Fusion des topics
# ===============================================

# ----- Regroupements : chargement + contrôles -----

REGROUPEMENTS = runpy.run_path(MAPPING_PATH)[
    "REGROUPEMENTS"
]  # lecture du dictionnaire (sans les commentaires)

info = topic_model.get_topic_info().set_index("Topic")

# NOTE : renvoyer des topics vers -1 n'est pas géré (volontairement).
# La validation ci-dessous exclut -1 des groupes. Si ce besoin apparaît :
# - mettre -1 EN PREMIER dans le groupe : merge_topics prend topic_group[0]
#   comme identifiant résultant (sinon les outliers sont absorbés par un topic) ;
# - donner à ce groupe le même libellé que mapping[-1] ("-1_Outliers"), sinon
#   l'assertion d'unicité de corr échoue (deux libellés pour l'id -1) ;
# - adapter la validation pour accepter -1 dans un groupe ;
# - documenter que -1 regroupe alors outliers d'origine et topics renvoyés
#   (topic_id_base conserve la distinction, pas topic_id_niveauN).

# Chaque topic du modèle (hors -1) doit apparaître exactement une fois à chaque niveau.
ids_modele = set(info.index) - {-1}
for niveau, groupes in REGROUPEMENTS.items():
    ids = [topic_id for topic_ids in groupes.values() for topic_id in topic_ids]
    assert len(ids) == len(set(ids)), (
        f"{niveau} : un topic apparaît dans plusieurs groupes"
    )
    assert set(ids) == ids_modele, (
        f"{niveau} : écart entre le modèle et les regroupements. "
        f"Absents des regroupements : {sorted(ids_modele - set(ids))}. "
        f"Absents du modèle : {sorted(set(ids) - ids_modele)}"
    )

# Correspondance topic de base -> libellé pour chacun des trois niveaux.
LIBELLES_TOPICS = {
    niveau: {
        int(topic_id): libelle
        for libelle, topic_ids in groupes.items()
        for topic_id in topic_ids
    }
    for niveau, groupes in REGROUPEMENTS.items()
}
# NOTE
# Injecter un libellé aux outliers (-1) : ne modifie pas le comportement
# des topic_model mais les labels renvoyés pour ajout info dans les datasets
# (et pas avoir des NaN silencieux). Selon choix plus bas (enlever ou
# non de corr), affectera ou non les customs names des TM pour -1
for mapping in LIBELLES_TOPICS.values():
    mapping[-1] = (
        "-1_Outliers"  # NOTE : choix - Si on ne fixe rien ici renverra des NaN dans df
    )

# Topics du modèle de base, qu'il soit chargé ou nouvellement entraîné.
# (ie topic_model.topics_ vs topics)
df["topic_id_base"] = topic_model.topics_

# Pour les topics 1, 2 et 3, chaque niveau reçoit le libellé du groupe
# attribué manuellement.
# Le topic -1 renvoie l'info fixée au dessus dans mapping[-1] = XXXXXX
for n in (1, 2, 3):
    niveau = f"niveau{n}"
    df[niveau] = df["topic_id_base"].map(LIBELLES_TOPICS[niveau])

# Table indexée par topic de base, utilisée pour l'export.
ref = pd.DataFrame(
    {f"niveau{n}": pd.Series(LIBELLES_TOPICS[f"niveau{n}"]) for n in (1, 2, 3)}
)
ref.index.name = "Topic"

ref.to_csv(
    "../models/bertopic/ref_topics.csv", index=True
)  # True pour trace de l'index topic


# ----- Fusion simple au niveau N -----


def merge_topics_niveau(n):
    """
    Fusionne les topics au niveau N directement,
    en utilisant les regroupements définis dans REGROUPEMENTS.
    """
    m = copy.deepcopy(topic_model)  # copier le TM pour ne pas modifier l'original
    # Regroupements manuels du niveau demandé
    niveau = f"niveau{n}"  # intercepter le niveau demandé
    groupes = REGROUPEMENTS[niveau]  # identifier les groupes du niveau demandé

    # Extraire uniquement les groupes avec plus d'un topic
    # (pas besoin de fusionner si seul topic du groupe)
    topics_to_merge = [
        topic_ids for topic_ids in groupes.values() if len(topic_ids) > 1
    ]

    # Fusion des topics
    m.merge_topics(docs, topics_to_merge)

    # Après fusion, les IDs ont pu changer.
    # On récupère donc les nouveaux IDs depuis m.topics_.
    corr = pd.DataFrame(
        {
            "id": m.topics_,
            "lib": df[f"niveau{n}"].values,
        }
    ).drop_duplicates()

    # Vérifie qu'un topic fusionné correspond à un seul label manuel
    # NOTE : décommenter corr = blablabla `-1` pour exclure les outliers
    # de la correspondance des labels et donc ne pas y toucher dans le
    # bertopic topic_model_nX
    # corr = corr[corr["id"] != -1] # selon ce que l'on veut
    assert corr["id"].is_unique, (
        f"Un topic fusionné correspond à plusieurs labels au niveau {n}"
    )

    # Remplace les noms automatiques par les labels manuels
    m.set_topic_labels(dict(zip(corr["id"], corr["lib"])))

    # Compteurs
    merged_count = len(topics_to_merge)
    unchanged_count = len([g for g in groupes.values() if len(g) == 1])
    print(f"Niveau {n} : {merged_count} fusions, {unchanged_count} inchangés")

    return m


# Appliquer
topic_model_n3 = merge_topics_niveau(3)
topic_model_n2 = merge_topics_niveau(2)
topic_model_n1 = merge_topics_niveau(1)

# %% [markdown]
# # Tests

# %%
# # Les principales fonctions à tester pour avoir un aperçu simple :

# topic_model.get_topic_info()
# topic_model.visualize_barchart()
# topic_model.visualize_topics()
# topic_model.visualize_hierarchy()
# topic_model.visualize_documents(df["Texte_clean"].to_list())

# Si modification du libellé et qu'on le veut passer l'argument :
# custom_labels=True


# %% [markdown]
# ## Topics

# %%
# topic_model.get_topic_info()

# %% [markdown]
# ## Barchart

# %%
# topic_model.visualize_barchart(
#     # n_words=10,  # Select the number of words to display per topic
#     # topics = [0,1,2,3,4], # Select specific topics to display
#     top_n_topics=8,  # Select the first n topics to display
#     # height = 300, # Adjust the height of the plot
#     # width = 800 # Adjust the width of the plot
# )

# %% [markdown]
# ## Visualisation documents / topics

# %%
# # TODO : si utilisé pour finaliser une carte, figer une projection umap 2D pour reproductibilité ?
# topic_model.visualize_documents(
#     docs=docs,
#     embeddings=precomputed_embeddings,
#     hide_annotations=True,  # better readability
#     # topics=[0, 1, 2, 3],  # Select topics to highlight
#     # height = 300, # Adjust the height of the plot
#     # width = 800 # Adjust the width of the plot
# )

# %% [markdown]
# ## Hiérarchie des topics

# %%
# topic_model.visualize_hierarchy()

# %%
# hierarchical_topics = topic_model.hierarchical_topics(docs)
# print(topic_model.get_topic_tree(hierarchical_topics))

# %%
# topic_model.visualize_heatmap()

# %% [markdown]
# # Vrac

# %% [markdown]
# Comparaison de modèles :
#
# https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#finding-similar-topics-between-models

# %% [markdown]
# ## Dynamic topic model

# %% [markdown]
# * `global_tuning`
#   * Whether to average the topic representation of a topic at time *t* with its global topic representation
# * `evolution_tuning`
#   * Whether to average the topic representation of a topic at time *t* with the topic representation of that topic at time *t-1*
# * `nr_bins`
#   * The number of bins to put our timestamps into. It is computationally inefficient to extract the topics at thousands of different timestamps. Therefore, it is advised to keep this value below 20.
#

# %%
# # NOTE : timestamps = df["dateSeance_day"].tolist()
# # Analyse temporelle
# topics_over_time = topic_model_n1.topics_over_time(
#     docs=docs,
#     timestamps=timestamps,
#     global_tuning=True,
#     evolution_tuning=True,
#     nr_bins=20,
# )

# fig_dynamic_topic = topic_model_n1.visualize_topics_over_time(
#     topics_over_time,
#     top_n_topics=10,
# )

# fig_dynamic_topic

# %%
# fig_dynamic_topic.write_html("../reports/figures/dynamic_topics.html")

# %% [markdown]
# ### Topics per class

# %%
# # ATTENTION, PLANTAIT À CAUSE DES NA
# # cf ajout précédent d'une classe "Inconnu" pour les NA dans affiliation_et_gouv
# topics_per_class = topic_model.topics_per_class(docs, classes=affiliations)

# fig_topics_per_class = topic_model.visualize_topics_per_class(
#     topics_per_class,
#     # topics=[0, 1, 2, 3],  # choix spécifique de topics à visualiser
#     top_n_topics=11,  # choix des X principaux topics
#     custom_labels=True,
# )
# fig_topics_per_class

# %%
# fig_topics_per_class.write_html("../reports/figures/topics_per_class.html")

# %%
# df["topic"] = topics
# df["probability"] = probs

# # joindre le nom/label du topic plutôt que juste l'ID
# topic_info = topic_model.get_topic_info()[["Topic", "Name"]]
# df = df.merge(topic_info, left_on="topic", right_on="Topic", how="left")

# # df.to_csv("../models/bertopic/export_final.csv", index=False)

# %% [markdown]
# ## Sauvegarde

# %%
# Sauvegarde info topics + sortie df
# TODO : path et names à affiner
dossier = Path("../models/bertopic/topic_info")
dossier.mkdir(parents=True, exist_ok=True)

modeles = {
    "base": topic_model,
    "niveau1": topic_model_n1,
    "niveau2": topic_model_n2,
    "niveau3": topic_model_n3,
}

for nom, modele in modeles.items():
    # écrire csv info
    # (si il est rechargé et pas entrainé lors de la session
    # le topic_baase n'aura pas de Representative_Docs)
    modele.get_topic_info().to_csv(dossier / f"topics_{nom}.csv", index=False)
    # ajouter id par niveau dans df (sauf base qu'on a déjà)
    if nom != "base":
        df[f"topic_id_{nom}"] = modele.topics_

df.to_csv("../models/bertopic/df_with_topics.csv", index=False)  # sortir le full csv

# %% [markdown]
# ## Tests reduce outliers

# %%
# https://maartengr.github.io/BERTopic/getting_started/outlier_reduction/outlier_reduction.html#chain-strategies
# https://github.com/MaartenGr/BERTopic/blob/master/docs/getting_started/outlier_reduction/outlier_reduction.md
# https://css-polytechnique.github.io/css-ipp-materials/pages/bertopic-tutorial.html#fnref11
# https://css-polytechnique.github.io/css-ipp-materials/pages/techy-notes.html#reduce-outliers-strategies

# # Train your BERTopic model
# topic_model = BERTopic()
# topics, probs = topic_model.fit_transform(docs)

# # Reduce outliers using the `embeddings` strategy
# new_topics = topic_model.reduce_outliers(
#     docs, topics, embeddings=precomputed_embeddings, strategy="embeddings"
# )

# %%
m = topic_model_n1  # pour tester entre les différents niveaux
ids = sorted(set(m.topics_) - {-1})
assert ids == list(range(len(ids)))  # ids contigus
print(
    "shape topic embeddings :",
    m.topic_embeddings_.shape,
    " -> [attendu : (n_topics + 1, 4096)]",
)
# permet de tester trheshold = l'effet du seuil sur le nombre de textes déplacés hors outliers
for t in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    nt = m.reduce_outliers(
        docs,
        m.topics_,
        strategy="embeddings",
        embeddings=precomputed_embeddings,
        threshold=t,
    )
    moved = sum(1 for o, n in zip(m.topics_, nt) if o == -1 and n != -1)
    print(t, moved)

# %%
# new_topics = reduce_outliers(topic_model, docs, topics, probabilities=probs,
#                              threshold=0.05, strategy="probabilities")
# topic_model.update_topics(docs, topics=new_topics)

# Une fois le seuil choisi :
m = topic_model_n1  # pour tester entre les différents niveaux
t = 0.6  # TODO: valeur à choisir après tests pour le treshold
nt = m.reduce_outliers(
    docs,
    m.topics_,
    strategy="embeddings",
    embeddings=precomputed_embeddings,
    threshold=t,
)
moved = sum(1 for o, n in zip(m.topics_, nt) if o == -1 and n != -1)
print(t, moved)

old, new = np.array(m.topics_), np.array(nt)
moved = (old == -1) & (new != -1)
noms = m.get_topic_info().set_index("Topic")["CustomName"]
print(pd.Series(new[moved]).value_counts().rename(index=noms))  # destinations
df.loc[moved, "texte"].sample(10)  # lire des cas

# %%
# Appliquer sur une copie
# dépendra du nt fixé juste au dessus (qui dépendra lui même du seuil choisi)
modele_update = copy.deepcopy(m)
modele_update.update_topics(
    docs, topics=nt, vectorizer_model=m.vectorizer_model, ctfidf_model=m.ctfidf_model
)

# %% [markdown]
# ## pour matthias et explo

# %%
# précalculer les deux dimension pour stabiliser la cart et gagner de temps sur les itérations

# We reduce our embeddings to 2D as it will allows us to quickly iterate later on
reduced_embeddings = UMAP(
    n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine", random_state=RANDOM_SEED
).fit_transform(precomputed_embeddings)

# %%
# ici tu mets le modele de référence
topic_model_n1.visualize_documents(
    docs,
    reduced_embeddings=reduced_embeddings,
    # embeddings=precomputed_embeddings, # sinon repartir direct mais bougera un peu à chaque fois
    hide_document_hover=True,
    hide_annotations=True,
)

# %%
# Et ici le modèle actualisé pour comparer
modele_update.visualize_documents(
    docs,
    reduced_embeddings=reduced_embeddings,
    # embeddings=precomputed_embeddings, # sinon repartir direct mais bougera un peu à chaque fois
    hide_document_hover=True,
    hide_annotations=True,
)

# %%
# et ici tu peux comparer ou vont les textes pour voir l'effet

noms = m.get_topic_info().set_index("Topic")["CustomName"]

res = df.loc[moved, ["texte"]].copy()
res["topic_dest"] = new[moved]
res["label_dest"] = res["topic_dest"].map(noms)
# res["texte"] = res["texte"].str[:300]     # extrait, les discours sont longs

res.to_csv(
    "../models/bertopic/topic_info/df_outliers_moved.csv", index=False
)  # sortir le full csv
res

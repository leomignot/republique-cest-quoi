# %% [markdown]
# # 5-2 - Topic modeling avec BERTopic (analyse thématique)

# %%
# TODO : modifier le chapo si besoin

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
import runpy
import copy
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
    "../models/embeddings/qwen3-8b_embeddings_reproductible"
)  # Chemin de la copie fournie des embeddings pour reproductibilité. Remplacer au besoin si recalcul.

MODEL_DIR = Path("../models/bertopic/qwen3-emb-8B-bertopic-base")

MAPPING_PATH = Path(
    "../files/regroupements_topics.py"
)  # path liste des regroupements de topics

# UMAP_PATH = Path("../models/bertopic/umap_nn15_nc5_md0.0_cosine_seed42.npy") # si besoin

RANDOM_SEED = 42


##############################################################
# Création du modèle BERTopic avec les embeddings pré-calculés
# ou chargement du modèle pré-entraîné si disponible
##############################################################

# ----- Charger docs et precomputed embeddings -----

df = pd.read_csv(DATASET_DIR / "dataset_repu.csv")
docs = df["texte"].tolist()

precomputed_embeddings = np.load(DATASET_DIR / "embeddings.npy")  # Shape : N * 4096
assert len(df) == len(precomputed_embeddings)

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


# ----- Modèle BERTopic de base : chargement ou création -----
if MODEL_DIR.exists():
    print(f"Chargement du modèle BERTopic de base : {MODEL_DIR}")

    topic_model = BERTopic.load(MODEL_DIR)

else:
    print("Modèle de base absent : création en cours...")
    # ----- Paramètres BERTopic-----

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
###################################################
# Fusion des topics
###################################################

# ----- Regroupements : chargement + contrôles -----


REGROUPEMENTS = runpy.run_path(MAPPING_PATH)[
    "REGROUPEMENTS"
]  # lecture du dictionnaire (sans les commentaires)

info = topic_model.get_topic_info().set_index("Topic")

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

# test sagouin
df["topic_base"] = (
    topic_model.topics_
)  # pas la variable `topics` de fit_transform (avant réaffectation)

# Le topic -1 reste comme -1 ; les autres topics reprennent
# directement le libellé du groupe manuel.
for n in (1, 2, 3):
    niveau = f"niveau{n}"
    # TODO: aviser pour les hors topics, mais pas de fillna sur -1 si on veut garder trace
    df[f"n{n}"] = df["topic_base"].map(LIBELLES_TOPICS[niveau])
    df[niveau] = df[f"n{n}"]

# Table indexée par topic de base, utilisée pour l'export.
ref = pd.DataFrame(
    {f"n{n}": pd.Series(LIBELLES_TOPICS[f"niveau{n}"]) for n in (1, 2, 3)}
)
ref.index.name = "Topic"

ref.to_csv("../files/ref_topics.csv", index=True)  # True pour trace de l'index topic

# %%
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
topic_model_n3.get_topic_info()

# %% [markdown]
# ## Barchart

# %%
topic_model_n3.visualize_barchart(
    # n_words=10,  # Select the number of words to display per topic
    # topics = [0,1,2,3,4], # Select specific topics to display
    top_n_topics=8,  # Select the first n topics to display
    # height = 300, # Adjust the height of the plot
    # width = 800 # Adjust the width of the plot
)

# %% [markdown]
# ## Visualisation documents / topics

# %%
# TODO : si utilisé pour finaliser une carte, figer une projection umap 2D pour reproductibilité ?
topic_model.visualize_documents(
    docs=docs,
    embeddings=precomputed_embeddings,
    hide_annotations=True,  # better readability
    # topics=[0, 1, 2, 3],  # Select topics to highlight
    # height = 300, # Adjust the height of the plot
    # width = 800 # Adjust the width of the plot
)

# %%
topic_model_n3.visualize_topics()

# %% [markdown]
# ## Hiérarchie des topics

# %%
topic_model_n2.visualize_hierarchy()

# %%
hierarchical_topics = topic_model.hierarchical_topics(docs)
print(topic_model.get_topic_tree(hierarchical_topics))

# %%
topic_model_n3.visualize_heatmap()

# %% [markdown]
# ## Sauvegarder

# %%
# topic_model.save(
#     path="../models/bertopic/qwen3-emb-8B-bertopic-base",
#     serialization="safetensors",
#     save_ctfidf=True,
# )

# %%
# topic_model = BERTopic.load("../models/bertopic/qwen3-emb-8B-bertopic-base")

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
# NOTE : timestamps = df["dateSeance_day"].tolist()
# Analyse temporelle
topics_over_time = topic_model_n1.topics_over_time(
    docs=docs,
    timestamps=timestamps,
    global_tuning=True,
    evolution_tuning=True,
    nr_bins=20,
)

fig_dynamic_topic = topic_model_n1.visualize_topics_over_time(
    topics_over_time,
    top_n_topics=10,
)

fig_dynamic_topic

# %%
# fig_dynamic_topic.write_html("../reports/figures/dynamic_topics.html")

# %% [markdown]
# ## Topic reduction

# %%
# topics_to_merge = [[X, Y],
#                    [Z, W]]
# topic_model.merge_topics(df["Texte_clean"], topics_to_merge)

# %%
# DONT : # topic_model.reduce_topics(df["Texte_clean"], nr_topics=40) # DONT, IT NUKES THE TOPICS IN MODEL
# # Access updated topics
# topics = topic_model.topics_

# %% [markdown]
# ### Topics per class

# %%
# ATTENTION, PLANTAIT À CAUSE DES NA
# cf ajout précédent d'une classe "Inconnu" pour les NA dans affiliation_et_gouv
topics_per_class = topic_model_n1.topics_per_class(docs, classes=affiliations)

fig_topics_per_class = topic_model_n1.visualize_topics_per_class(
    topics_per_class,
    # topics=[0, 1, 2, 3],  # choix spécifique de topics à visualiser
    top_n_topics=11,  # choix des X principaux topics
    custom_labels=True,
)
fig_topics_per_class

# %%
topic_model_n1.get_topic_info()

# %%
# fig_topics_per_class.write_html("../reports/figures/topics_per_class.html")

# %%
# TODO: revoir regroupement des topics
# TODO: revoir Topic distribution
# TODO: aviser alternatives representation sur le nom des topics ?

# %%
# df["topic"] = topics
# df["probability"] = probs

# # joindre le nom/label du topic plutôt que juste l'ID
# topic_info = topic_model.get_topic_info()[["Topic", "Name"]]
# df = df.merge(topic_info, left_on="topic", right_on="Topic", how="left")

# # df = df.drop(columns=["embedding"], errors="ignore")
# # df.to_csv("../models/embeddings/export_final.csv", index=False)

# %% [markdown]
# # 5-2 - Topic modeling avec BERTopic (analyse thématique)

# %%
# TODO : se passer de dataset ?
# TODO : rédiger le chapo

# %% [markdown]
# des embeddings de texte via un modèle Qwen3-Embedding-8B servi par Ollama,
#
# Lit les embeddings issus de 5_1 et construit un topic model BERTopic à partir
# des embeddings pré-calculés et des paramètres fixés (UMAP, HDBSCAN et c-TF-IDF
# standards, stopwords français). Pas de réduction automatique du nombre de topics,
# afin de privilégier une lecture manuelle fine.
#
# NOTE : UMAP n'étant pas strictement reproductible entre environnements
# (uv/conda car BLAS/LAPACK diffèrent), la réduction UMAP pré-calculée est
# également fournie. N.B : le calcul initial umap a été réalisé dans un
# environnement conda (cf bertopic_repu_env.yml)
#
# Explore les topics (mots-clés, hiérarchie, similarités), puis les fusionne sur la base
# des similarités et de la lecture des textes pour obtenir 3 niveaux de classification.
# Analyse les résultats (topics par affiliation, dynamique temporelle).
#
# Écrit le modèle BERTopic, les figures HTML et l'export final des documents avec topic.

# %%
from datasets import load_from_disk
import pandas as pd
import numpy as np
from umap import UMAP  # attention umap-learn pour l'installation
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.feature_extraction.text import CountVectorizer
from stopwordsiso import stopwords
from pathlib import Path
# from bertopic.vectorizers import ClassTfidfTransformer


DATASET_DIR = Path(
    "../models/embeddings/qwen3-8b_embeddings_2026-07-21/dataset_with_embeddings"
)  # Chemin de la copie fournie des embeddings pour reproductibilité. Remplacer au besoin si recalcul.
UMAP_PATH = Path("../models/bertopic/umap_nn15_nc5_md0.0_cosine_seed42.npy")

RANDOM_SEED = 42

# True  : charge la réduction UMAP fournie (reproductible entre environnements)
# False : recalcule UMAP (résultats pouvant différer légèrement selon l'environnement)
USE_PRECOMPUTED_UMAP = True
# True : recalcule UMAP et écrase UMAP_PATH (à utiliser pour régénérer le fichier fourni)
SAVE_UMAP = False

# ----- Charger docs et precomputed embeddings -----

# dataset = load_from_disk(DATASET_DIR)
# docs = dataset[
#     "texte"
# ]  # NOTE: list pas obligatoire mais assure ? list(dataset["texte"])
# precomputed_embeddings = np.array(
#     dataset["embedding"]
# )  # Shape : XXX * 4096 pour qwen3-embedding:8b

# ----- Charger docs et precomputed embeddings -----

dataset = load_from_disk(DATASET_DIR)
df = dataset.to_pandas()

df["dateSeance_ts"] = pd.to_datetime(
    df["dateSeance"].astype(str),
    format="%Y%m%d%H%M%S%f",
)

df["dateSeance_day"] = df["dateSeance_ts"].dt.normalize()

df["affiliation_et_gouv"] = df["affiliation_et_gouv"].fillna("Inconnu")

docs = df["texte"].tolist()
precomputed_embeddings = np.array(df["embedding"].tolist())  # Shape : N * 4096
timestamps = df["dateSeance_ts"].tolist()
affiliations = df["affiliation_et_gouv"].tolist()


# ----- Paramètres BERTopic-----

# UMAP
# Pré-calcul : l'aléatoire d'UMAP n'est pas parfaitement reproductible entre
# environnements (uv vs conda ; BLAS/LAPACK : Accelerate vs OpenBLAS),
# même avec random_state fixé. Cela peut changer les topics.
# On fige donc la sortie UMAP (N * 5) dans un .npy fourni avec le projet.

umap_params = dict(
    n_neighbors=15,  # default=15
    n_components=5,  # default=5
    min_dist=0.0,  # default=0.0
    metric="cosine",  # default="cosine"
    random_state=RANDOM_SEED,  # pour reproductibilité
)

if SAVE_UMAP or not USE_PRECOMPUTED_UMAP:
    # Calcul (ou recalcul) de la réduction sur les embeddings d'origine
    reduced = UMAP(**umap_params).fit_transform(precomputed_embeddings)
    if SAVE_UMAP:
        np.save(UMAP_PATH, reduced)
else:
    reduced = np.load(UMAP_PATH)  # utiliser le UMAP pré-calculé

assert reduced.shape[0] == len(docs), "UMAP pré-calculé incohérent avec le dataset"

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
    umap_model=BaseDimensionalityReduction(),  # réduction umap déjà faite (reduced)
    vectorizer_model=vectorizer_model,  # enlever les stopwords après embbedings/clustering
    # ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True) # nope mais pour test
)

# Fit + transform en utilisant les reduced dimensions (pré-calculées sur les precomputed_embeddings)
topics, probs = topic_model.fit_transform(documents=docs, embeddings=reduced)

# %% [markdown]
# # Tests

# %%
# # Les principales fonctions à tester pour avoir un aperçu simple :

# topic_model.get_topic_info()
# topic_model.visualize_barchart()
# topic_model.visualize_topics()
# topic_model.visualize_hierarchy()
# topic_model.visualize_documents(df["Texte_clean"].to_list())

# %% [markdown]
# ## Topics

# %%
topic_model.get_topic_info()

# %% [markdown]
# ## Barchart

# %%
topic_model.visualize_barchart(
    # n_words=10,  # Select the number of words to display per topic
    # topics = [0,1,2,3,4], # Select specific topics to display
    top_n_topics=8,  # Select the first n topics to display
    # height = 300, # Adjust the height of the plot
    # width = 800 # Adjust the width of the plot
)

# %%
## Visualisation documents / topics

# %%
topic_model.visualize_documents(
    docs=docs,
    embeddings=precomputed_embeddings,
    hide_annotations=True,  # better readability
    topics=[0, 1, 2, 3],  # Select topics to highlight
    # height = 300, # Adjust the height of the plot
    # width = 800 # Adjust the width of the plot
)

# %%
topic_model.visualize_topics()

# %% [markdown]
# ## Hiérarchie des topics

# %%
topic_model.visualize_hierarchy()

# %%
hierarchical_topics = topic_model.hierarchical_topics(docs)
print(topic_model.get_topic_tree(hierarchical_topics))

# %%
topic_model.visualize_heatmap()

# %% [markdown]
# ## Sauvegarder

# %%
# topic_model.save(
#     path="../models/bertopic/qwen3-emb-8B-bertopic-default-with-ctfidf",
#     serialization="safetensors",
#     save_ctfidf=True,
# )

# %%
# topic_model = BERTopic.load("../models/bertopic/qwen3-emb-8B-bertopic-default-with-ctfidf")

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
topics_over_time = topic_model.topics_over_time(
    docs=docs,
    timestamps=timestamps,
    global_tuning=True,
    evolution_tuning=True,
    nr_bins=20,
)

fig_dynamic_topic = topic_model.visualize_topics_over_time(
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
topics_per_class = topic_model.topics_per_class(docs, classes=affiliations)

fig_topics_per_class = topic_model.visualize_topics_per_class(
    topics_per_class,
    # topics=[0, 1, 2, 3],  # choix spécifique de topics à visualiser
    top_n_topics=10,  # choix des X principaux topics
)
fig_topics_per_class

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

# %% [markdown]
# # Test Bertopic

# %% [markdown]
# ## décider de comment on découpe
# - sentence ?
# - paragraphe (existe pas ici) -> mais possible avec wtpsplit : https://github.com/segment-any-text/wtpsplit
# - dire que tant pis ? Ok si utilise modèle avec fenetre suffisante (ici ok avec qwen)
# - Passer à l'échelle de la phrase pour tout ce qui va être pour sentiment, réseau de mots, proba des termes, etc.
# - etc.

# %% [markdown]
# Stopwords :
# - https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#document-length
# - Instead, we can use the CountVectorizer to preprocess our documents after having generated embeddings and clustered our documents. **Personally, I have found almost no disadvantages to using the CountVectorizer to remove stopwords and it is something I would strongly advise to try out**
# - We can also use the ClassTfidfTransformer to reduce the impact of frequent words. The end result is very similar to explicitly removing stopwords but this process does this automatically:
#
# ```python
# from bertopic import BERTopic
# from bertopic.vectorizers import ClassTfidfTransformer
#
# ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
# topic_model = BERTopic(ctfidf_model=ctfidf_model)
# ```

# %%
from datasets import load_from_disk
import pandas as pd
import numpy as np
from umap import UMAP
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from stopwordsiso import stopwords
from pathlib import Path
# from bertopic.vectorizers import ClassTfidfTransformer


DATASET_DIR = Path(
    "../models/embeddings/qwen3-8b_embeddings_2026-07-21/dataset_with_embeddings"
)

RANDOM_SEED = 42

# ----- Charger docs et precomputed embeddings -----

# dataset = load_from_disk(DATASET_DIR)
# docs = dataset[
#     "texte"
# ]  # NOTE: list pas obligatoire mais assure ? list(dataset["texte"])
# precomputed_embeddings = np.array(
#     dataset["embedding"]
# )  # Shape : XXX * 4096 pour qwen3-embedding:8b

# ----- Charger docs et precomputed embeddings + pandas -----
# TODO: Si tout en pandas pour faciliter les manipulations
dataset = load_from_disk(DATASET_DIR)

df = dataset.to_pandas()

df["dateSeance_ts"] = pd.to_datetime(
    df["dateSeance"].astype(str),
    format="%Y%m%d%H%M%S%f",
)

df["dateSeance_day"] = df["dateSeance_ts"].dt.normalize()

df["affiliation_et_gouv"] = df["affiliation_et_gouv"].fillna("Inconnu")

docs = df["texte"].tolist()
precomputed_embeddings = np.array(df["embedding"].tolist())  # reconversion nécessaire
timestamps = df["dateSeance_ts"].tolist()
affiliations = df["affiliation_et_gouv"].tolist()


# ----- Paramètres -----

# umap (pour reproductibilité, on fixe la seed)
umap_model = UMAP(
    n_neighbors=15,  # default=15
    n_components=5,  # default=5
    min_dist=0.0,  # default=0.0
    metric="cosine",  # default="cosine"
    random_state=RANDOM_SEED,  # pour reproductibilité
)

# TODO : HDBSCAN


# vectorizer_model et french stopwords (avec stopwordsiso)
# TODO : ou avec spacy ou nltk au choix ?
french_stopwords = list(stopwords("fr"))
vectorizer_model = CountVectorizer(stop_words=french_stopwords)

# ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# ----- Créer le modèle -----

# Créer le modèle avec paramètres souhaités +
# train avec les precomputed embeddings

topic_model = BERTopic(
    language="french",  # affecte l'affichage des caractères
    embedding_model=None,  # NOTE : pas d'embedding_model puisque précalculés
    umap_model=umap_model,
    vectorizer_model=vectorizer_model,  # enlever les stopwords après embbedings/clustering
    # ctfidf_model=ctfidf_model, # TODO : ou sinon utiliser ctfidf = reduce the impact of frequent word
    # NOTE : pas de embedding_model puisque précalculés,
    # et language= n'est pas utilisé si on passe les embeddings
)

# Fiter le modèle + transform pour extraire les topics et probabilités
topics, probabilities = topic_model.fit_transform(
    documents=docs, embeddings=precomputed_embeddings
)

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
display(topics_per_class)

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
# df["probability"] = probabilities

# # joindre le nom/label du topic plutôt que juste l'ID
# topic_info = topic_model.get_topic_info()[["Topic", "Name"]]
# df = df.merge(topic_info, left_on="topic", right_on="Topic", how="left")

# # df = df.drop(columns=["embedding"], errors="ignore")
# # df.to_csv("../models/embeddings/export_final.csv", index=False)

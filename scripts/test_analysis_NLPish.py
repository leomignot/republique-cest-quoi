# %% [markdown]
# ## tests NLP(ish)

# %% [markdown]
# https://maartengr.github.io/BERTopic/getting_started/quickstart/quickstart.html

# %% [markdown]
# ## décider de comment on découpe
# - sentence ?
# - paragraphe (existe pas ici) -> mais possible avec wtpsplit : https://github.com/segment-any-text/wtpsplit
# - dire que tant pis ? Ok si utilise modèle avec fenetre suffisante (qwen ? https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
# - faire un check global du nb token par intervention et cut que ce qui dépasse ?
# - Passer à l'échelle de la phrase pour tout ce qui va être pour sentiment, réseau de mots, proba des termes, etc.
# - etc.

# %%
# TODO: remplacer texte_clean par texte (corrigé dans nb3)
# TODO: aviser si vire id_orateur et utiliser id_acteur partout

# TODO: remove unused imports when finalized
import pandas as pd
import spacy
# import nltk
# from nltk.corpus import stopwords

# import bertopic
from bertopic import BERTopic

# from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# # %pip install einops
# # !python -m spacy download fr_core_news_sm

# %%
df = pd.read_csv(
    "../data/interim/df_repu.csv",
    low_memory=False,
    dtype={
        "ID_orateur": str
    },  # désormais géré avant (ajout PA et id_acteur) vire quand mettra au propre
)

# %%
df["DateSeance_ts"] = pd.to_datetime(df["DateSeance"], format="%Y%m%d%H%M%S%f")
df["DateSeance_day"] = df["DateSeance_ts"].dt.normalize()  # guess it works


# %% [markdown]
# ## Bert things

# %%
#########
# Si besoin, revenir au plus simple :
#########

# nltk.download("stopwords")
# french_stopwords = list(set(stopwords.words("french")))
# vectorizer_model = CountVectorizer(stop_words=french_stopwords)
# topic_model = bertopic.BERTopic(language="french", vectorizer_model=vectorizer_model)
# # topics, probs = topic_model.fit_transform(df["Texte_clean"])

# %%
# TODO: redo with the good sentencetransformer

# %% [markdown]
# Stopwords :
# - https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#document-length
# - Instead, we can use the CountVectorizer to preprocess our documents after having generated embeddings and clustered our documents. Personally, I have found almost no disadvantages to using the CountVectorizer to remove stopwords and it is something I would strongly advise to try out:
# - We can also use the ClassTfidfTransformer to reduce the impact of frequent words. The end result is very similar to explicitly removing stopwords but this process does this automatically:

# %%
# TODO: Tester Flaubert et autres, qwen, etc. ALibaba = cry in GPU, , etc. Aviser dans colab ou humanum ?
# Qwen pour sa Context Length ? + est multilingue ?
# pousser vers leur 4B ou 8B si ressources suffisantes ?
# ou https://huggingface.co/jinaai/jina-embeddings-v3

# "dangvantuan/sentence-camembert-large"
# "all-MiniLM-L6-v2"
# Maybe https://huggingface.co/jinaai/jina-embeddings-v3

# embedding_model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

#######
# ICI
#######
# Les tests à l'arrache active tigger donnent des trucs pas mal avec
# "Alibaba-NLP/gte-multilingual-base"
# https://huggingface.co/Alibaba-NLP/gte-multilingual-base


# %%
# vectorizer_model et french stopwords
# Avec spacy
nlp = spacy.load("fr_core_news_sm")  # !python -m spacy download fr_core_news_sm
french_stopwords = list(nlp.Defaults.stop_words)
vectorizer_model = CountVectorizer(stop_words=french_stopwords)

# # Ou avec nltk
# import nltk
# from nltk.corpus import stopwords
# nltk.download("stopwords")
# french_stopwords = list(set(stopwords.words("french")))
# vectorizer_model = CountVectorizer(stop_words=french_stopwords)


#####################################
# TODO: Choisir le modèle d'embedding final
#####################################


embedding_model = SentenceTransformer(
    # "Alibaba-NLP/gte-multilingual-base", # cry in gpu, something wrong ?
    # "dangvantuan/sentence-camembert-large",
    "all-MiniLM-L6-v2",  # celui par défaut ?
    # "jinaai/jina-embeddings-v3",
    trust_remote_code=True,
)

print(
    "device used :", embedding_model.device
)  # Vérifie si le modèle est sur GPU ou CPU

# ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# créer le modèle
topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,  # remove stopwords after embbedings
    # ctfidf_model=ctfidf_model, # (or) reduce the impact of frequent word
)

# %%
# Fiter le modèle
topics, probs = topic_model.fit_transform(
    df["Texte_clean"]
)  # .tolist() ? pas obligatoire ?

# %% [markdown]
# # Test taille batch

# %%
# Reste quand même trop gros, faudrait couper les interventions sans doute.

# # from sentence_transformers import SentenceTransformer
# import numpy as np

# # embedding_model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
# texts = df["Texte_clean"].tolist()

# batch_size = 8
# embeddings = []

# for i in range(0, len(texts), batch_size):
#     batch_texts = texts[i:i+batch_size]
#     batch_embeds = embedding_model.encode(batch_texts, show_progress_bar=False)
#     embeddings.append(batch_embeds)

# embeddings = np.vstack(embeddings)

# topics, probs = topic_model.fit_transform(df["Texte_clean"], embeddings=embeddings)


# %%
# # V1 safetensors
# topic_model.save(
#     "../models/bert/camembert_safetensors",
#     serialization="safetensors",
#     save_ctfidf=True,
#     save_embedding_model=embedding_model,
# )

# # V2 pytorch
# topic_model.save(
#     "../models/bert/camembert_pytorch",
#     serialization="pytorch",
#     save_ctfidf=True,
#     save_embedding_model=embedding_model,
# )

# # V3 pickle (pas recommandé ?)
# topic_model.save("../models/bert/############", save_embedding_model=True)


# %%
# Load from directory
# loaded_model = BERTopic.load("../models/bert/############")


# %%
# TODO: does mean pooling thing really works ?

# %% [markdown]
# Pour "dangvantuan/sentence-camembert-large"
# - No sentence-transformers model found with name dangvantuan/sentence-camembert-large. Creating a new one with MEAN pooling.
# - https://github.com/UKPLab/sentence-transformers/issues/2779 In short: even with a will create a new model with mean pooling warning, your model uses the tokenizer from https://huggingface.co/dangvantuan/sentence-camembert-large.
#
#
# DOES IT REALLY WORKS ?

# %%
# help(topic_model)

# %%
topic_model.get_topic_info()

# %% [markdown]
# ### Tester des trucs rapides

# %%
# # Les principales fonctions à tester pour avoir un aperçu simple :

# topic_model.get_topic_info()
# topic_model.visualize_barchart()
# topic_model.visualize_topics()
# topic_model.visualize_hierarchy()
# topic_model.visualize_documents(df["Texte_clean"].to_list())


# %% [markdown]
# ### Tester plus

# %%
# Optional: visualize
fig_topic_distance_map = topic_model.visualize_topics()
fig_topic_distance_map

# %%
fig_topic_distance_map.write_html("../reports/figures/topic_distance_map.html")

# %%
table_topic = topic_model.get_topic_info()
table_topic[:20]

# %%
table_topic.to_csv("../data/interim/table_topics.csv", index=False)

# %%
df["Topic"] = topics

# %%
import csv

df.to_csv("../data/interim/df_repu_with_topics.csv", index=False, quoting=csv.QUOTE_ALL)

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
topics_over_time = topic_model.topics_over_time(
    docs=df["Texte_clean"],
    timestamps=df["DateSeance_ts"],
    global_tuning=True,
    evolution_tuning=True,
    nr_bins=20,
)

# %%
fig_dynamic_topic = topic_model.visualize_topics_over_time(
    topics_over_time, top_n_topics=10
)
fig_dynamic_topic

# %%
fig_dynamic_topic.write_html("../reports/figures/dynamic_topics.html")

# %% [markdown]
# ## Hierarchical topics

# %%
hierarchical_topics = topic_model.hierarchical_topics(df["Texte_clean"])

# %%
fig_hierarchical = topic_model.visualize_hierarchy(
    hierarchical_topics=hierarchical_topics
)
fig_hierarchical

# %%
fig_hierarchical.write_html("../reports/figures/hierarchical_topics.html")

# %%
tree = topic_model.get_topic_tree(hierarchical_topics)
print(tree)

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
df.columns

# %%
df["groupeAbrev"] = df["groupeAbrev"].fillna("gouv_TEMP")

# %%
# ATTENTION, PLANTAIT À CAUSE DES NA
topics_per_class = topic_model.topics_per_class(
    df["Texte_clean"], classes=df["groupeAbrev"]
)

# %%
fig_topics_per_class = topic_model.visualize_topics_per_class(
    topics_per_class, top_n_topics=10
)
fig_topics_per_class

# %%
fig_topics_per_class.write_html("../reports/figures/topics_per_class.html")

# %%
# TODO: revoir regroupement des topics
# TODO: revoir Topic distribution
# TODO: aviser genAI sur le nom des topics ?

# %%
df.shape

# %%
df["Texte_clean"].str.len().describe()

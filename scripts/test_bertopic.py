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
# ````
# from bertopic import BERTopic
# from bertopic.vectorizers import ClassTfidfTransformer
#
# ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
# topic_model = BERTopic(ctfidf_model=ctfidf_model)
# ```

# %%
from datasets import load_from_disk
import numpy as np
from umap import UMAP
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from stopwordsiso import stopwords
from pathlib import Path

DATASET_DIR = Path(
    "../models/embeddings/qwen3-8b_embeddings_2026-07-21/dataset_with_embeddings"
)

RANDOM_SEED = 42

# ----- Charger docs et precompuited embeddings -----

dataset = load_from_disk(DATASET_DIR)
docs = dataset[
    "texte"
]  # TODO: list pas obligatoire mais assure ? list(dataset["texte"])
precomputed_embeddings = np.array(
    dataset["embedding"]
)  # Shape : XXX * 4096 pour qwen3-embedding:8b

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
topic_model.visualize_barchart()

# %%
topic_model.visualize_barchart(
    # n_words=10,  # Select the number of words to display per topic
    # topics = [0,1,2,3,4], # Select specific topics to display
    top_n_topics=10,  # Select the first n topics to display
    # height = 300, # Adjust the height of the plot
    # width = 800 # Adjust the width of the plot
)

# %%
topic_model.visualize_topics()

# %%
topic_model.visualize_hierarchy()

# %%
topic_model.visualize_documents(docs, embeddings=precomputed_embeddings)

# %%
# genre crest :
(
    topic_model.visualize_documents(
        docs=docs,
        embeddings=precomputed_embeddings,
        hide_annotations=True,  # better readability
        topics=[0, 1, 2, 3],  # Select topics to highlight
        # height = 300, # Adjust the height of the plot
        # width = 800 # Adjust the width of the plot
    )
)

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

# %%
df["DateSeance_ts"] = pd.to_datetime(df["DateSeance"], format="%Y%m%d%H%M%S%f")
df["DateSeance_day"] = df["DateSeance_ts"].dt.normalize()  # guess it works

# %% [markdown]
# Comparaison de modèles :
#
# https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#finding-similar-topics-between-models

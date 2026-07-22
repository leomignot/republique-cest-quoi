# %% [markdown]
# # Test Bertopic

# %%
from datasets import load_from_disk
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from stopwordsiso import stopwords
from pathlib import Path

DATASET_DIR = Path(
    "../models/embeddings/qwen3-8b_embeddings_2026-07-21/dataset_with_embeddings"
)

# RANDOM_SEED = TODO: fixer une seed et passer dans umap ?

# Charger les embeddings pour éviter recalcul long
dataset = load_from_disk(DATASET_DIR)
docs = dataset["texte"]  # XXX lignes # TODO: list(dataset["texte"]) ??
embeddings = np.array(
    dataset["embedding"]
)  # Shape : XXX * 4096 pour qwen3-embedding:8b

# vectorizer_model et french stopwords
# Avec stopwordsiso (# NOTE : ou avec spacy ou nltk au choix)
french_stopwords = list(stopwords("fr"))
vectorizer_model = CountVectorizer(stop_words=french_stopwords)


# ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

topic_model = BERTopic(
    language="french",  # NOTE : doute change rien si on passe les autres arg ? recreuser
    vectorizer_model=vectorizer_model,  # remove stopwords after embbedings
    # ctfidf_model=ctfidf_model, # (or) reduce the impact of frequent word
)

# Fiter le modèle
topic_model.fit(documents=docs, embeddings=embeddings)

# et transform pour extraire les topics et probabilités
topics, probabilities = topic_model.transform(documents=docs, embeddings=embeddings)

# %% [markdown]
# # Tests

# %%
# # Les principales fonctions à tester pour avoir un aperçu simple :

topic_model.get_topic_info()

# %%
topic_model.visualize_barchart()

# %%
topic_model.visualize_topics()

# %%
topic_model.visualize_hierarchy()

# %%
topic_model.visualize_documents(docs, embeddings=embeddings)

# %%
# # %pip install umap-learn

# %%
# genre crest :
(
    topic_model.visualize_documents(
        docs=docs,
        embeddings=embeddings,
        hide_annotations=True,  # better readability
        topics=[0, 1, 2, 3],  # Select topics to highlight
        # height = 300, # Adjust the height of the plot
        # width = 800 # Adjust the width of the plot
    )
)

# %%
topic_model.visualize_barchart(
    n_words=10,  # Select the number of words to display per topic
    # topics = [0,1,2,3,4], # Select specific topics to display
    # top_n_topics = 6, # Select the first n topics to display
    # height = 300, # Adjust the height of the plot
    # width = 800 # Adjust the width of the plot
)

# %% [markdown]
# ## Sauvegarder

# %%
# topic_model.save(
#     path="../models/bertopic/qwen-bertopic-default-with-ctfidf",
#     serialization="safetensors",
#     save_ctfidf=True,
# )

# %%
# topic_model = BERTopic.load("./bertopic-default")

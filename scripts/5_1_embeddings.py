# %% [markdown]
# # 5-1 - Calcul des embeddings (serveur Ollama Huma-Num)
# Lit les interventions mentionnant la République (sortie de 3-1). Calcule
# des embeddings de texte via un modèle Qwen3-Embedding-8B servi par Ollama,
# avec reprise incrémentale en cas d'interruption (checkpoint JSONL).
# Un scan préalable de la taille des textes en tokens permet de vérifier
# que la fenêtre de contexte choisie est adaptée avant de lancer le calcul.
# Écrit un Dataset HuggingFace (texte + embedding) sur disque.

# %%
# TODO : se passer de dataset ?

# %%
import os
import time
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from ollama import Client
from datasets import Dataset
from transformers import AutoTokenizer

PATH_ENTREE = "../data/interim/3_1_df_repu.csv"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST")
client = Client(host=OLLAMA_HOST)

MODEL_NAME = "qwen3-embedding:8b"
# valeur documentée par Qwen, pas identique métadonnées Ollama/trasnformers HF
# (cf. section scan ci-dessous)
QWEN3_EMBEDDING_RECOMMENDED_MAX = 32768


BATCH_SIZE = 16  # ou monter à 32
MAX_RETRIES = 3
RETRY_DELAY = 5  # secondes

CHECKPOINT_DIR = Path("../models/embeddings_checkpoint")
CHECKPOINT_DIR.mkdir(exist_ok=True)
EMBEDDINGS_JSONL = CHECKPOINT_DIR / "embeddings.jsonl"
FINAL_DATASET_DIR = Path("../models/embeddings/dataset_with_embeddings")

# %%
# Charger le dataset cible
df = pd.read_csv(PATH_ENTREE)
print("Shape du df chargé : ", df.shape)

# %% [markdown]
# ## Estimation des tailles de textes en nombre de tokens
# NOTE : le tokenizer chargé ici (transformers) n'est pas celui réellement
# utilisé par Ollama pour l'embedding, mais donne une estimation utile du
# nombre de tokens par texte (information absente de l'API Ollama).


# %%
# ====================================================
# Estimation des tailles de textes en nombre de tokens
# ====================================================

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B")

# ---------- fonctions diag taille contexte et nb tokens par texte ----------


def get_model_context_length(model_name: str = MODEL_NAME) -> int:
    """Récupère la longueur de contexte max supportée par le modèle, depuis les métadonnées Ollama."""
    info = client.show(model_name)
    model_info = info.get("modelinfo", {})

    # Cherche dynamiquement la clé qui se termine par "context_length"
    # (le préfixe varie selon l'architecture : qwen3, llama, mistral, etc.)
    context_key = next((k for k in model_info if k.endswith("context_length")), None)

    if context_key is None:
        raise ValueError(
            f"Impossible de trouver la context_length dans les métadonnées de {model_name}"
        )

    return model_info[context_key]


def analyze_token_lengths(texts: list[str]) -> pd.DataFrame:
    """Calcule la longueur en tokens de chaque texte."""
    lengths = [len(tokenizer.encode(text)) for text in texts]
    return pd.Series(lengths)


def print_token_stats(texts: list[str]):
    """Affiche les statistiques de longueur en tokens et les seuils de dépassement."""
    lengths = analyze_token_lengths(texts)

    print(f"Nombre de textes : {len(lengths)}")
    print(f"Min             : {lengths.min()}")
    print(f"Max             : {lengths.max()}")
    print(f"Moyenne         : {lengths.mean():.1f}")
    print(f"Médiane         : {lengths.median():.1f}")
    print(f"Percentile 90   : {lengths.quantile(0.90):.0f}")
    print(f"Percentile 95   : {lengths.quantile(0.95):.0f}")
    print(f"Percentile 99   : {lengths.quantile(0.99):.0f}")

    # Nombre de textes qui dépasseraient certains seuils de tokens
    for threshold in [512, 1024, 2048, 4096, 8192, 32768]:
        n_over = (lengths > threshold).sum()
        pct = 100 * n_over / len(lengths)
        print(f"Textes > {threshold} tokens : {n_over} ({pct:.1f}%)")

    return lengths


# %%
# passage des textes en liste
texts = df["texte"].tolist()

# ---------- Renvoi des infos ----------

# Renvoi des infos taille modèle
max_context = get_model_context_length()
print("---------- Identification taille contexte modèle ----------")
print(f"Fenêtre de contexte max du modèle :")
print(f"Max architectural (Ollama) : {max_context} tokens")
print(f"Max architectural (transformers HF) : {tokenizer.model_max_length} tokens")
print(f"Max recommandé (doc Qwen)  : {QWEN3_EMBEDDING_RECOMMENDED_MAX} tokens")

print("\n-------------------------------------------------------------")
print("Les infos récupérées automatiquement semblent ici peu fiables.")
print(
    "(valeur par défaut / mal renseigné / hérité d'un autre modèle (base vs emdeddings ?))"
)
print(
    f"\nSÉCURITÉ : Rester sur 'QWEN3_EMBEDDING_RECOMMENDED_MAX' : {QWEN3_EMBEDDING_RECOMMENDED_MAX} tokens"
)

# Renvoi des infos taille token par texte
print("\n-------------------------------------------------------------")
print("Calcul tokens par textes :")


lengths = print_token_stats(texts)

# %% [markdown]
# ## Calcul des embeddings par serveur Ollama
# NOTE : reprise incrémentale gérée via le nombre de lignes déjà écrites dans
# EMBEDDINGS_JSONL (pas de fichier de progression séparé) : relancer ce
# script après une interruption reprend automatiquement où il s'était arrêté.

# %%
# =========================================
# CALCUL EMBEDDINGS SERVEUR OLLAMA
# =========================================


# ----- Fonctions pour le calcul des embeddings -----


def embed_batch_with_retry(
    batch: list[str], max_retries: int = MAX_RETRIES
) -> list[list[float]]:
    """Calcule les embeddings d'un batch de textes, avec re-tentatives en cas d'erreur serveur.
    NOTE : si un jour textes plus (=trop) longs, implémenter un fallback propre en cas d'erreur
    levée par truncate=false, plutôt que juste lever une erreur. Ici ok car < context max recommandé.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = client.embed(
                model=MODEL_NAME,
                input=batch,
                truncate=False,  # lever une erreur si dépassement au lieu de tronquer silencieusement.
                # NOTE : Ollama utilise num_ctx pour la taille max de contexte. Pour un
                # modèle d'embeddings, correspond à la longueur max de texte encodable
                # avant troncature.
                options={
                    "num_ctx": QWEN3_EMBEDDING_RECOMMENDED_MAX
                },  # ici = 32768, la valeur recommandée à 32k
            )
            return response["embeddings"]
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(f"Échec après {max_retries} tentatives : {e}")
            wait = RETRY_DELAY * attempt
            print(
                f"[Tentative {attempt}/{max_retries}] Erreur: {e}. Nouvelle tentative dans {wait}s..."
            )
            time.sleep(wait)


def append_embeddings(batch_embeddings: list[list[float]]):
    """Ajoute un batch d'embeddings au fichier de checkpoint JSONL."""
    with open(EMBEDDINGS_JSONL, "a") as f:
        for emb in batch_embeddings:
            f.write(json.dumps(emb) + "\n")


def count_saved_embeddings() -> int:
    """Reprise basée sur le nombre de lignes déjà écrites (pas de fichier progress séparé)."""
    if not EMBEDDINGS_JSONL.exists():
        return 0
    with open(EMBEDDINGS_JSONL) as f:
        return sum(1 for _ in f)


def load_embeddings_jsonl() -> list[list[float]]:
    """Recharge tous les embeddings déjà calculés depuis le checkpoint JSONL."""
    with open(EMBEDDINGS_JSONL) as f:
        return [json.loads(line) for line in f]


def compute_embeddings_incremental(texts: list[str], batch_size: int = BATCH_SIZE):
    """Calcule les embeddings manquants par batch, avec reprise automatique."""
    start_index = count_saved_embeddings()

    if start_index > 0:
        print(
            f"Reprise à l'index {start_index} ({start_index} embeddings déjà calculés)"
        )

    remaining_texts = texts[start_index:]

    for i in tqdm(range(0, len(remaining_texts), batch_size)):
        batch = remaining_texts[i : i + batch_size]
        try:
            batch_embeddings = embed_batch_with_retry(batch)
        except RuntimeError as e:
            print(
                f"Arrêt du job à l'index {start_index + i}. Relance le script pour reprendre."
            )
            raise e

        append_embeddings(batch_embeddings)

    print("Tous les embeddings ont été calculés.")


def build_final_dataset(df: pd.DataFrame) -> Dataset:
    """Assemble texte + embeddings dans un Dataset HuggingFace, une fois le calcul terminé."""
    embeddings = load_embeddings_jsonl()
    assert len(embeddings) == len(df), (
        f"Désalignement : {len(embeddings)} embeddings vs {len(df)} lignes du df"
    )

    dataset = Dataset.from_pandas(df.reset_index(drop=True))
    dataset = dataset.add_column("embedding", embeddings)
    dataset.save_to_disk(FINAL_DATASET_DIR)
    print(f"Dataset final sauvegardé dans {FINAL_DATASET_DIR}")
    return dataset


# %% [markdown]
# ## Exécution

# %%
# %%
# ================================================
# EXÉCUTION DU CALCUL DES EMBEDDINGS ET SAUVEGARDE
# ================================================

# NOTE: texts est défini plus haut, à partir du df complet.
# Si on veut lancer sur un sous-ensemble, il faut redéfinir texts ici.

# I.e : pour tests, réduire le nombre de textes à traiter :
# df = df.head(1000)
# texts = df["texte"].tolist()


# ----- Step 1 : calcul avec reprise (peut être relancé si ça plante) -----
compute_embeddings_incremental(texts)

# ----- Step 2 : une fois terminé, on construit le dataset final propre -----
dataset = build_final_dataset(df)

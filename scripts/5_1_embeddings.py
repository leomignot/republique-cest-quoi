# %% [markdown]
# # 5-1 - Calcul des embeddings (serveur Ollama Huma-Num)
# Lit les interventions mentionnant la République (sortie de 3-1). Calcule
# des embeddings de texte via un modèle Qwen3-Embedding-8B servi par Ollama,
# avec reprise incrémentale en cas d'interruption (checkpoint JSONL).
# Un scan préalable de la taille des textes en tokens permet de vérifier
# que la fenêtre de contexte choisie est adaptée avant de lancer le calcul.
# Écrit un CSV (dataset) + .npy (embedding) + un JSON de métadonnées sur disque.
#
# NOTE : dans le cadre d'une ré-exécution du pipeline, le choix est laissé
# (switch) de réutiliser les embeddings déjà calculés pour reconstruire le
# dataset sans relancer Ollama. Ne vaut que si la structure des lignes et
# les textes n'ont pas été modifiés depuis le calcul des embeddings.

# %%
# TODO : check si les hash tiennent bien.

# %%
import hashlib
import json
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

PATH_ENTREE = "../data/interim/3_1_df_repu.csv"
FINAL_DIR = Path("../models/embeddings")
META_PATH = FINAL_DIR / "embeddings_meta.json"
# SWITCH POUR LE CALCUL DES EMBEDDINGS
# (PRODUCTION ORIGINELLE VS REPRODUCTIBILITÉ)
# True pour relancer le pipeline originel des embeddings
# False pour réutiliser les embeddings existants (reproduction)
RUN_EMBEDDING_COMPUTATION = True


def hash_texts(texts) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(str(t).encode("utf-8"))
        h.update(b"\x00")  # séparateur, évite que ["ab","c"] == ["a","bc"]
    return h.hexdigest()


def write_meta(df, embeddings, model="qwen3-embedding:8b", num_ctx=32768):
    meta = {
        "texts_sha256": hash_texts(df["texte"]),
        "n_rows": len(df),
        "embedding_shape": list(embeddings.shape),
        "model": model,
        "num_ctx": num_ctx,
        "date": datetime.date.today().isoformat(),
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


# %% [markdown]
# ## Production historique des embeddings


# %%
# ===========================================================
# PRODUCTION HISTORIQUE DES EMBEDDINGS
# Cette fonction conserve le code de production original.
# Elle n'est appelée que si RUN_EMBEDDING_COMPUTATION = True.
# ===========================================================


def run_embedding_production():
    """
    Exécute la production des embeddings à partir du dataframe issu
    de l'étape 3_1.
    Les embeddings sont calculés à l'aide du modèle
    Qwen3-Embedding-8B, servi par Ollama sur le serveur Huma-Num.
    Le calcul est effectué par lots, avec un mécanisme de reprise
    incrémentale reposant sur un checkpoint JSONL, afin de pouvoir
    reprendre le traitement en cas d'interruption.
    Cette fonction comprend également les diagnostics préalables
    sur la longueur des textes et la fenêtre de contexte du modèle.

    Les résultats sont sauvegardés dans FINAL_DIR :
        - dataset_repu.csv : dataframe enrichi avec les emb_id ;
        - embeddings.npy : matrice des embeddings calculés.
        - embeddings_meta.json : hash des textes, shape, modèle, num_ctx, date.

    Cette fonction constitue la procédure historique de production.
    Elle n'est appelée que lorsque RUN_EMBEDDING_COMPUTATION vaut True.

    Note
    ----
    L'exécution nécessite un accès au serveur Ollama et au modèle
    d'embeddings configuré.
    """

    # ====================================================
    # Imports, initialisations et path
    # ====================================================
    import os
    import time
    from tqdm import tqdm
    from ollama import Client
    from transformers import AutoTokenizer
    # import json # redéfini plus haut avec stratégie switch
    # import numpy as np # redéfini plus haut avec stratégie switch
    # from pathlib import Path # redéfini plus haut avec stratégie switch
    # import pandas as pd # redéfini plus haut avec stratégie switch

    # FINAL_DIR = Path("../models/embeddings") # redéfini plus haut avec stratégie switch

    # PATH_ENTREE = "../data/interim/3_1_df_repu.csv" # redéfini plus haut avec stratégie switch

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
    CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
    EMBEDDINGS_JSONL = CHECKPOINT_DIR / "embeddings.jsonl"
    CHECKPOINT_HASH = CHECKPOINT_DIR / "texts_sha256.txt"

    # ----- Charger le dataset cible -----
    df = pd.read_csv(PATH_ENTREE)
    print("Shape du df chargé : ", df.shape)

    current_hash = hash_texts(df["texte"])
    if EMBEDDINGS_JSONL.exists() and (
        not CHECKPOINT_HASH.exists() or CHECKPOINT_HASH.read_text() != current_hash
    ):
        raise RuntimeError(
            f"Checkpoint obsolète ou sans hash : supprimer {CHECKPOINT_DIR} "
            "pour repartir de zéro."
        )
    CHECKPOINT_HASH.write_text(current_hash)

    # ====================================================
    # Estimation des tailles de textes en nombre de tokens
    # ====================================================

    # ## Estimation des tailles de textes en nombre de tokens
    # NOTE : le tokenizer chargé ici (transformers) n'est pas celui réellement
    # utilisé par Ollama pour l'embedding, mais donne une estimation utile du
    # nombre de tokens par texte (information absente de l'API Ollama).

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B")

    # ---------- fonctions diag taille contexte et nb tokens par texte ----------

    def get_model_context_length(model_name: str = MODEL_NAME) -> int:
        """Récupère la longueur de contexte max supportée par le modèle, depuis les métadonnées Ollama."""
        info = client.show(model_name)
        model_info = info.get("modelinfo", {})

        # Cherche dynamiquement la clé qui se termine par "context_length"
        # (le préfixe varie selon l'architecture : qwen3, llama, mistral, etc.)
        context_key = next(
            (k for k in model_info if k.endswith("context_length")), None
        )

        if context_key is None:
            raise ValueError(
                f"Impossible de trouver la context_length dans les métadonnées de {model_name}"
            )

        return model_info[context_key]

    def analyze_token_lengths(texts: list[str]) -> pd.Series:
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

        # NOTE : retourner les longueurs en plus des prints si usage ultérieur
        return lengths  # si jamais besoin identifier et filtrer

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

    # Positions disponibles pour examiner ou découper les textes si nécessaire.
    indices_trop_longs = lengths[lengths > QWEN3_EMBEDDING_RECOMMENDED_MAX].index
    if len(indices_trop_longs):
        print(f"Positions des textes trop longs : {indices_trop_longs.tolist()}")

    # =========================================
    # CALCUL EMBEDDINGS SERVEUR OLLAMA
    # =========================================

    # ### Calcul des embeddings par serveur Ollama
    # NOTE : reprise incrémentale gérée via le nombre de lignes déjà écrites dans
    # EMBEDDINGS_JSONL (pas de fichier de progression séparé) : relancer ce
    # script après une interruption reprend automatiquement où il s'était arrêté.

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

        # Vérification de la cohérence du checkpoint
        if start_index > len(texts):
            raise ValueError(
                f"Checkpoint incohérent : {start_index} embeddings "
                f"pour seulement {len(texts)} textes."
            )

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

    def save_outputs(df: pd.DataFrame) -> None:
        """Sauvegarde df (CSV) + embeddings (.npy) une fois le calcul terminé."""
        embeddings = np.array(load_embeddings_jsonl())
        assert len(embeddings) == len(df), (
            f"Désalignement : {len(embeddings)} embeddings vs {len(df)} lignes du df"
        )

        FINAL_DIR.mkdir(parents=True, exist_ok=True)
        # emb_id : indice du vecteur correspondant dans embeddings.npy
        # (simple pointeur de position, sans contrôle de contenu, aide à
        # retrouver le bon vecteur si des lignes du df sont supprimées ou réordonnées
        df["emb_id"] = np.arange(len(df))
        df.to_csv(FINAL_DIR / "dataset_repu.csv", index=False)
        np.save(FINAL_DIR / "embeddings.npy", embeddings)
        write_meta(df, embeddings)
        print(f"dataset et embeddings sauvegardés dans {FINAL_DIR}")

    # ================================================
    # EXÉCUTION DU CALCUL DES EMBEDDINGS ET SAUVEGARDE
    # ================================================

    # ----- Step 1 : calcul avec reprise (peut être relancé si ça plante) -----
    compute_embeddings_incremental(texts)

    # ----- Step 2 : une fois terminé, on construit le dataset final propre -----
    save_outputs(df)


# %% [markdown]
# ## B. Ré-exécution du pipeline sans recalcul embeddings

# %%
# ========================================================================
# Cas ré-exécution du pipeline : réutilisation des embeddings pré-calculés
# ========================================================================

# NOTE: cette fonction permet de relancer le notebook sans accès au
# serveur Ollama, en réutilisant les embeddings déjà calculés.
# Utilisé ici pour la reproductibilité du pipeline.
# Peut aussi être utile si les colones conservées changent et que l'on
# veut reconstruire dataset_repu.csv sans recalculer les embeddings,
# mais non utilisable si le df est modifié (ajout/suppression de lignes)
# ou si les textes sont modifiés (nettoyage, etc.).
# Dans ce cas, relancer l'embedding, ou envisager un fallback partiel
# sur la base des emb_id existants dans le dataset_repu.csv.


def reuse_embedding():
    """
    Ne relance pas Ollama. Reconstruit dataset_repu.csv à partir du df courant
    en réutilisant les embeddings pré-calculés (embeddings.npy).
    Vérifie via embeddings_meta.json (hash des textes, shape) que les textes
    et leur ordre n'ont pas changé depuis le calcul des embeddings, puis
    attribue emb_id = position de la ligne.
    NOTE: donc uniquement valable si la structure des lignes et les textes n'ont
    pas été modifiés depuis le calcul des embeddings.
    """

    df = pd.read_csv(PATH_ENTREE)

    if not (FINAL_DIR / "embeddings.npy").exists():
        raise FileNotFoundError(
            f"embeddings.npy introuvable dans {FINAL_DIR}. Téléchargez-le ou lancez "
            "avec RUN_EMBEDDING_COMPUTATION = True (accès Ollama requis)."
        )

    precomputed_embeddings = np.load(FINAL_DIR / "embeddings.npy")

    if len(df) != len(precomputed_embeddings):
        raise ValueError(
            f"Le df contient {len(df)} lignes, "
            f"mais {len(precomputed_embeddings)} embeddings sont disponibles."
        )

    # Check si les textes n'ont pas changé depuis le calcul des embeddings
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"{META_PATH} introuvable : impossible de vérifier l'alignement textes/embeddings."
        )
    meta = json.loads(META_PATH.read_text())
    if hash_texts(df["texte"]) != meta["texts_sha256"]:
        raise ValueError(
            "Les textes de 3_1_df_repu.csv ont changé depuis le calcul des embeddings "
            "(contenu ou ordre). Relancer avec RUN_EMBEDDING_COMPUTATION = True (accès Ollama requis)."
        )
    if list(precomputed_embeddings.shape) != meta["embedding_shape"]:
        raise ValueError("embeddings.npy ne correspond pas aux métadonnées.")

    # NOTE: valable que si la structure des lignes et les textes
    # n'ont pas été modifiés depuis le calcul des embeddings.
    # Alignement vérifié par hash ci-dessus : emb_id = position de la ligne
    df["emb_id"] = np.arange(len(df))

    df.to_csv(
        FINAL_DIR / "dataset_repu.csv",
        index=False,
    )

    print("Dataset reconstruit sans recalcul des embeddings.")


# %% [markdown]
# ## Exécution principale :
# Selon le switch, on relance le calcul ou on réutilise les embeddings existants.

# %%
if RUN_EMBEDDING_COMPUTATION:
    run_embedding_production()
else:
    reuse_embedding()

# %%

# %%
# bug incompatibilité dans myenv_clone, ok dans nlp_env
# faire du nettoyage dans les envs un jour …
import os
from ollama import Client

OLLAMA_HOST = os.environ.get("OLLAMA_HOST")
client = Client(host=OLLAMA_HOST)


# %%
import pandas as pd

df = pd.read_csv(
    "../data/interim/annotations_republique-annotation_previous_annotation_test.csv"
)

# %%
df.head()

# %%
df["previous_annotation"].value_counts()

# %%
from tqdm import tqdm


def do_predictions(prompt_generator, texts, model):
    """
    Prediction avec Ollama : prompt_generator, liste de textes, modèle.
    Affiche la progression (tqdm).
    """
    results = []
    total = len(texts)

    with tqdm(
        total=total,
        desc="Progress",
        unit="item",
        bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{percentage:3.0f}%]",
    ) as pbar:
        for i, text in texts.items():
            try:
                messages = prompt_generator(text)
                response = client.chat(model=model, messages=messages)
                content = response.get("message", {}).get("content", "").strip()
                results.append(content)
            except Exception as e:
                print(f"Erreur ligne {i} : {e}")
                results.append(None)
            pbar.update(1)

    return results



# %%
# TODO: need to check prompt struct to be sure of what is going to ollama
def get_prompt_simple(text):
    return [
        {
            "role": "system",
            "content": (
                "Tu es un expert en sciences politique. "
                "Classifie chaque intervention comme relevant du thème 'REPUBLIQUE' ou 'AUTRE'"
                "Réponds uniquement par un de ces deux mots"
            ),
        },
        {"role": "user", "content": f'Classe cette intervention :\n"{text}"\nLabel:'},
    ]



# %%
# TODO: need to check prompt struct to be sure of what is going to ollama
def get_prompt_exclure_pays(text):
    return [
        {
            "role": "system",
            "content": (
                "Tu es un expert en sciences politique. "
                "Tu dois classifier chaque intervention comme relevant bien du thème de la 'REPUBLIQUE' française ou 'AUTRE"
                "Fais attention à exclure celles qui ne parlent que d'autres pays (=AUTRE)"
                "Réponds uniquement par un de ces deux mots ('REPUBLIQUE' ou 'AUTRE')"
            ),
        },
        {"role": "user", "content": f'Classe cette intervention :\n"{text}"\nLabel:'},
    ]



# %%
# # S'inspirer de ça si besoin ?
# # The structure is a bit different
# def build_prompt(text):
#     system_prompt = (
#         "You are a helpful and accurate news headline classifier. "
#         "Your job is to classify news headlines as either 'POLITICS' or 'OTHER'. "
#         "Only respond with exactly one of those two labels."
#     )
#     user_prompt = f"Classify this headline:\n\"{text}\"\nLabel:"
#     return f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n[INST] {user_prompt} [/INST]"

# %%
import time
# si abesoin ajout pause, un truc du genre ?

# # test avec un extrait
# texts = df["Texte"][:100]  # ou .iloc[:blabla]

start = time.time()
# Appel
r_ollama = do_predictions(get_prompt_exclure_pays, df["text"], model="llama3.1:70b")
end = time.time()
print(f"Temps d'exécution : {end - start:.2f} secondes")

# # Ajout dans le DataFrame
# df.loc[texts.index, "prediction_simple"] = r_ollama

# guess it kinda work ? puisque à tout pris.
# Si slice ici aussi renvoyer le slice.index
df.loc[df["text"].index, "prediction_llama_exclure_pays"] = r_ollama


# %%
df.to_csv("../data/interim/df_testset_pred_simple_llama.csv", index=False)

# %% [markdown]
# ## longer prompt ?

# %%
# # TODO: need to check prompt struct to be sure of what is going to ollama


# def get_prompt_long(text):
#     return [
#         {
#             "role": "system",
#             "content": (
#                 "Dans le cadre d’un article académique en science politique, tu dois distinguer les interventions relatives à la 'République' des 'autres' "
#                 "Il s’agit d’un corpus d’interventions de députés à l’assemblée nationale en France lors de la 16e législature."
#                 " La « République » renvoie ici à sa dimension idéelle ou idéologique. Est donc exclut les références à des pays, au nom d’un parti politique, d’un journaliste ou d’une institution, ou à la fonction de président de la République. Est inclut en plus du champ lexical de la « République », à la devise « liberté, égalité, fraternité » et aux principes de « laïcité », « universalisme » et « indivisibilité ». "
#                 "Classifie chaque intervention comme 'REPUBLIQUE' ou 'AUTRE'"
#                 "Réponds uniquement par un de ces deux mots."
#             ),
#         },
#         {"role": "user", "content": f'Classe cette intervention :\n"{text}"\nLabel:'},
#     ]


# %%
# import time

# # test avec un extrait
# texts = df["Texte"][:100]  # ou .iloc[:10]

# start = time.time()
# # Appel
# r_ollama = do_predictions(get_prompt_long, texts, model="llama3.1:70b")
# end = time.time()
# print(f"Temps d'exécution : {end - start:.2f} secondes")

# # Ajout dans le DataFrame
# df.loc[texts.index, "prediction_long"] = r_ollama

# %% [markdown]
# # Experimentation

# %% [markdown]
# ## Scores

# %%
df["prediction_llama_simple"] = df["prediction_llama_simple"].replace(
    {"REPUBLIQUE": "Republique", "AUTRE": "Autre"}
)

df["prediction_llama_exclure_pays"] = df["prediction_llama_exclure_pays"].replace(
    {"REPUBLIQUE": "Republique", "AUTRE": "Autre"}
)


# %%
df[["previous_annotation", "prediction_llama_simple", "prediction_llama_exclure_pays"]]

# %%
import pandas as pd
from sklearn.metrics import classification_report, f1_score

# df_score = pd.read_csv("../data/interim/df_merged_pred.csv")
df_score = df[["id", "text", "previous_annotation", "prediction_llama_simple", "prediction_llama_exclure_pays"]]
df_score.head()

# %%
# pour le principe, puisque ici pas de gold standard, juste les modèles entre eux
print(
    classification_report(
        df_score["previous_annotation"], df_score["prediction_llama_exclure_pays"], digits=3
    )
)

# %%
f1_score(
    df_score["previous_annotation"],
    df_score["prediction_llama_exclure_pays"],
    average="macro",
)

# %%
# comparer les modèles
pd.crosstab(
    df_score["previous_annotation"],
    df_score["prediction_llama_exclure_pays"],
    margins=True,
    margins_name="Total",
)

# %% [markdown]
# ## Few-shot ?

# %%
# # Let's build the prompt, with some examples
# few_shot_examples = [
#         ("Biden signs executive order on student debt relief", "POLITICS"),
#         ("Amazon launches new AI-powered Alexa features", "OTHER"),
#         ("Congress debates military spending bill", "POLITICS"),
#         ("PSG wins much-anticipated Champions League", "OTHER")
#         ]

# def get_prompt_few_shots(text: str, examples: list[tuple] = few_shot_examples):
#   examples = "\n".join([f"Classify this headline:\n{headline}\nLabel: {label}\n\n" for headline, label in examples])
#   return [{"role":"system",
#            "content":"You are a strict news classifier. You must respond with one word only — either 'POLITICS' or 'OTHER'." +
#            "\n Do not explain. Do not output anything else."

#            },
#            {"role":"user","content": f"{examples}\nClassify this headline:\n\"{text}\"\nLabel:"}
#   ]

# %% [markdown]
# ## Refacto

# %%
# # Inspiration GPT pour avoir des checkpoints :
# results = []
# indexes = []
# batch_size = 20

# for i in range(0, len(df), batch_size):
#     batch = df["Texte"].iloc[i:i+batch_size]
#     try:
#         r_batch, _ = do_predictions(get_prompt, batch, model="llama3.1:70b")
#         results.extend(r_batch)
#         indexes.extend(batch.index)

#         # Checkpoint every 100 samples
#         if i % 100 == 0:
#             df.loc[indexes, "prediction_simple"] = results
#             df.to_csv("checkpoint_predictions.csv", index=False)
#             print(f"Checkpoint sauvegardé à {i}.")

#         time.sleep(1)  # optional pause

#     except Exception as e:
#         print(f"Erreur au batch {i}: {e}")
#         continue

# df.loc[indexes, "prediction_simple"] = results
# df.to_csv("final_predictions.csv", index=False)


# %%
# # Et même inspiration pour un truc encore plus complet, mais overkill pour l'instant

# import os
# import time
# import pandas as pd
# from datetime import datetime


# def ensure_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)


# def timestamp():
#     return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# def do_predictions(prompt_func, texts, model):
#     prompt_name = prompt_func.__name__
#     results = [prompt_func(text) for text in texts]
#     return results, prompt_name


# def run_llm_batches(df, text_column, prompt_func, model,
#                     column_name=None, batch_size=20, sleep_time=1,
#                     checkpoint_every=100, output_dir="outputs"):

#     # Setup
#     start = time.time()
#     all_results = []
#     all_indexes = []
#     name = prompt_func.__name__ if column_name is None else column_name

#     ensure_dir(output_dir)

#     print(f"🚀 Début du traitement avec le prompt '{name}' et le modèle '{model}'")

#     # Boucle par batch
#     for i in range(0, len(df), batch_size):
#         batch = df[text_column].iloc[i:i + batch_size]
#         idx = batch.index

#         try:
#             batch_start = time.time()
#             r_batch, _ = do_predictions(prompt_func, batch, model=model)
#             all_results.extend(r_batch)
#             all_indexes.extend(idx)
#             print(f"✅ Batch {i}-{i + batch_size} traité en {time.time() - batch_start:.2f}s")

#             # Sauvegarde périodique
#             if (i // batch_size) % (checkpoint_every // batch_size) == 0:
#                 df.loc[all_indexes, name] = all_results
#                 ckpt_file = os.path.join(output_dir, f"checkpoint_{name}_{timestamp()}.csv")
#                 df.to_csv(ckpt_file, index=False)
#                 print(f"💾 Checkpoint enregistré : {ckpt_file}")

#             time.sleep(sleep_time)

#         except Exception as e:
#             print(f"⚠️ Erreur au batch {i}-{i + batch_size}: {e}")
#             continue

#     # Résultat final
#     df.loc[all_indexes, name] = all_results
#     final_file = os.path.join(output_dir, f"final_{name}_{timestamp()}.csv")
#     df.to_csv(final_file, index=False)

#     print(f"🏁 Traitement terminé en {time.time() - start:.2f}s")
#     print(f"📝 Résultat final enregistré : {final_file}")
#     return df


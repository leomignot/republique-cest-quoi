# La République c'est quoi ?

Première ébauche d'analyse des comptes rendus des débats à l'Assemblée nationale (2017-2024) faisant mention de la famille du mot "République".  
Collaboration entre [Léo Mignot](https://www.centreemiledurkheim.fr/notre-equipe/leo-mignot/) et [Matthias Levalet](https://www.centreemiledurkheim.fr/notre-equipe/matthias-levalet/) sur l'objet de thèse de ce dernier.  
Projet initié dans le cadre de [SICSS-Paris 2025](https://sicss.io/2025/paris/).

> [!NOTE]
> Projet en cours : première ébauche, résultats non stabilisés.
>
> **Reproductibilité** : le pipeline n'est pas encore entièrement reproductible.
> - L'étape 2_3 utilise des données générales sur les députés (fichier Datan) dont la mise à disposition est en attente d'autorisation.
> - L'étape 5_1 (embeddings) nécessite une clé d'accès au serveur Ollama (Huma-Num). Le fichier des embeddings calculés est mis à disposition (TODO lien à ajouter).
> - **TODO** : l'environnement Python (uv ou conda) reste à stabiliser.

## Pipeline

| Étape | Scripts (`scripts/`) | Description |
| --- | --- | --- |
| 1 | `1_1` à `1_3` | Extraction des XML de l'AN, fusion des législatures, regroupement des interventions |
| 2 | `2_1` à `2_5` | Filtrage, identification des acteurs, appariement aux députés, affiliations partisanes et commissions |
| 3 | `3_1` | Repérage des occurrences de « République » |
| 4 | `4_1`, **TODO** | Statistiques |
| 5 | `5_1`, **TODO** | Embeddings, topic modeling |

Chaque script `.py` est jumelé à un notebook dans `notebooks/` (via [jupytext](https://jupytext.readthedocs.io/)).

## Détail des scripts

```bash
run_pipeline.py / run_pipeline.sh           # orchestration : enchaîne les scripts ci-dessous (journal dans logs/ pour .sh)

scripts/1_1_extraction.py                   # lecture des XML de l'AN, exclusion doublons/congrès
scripts/1_2_fusion_legislatures.py          # concaténation 15e + 16e, déduplication
scripts/1_3_regroupement_interventions.py   # fusion interventions interrompues
scripts/2_1_filtrage.py                     # exclusion interventions non exploitables, nettoyage textes
scripts/2_2_identification_acteurs.py       # fusion id_acteur / id_orateur, correction erreurs d'identification
scripts/2_3_match_deputes.py                # jointure données députés (groupe, mandat, etc.)
scripts/2_4_affiliations.py                 # affiliation (groupe/gouv) à date de l'intervention
scripts/2_5_commissions.py                  # affiliation commission à date de l'intervention
scripts/3_1_identification_republique.py    # repérage des mentions de « républi* », exclusion faux positifs
scripts/4_1_analysis_stats.py               # statistiques descriptives, dynamique temporelle (exploratoire)
scripts/5_1_embeddings.py                   # embeddings des interventions
**TODO** LA SUITE
```

<!--

Sorties intermédiaires (`data/interim/`) : `1_1_extract_{15,16}.csv` → `1_2_extract_15_16_concat.csv` → `1_3_interventions_regroupees.csv` → `2_1_filtrage_nettoyage.csv` → `2_2_identification_acteurs.csv` → `2_3_match_deputes.csv` → `2_4_interventions_nettoyees.csv` → `3_1_df_repu.csv` (lignes avec mention) et `3_1_df_repu_proportion.csv` (toutes lignes, avec colonnes de match).

-->

**TODO**: `run_pipeline.*` ne lance pour l'instant que les étapes 1_1 à 3_1 ; `4_1` et `5_1` s'exécutent à part + enjeu clé API

Fichiers `test_*` dans `scripts/` et `notebooks/` : essais exploratoires, hors pipeline.

## Utilisation

```bash
git clone https://github.com/leomignot/republique-cest-quoi.git
cd republique-cest-quoi
# TODO : environnement Python (uv ou conda) à stabiliser ; en attendant, installer les dépendances à la main
# télécharger les XML (voir « Données ») dans data/raw/15-xml et data/raw/16-xml
# + fichiers Datan (députés) dans data/raw/ : en attente d'autorisation de mise à disposition (étape 2_3)
python run_pipeline.py   # ou ./run_pipeline.sh (journal dans logs/)
```

## Structure

```
data/       raw/ (sources), interim/, processed/
scripts/    pipeline numéroté
notebooks/  versions notebook des scripts et explorations
reports/    figures
debug/      vérifications (ex. comparaison avec NosDéputés)
archive/    anciennes versions
```

## Données

- Comptes rendus des débats en séance publique ([open data de l'Assemblée nationale](https://data.assemblee-nationale.fr/)) :
  - [15e législature](https://data.assemblee-nationale.fr/archives-anterieures/archives-15e/debats-en-seance-publique)
  - [16e législature](https://data.assemblee-nationale.fr/archives-16e/debats)
- Contrôles de cohérence : [NosDéputés.fr](https://www.nosdeputes.fr/) (Regards Citoyens)

## Licence et citation

**TODO** :)

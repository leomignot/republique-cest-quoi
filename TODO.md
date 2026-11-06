# TODO

## Général

## 1-data-extraction

## Global

- passer tout sous UV
- finaliser pipeline
- ajouter infos sur téléchargement fichiers et ou option exécution code
- ajouter liste pays txt pour reproductibilité
- REPRENDRE DEMANDE AJOUT COMMISSIONS ETC DÉPUTÉS POUR MATTHIAS
- Aviser si fichier datan diffusables. Sinon voir pour reproductibilité
- REPRENDRE LE **README** !!!!

## À voir

## 1-extraction

## 2-clean&filter

- [ ] 2.4 affiliation : depuis le doc mandats de la datan, repasser leur commission d'appartenance + leur "codeQualite" et libelleAbrev = REPRENDRE DEMANDE AJOUT COMMISSIONS ETC DÉPUTÉS POUR MATTHIAS
  - [ ] FAIT EN V0, ATTENTE CONFIRMATION MATTHIAS !!!

## 3-identify-republic

### nettoyer les textes ?

- [ ] RAS ?

## 4-analysis ?

- [ ] repartir de ce qu'a fait matthias, mettre à plat, vérifier, stabiliser, améliorer, etc.
- [ ] Analyses.ipynb pour le fichier ok de matthias.
- [ ] envisager stat en nb occurrence, % des interventions, et même chose hors interruption(doc actualiser numérateur et dénominateur)
- [ ] aviser en % nb mots / ngrams ?
- [ ] faire des sorties vizu pour avoir une base de discussion de ce qu'on veut (= avoir des exemples pour imaginer ce qui est réalisable et utile)

### dates

- [ ] Vérif la conversion : visiblement des outliers -> vérif ??

## 5-embeddings et topic-modeling

### 5_1_embeddings

- [ ] aviser si nécessaire de refaire tourner une fois le fichier final export repu consolidé
  - [ ] à ce jour : toujours ok sur nb interventions, mais probalbme ajout de colones info députés entre temps (comission, etc.), voir pour une fusuion si nécessaire
- [ ] Aviser les possibles embeddings qwen, alibaba et explo topic modelling depuis activetigger
- [ ] **decider du focus.découpe sentence/paragraph/intervention/etc.**
  - [ ] Option retenue à ce jour = full intervention (ok taille modèle, mais on mesure en effet pas la meme chose : thème interv vs thème lorsque mobilise précisément république. = à quelle occasion plus que purement pour dire quoi)
    - sentence ?
    - paragraphe (existe pas ici) -> mais possible avec wtpsplit : https://github.com/segment-any-text/wtpsplit
    - dire que tant pis ? Ok si utilise modèle avec fenetre suffisante (ici ok avec qwen)
    - Passer à l'échelle de la phrase pour tout ce qui va être pour sentiment, réseau de mots, proba des termes, etc.
    - etc.

### 5_2_topic_modeling

- [ ] revoir regroupement des topics
  - [ ] idée de le faire chacun séparément pour confronter ensuite
  - [ ] jouer sur la seed (pas pour optimiser mais à l'inverser vérifier risque variabilité)
  - [ ] jouer sur paramètres UMAP / HDBSCAN : mais ici on préfère un travail de recoupement fin à la main.
- [ ] aviser genAI sur le nom des topics ? -> meh.

## Pistes, etc.

Analyses.ipynb pour le fichier ok de matthias.

## bertopic regroupement

VÉRIF NOTRE DAME DES LANDES VS CATHÉDRALE NOTRE DAME ET EXPLOSER LA CATÉGORIE violet.
90 = aéroport
vs cathédrale 141 -> non ok surtout nahel et émeutes
vs 85 pompiers - > plus chelou, sécurité civile, etc. -> mais il est a un autre embrachement à part quand tout le reste fait un groupe. Ne pas le fusionner à la couleur et grader à part puis voir si mieux ailleurs ?

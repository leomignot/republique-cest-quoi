# TODO

## Général

## 1-data-extraction

## Global

- passer tout sous UV
- finaliser pipeline
- ajouter infos sur téléchargement fichiers et ou option exécution code
- ajouter liste pays txt pour reproductibilité

## À voir

## 1-extraction
- [ ] SOUCIS REGROUPEMENT INTERVENTIONS FOIRE -> nouvelle version semble okayish
  - toujours un mini écart, semble plus stable de pas trier les fichiers par numéro ordre  :
  - les interventions sont déjà dans l'ordre, les id numérotés semblent pas super stables ?
  - et donc au final le résultat semble meilleur sans réappliquer de tri. Mais pq cet écart ?
  - tester avec le id_syceron ? -> test réalisé, c'est naze.
- [X] EXCLURE LAMARTINE OU PAS ? -> DONE = non : passage des acteurs restant en externes
- [X] récupération des points de contexte parents(cf tentative Matthias) = DONE
- [ ] check par matthias si contexte est OK.
- [ ] check repu against ND
- [ ] IMPORTANT : check si implementation gestion balises <br/> ET italique est ok
  - [ ] aviser si veut pareil pour balises exposants (AN gère pas toujours de manière constante le fait de mettre un espace final avant fermeture de l'exposant - )parfois voulu parfois non ?)
    - [ ] ex : Vous êtes le plus jeune vice-président de l’histoire de la V<exposant>e</exposant><exposant/>République.
    - [ ] cas des n°-> l’amendement n<exposant>o </exposant>359. -> no 359 VS dans ND n°359 (ou n° 359)
  - [ ] les … sont pas toujours suivis d'espaces chez nous (alors que ND oui ?), mais c'est comme ça dans les xml d'origine (variation parfois oui parfois non en début d'intervention coupée, on peut pas tout avoir)
  - [ ] == cas limites (rares ?) :
  - [ ] In [40]: df_extract.texte.str.contains("VeRépubli").sum() Out[40]: 7
  - 
- [X] sans doute devoir l'implementer aussi pour récupération du niv point : des trucs qui passent sur plusieurs lignes
  - [X] IE :voir ce qui foire pour rappel réglement sur plusieurs lignes et aussi les points avec (suite) sur autre ligne
  - [X] ex : CRSANR5L15S2017E1N007, CRSANR5L15S2018E1N027, CRSANR5L15S2018O1N284
  - [X] en fait comme ça en partie dans fichiers originaux, correction : passage dans fonction gestion balises + supr séparé des espace multiples et retour ligne
  - 


## 2-clean&filter

- [X] voir la liste que je sors des sans affiliations (pas nombreux)
- [X] Aviser cas gouvernement
- [X] Aviser cas affiliation multiples (ex gauche sans groupe comme RN, etc.)
- [X] Aviser cas houplain NI/RN
- [X] Aviser lamartine et pb soucis identification -> DONE avec le passage en externes
- [X] syceron 2827575 2827576 2827577 = M. Lionel Tivoli = PA793298 ?
  - [X] sans doute pas : cas ultra spécifique et doit y en avoir d'autres (voir point suivant)
  - [X] désormais géré par le fait que conserve une trace sur-imprimée de nom_orateur sur nom_orateur_clean quand on en a pas si PA0
- [X] Vérif cas de nom orateur sans nom orateur clean plus qu'1 (cf depuis gestion en cas de PA0)
- [X] Vérif cas de nom orateur clean sans nom orateur -> pas concluant ~13 cas (président séances autres mal identif (chenu, laporte)


## 3-identify-republic

- [X] ON PERD DES OCCURRENCES AVEC LE NETTOYAGE TEXTE -> normal, espaces multiples mal gérés par regex si pas nettoyé avant
- [X] check "\t" vs rien dans liste pays république -> DONE
- [X] ajouter les nouveaux cas identifiés
- [X] aviser avec la nouvelle remontée d'exclusions possibles.
- [X] NOTE : quelques (~10) "république islamique" sans précision pour parler de l'Iran, mais risque de supprimer d'autres occurrences que l'on veut garder. Ou alors aviser majuscule a République vs sans ? -> niche, pas fait, check matthias
- [ ] voir matthias pour un check des ajouts réalisés avec si c'est ok
- [ ] check repu against ND

### nettoyer les textes ?

- [X] ENJEU DES ACCENTS À TESTER ! DONE fait avec unicodedata, pas de diff -> introduit quand même dans nettoyage 2_1
- [ ] RAS ?

## 4-analysis ?

- [ ] repartir de ce qu'a fait matthias, mettre à plat, vérifier, stabiliser, améliorer, etc.
- [ ] envisager stat en nb occurrence, % des interventions, et même chose hors interruption(doc actualiser numérateur et dénominateur)

### dates

- [ ] Vérif la conversion : visiblement des outliers -> vérif ??

### Bert, etc

- [ ] Aviser les possibles embeddings qwen, alibaba et explo topic modelling depuis activetigger
- [ ] **decider du focus sentence/paragraph/intervention/etc.**
- [ ] revoir regroupement des topics
- [ ] revoir Topic distribution
- [ ] aviser genAI sur le nom des topics ? -> meh.

## Pistes, etc.


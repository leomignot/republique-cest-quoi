# Journal de comparaison : mentions de "République"

Extrait du script `3_1_identification_republique.py`, pour ne pas alourdir
le code. Comptages obtenus lors des premiers tests de la fonction
`contains_lexical_outside_excl` (avec exclusions), selon la source du corpus.  
Actualisé depuis changement extraction (br/italique) / changement nettoyage / changement regex

## 1. Résumé global regex repu vs différents fichiers

### 1.1 Match regex avec exclusions lexicales

NOTE : colle pas pile à la shape df avec Nan,
(ou incongruences entre versions si vire ou pas selon présence combinée id_orateur/id_acteur/nom_orateur, etc.)

| Source                                    | Shape       | False       | True   |
|--------------------------------------------|------------:|------------:|-------:|
| Fichier NosDéputés (ND15+16_interventions_hemicycle_rich.tsv)      | 1391207   | 1377532 (1377532)   | 13675 (13675 nettoyé) |
| Extraction brute (1_2_extract_15_16_concat.csv)        | 1127829   | 1114011 (1114734)   | 13818 (13095 nettoyé) |
| Fichier maison (2_4_interventions_nettoyees)          | 517023   | (506192) 505725   | (11298 brut) 10831 (net = de base) |

**Écart observé :**
- NosDéputés : 13675 mentions valides (nettoyé comme brut)
- Extraction nettoyée : 13095 mentions valides sur texte nettoyé (13818 sur brut)
- Différence : 580 occurrences
- (pour rappel sur fusion interruptions : 10831 (11298 sur brut))

**TODO :**
- comprendre les ~600 occurrences présentes côté NosDéputés mais absentes de l’extraction nettoyée
- distinguer :
  - différence de périmètre ;
  - découpage différent des interventions ;
  - perte lors extraction/nettoyage.

### 1.2 Effet du nettoyage texte brut vs texte nettoyé

Dans le texte brut, les espaces multiples empêchent certaines exclusions.

Ex d'cart identifié : "le Président de                            la République"

Conclusion :

- la normalisation des espaces modifie le résultat de la regex ;
- le texte brut produit davantage de faux positifs car certaines expressions exclues ne sont plus reconnues.

### 1.3 Impact de la présence d'un speaker sur les matchs regex

La comparaison avec et sans restriction aux interventions disposant d'un speaker montre que les matchs valides de la regex repu dépendent très peu des interventions sans speaker (50aine de cas).  

> À noter que ce résultat est très différent de l’analyse des écarts de volumétrie brute entre fichiers : les lignes sans intervenant expliquent une grande partie des différences de structure (voir journal dédié vs nd), mais contribuent (très) peu aux matchs regex valides.

| Texte        | Corpus     | Sans filtre speaker | Avec speaker | Écart |
| ------------ | ---------- | ------------------: | -----------: | ----: |
| `texte_brut` | NosDéputés |              13 675 |       13 623 |   -52 |
| `texte_brut` | Extraction |              13 818 |       13 817 |    -1 |
| `texte_net`  | NosDéputés |              13 675 |       13 623 |   -52 |
| `texte_net`  | Extraction |              13 095 |       13 095 |     0 |

> NOTE : la 50 aine de cas côté NosDéputés semble être liée à la "confusion" entre texte vs niveau point (voir plus bas)

Les écarts observés entre NosDéputés et l'extraction ne semblent donc pas principalement liés à la présence ou absence d'un orateur associé aux interventions.

Les divergences proviennent plutôt (voir plus bas) :

- des différences de découpage des interventions
- des différences de contenu textuel
- des traitements de nettoyage et normalisation
- des modalités d'extraction des textes
- des vrais soucis de diff (à identifier)

### 1.4 Match "républi" brut sans aucune exclusion de termes (pour trace)

Pour trace et idée (sans nettoyage spécifiques ni verif speakers etc.)
NOTE : colle pas pile à la shape df avec Nan etc.

Pour référence, `pattern_lexical = re.compile(r"républi", re.I)` seul,
sans passer par `contains_lexical_outside_excl` :

(NOTE : avant modif extraction et gestion balises, mais doit pas compter ici car pas exclusion)

| Source                                    | False       | True   |
|--------------------------------------------|------------:|-------:|
| Interventions groupées                      |   485 683   | 31 340 |
| Idem, avec normalisation NFC                |   485 683   | 31 340 |
| Extraction brute (sans regroupement)        | 1 083 667   | 44 089 |
| NosDéputés 15+16                            | 1 345 638   | 45 569 |

Comparaison externe (non recalculée dans ce projet) :
site [an-4931d4.gitpages.huma-num.fr/debats-AN](https://an-4931d4.gitpages.huma-num.fr/debats-AN#tableau-complet)
annonçant ~2 010 738 interventions en mémoire (2011-2026) pour 33 694
interventions matchées — période plus longue que la nôtre mais total de
matches inférieur, à creuser (source potentiellement moins exhaustive sur
cette période, ou méthode de comptage différente).

## 2. Analyse des divergences regex repu

Exploration des cas LIMITES.

### 2.1 Comparaison absence par pnum vs_idsyceron

> NOTE : comparaison effectuée dès première version sur les textes bruts, affiner avec les exclusions depuis textes nettoyés

Pas forcément la vraie/seule source différence mais aide comparaison par pnum / id_syceron
(cf exploration manuelle de `nd_pnum_absentes_de_brut_match.csv`)

- Flag contenu dans des **fichiers que l'on exclut** de notre côté (**congrès**, doublons, etc.)
- Même sans fusion interventions, **pas même gestion du nb lignes par interventions** (parenthèses applaudissements dans l'intervention chez nous, renvoi à une autre inter chez ND)
  - ex : 15 vs 18 interv pour la déclaration politique générale du 4 juillet 2017
- **Mauvais pnum** mais bien dans le df maison qui match :
  - liste par ex : pnum 1381312 (vs id 1381415), 1381313 (vs encore 1381415), 1433638 (vs 1433583), 1627250, 2396860, 2397067, 2397071, 2480254
- **Texte vs niveau point** : "erreur" ND avec passage en texte de l'intervention ce qui est en fait pour nous le niveau de discussion (et donc pas d'orateur associé)
  - voir ex dessous (Respect des principes de la République, Valeurs républicaines à l'école, etc.)
  - cf pnum 2337471, 2384869, 2385355, 2388583, 2388644, 2390359, 2391120, 2391958, 2393319, 2394501, 2394572, 2396343, 2397284, 2399758, 2402544, 2403554, 2404746, 2405372, 2407066, 2407841, 2408605, 2410818, 2411493, 2416482, 2567101, 2568277, 2569825, 2569941, 2571429, 2571899, 2572873, 2574330, 2597558, 2771128, 2975816, 2976329, 3096183, 3121590, 3194896, 3453420, 1581096, 1611606, 1626627, 1805669, 2239001, 2239014

Exemple :
```
<p>Prééminence des lois de la République</p>, <p>Respect des principes de la République</p>, <p>respect des principes de la république</p>, <p>Convention relative à la nationalité entre la République française et le royaume d'Espagne</p>, <p>Dissolution des groupuscules fascistes et antirépublicains</p>, <p>État de l'école de la République</p>, <p>Fonds Marianne pour la République</p>, <p>Valeurs républicaines à l'école</p>, <p>Arc républicain et extrême droite</p>, <p>Prestation de serment d'une juge suppléante à la Cour de la République</p>,
# et ceux là (pris après nettoyage, mais sinon y a les balises p aussi normalement)
Attaques contre les élus de la République, Attaques contre la République et les institutions démocratiques, Quartiers de reconquête républicaine en Seine-Saint-Denis, Quartiers de reconquête républicaine, Crise de la République, Valeurs de la République à l'école, Valeurs de la République à l'école
```

### 2.2 Comparaison par snippets de texte
> NOTE : ici sur les textes nettoyés pour commensurabilité

> NOTE : observations reposant sur automatisation script + exploration manuelle des fichiers sortie

- CHECK : des cas ou snippet peut pas bien comparer car **normalisation des textes** (avant même fonction nettoyage) colle pas ?
  - TODO souci fonction nettoyage qui marche bien mais qui plante dans le fichier extract ?
  - fait pour br et italique -> retester
- Snippet échoue lorsque texte extract **contenait des parenthèses** avant normalisation : elles virent bien avec le nettoyage, mais témoigne d'une différence de structure des fichiers =  **match pas avec format ND** qui renvoie **interventions séparées** si parenthèses de didascalies
  - Nombre de snippets ND absents de brut avec parenthèses : 32
  - Nombre de snippets brut absents de ND avec parenthèses : 2032
- Même idée **points suspension** ?
  - Fait (cf 2.3) : le check confirme que les points de suspension expliquent une part substantielle des "absents", plus importante en proportion que les parenthèses côté ND->extract (617 vs 32), et comparable côté extract->ND (1049 vs 2032). Chevauchement partiel entre les deux patterns (union croisée < somme simple, cf 2.3) : certaines lignes cumulent les deux causes.
  

Ex de cas de parenthèses : 
CRSANR5L15S2017E1N012,,,20170713150000000,jeudi 13 juillet 2017,2,12,AN,15,Première session extraordinaire 2017,20171012,Présidence de M. François de Rugy,Renforcement du dialogue social > Discussion des articles (suite) > Après l’article 3 (suite),4,Renforcement du dialogue social,Discussion des articles (suite),Après l’article 3 (suite),,Après l’article 3 (suite),(n[[o]] 19),Après_ 3,DISC_ARTICLES_3_1,1,1,62,PA717379,PM723282,DISC_ARTICLES_1_30_1,NORMAL,PAROLE_1_2,991112,,M. Sylvain Maillard,,717379.0,,"Je tenais à saluer votre première présidence de séance, monsieur le président. (Applaudissements sur les bancs des groupes REM et MODEM.) Nous sommes fiers de vous voir à cette place. Vous êtes le plus jeune vice-président de l’histoire de la VeRépublique. Félicitations ! Nous comptons sur vous. (Applaudissements sur les bancs du groupe REM.)","Je tenais à saluer votre première présidence de séance, monsieur le président. Nous sommes fiers de vous voir à cette place. Vous êtes le plus jeune vice-président de l'histoire de la VeRépublique. Félicitations ! Nous comptons sur vous.",1,True,"Je tenais à saluer votre première présidence de séance, monsieur le président. Nous sommes fiers de vous voir à cette place. Vous êtes le plus jeune v",False

### 2.3 Confrontation systématique (script factorisé) : orig/net × tous/speaker

Résultats du script de confrontation factorisé (4 configurations :
texte_brut / texte_net × tous / avec-speaker uniquement).

NOTE : la comparaison par pnum/id_syceron (identifiants) est valide dans les 4 configs.
La comparaison par snippets, elle, repose sur une égalité de sous-chaîne exacte entre les deux corpus.
Et elle n'est donc pertinente que sur texte_net, où les deux textes sont normalisés de façon comparable (balises, apostrophes,; espaces). (Et encore, ça foire pour d'autres raisons, plus pas parfait avec gestion des exposants, etc.)
Sur texte_brut, elle donne un taux d'"introuvable" artificiellement proche de 100%, donc non calculée / non retenue ici. (tests réalisés pour la science quand même et pour avoir une idée)

/!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ 
# TODO : creuser tout ce qui est ci-dessous
/!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ 

#### a) pnum vs id_syceron

> NOTE / TODO : ajouter une colone des matchs repu total, ça ajoute de la confusion sinon

| Configuration        | Lignes ND | Absentes de extraction | Dont mention valide de République |
|-----------------------|----------:|--------------------------:|-----------------------------------:|
| texte_brut             | 1 391 207 |                     45 410 |                                 130 |
| texte_brut_speaker     | 1 088 105 |                        841 |                                  81 |
| texte_net              | 1 391 207 |                     45 410 |                                 130 |
| texte_net_speaker      | 1 088 105 |                        841 |                                  81 |



**Constats :**


- Les comptes sont identiques entre `texte_brut` et `texte_net` (130 / 81) :
  sur ce sous-ensemble précis (lignes ND absentes de l'extraction), les
  variantes brut/net de la regex ne divergent pas
- Le filtre "avec speaker" fait chuter les lignes absentes de 45410 à 841, mais beaucoup moins concernant les match repu valides. Cf : cohérent avec identification précédente sur le fait que ne semble pas se jouer au niveau de l'absence de speakers, sauf cas marginaux (textes vs niv point, etc.).

> **NOTE / TODO : ALLER CREUSER LES 81 QUI TRAINENT ICI**
> Et creuser l'écart entre les quasi 600 et juste 81 ici = qui sont les autres.

#### b) Snippets (texte_net uniquement)

| Config             | Direction       | Introuvable / total | dont parenthèses | dont points_suspension | dont ≥ 1 pattern (croisé) |
|--------------------|-----------------|--------------------:|-----------:|-------------:|--------------:|
| texte_net          | ND -> extract    |      1 229 / 13 675 |         32 |          617 |           642 |
| texte_net          | extract -> ND    |      2 577 / 13 095 |      2 032 |        1 049 |         2 362 |
| texte_net_speaker  | ND -> extract    |      1 209 / 13 623 |         32 |          617 |           642 | 
| texte_net_speaker  | extract -> ND    |      2 579 / 13 095 |      2 033 |        1 050 |         2 364 |

**Constats :**

- Le filtre speaker ne fait presque pas bouger les deux comptes ND->extract (-20) ou extract -> ND (+2). Note : l'augmentation marginale > possibles matchs qui étaient dans le big blob et en sont supprimés avec leur intervention sans speakers (avec découpe intervention qui n'est pas la même, etc.)
- TODO DÉTAILLER RAPIDEMENT PATTERNS AVANT POINT DESSOUS
- Union croisée (`check_au_moins_un`) des patterns : léger chevauchement de lignes qui cumulent parenthèses et points de suspension en en ND->extract (7 lignes) plus "franc" pour extract->ND (719) en ND->extract [fonction appliquée sur les textes originels/bruts].
- Vérifier si ces patterns révèlent bien une différence de structure sous-jacente et donc un découpage des interventions qui empêchent le match.
- Il reste dans tous les cas un residu non expliqué par ces deux patterns : **587** côté ND->extract et **215** côté extract->ND (texte_net).
- Explorer les cas :
  - exploration manuelle des fichiers
  - et c'est ce résidu qui mériterait l'exploration manuelle prioritaire

> TODO : EXPLORER LES CAS RESTANTS

#### c) Focus patterns

**Cas liés aux parenthèses :**  

ND absent dans extraction Total : 1229  
Avec parenthèses dans texte original : 32  

Extraction absente dans ND Total : 2577  
Avec parenthèses dans texte original : 2032  

**Cas liés aux points de suspension :**

ND absent extraction Avec points de suspension : 617  
Extraction absente ND Avec points de suspension : 1049  

Hypothèse : les points de suspension peuvent donc révéler :

- découpage différent des interventions
- troncature
- fusion/séparation de blocs.

/!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ 
# TODO :
- explorer les cas, notamment après nettoyage speakers
- exploration manuelle des fichiers correspondants
- prioritairement ND abs de extract (l'autre sens à la limite pour la science et identifier des patterns)

# Analyse complète en vrac


pnum / syceron


**SNIPPETS**
Exploration des cas ND absents de notre extraction sur base snippets sur texte net + speaker.
Première vérif
== fichier nd_snippets_absent_de_extract_texte_net_speaker.csv
Bien des bugs identification sur … pas toujours suivis chez nous d'espaces en debut d'intervention coupée (comme ça dans xml, varie), alors que ND est stable ?

Puis réduit à ceux qui présentait pas de spécificité parenthèses ou … (check_au_moins_un == False)


- Fichiers congrès
une fois réduit à sans pattern spécifique :
2018-07-09    50
2017-07-03    13
(14 et 56 avant réduction)

**Mise en forme et patterns et différence snippet**

- le cas des exposants :
  - cas des n°-> l’amendement n<exposant>o </exposant>X. -> no X VS n°X (ou n° X)
    - 17 "°"
  - des exposants e sans espaces ?
    - 7 " Ve " (vs parfois " VeRépublique") et 1 (" VeR").sum()
- les espaces après tirets
  - 81 ("–"), dont 48 ("– ,") 80 ("– .")
  - "– il n'est pas là – ," vs nous qui devient bien "finances – il n'est pas là –, "
  - (ils ont peut être bourré espace après balises partout pour être surs ?)
- caractères spéciaux ?
  - œ nous vs oe ND (coeur, oeuvre, voeu, etc.) -> 73
  - pattern = "coeur | oeuvr | voeux | voeu | Woerth | oeuvr | soeur | oeillères | manoeuvre | oeuvré | soeurs | oeuvrais | Coeur | oeuvres | oecuménisme | manoeuvres | oeil" 
  - In [68]: df.snippet.str.lower().str.contains(pattern).sum() Out[68]: 73
    - dont : In [56]: df.snippet.str.contains("coeur").sum() Out[56]: 69
    - In [58]: df.snippet.str.contains("voeu").sum() Out[58]: 7
    - In [59]: df.snippet.str.contains("oeuvre").sum() Out[59]: 21


**Après intégration fonction pattern :**

ND (repu) introuvable dans extract : 1209 / 13623
  dont patterns (dans le texte original) :
    dont avec parentheses : 32
    dont avec points_suspension : 617
    dont avec caractere_œ : 33
    dont avec mots_oe : 234
    dont avec tiret_avant_ponctuation : 145
  dont avec au moins un pattern (croisé) : 851

Différence de fichiers :
- œ (nous) vs oe (eux) : œuvre
- rapporteure vs rapporteuse
- no vs nos (numéros)
  - avoir accepté de réintroduire dans la discussion les amendements nos (308 et 309) "no 308 et 309"
- madame la rapporteure vs nous : madame la rapporteuse. notre version en ligne (dans madame la rapporteure, mes chers collègues, il faut donc que tout change pour que rien ne change) https://www.assemblee-nationale.fr/dyn/15/comptes-rendus/seance/session-extraordinaire-de-2016-2017/premiere-seance-du-lundi-24-juillet-2017#P998688
Autres diff fichier :
qui veut la faire haïr ? « Si je vs VS la faire haïr ? Si je n'étais (notre version qui est en ligne) https://www.assemblee-nationale.fr/dyn/15/comptes-rendus/seance/2e-session-extraordinaire-de-2016-2017/deuxieme-seance-du-mardi-26-septembre-2017#P1024732
nous signalements pour VS signalementspour (nous en ligne)


TESTS fuzzy snippets

ND absents analysés fuzzy : 1209  
Dont fuzzy >= 95 : 966  
Dont fuzzy >= 85 (vérif manuelle ok): 997  
Extract absents analysés fuzzy : 2579  
Dont fuzzy >= 95 : 980  
Dont fuzzy >= 85 (vérif manuelle ok): 1039  


# TODO : on devrait meme en fait déterminer des snipets autour de république, mais bon

# TODO : on devrait faire un total des count de match valides (vrai nb, pas présent abs) et voir si cohérent !
# BOURRIN :

NOTE : refaire au propre si veut être sur mais noramlement :


une fois réduits aux matchs :
df["nombre_mentions_repu_net"].sum()
18654

df_extract["nombre_mentions_repu_net"].sum()
18794

df_ND1516["nombre_mentions_repu_net"].sum()
18984

une fois réduit aux speakers et sans les congrès :

df_ND1516_with_speaker["nombre_mentions_repu"].sum()
18791

== on en a plus.
ARRÉTEZ TOUT !!!!

df_extract_with_speaker["nombre_mentions_repu_net"].sum()
18794

df_ND1516_with_speaker["nombre_mentions_repu_net"].sum()
18791

df["nombre_mentions_repu_net"].sum()
18654
-> mais dans le fichier regroupement on a certaines supressions et tout qui sont faites.






============================================================
CONFIGURATION : texte_brut
============================================================
---------- pnum vs id_syceron [texte_brut] ----------
Lignes ND : 1391207 | absentes de l'extraction : 45410
Dont mention valide de République : 130
============================================================
CONFIGURATION : texte_brut_speaker
============================================================
---------- pnum vs id_syceron [texte_brut_speaker] ----------
Lignes ND : 1088105 | absentes de l'extraction : 841
Dont mention valide de République : 81
============================================================
CONFIGURATION : texte_net
============================================================
---------- pnum vs id_syceron [texte_net] ----------
Lignes ND : 1391207 | absentes de l'extraction : 45410
Dont mention valide de République : 130
---------- snippets [texte_net] ----------
Lignes ND avec mention repu valide       : 13675
Lignes extract avec mention repu valide  : 13095
ND (repu) introuvable dans extract : 1229 / 13675
  dont patterns (dans le texte original) :
    dont avec parentheses : 32
    dont avec points_suspension : 617
    dont avec caractere_œ : 33
    dont avec mots_oe : 234
    dont avec tiret_avant_ponctuation : 145
  dont avec au moins un pattern (croisé) : 851
Extract (repu) introuvable dans ND : 2577 / 13095
  dont patterns (dans le texte original) :
    dont avec parentheses : 2032
    dont avec points_suspension : 1049
    dont avec caractere_œ : 492
    dont avec mots_oe : 7
    dont avec tiret_avant_ponctuation : 0
  dont avec au moins un pattern (croisé) : 2413
============================================================
CONFIGURATION : texte_net_speaker
============================================================
---------- pnum vs id_syceron [texte_net_speaker] ----------
Lignes ND : 1088105 | absentes de l'extraction : 841
Dont mention valide de République : 81
---------- snippets [texte_net_speaker] ----------
Lignes ND avec mention repu valide       : 13623
Lignes extract avec mention repu valide  : 13095
ND (repu) introuvable dans extract : 1209 / 13623
  dont patterns (dans le texte original) :
    dont avec parentheses : 32
    dont avec points_suspension : 617
    dont avec caractere_œ : 33
    dont avec mots_oe : 234
    dont avec tiret_avant_ponctuation : 145
  dont avec au moins un pattern (croisé) : 851
Extract (repu) introuvable dans ND : 2579 / 13095
  dont patterns (dans le texte original) :
    dont avec parentheses : 2033
    dont avec points_suspension : 1050
    dont avec caractere_œ : 492
    dont avec mots_oe : 7
    dont avec tiret_avant_ponctuation : 0
  dont avec au moins un pattern (croisé) : 2415
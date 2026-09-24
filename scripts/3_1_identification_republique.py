# %% [markdown]
# # 3-1 - Identifier les textes mentionnant la République
# Lit `2_4_interventions_nettoyees.csv` (issu de 2_4). Repère les mentions
# valides de la famille du mot "républi" en excluant les faux positifs
# (Les Républicains, président de la République, pays nommés "République de…", etc.)
# via une logique d'exclusion par positions.
# Écrit `df_repu_proportion.csv` (toutes lignes, avec colonnes de match) et
# `df_repu.csv` (seules les lignes matchées).

# %%
import pandas as pd
import re

PATH_ENTREE = "../data/interim/2_4_interventions_nettoyees.csv"
PATH_SORTIE_PROPORTION = "../data/interim/3_1_df_repu_proportion.csv"
PATH_SORTIE_MATCHES = "../data/interim/3_1_df_repu.csv"
PATH_LISTE_PAYS = "../data/raw/liste_pays_republique_stable.txt"

# %%
df = pd.read_csv(PATH_ENTREE, low_memory=False)
print("Shape du df chargé : ", df.shape)

# %% [markdown]
# ## Nettoyage du texte : déjà fait en amont
# nb : le nettoyage basique du texte (balises, parenthèses, espaces,
# apostrophes) a déjà été appliqué à l'étape 2-1 (voir
# `2_1_filtrage.py`). La fonction `nettoyer_texte()` utilisée là-bas est
# strictement identique à celle qui était ici à l'origine (vérifié par diff
# le 07/07/2026) : on ne la réapplique donc pas, pour éviter un travail
# redondant et le risque d'écraser `texte_brut` sans raison.
# Conservée ci-dessous en commentaire pour référence/traçabilité, à
# réactiver uniquement si ce script est un jour utilisé à partir d'un
# fichier qui n'aurait pas déjà été nettoyé par 2_1.

# %%
# NOTE : après tests, pas de différence après normalisation unicodedata NFC :
# Néanmoins conservé pour harmonisation des données

# Trace fonction nettoyage (voir 2_1_filtrage.py)
# def nettoyer_texte(texte):
#     if not isinstance(texte, str):
#         return ""
#     # Normaliser les caractères Unicode
#     texte = unicodedata.normalize("NFC", texte)
#     # Décoder les entités HTML
#     texte = html.unescape(texte)
#     # Supprimer les balises HTML/XML > espace (éviter collage de mots)
#     texte = re.sub(r"<[^>]+>", " ", texte)
#     # Supprimer contenu entre parenthèses
#     # NOTE : CHOIX FORT SELON CE QUI VEUT ÊTRE ÉTUDIÉ
#     # Supprime des didascalies ("Applaudissements", etc.)
#     # mais aussi tout autre contenu entre parenthèses
#     # ne gère pas les parenthèses imbriquées mais sont extrêmement rares (parfois sur (e))
#     texte = re.sub(r"\([^()]*\)", "", texte)
#     # Uniformiser apostrophes (utile pour regex)
#     texte = texte.replace("’", "'").replace("\u02bc", "'")
#     # Normaliser les espaces (après unescape(), couvre \xa0, \t, \n)
#     # et supprimer les espaces multiples
#     texte = re.sub(r"\s+", " ", texte).strip()

#     return texte


# df["texte_brut"] = df["texte"]  # garder une version brute du texte
# df["texte"] = df["texte"].apply(nettoyer_texte)

# Garde-fou générique (indépendant du nettoyage) : au cas où des lignes
# auraient un texte manquant ou vide à ce stade du pipeline.
# (normalement déjà filtré en amont, mais on double-check ici pour éviter des erreurs)
df = df[df["texte"].notna() & (df["texte"] != "")]
print("Shape après garde-fou texte manquant/vide : ", df.shape)

# %% [markdown]
# ## Regex
# Logique de l'identification des mentions valides de République :
# - regex sur la famille du mot "républi"
# - mais exclusion de certains termes (positions, pas de chaînage) car les
# termes exclus peuvent apparaître aussi avec les termes voulus (les idées
# républicaines sont menacées par Les Républicains) : chaîner risquerait
# de virer des occurrences qu'on aurait voulu garder.

# %%
# préparer les pays à exclure
with open(PATH_LISTE_PAYS, "r", encoding="utf-8") as f:
    liste_pays = [line.strip() for line in f]

# pattern regex pour les pays, rendu non capturant plus bas
pattern_pays = r"|".join(re.escape(p) for p in liste_pays)

# Regex de la famille du mot République (simplifié ici)
pattern_lexical = re.compile(
    r"républi",  # même au milieu des mots
    re.I,
)

# Regex des expressions à exclure
# logique : groupes (?:…) non capturant, utilisés juste pour les positions

# Expressions à exclure - casse exacte
# possible cas du féminin… mais pas d'occurrence dans la base avec nos exclusions
pattern_excl_case_sensitive = re.compile(
    # --- Exclusion des occurrences liées au parti les Républicains ---
    r"(?:\b[LlDd]es Républicains\b)"  # garde la casse pour identifier le parti (et pas un adjectif)
    r"|(?:\baux Républicains\b)"  # idem majuscule pour le groupe
    r"|(?:\b[Cc]ollègues? Républicains?\b)"  # cas avec et sans maj pour collègues
    r"|(?:\bsénateurs? Républicains?\b)"  # pas de maj sénateurs ou féminin dans la base après exclu, mais aviser
    r"|(?:\bdéputés? Républicains?\b)"  # pas de maj députés ou féminin dans la base après exclu, mais aviser
    # --- Spécifique corpus AN choisi ---
    r"|(?:\bLes Républicain\b)"  # typo manque s = spécifique corpus AN (4 occurrences)
    r"|(?:\bentre Républicains?\b)"  # spécifique corpus AN (4 occurrences)
    r"|(?:\bex-Républicains?\b)"  # spécifique corpus AN (4 occurrences)
    r"|(?:\banciens? Républicains?\b)"  # spécifique corpus AN (1 occurrence)
    r"|(?:\bseuls Républicains\b)"  # spécifique corpus AN (1 occurrence)
    r"|(?:\bparlementaires Républicains\b)"  # spécifique corpus AN (1 occurrence)
    r"|(?:\bélus Républicains\b)"  # spécifique corpus AN (1 occurrence)
    r"|(?:\bgroupeLes Républicains\b)"  # spécifique corpus AN (1 occurrence)
    r"|(?:\b[Nn]ous Républicains\b)"  # spécifique corpus AN (2 occurrences)
    r"|(?:\bcertains Républicains\b)"  # spécifique corpus AN (4 occurrences)
    r"|(?:\bamis Républicains\b)"  # spécifique corpus AN (1 occurrence)
    # --- Occurrences plusieurs partis ---
    r"|(?:\bdroite, Républicains et macronistes\b)"  # spécifique corpus AN (1 occurrence)
    r"|(?:\bRépublicains-Front national\b)"  # spécifique corpus AN (1 occurrence)
    r"|(?:\bMacronistes, Républicains, lepénistes\b)"  # spécifique corpus AN (1 occurrence)
    r"|(?:\bRassemblement national, Républicains et macronistes\b)"  # spécifique corpus AN (1 occurrence)
    r"|(?:\bparti Républicain\b)"  # spécifique corpus AN (1 occurrence : US)
    # --- Titres de presse ---
    r"|(?:\bL[’']Est républicain\b)"  # le journal
    r"|(?:\bLa Nouvelle République\b)"  # le journal
)

# Expressions à exclure - ignorer la casse
pattern_excl_case_insensitive = re.compile(
    # --- Partis et groupes politiques ---
    r"(?:\bgauche démocrate et républicaine\b)"  # premier sans |
    r"|(?:\bGauche démocrate républicaine\b)"  # variante
    r"|(?:\bGauche démocratique et Républicaine\b)"  # variante
    r"|(?:\bgauche démocrate et républicaine-NUPES\b)"
    r"|(?:\bsocialiste, écologiste et républicain\b)"
    r"|(?:\bgroupe socialiste et républicain\b)"  # garder "groupe" pour limiter flag
    r"|(?:\bcommuniste républicain citoyen et écologiste\b)"
    r"|(?:\brépublique en marche\b)"
    r"|(?:\bconstructifs : républicains, UDI, indépendants\b)"
    r"|(?:\bconstructifs : républicains, UDI et apparentés\b)"  # variante
    r"|(?:\bLes Indépendants - République et Territoires\b)"
    r"|(?:\bLes Indépendants-République et Territoires\b)"  # variante
    # NOTE : ont également été testés (0 cas ici, mais voir selon autres législatures)
    # "Union des démocrates pour la République" UDR  / "Union des droites pour la République"
    # "Union pour une Nouvelle République" / "Debout la République"
    # "Forum des républicains sociaux" / "Identité et République"
    # "Rassemblement pour la République" RPR =  1 cas -> mais risque faux positif (appel au rassemblement)
    # --- Fonctions et institutions ---
    r"|(?:\bprésidents? de la république\b)"
    r"|(?:\bprésidentes? de la république\b)"  # 7 cas pour féminiser la fonction ou souhaiter élection MLP
    r"|(?:\bprésidences? de la république\b)"
    r"|(?:\bprocureurs? de la république\b)"
    r"|(?:\bcours? de justice de la république\b)"  # nb : cours de sûreté est lui gardé car projet loi LR et pas une institution
    r"|(?:\badministration générale de la république\b)"
    r"|(?:\bgouvernement de la république française\b)"  # pas de pluriel dans corpus
    r"|(?:\bInstitut supérieur des langues de la République française\b)"
    r"|(?:\bHaut-commissariat de la République\b)"  # préfets en Kanaky et Polynésie française uniquement
    r"|(?:\bHaut-commissaire de la République\b)"  # ibid
    r"|(?:\bcompagnies? républicaines? de sécurité\b)"
    r"|(?:\bgarde républicaine\b)"
    r"|(?:\bgardes? républicains?\b)"
    r"|(?:\buniversités? de la République\b)"  # sur corpus 2017-2024 réf à une commission d'enquête
    r"|(?:\binstitut Famille et République\b)"  # institut privé de la galaxie LMPT
    # --- Titres de lois ---
    r"|(?:\bnouvelle organisation territoriale de la République\b)"
    r"|(?:\bconfortant le respect des principes de la République\b)"
    r"|(?:\bpour une république numérique\b)"
    # --- Expression et législations ---
    # NOTE: Le choix a été fait de ne pas exclure ces formes qui sont pertinentes à garder
    # Elles sont conservées ici en commentaires pour traçabilité
    # r"|(?:\bcontrat d[’']engagement républicain\b)"
    # r"|(?:\bcontrat d[’']engagement au respect des principes de la République\b)"
    # r"|(?:\bcontrat d[’']intégration républicaine\b)"
    # r"|(?:\bquartier[s]? de reconquête républicaine\b)"
    # --- Lieux (places, monuments) ---
    r"|(?:\bplace de la République\b)"  # (45 occurrences)
    # --- Pays, territoires, entités, etc. ---
    r"|(?:\brépubliques? soviétiques?\b)"
    r"|(?:\bex-républiques? soviétiques?\b)"
    r"|(?:\brépublique de Weimar\b)"
    r"|(?:\bRépublique yougoslave\b)"  # spécifique corpus AN
    r"|(?:\brépublique du Haut-Karabakh\b)"  # spécifique corpus AN
    r"|(?:\brépublique d[’']Artsakh\b)"  # spécifique corpus AN
    r"|(?:\brépublique de l[’']Artsakh\b)"  # spécifique corpus AN
    r"|(?:\brépublique des Fidji\b)"  # spécifique corpus AN
    r"|(?:\bRépublique de Chine\b)"  # spécifique corpus AN
    r"|(?:\brépubliques? du Donbass\b)"  # spécifique corpus AN
    r"|(?:\brépublique de Crimée\b)"  # spécifique corpus AN
    r"|(?:\bRépublique démocratique d[’']Arménie\b)"  # spécifique corpus AN
    r"|(?:\bRépubliques du Bénin et du Sénégal\b)"  # spécifique corpus AN
    r"|(?:\b(?:" + pattern_pays + r")\b)",  # ajout des exclusions de pays (liste)
    re.I,
)

# NOTE : quelques (~10) "république islamique" sans précision pour parler de l'Iran
# mais risque de supprimer d'autres occurrences que l'on veut garder.


# Fonction de décompte des occurrences
def count_lexical_outside_excl(text):
    """
    Compte les occurrences valides de la famille du mot "républi" (hors zones
    d'exclusion). Early-exit via pattern_lexical.search() avant de calculer
    les positions d'exclusion (coûteux, notamment la liste de pays) : utile
    car la grande majorité des textes ne contient aucune occurrence.

    NOTE : l'ancienne fonction contains_lexical_outside_excl() a été
    supprimée (07/07/2026) : elle est strictement équivalente à
    (count_lexical_outside_excl(text) > 0), donc redondante. Équivalence
    vérifiée par test, même nombre de matchs.

    NOTE : on pourrait optimiser in_excl() via spans triés + bisect, pas
    indispensable ici et plus complexe.
    + probablement pas rentable car pas assez occurrences par texte ?
    """
    if pd.isna(text) or not pattern_lexical.search(text):
        return 0
    # Trouver les positions des expressions exclues
    excl_positions = [m.span() for m in pattern_excl_case_sensitive.finditer(text)] + [
        m.span() for m in pattern_excl_case_insensitive.finditer(text)
    ]

    # Fonction pour vérifier si une position est dans une zone exclue
    def in_excl(pos):
        for start, end in excl_positions:
            if start <= pos < end:
                return True
        return False

    return sum(1 for m in pattern_lexical.finditer(text) if not in_excl(m.start()))



# %%
# bloc d'essai
mon_texte = "Les députés Républicains sont très républicains. Vive la république !"
print(
    "Test bloc d'essai (nb mentions valides) :", count_lexical_outside_excl(mon_texte)
)

# %% [markdown]
# ## Application
# nb : une seule fonction (count_lexical_outside_excl) suffit désormais.
# repu_match_valide est dérivé de façon vectorisée à partir du comptage,
# sans second passage sur le texte (voir note dans count_lexical_outside_excl).

# %%
# ========== Appliquer comptage et identification mentions valides ==========
df["nombre_mentions_repu"] = df["texte"].apply(count_lexical_outside_excl)
df["repu_match_valide"] = df["nombre_mentions_repu"] > 0
print(df["repu_match_valide"].value_counts())

# %% [markdown]
# ## Exports

# %%
# ========== Exports ===========

# Fichier complet avec colonne de match, pour calcul de proportions
df.to_csv(PATH_SORTIE_PROPORTION, index=False)
print("Export vers :", PATH_SORTIE_PROPORTION)

# Export des seuls cas contenant une mention valide de République
df_match = df[df["repu_match_valide"]]
df_match.to_csv(PATH_SORTIE_MATCHES, index=False)
print("Export vers :", PATH_SORTIE_MATCHES)

# %% [markdown]
# # 1-1 - Extraction des données XML
# Extrait les paragraphes des comptes rendus de l'Assemblée nationale à partir
# des fichiers XML (lxml), pour les 15e et 16e législatures.
# Exclut les uids identifiés comme doublons/congrès et exporte un csv par législature.
# Écrit `1_1_extract_15.csv` et `1_1_extract_16.csv`, utilisés par l'étape suivante (1_2).

# %% [markdown]
# NOTE : /!\ comportement isna() vs "" avec pandas
# Une étape de stabilisation ("" -> None), est ici appliquée a
# toutes les colones en sortie (activable/désactivable dans
# `traiter_dossier_compte_rendu_lxml` (bloc Stabilisation).
# Reproduit ici le comportement que pandas applique à l'écriture/lecture du CSV
# (conversion "" -> vide/NaN par défaut). Sans effet sur le csv écrit,
# mais évite les faux négatifs si un jour tout passe dans un seul script
# (sans CSV entre les étapes), ou lors d'un check manuel directement sur le DataFrame.
# Perte assumée de la nuance "absent" (None) vs "vide" ("") sur certaines
# colonnes natives du XML. Désactiver si besoin de check avant écriture CSV.

# %%
# # TODO: check against NosDéputés/RegardsCitoyens <3

# %%
import os
import glob
from lxml import etree
import pandas as pd
import re

PATH_XML_15 = "../data/raw/15-xml/compteRendu/"
PATH_XML_16 = "../data/raw/16-xml/compteRendu/"

PATH_SORTIE_15 = "../data/interim/1_1_extract_15.csv"
PATH_SORTIE_16 = "../data/interim/1_1_extract_16.csv"

# %% [markdown]
# ## Fonctions d'extraction


# %%
# ==================================================================
# FONCTIONS D'EXTRACTION DES DONNÉES
# ==================================================================

# ======== Fonctions extraction infos depuis fichier XML =========


def _extraire_texte_avec_espacement(elem, ns):
    """
    Reconstruit le texte d'un élément XML en corrigeant deux pertes
    d'espacement propres au format des comptes rendus AN :
    - <br/> : ignoré par itertext(), on restaure l'espace via son tail.
    - <italique/> vide (sans texte ni enfant) : ne porte aucune mise en
      forme, sert uniquement à séparer des mots/segments (ex. didascalies
      ou titres découpés mot par mot en 15e législature).
    Ne gère PAS la normalisation des espaces multiples/retours à la ligne
    issus de l'indentation du XML source : à la charge de l'appelant,
    pour préserver la trace brute utile en debug.
    """
    for br in elem.findall(".//ns:br", namespaces=ns):
        # tail = suite texte, `or ""`` gère cas où tail est None
        br.tail = " " + (br.tail or "")

    for it in elem.findall(".//ns:italique", namespaces=ns):
        # balise vide : aucun texte et aucun élément enfant (<italique/> ou <italique></italique>)
        if (not it.text or not it.text.strip()) and len(it) == 0:
            it.tail = " " + (it.tail or "")

    return "".join(elem.itertext()).strip()


# Fonction extraction de la hiérarchie complète des points parents


# Hiérarchie LOGIQUE via nivpoint, pré-calculée une
# fois par fichier (ordre du document), pas par remontée d'ancêtres.
def construire_contexte_nivpoint(root, ns):
    """
    Reconstruit la hiérarchie LOGIQUE des points d'un compte rendu, indexée
    par nivpoint plutôt que par imbrication physique dans l'arbre XML.

    Contrairement à l'ancienne approche par ancêtres physiques
    (extraire_hierarchie via getparent), ici on capture le vrai nivpoint=1
    du document même quand celui-ci est un FRÈRE physique du point où se
    trouve le paragraphe, et non un ancêtre réel dans l'arbre (cas fréquent :
    les points de haut niveau sont souvent juxtaposés sous <contenu>, pas
    imbriqués > cas reprise discussion loi séance précédente, etc.).

    Principe : parcourt <contenu> en DFS, dans l'ordre du document.
    À chaque nouveau <point> de niveau nivpoint=N rencontré, tout
    contexte de niveau >= N précédemment enregistré est invalidé
    (un point de niveau N ne peut pas être un sous-niveau d'un point de
    niveau N ou plus profond déjà vu).
    nivpoint="99" (suspensions de séance) est délibérément ignoré :
    il ne modifie jamais le contexte en cours, pour ne pas interrompre
    la hiérarchie logique avec une pause procédurale.

    Retourne {id_syceron: hierarchy_list}, une liste ordonnée du niveau
    le plus haut au plus bas, pour être utilisé avec _hierarchie_to_colonnes().
    """
    contenu = root.find("ns:contenu", namespaces=ns)
    if contenu is None:
        return {}

    contexte_par_niveau = {}
    resultat = {}

    def walk(elem):
        for child in elem:
            tag = etree.QName(child).localname

            if tag == "point":
                niv_str = child.get("nivpoint")
                niv = int(niv_str) if niv_str and niv_str.isdigit() else None

                if (
                    niv is not None and niv != 99
                ):  # ne pas rompre contexte pour nivpoint=99 (suspension de séance)
                    titre_texte_elem = child.find("ns:texte", namespaces=ns)
                    titre = (
                        _extraire_texte_avec_espacement(titre_texte_elem, ns)
                        if titre_texte_elem is not None
                        else ""
                    )  # /!\ -> "" plutôt que None ici sinon TypeError avec re.sub()
                    titre = re.sub(
                        r"\s+", " ", titre
                    ).strip()  # normalisation espaces multiples et retours à la ligne

                    # un nouveau point de niveau N invalide tout contexte >= N
                    for k in [k for k in contexte_par_niveau if k >= niv]:
                        del contexte_par_niveau[k]
                    contexte_par_niveau[niv] = {
                        "niveau": niv,
                        "code": child.get("code_grammaire"),
                        "titre": titre,
                        "valeur_ptsodj": child.get("valeur_ptsodj"),
                        "art": child.get("art"),
                        "adt": child.get("adt"),
                        "bibard": child.get("bibard"),
                    }

                walk(child)  # gère à la fois sous-points imbriqués ET paragraphe direct

            elif tag == "paragraphe":
                id_syc = child.get("id_syceron")
                if id_syc:
                    resultat[id_syc] = [
                        contexte_par_niveau[k] for k in sorted(contexte_par_niveau)
                    ]
            else:
                walk(child)  # interExtraction, ouvertureSeance, finSeance, etc.

    walk(contenu)
    return resultat


# Fonction transformation hiérarchie vers colonnes
def _hierarchie_to_colonnes(hierarchy):
    """
    Aplati une hiérarchie de points (liste ordonnée du niveau le plus haut
    au plus bas) en colonnes exploitables dans un DataFrame.
    """
    titres = [h["titre"] for h in hierarchy if h["titre"]]
    structure = " > ".join(titres)

    return {
        "point_structure_complete": structure,
        "point_nb_niveaux": len(hierarchy),
        "point_niveau_1": hierarchy[0]["titre"] if len(hierarchy) > 0 else "",
        "point_niveau_2": hierarchy[1]["titre"] if len(hierarchy) > 1 else "",
        "point_niveau_3": hierarchy[2]["titre"] if len(hierarchy) > 2 else "",
        "point_niveau_last": hierarchy[-1]["titre"] if hierarchy else "",
        "point_niveau_last_known": next(
            (h["titre"] for h in reversed(hierarchy) if h["titre"]), ""
        ),
        "point_bibard": hierarchy[-1].get("bibard") if hierarchy else None,
        "point_art": hierarchy[-1].get("art") if hierarchy else None,
    }


# Fonction d'extraction des infos pour les paragraphes
def extraire_paragraphes_lxml(fichier_xml: str) -> pd.DataFrame:
    """
    Extrait les paragraphes d'un fichier XML de compte rendu en utilisant lxml.
    """
    try:
        tree = etree.parse(fichier_xml)
        root = tree.getroot()
        ns = {"ns": "http://schemas.assemblee-nationale.fr/referentiel"}

        meta = {
            "uid": root.findtext("ns:uid", namespaces=ns),
            "SeanceRef": root.findtext("ns:seanceRef", namespaces=ns),
            "SessionRef": root.findtext("ns:sessionRef", namespaces=ns),
        }
        meta_tags = [
            "dateSeance",
            "dateSeanceJour",
            "numSeanceJour",
            "numSeance",
            "typeAssemblee",
            "legislature",
            "session",
            "nomFichierJo",
            "presidentSeance",
        ]
        for tag in meta_tags:
            meta[tag] = root.findtext(f".//ns:{tag}", namespaces=ns)

        # pré-calcul de la hiérarchie nivpoint (une seule passe par fichier)
        contexte_nivpoint = construire_contexte_nivpoint(root, ns)

        rows = []

        for paragraphe in root.xpath(".//ns:paragraphe", namespaces=ns):
            # Hiérarchie nivpoint
            id_syc = paragraphe.get("id_syceron")
            hierarchy_nivpoint = contexte_nivpoint.get(id_syc, [])
            cols_nivpoint = _hierarchie_to_colonnes(hierarchy_nivpoint)
            # point_type = code_grammaire du <point> parent (= dernier niveau hiérarchie non nivpoint=99)
            point_type = (
                hierarchy_nivpoint[-1].get("code") if hierarchy_nivpoint else None
            )
            # texte du paragraphe (sans normalisation espaces/retours à ligne bruts du XML)
            # utiles en debug et géré par nettoyer_texte (1_2) ensuite.
            # TODO : tester et explorer les textes suite intégration modif
            texte_elem = paragraphe.find("ns:texte", namespaces=ns)
            texte = (
                _extraire_texte_avec_espacement(texte_elem, ns)
                if texte_elem is not None
                else None
            )

            # stime du paragraphe (attribut de <texte>, pas de <paragraphe>)
            stime = texte_elem.get("stime") if texte_elem is not None else None

            # Récupérer les informations de l'orateur
            # (= celles présentes dans la balise <orateur>, pas forcément dans les attributs du paragraphe)
            orateur = paragraphe.find(".//ns:orateur", namespaces=ns)
            nom_orateur = (
                orateur.findtext("ns:nom", namespaces=ns)
                if orateur is not None
                else None
            )
            qualite_orateur = (
                orateur.findtext("ns:qualite", namespaces=ns)
                if orateur is not None
                else None
            )
            id_orateur = (
                orateur.findtext("ns:id", namespaces=ns)
                if orateur is not None
                else None
            )

            # toper désormais toutes les infos
            # garder apparent pour éventuels choix ou recodages des noms plutôt que des machins type `**meta`
            rows.append(
                {
                    # ===== Métadonnées de la séance =====
                    "uid": meta["uid"],
                    "SeanceRef": meta["SeanceRef"],
                    "SessionRef": meta["SessionRef"],
                    "dateSeance": meta["dateSeance"],
                    "dateSeanceJour": meta["dateSeanceJour"],
                    "numSeanceJour": meta["numSeanceJour"],
                    "numSeance": meta["numSeance"],
                    "typeAssemblee": meta["typeAssemblee"],
                    "legislature": meta["legislature"],
                    "session": meta["session"],
                    "nomFichierJo": meta["nomFichierJo"],
                    "presidentSeance": meta["presidentSeance"],
                    # ===== Hiérarchie NIVPOINT - données contexte =====
                    "point_structure_complete": cols_nivpoint[
                        "point_structure_complete"
                    ],
                    "point_nb_niveaux": cols_nivpoint["point_nb_niveaux"],
                    "point_niveau_1": cols_nivpoint["point_niveau_1"],
                    "point_niveau_2": cols_nivpoint["point_niveau_2"],
                    "point_niveau_3": cols_nivpoint["point_niveau_3"],
                    "point_niveau_last": cols_nivpoint["point_niveau_last"],
                    "point_niveau_last_known": cols_nivpoint["point_niveau_last_known"],
                    "point_bibard": cols_nivpoint["point_bibard"],  # TODO : virer ?
                    "point_art": cols_nivpoint["point_art"],  # TODO : virer ?
                    "point_type": point_type,
                    # ===== Données du paragraphe =====
                    "valeur_ptsodj": paragraphe.get("valeur_ptsodj"),
                    "ordinal_prise": paragraphe.get("ordinal_prise"),
                    "ordre_absolu_seance": paragraphe.get("ordre_absolu_seance"),
                    "id_acteur": paragraphe.get("id_acteur"),
                    "id_mandat": paragraphe.get("id_mandat"),
                    "code_grammaire": paragraphe.get("code_grammaire"),
                    "code_style": paragraphe.get("code_style"),
                    "code_parole": paragraphe.get("code_parole"),
                    "id_syceron": paragraphe.get("id_syceron"),
                    "roledebat": paragraphe.get("roledebat"),
                    # ===== Données orateur + texte =====
                    "nom_orateur": nom_orateur,
                    "qualite_orateur": qualite_orateur,
                    "id_orateur": id_orateur,
                    "stime": stime,
                    "texte": texte,
                }
            )

        return pd.DataFrame(rows)

    except Exception as e:
        print(f" Erreur dans {fichier_xml} : {e}")
        return pd.DataFrame()


# ======== Fonction traitement d'un dossier contenant les XML =========
def traiter_dossier_compte_rendu_lxml(
    dossier_path: str, pattern: str = "*.xml"
) -> pd.DataFrame:
    """
    Traite tous les fichiers XML d'un dossier avec la fonction extraire_paragraphes_lxml().
    """
    # lecture des fichiers avec un sorted pour reproductibilité
    fichiers = sorted(glob.glob(os.path.join(dossier_path, pattern)))

    if not fichiers:
        print(f"Aucun fichier XML trouvé dans {dossier_path}")
        return pd.DataFrame()

    df_cumul = []
    total = len(fichiers)
    print(f"Traitement de {total} fichiers XML...\n")

    for i, fichier in enumerate(fichiers, 1):
        nom = os.path.basename(fichier)
        print(f"[{i}/{total}] {nom}...", end=" ")

        df_temp = extraire_paragraphes_lxml(fichier)
        if not df_temp.empty:
            print(f"{len(df_temp)} lignes")
            df_cumul.append(df_temp)
        else:
            print("Vide ou erreur")

    if df_cumul:
        df_extraction = pd.concat(df_cumul, ignore_index=True)
        # Stabilisation "" -> None : reproduit ici le comportement que pandas
        # applique déjà automatiquement à l'écriture/lecture du CSV (conversion
        # "" -> vide/NaN par défaut). Sans effet sur le CSV écrit, mais évite les
        # faux négatifs si un jour tout passe dans un seul script (sans CSV entre
        # les étapes), ou lors d'un check manuel directement sur le DataFrame.
        # Perte assumée de la nuance "absent" (None) vs "vide" ("") sur certaines
        # colonnes natives du XML.
        for col in df_extraction.columns:
            df_extraction[col] = df_extraction[col].replace("", None)

        print(f"\n Extraction terminée : {len(df_extraction)} lignes consolidées")
        return df_extraction
    else:
        return pd.DataFrame()


# %% [markdown]
# ## Traitement des législatures souhaitées

# %%
# ==================================================================
# TRAITEMENT DES LÉGISLATURES SOUHAITÉES
# ==================================================================

# ========== Traitement des législatures ==========

df_16 = traiter_dossier_compte_rendu_lxml(PATH_XML_16)
df_15 = traiter_dossier_compte_rendu_lxml(PATH_XML_15)


# %% [markdown]
# ## Nettoyage fichiers doublons et congrès
# nb traçabilité : exclusion manuelle ici, mais pourrait s'automatiser sur base
# de str.contains("Congrès du Parlement") dans `session`, complétée par une
# déduplication sur id_syceron + texte (imparfaite par rapport à l'exclusion
# ciblée de fichiers, voir 1-2).

# %%
# ========== Nettoyage fichiers doublons et congrès ==========

uids_a_exclure = {
    "CRSANR5L16S2021O1N144",  # "faux" fichier en 16e (doublon de "CRSANR5L15S2021O1N144" de 2021)
    "CRSJOCGR5L15S2017E1N001",  # JO "Congrès du Parlement du 3 juillet 2017"
    "CRSANR5L15S2017O1N001",  # doublon AN JO "Congrès du Parlement du 3 juillet 2017"
    "CRSJOCGR5L15S2018E1N001",  # JO "Congrès du Parlement du 9 juillet 2018"
    "CRSCGR5L16S2024O1N001",  # CG "Congrès du Parlement du 4 mars 2024"
    "CRSANR5L15S2022O1N168",  # séance spéciale congrès intervention Zelensky
}

# Pour affichage (pas indispensable)
uids_trouvees_15 = uids_a_exclure & set(df_15["uid"])
uids_trouvees_16 = uids_a_exclure & set(df_16["uid"])

print(
    f"UIDs à exclure trouvés en df_15 : {len(uids_trouvees_15)}/{len(uids_a_exclure)}"
)
for uid in uids_trouvees_15:
    print(f"  - {uid}")

print(
    f"UIDs à exclure trouvés en df_16 : {len(uids_trouvees_16)}/{len(uids_a_exclure)}"
)
for uid in uids_trouvees_16:
    print(f"  - {uid}")

# puis suppression des lignes correspondantes
n15_avant, n16_avant = len(df_15), len(df_16)
df_15 = df_15[~df_15["uid"].isin(uids_a_exclure)]
df_16 = df_16[~df_16["uid"].isin(uids_a_exclure)]

print(f"Suppression UID ciblés - df_15 : {n15_avant - len(df_15)} ligne(s)")
print(f"Suppression UID ciblés - df_16 : {n16_avant - len(df_16)} ligne(s)")

# %% [markdown]
# ## Export

# %%
# ========== Exports ==========
df_15.to_csv(PATH_SORTIE_15, index=False, encoding="utf-8")
print(f"\n Export CSV df_15: ({df_15.shape[0]} lignes) -> {PATH_SORTIE_15}")

df_16.to_csv(PATH_SORTIE_16, index=False, encoding="utf-8")
print(f"\n Export CSV df_16: ({df_16.shape[0]} lignes) -> {PATH_SORTIE_16}")

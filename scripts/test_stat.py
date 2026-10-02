# %%
import pandas as pd

df = pd.read_csv("../data/interim/3_1_df_repu_proportion.csv", low_memory=False)

# %%
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

COL_ACTEUR = "id_acteur"
COL_GROUPE = "affiliation_et_gouv"
MIN_INTERV = 50  # seuil "députés retenus" pour Lorenz / Gini
MIN_AVEC_REPU = 1  # min d'interventions avec le terme (graphiques de variabilité)


def agreger_deputes(df):
    """Une ligne par (id_acteur, groupe) : un député qui change de groupe compte dans chacun."""
    d = df.dropna(subset=[COL_ACTEUR, COL_GROUPE]).copy()
    d["avec_terme"] = d["repu_match_valide"].astype(int)
    out = (
        d.groupby([COL_ACTEUR, COL_GROUPE])
        .agg(
            nom=(
                "nom_orateur_clean",
                lambda s: s.mode().iat[0] if s.notna().any() else None,
            ),
            n_interv=("avec_terme", "size"),
            n_avec=("avec_terme", "sum"),
            n_mentions=("nombre_mentions_repu", "sum"),
        )
        .reset_index()
        .rename(columns={COL_GROUPE: "groupe"})
    )
    out["pct"] = 100 * out["n_avec"] / out["n_interv"]
    return out


dep = agreger_deputes(df)
dep_ret = dep[dep["n_interv"] >= MIN_INTERV]
print(
    len(dep),
    "couples député-groupe ;",
    len(dep_ret),
    f"retenus (>= {MIN_INTERV} interventions & >= {MIN_AVEC_REPU} mention répu)",
)


# %%
# 1. Entonnoir : % d'interventions avec le terme vs nombre d'interventions (log)
# Les 0 % ne sont pas représentables en log : exclus de ce graphique
fig1 = px.scatter(
    dep[(dep["pct"] > 0) & (dep["n_avec"] >= MIN_AVEC_REPU)],
    x="n_interv",
    y="pct",
    color="groupe",
    symbol="groupe",
    opacity=0.6,
    log_x=True,
    log_y=True,
    hover_name="nom",
    hover_data=["n_avec"],
    labels={
        "n_interv": "Nombre d'interventions (log)",
        "pct": "% interventions avec le terme",
    },
    title="Entonnoir : proportion individuelle selon le nombre d'interventions",
)
# Enveloppe binomiale à 95 % autour du taux global
p0 = dep["n_avec"].sum() / dep["n_interv"].sum()
n = np.logspace(0, np.log10(dep["n_interv"].max()), 200)
se = np.sqrt(p0 * (1 - p0) / n)
for sign in (1, -1):
    fig1.add_trace(
        go.Scatter(
            x=n,
            y=100 * np.clip(p0 + sign * 1.96 * se, 1e-4, 1),
            mode="lines",
            line=dict(color="grey", dash="dash"),
            showlegend=False,
        )
    )
fig1.add_hline(y=100 * p0, line_dash="dot", line_color="grey")

fig1.show()

# # 2. Diffusion : part des députés ayant au moins une mention, par groupe
# # Plutôt direct dans le  tableau
# dif = (
#     dep.assign(a_mention=dep["n_avec"] > 0)
#     .groupby("groupe")["a_mention"]
#     .mean()
#     .mul(100)
#     .sort_values(ascending=False)
#     .rename("diffusion")
#     .reset_index()
# )
# fig2 = px.bar(
#     dif,
#     x="groupe",
#     y="diffusion",
#     labels={"diffusion": "% de députés avec ≥ 1 mention", "groupe": "Groupe"},
#     title="Diffusion : part des députés ayant au moins une mention",
# )

# fig2.show()


# %%
# 3. Concentration : courbes de Lorenz des mentions (députés retenus) + Gini
def lorenz(values):
    v = np.sort(np.asarray(values, dtype=float))  # des moins aux plus "républicains"
    cum = np.insert(np.cumsum(v), 0, 0) / v.sum()
    x = np.linspace(0, 1, len(cum))
    return x, cum


def gini(values):
    x, y = lorenz(values)
    return 1 - 2 * np.trapz(y, x)


fig3 = go.Figure()
fig3.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(color="grey", dash="dash"),
        name="Égalité parfaite",
    )
)
x, y = lorenz(dep_ret["n_mentions"])
fig3.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=f"Tous (Gini = {gini(dep_ret['n_mentions']):.2f})",
        line=dict(width=3, color="black"),
    )
)
for g, sub in dep_ret.groupby("groupe"):
    if len(sub) >= 5 and sub["n_mentions"].sum() > 0:
        x, y = lorenz(sub["n_mentions"])
        fig3.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"{g} (Gini = {gini(sub['n_mentions']):.2f})",
            )
        )
fig3.update_layout(
    title=f"Courbes de Lorenz des mentions (députés avec ≥ {MIN_INTERV} interventions)",
    xaxis_title="Part cumulée des députés (des moins aux plus « républicains »)",
    yaxis_title="Part cumulée des mentions",
)
fig3.show()


# Tableau de synthèse par groupe (même seuil que tableau_bis)
def synth(sub):
    tot_avec = sub["n_avec"].sum()
    return pd.Series(
        {
            "Unités": len(sub),
            "Interventions": sub["n_interv"].sum(),
            "Avec le terme": tot_avec,
            "% agrégé": 100 * tot_avec / sub["n_interv"].sum(),
            "% médian": sub["pct"].median(),
            "Diffusion (%)": 100 * (sub["n_avec"] > 0).mean(),
            "Part du 1er locuteur (%)": 100 * sub["n_avec"].max() / tot_avec
            if tot_avec
            else np.nan,
            "Gini": gini(sub["n_avec"]) if tot_avec else np.nan,
        }
    )


SEUIL_TABLEAU = 25
dep_t = dep[dep["n_interv"] >= SEUIL_TABLEAU]
tableau = dep_t.groupby("groupe").apply(synth).sort_values("% agrégé", ascending=False)
tableau.loc["TOTAL"] = synth(dep_t)
tableau.round(1).astype({"Unités": int, "Interventions": int, "Avec le terme": int})

# %%
# EN pas interractif et cette fois nb occurences pour 100 interventions

# %%
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

# Agrégation au niveau député.
df["speaker_key"] = df["id_orateur"].fillna(
    "NAME:" + df["nom_orateur_clean"].fillna("UNKNOWN")
)
agg = (
    df.groupby(["speaker_key", "nom_orateur_clean"], dropna=False)
    .agg(
        interventions=("uid", "count"),
        mentions=("nombre_mentions_repu", "sum"),
        groupe=(
            "affiliation_et_gouv",
            lambda s: s.dropna().mode().iat[0]
            if not s.dropna().empty
            else "Non renseigné",
        ),
    )
    .reset_index()
)
agg["mentions_100"] = 100 * agg["mentions"] / agg["interventions"]

# Groupes suffisamment représentés dans cet échantillon pour une lecture graphique.
valid_groups = agg.groupby("groupe").size().loc[lambda s: s >= 10].index
g = agg[agg["groupe"].isin(valid_groups)].copy()

fig, axes = plt.subplots(3, 1, figsize=(12, 15), constrained_layout=True)

# 1. Variabilité individuelle : taux vs exposition.
ax = axes[0]
for grp in sorted(g["groupe"].unique()):
    sub = g[g["groupe"] == grp]
    ax.scatter(sub["interventions"], sub["mentions_100"], alpha=0.55, s=35, label=grp)
ax.set_xscale("log")
ax.set_xlabel("Nombre d'interventions du député (échelle log)")
ax.set_ylabel("Mentions de « République » pour 100 interventions")
ax.set_title("Variabilité individuelle : fréquence de mention et volume de parole")
ax.grid(alpha=0.25)
ax.legend(ncol=4, fontsize=8)

# 2. Distribution des taux par groupe (boxplot + points).
ax = axes[1]
groups = sorted(g["groupe"].unique())
data = [g.loc[g["groupe"] == grp, "mentions_100"] for grp in groups]
ax.boxplot(data, labels=groups, showfliers=False)
for i, grp in enumerate(groups, 1):
    vals = g.loc[g["groupe"] == grp, "mentions_100"].values
    rng = np.random.default_rng(100 + i)
    jitter = rng.uniform(-0.13, 0.13, len(vals))
    ax.scatter(i + jitter, vals, alpha=0.35, s=18)
ax.set_ylabel("Mentions pour 100 interventions")
ax.set_title("Distribution des taux individuels par groupe")
ax.grid(axis="y", alpha=0.25)

# 3. Courbes de Lorenz : concentration des mentions.
ax = axes[2]
for grp in sorted(g["groupe"].unique()):
    sub = g[g["groupe"] == grp]
    x = np.sort(sub["mentions"].to_numpy(dtype=float))
    if x.sum() == 0:
        continue
    y = np.cumsum(x) / x.sum()
    xpop = np.arange(1, len(x) + 1) / len(x)
    ax.plot(np.r_[0, xpop], np.r_[0, y], label=grp)
ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Égalité parfaite")
ax.set_xlabel("Part cumulée des députés (du moins au plus mentionnant)")
ax.set_ylabel("Part cumulée des mentions")
ax.set_title("Concentration des mentions entre députés")
ax.grid(alpha=0.25)
ax.legend(ncol=4, fontsize=8)

plt.show()


# %%
import pandas as pd
import numpy as np
import plotly.express as px

# Agrégation au niveau député
dep = (
    df.groupby(["id_orateur", "nom_orateur_clean", "affiliation_et_gouv"], dropna=False)
    .agg(
        interventions=("uid", "count"),
        mentions=("nombre_mentions_repu", "sum"),
        mots=("len_texte_brut", "sum"),
    )
    .reset_index()
)

# Indicateurs
dep["mentions_100_interventions"] = 100 * dep["mentions"] / dep["interventions"]

dep["mentions_10000_mots"] = 10000 * dep["mentions"] / dep["mots"]

# Seuil minimal
seuil = 25
d = dep[dep["interventions"] >= seuil].copy()

fig = px.scatter(
    d,
    x="interventions",
    y="mentions_100_interventions",
    color="affiliation_et_gouv",
    hover_name="nom_orateur_clean",
    hover_data={
        "affiliation_et_gouv": True,
        "interventions": True,
        "mentions": True,
        "mentions_100_interventions": ":.2f",
        "mentions_10000_mots": ":.2f",
    },
    log_x=True,
    labels={
        "interventions": "Nombre d'interventions",
        "mentions_100_interventions": "Mentions de « République » pour 100 interventions",
        "affiliation_et_gouv": "Groupe",
    },
    title=f"Variabilité individuelle de la mention de « République » — ≥ {seuil} interventions",
)

fig.update_traces(marker=dict(size=9, opacity=0.65))

fig.update_layout(height=700, template="plotly_white", legend_title="Groupe")

fig.show()

# %%
fig = px.box(
    d,
    x="affiliation_et_gouv",
    y="mentions_100_interventions",
    color="affiliation_et_gouv",
    points="all",
    hover_name="nom_orateur_clean",
    hover_data={
        "interventions": True,
        "mentions": True,
        "mentions_100_interventions": ":.2f",
    },
    labels={
        "affiliation_et_gouv": "Groupe",
        "mentions_100_interventions": "Mentions pour 100 interventions",
    },
    title="Variabilité individuelle au sein des groupes",
)

fig.update_layout(height=650, template="plotly_white", showlegend=False)

fig.show()

# %%
# Redéfinit dep
dep = agreger_deputes(df)
dep_ret = dep[dep["n_interv"] >= MIN_INTERV]


# %%
def gini(values):
    """
    Coefficient de Gini pour une distribution non négative.
    Les zéros sont conservés.
    """
    x = np.asarray(values, dtype=float)

    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    if np.any(x < 0):
        raise ValueError("Le Gini nécessite des valeurs positives ou nulles.")

    if x.sum() == 0:
        return 0.0

    x = np.sort(x)

    n = len(x)

    return (2 * np.sum(np.arange(1, n + 1) * x)) / (n * x.sum()) - (n + 1) / n


def lorenz(values):
    """
    Renvoie X, Y pour une courbe de Lorenz.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.array([0, 1]), np.array([0, 1])

    x = np.sort(x)

    total = x.sum()

    if total == 0:
        return (np.linspace(0, 1, len(x) + 1), np.zeros(len(x) + 1))

    cumulative = np.cumsum(x)

    X = np.r_[0, np.arange(1, len(x) + 1) / len(x)]

    Y = np.r_[0, cumulative / total]

    return X, Y


# %%
def synthese_groupes(dep, seuil=25):
    d = dep[dep["n_interv"] >= seuil]

    lignes = []
    for groupe, sub in d.groupby("groupe"):
        interventions = sub["n_interv"].sum()
        avec = sub["n_avec"].sum()
        lignes.append(
            {
                "Unités": len(sub),
                "Groupe": groupe,
                "Interventions": interventions,
                "Avec le terme": avec,
                "% agrégé": 100 * avec / interventions if interventions > 0 else np.nan,
                "% médian": sub["pct"].median(),
                "Diffusion (%)": 100 * (sub["n_avec"] > 0).mean(),
                "Part du 1er locuteur (%)": 100 * sub["n_avec"].max() / avec
                if avec > 0
                else 0,
                "Gini": gini(sub["n_avec"]),
            }
        )

    return pd.DataFrame(lignes).sort_values("Groupe").reset_index(drop=True)


tableau_bis = synthese_groupes(dep, seuil=25)

tableau_bis

# %%
tableau

# %%
import plotly.express as px

# ex palette à la noix
palette_groupes = {
    "LAREM": "#f0d213",
    "LR": "#022dd9",
    "SOC-A": "#c4187f",
    "LFI": "#d01f08",
    "RN": "#0b0b0b",
}

fig = px.box(
    d,
    x="affiliation_et_gouv",
    y="mentions_100_interventions",
    color="affiliation_et_gouv",
    # Points individuels
    points="all",
    hover_name="nom_orateur_clean",
    hover_data={
        "affiliation_et_gouv": False,
        "interventions": True,
        "mentions": True,
        "mentions_100_interventions": ":.2f",
    },
    color_discrete_map=palette_groupes,
    labels={
        "affiliation_et_gouv": "",
        "mentions_100_interventions": "Mentions de « République » pour 100 interventions",
    },
    title="Variabilité individuelle de la fréquence de mention",
)

fig.update_traces(
    # Box
    line=dict(width=1.5),
    # Points
    jitter=0.25,
    pointpos=0,
    marker=dict(size=5, opacity=0.45, line=dict(width=0)),
    # Afficher les valeurs extrêmes
    boxmean=False,
)

fig.update_layout(
    template="simple_white",
    # Dimensions
    width=1000,
    height=600,
    # Typographie
    font=dict(family="Arial", size=13, color="#222222"),
    title=dict(font=dict(size=18, color="#222222"), x=0, xanchor="left"),
    # Axes
    xaxis=dict(
        showline=True,
        linewidth=1,
        linecolor="#333333",
        mirror=False,
        ticks="outside",
        ticklen=5,
        tickwidth=1,
        showgrid=False,
        zeroline=False,
    ),
    yaxis=dict(
        showline=True,
        linewidth=1,
        linecolor="#333333",
        ticks="outside",
        ticklen=5,
        tickwidth=1,
        showgrid=True,
        gridcolor="#E5E5E5",
        gridwidth=1,
        zeroline=False,
    ),
    # Légende inutile puisqu'elle répète l'axe X
    showlegend=False,
    # Marges
    margin=dict(l=80, r=30, t=80, b=70),
)

fig.show()

# %%
# Même boxplot, en % d'interventions contenant le terme (colonnes de agreger_deputes)
d_pct = dep[
    (dep["n_interv"] >= SEUIL_TABLEAU) & (dep["n_avec"] >= MIN_AVEC_REPU)
].copy()

fig = px.box(
    d_pct,
    x="groupe",
    y="pct",
    color="groupe",
    points="all",
    hover_name="nom",
    hover_data={
        "groupe": False,
        "n_interv": True,
        "n_avec": True,
        "pct": ":.2f",
    },
    color_discrete_map=palette_groupes,
    labels={
        "groupe": "",
        "n_interv": "Interventions",
        "n_avec": "Interventions avec le terme",
        "pct": "% d'interventions mentionnant « République »",
    },
    title=f"Variabilité individuelle de la part d'interventions mentionnant « République » (≥ {SEUIL_TABLEAU} interventions, ≥ {MIN_AVEC_REPU} avec le terme)",
)

fig.update_traces(
    line=dict(width=1.5),
    jitter=0.25,
    pointpos=0,
    marker=dict(size=5, opacity=0.45, line=dict(width=0)),
    boxmean=False,
)

fig.update_layout(
    template="simple_white",
    width=1000,
    height=600,
    font=dict(family="Arial", size=13, color="#222222"),
    title=dict(font=dict(size=18, color="#222222"), x=0, xanchor="left"),
    xaxis=dict(
        showline=True,
        linewidth=1,
        linecolor="#333333",
        ticks="outside",
        ticklen=5,
        tickwidth=1,
        showgrid=False,
        zeroline=False,
    ),
    yaxis=dict(
        showline=True,
        linewidth=1,
        linecolor="#333333",
        ticks="outside",
        ticklen=5,
        tickwidth=1,
        showgrid=True,
        gridcolor="#E5E5E5",
        gridwidth=1,
        zeroline=False,
    ),
    showlegend=False,
    margin=dict(l=80, r=30, t=80, b=70),
)

fig.show()


# %%
df["nom_orateur_clean"].str.contains("Poulliat").sum()

# %%
poupou = df[df["nom_orateur_clean"].str.contains("Poulliat") == True]

# %%
poupou["texte"].to_csv("poupou.csv")

# %%

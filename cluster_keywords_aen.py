#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyword Strategy AI – Metodo A&N (base Conforti)
Clustering + Parent Topic + Estensioni NVivo & Mappa Temi + Prompt + Internal Linking

USO TIPICO
----------
python cluster_keywords_aen.py --input keywords.xlsx --sheet Sheet1 --text-col keyword --volume-col volume --auto-k
python cluster_keywords_aen.py --input keywords.xlsx --k 12

OUTPUT (fogli Excel)
--------------------
- Keywords_Clustered        → ogni keyword con cluster, parent_topic, rank, distanza
- Cluster_Summary           → riepilogo per cluster
- K_Selection               → tabella scelta k (WCSS, Silhouette, ElbowGain)
- Run_Params                → metadati esecuzione
- Cluster_Labels            → etichetta “umana” del cluster (euristica top-term)
- Prompt_Per_Cluster        → prompt pillar, brief, 5W+How, social per ogni hub
- Internal_Link_Map         → mappa hub → top-10 spoke (vicinanza al centro)
- Analisi_Qualitativa_NVivo → tab “qualitativa” in stile NVivo (tema, sotto-tema, insight)
- Mappa_Temi                → conteggi e % per tema/hub (+ somma volumi se presente)

NOTE
----
- Lingua italiana: stopword IT custom + normalizzazione; nessun accesso a internet.
- “nascondi il centroide”: NON esponiamo vettori o pesi; usiamo solo la keyword più vicina come parent_topic.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


# -----------------------
# Config
# -----------------------
ITALIAN_STOPWORDS: frozenset[str] = frozenset({
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra", "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "e", "ed", "o", "oppure", "che", "come", "anche", "ma", "non", "più", "meno", "della", "dell", "degli", "dai",
    "al", "ai", "agli", "alle", "allo", "alla", "del", "dei", "delle", "dallo", "dalla", "dagli", "dalle", "dei",
    "mi", "ti", "si", "ci", "vi", "loro", "tu", "voi", "noi", "sono", "sei", "siamo", "siete", "era", "ero", "eri",
    "essere", "avere", "ha", "hanno", "ho", "abbiamo", "avete", "fare", "fa", "fanno", "fatto", "può", "puoi",
    "solo", "tutto", "tutta", "tutti", "tutte", "questo", "questa", "questi", "queste", "quello", "quella",
    "quelli", "quelle", "qui", "qua", "lì", "là", "dove", "quando", "perché", "quanto", "quanti", "dunque"
})
TOKEN_PATTERN = re.compile(r"[a-zA-ZàèéìíòóùúÀÈÉÌÍÒÓÙÚ0-9]+")
DEFAULT_KEYWORD_COLUMN = "keyword"


# -----------------------
# Utils testuali
# -----------------------
def normalize_text(value: str) -> str:
    """Lowercase, strip and collapse whitespace while keeping accented chars."""

    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value).strip().lower())
    return re.sub(r"\s+", " ", normalized)


def tokenize_it(text: str) -> List[str]:
    """Tokenize Italian text removing custom stopwords."""

    tokens = TOKEN_PATTERN.findall(text.lower())
    return [token for token in tokens if token not in ITALIAN_STOPWORDS and len(token) > 1]


# -----------------------
# TF-IDF
# -----------------------
def build_vectorizer(
    *,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.9,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        tokenizer=tokenize_it,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        norm="l2",
    )


# -----------------------
# Scelta automatica di k
# -----------------------
@dataclass
class KSelectionResult:
    best_k: int
    table: pd.DataFrame


def choose_k_auto(
    X, *, k_min: int = 2, k_max: int = 20, random_state: int = 42
) -> KSelectionResult:
    if X.shape[0] < 2:
        raise ValueError("Servono almeno due keyword per creare dei cluster.")

    results: List[Dict[str, float]] = []
    wcss_prev: Optional[float] = None
    upper_bound = min(k_max, X.shape[0] - 1)
    for k in range(k_min, upper_bound + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=random_state)
        labels = km.fit_predict(X)
        wcss = float(km.inertia_)
        silhouette = float(silhouette_score(X, labels, metric="euclidean")) if k > 1 else np.nan
        elbow_gain = None
        if wcss_prev is not None and wcss_prev > 0:
            elbow_gain = (wcss_prev - wcss) / wcss_prev
        wcss_prev = wcss
        results.append({
            "k": k,
            "WCSS": wcss,
            "Silhouette": silhouette,
            "ElbowGain": elbow_gain,
        })

    table = pd.DataFrame(results)
    sil_idx = table["Silhouette"].idxmax()
    k_sil = int(table.loc[sil_idx, "k"]) if not np.isnan(table.loc[sil_idx, "Silhouette"]) else k_min

    elbow_k = k_sil
    gains = table["ElbowGain"].dropna().values
    if gains.size >= 3:
        threshold = max(0.10, float(np.mean(gains[:3]) * 0.5))
        for _, row in table.iloc[1:].iterrows():
            if row["ElbowGain"] is not None and row["ElbowGain"] < threshold:
                elbow_k = int(row["k"])
                break

    auto_k = int(round(np.mean([k_sil, elbow_k])))
    auto_k = max(k_min, min(auto_k, upper_bound))
    return KSelectionResult(best_k=auto_k, table=table)


# -----------------------
# Centroide: parent_topic + distanze
# -----------------------
def centroid_and_distances(X, labels: Iterable[int], texts: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame({"keyword_norm": texts, "cluster": labels})
    centroids: List[Dict[str, float | str | int]] = []
    for cluster_id in np.unique(labels):
        idx = np.where(labels == cluster_id)[0]
        sub = X[idx]
        centroid_vec = sub.mean(axis=0)
        denom = np.linalg.norm(centroid_vec)
        if denom == 0:
            dists = np.full(len(idx), 1.0)
        else:
            dots = sub @ centroid_vec.T
            dists = 1 - (dots.A1 / (np.linalg.norm(sub, axis=1) * denom + 1e-9))
        order = np.argsort(dists)
        best_index = idx[order[0]]
        centroids.append(
            {
                "cluster": int(cluster_id),
                "parent_topic_norm": texts[best_index],
                "centroid_distance": float(dists[order[0]]),
                "cluster_size": len(idx),
                "avg_distance": float(np.mean(dists)),
                "p90_distance": float(np.percentile(dists, 90)),
            }
        )
        df.loc[idx, "distance_to_centroid"] = dists
    cent_df = pd.DataFrame(centroids).sort_values("cluster").reset_index(drop=True)
    return cent_df, df


# -----------------------
# Labeling “umano” del cluster (euristica top-term)
# -----------------------
def label_clusters_kws(
    X, labels: Iterable[int], vectorizer: TfidfVectorizer, *, topn: int = 3
) -> pd.DataFrame:
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    rows: List[Dict[str, object]] = []
    for cluster_id in np.unique(labels):
        idx = np.where(labels == cluster_id)[0]
        sub = X[idx]
        mean_vec = np.asarray(sub.mean(axis=0)).ravel()
        top_idx = np.argsort(mean_vec)[::-1][: topn + 5]
        terms: List[str] = []
        for term in feature_names[top_idx]:
            if term.isnumeric() and len(terms) < topn:
                terms.append(term)
            elif len(term) > 2 and term not in ITALIAN_STOPWORDS:
                terms.append(term)
            if len(terms) >= topn:
                break
        label = " ".join(terms[:topn]).strip()
        rows.append({"cluster": int(cluster_id), "cluster_label": label})
    return pd.DataFrame(rows).sort_values("cluster")


# -----------------------
# Prompt factory (foglio Prompt_Per_Cluster)
# -----------------------
def make_prompts_for_hub(hub: str) -> Dict[str, str]:
    title = f"{hub.capitalize()}: guida completa e casi d’uso"
    pillar = (
        f"Scrivi un pillar SEO autorevole su “{hub}”.\n"
        f"- Target: utenti italiani con intento informativo/commerciale.\n"
        f"- Struttura: H1 (con USP), intro forte, H2/H3 in logica hub→spoke, FAQ (schema), "
        f"glossario essenziale, call to action soft.\n"
        f"- Stile: chiaro, denso, esempi pratici, evita fluff.\n"
        f"- SEO: inserisci dati strutturati consigliati, link interni a spoke pertinenti."
    )
    brief = (
        f"Crea un content brief editoriale su “{hub}” con: persona, intento, titoli H2/H3, punti chiave, asset visual, "
        f"SERP features da coprire, E-E-A-T, internal linking, KPI (CTR, dwell time), CTA e next step."
    )
    fivew = (
        f"Genera una sezione 5W+How per “{hub}”: What, Why, Who, Where, When, How. "
        f"Per ogni W includi 2–3 domande frequenti che l’utente cerca davvero e risposte sintetiche."
    )
    social = (
        f"Scrivi 3 caption Instagram e 2 script brevi TikTok (italiano) per promuovere un articolo su “{hub}”. "
        f"Usa hook forte nei primi 3 sec, chiusura con invito all’azione non commerciale."
    )
    return {
        "seo_title_suggerito": title,
        "prompt_pillar": pillar,
        "prompt_brief": brief,
        "prompt_5w_how": fivew,
        "prompt_social": social,
    }


# -----------------------
# Internal link map (hub → top-10 spoke)
# -----------------------
def build_internal_link_map(df_kw: pd.DataFrame, *, keyword_col: str, max_spoke: int = 10) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for cluster_id, group in df_kw.groupby("cluster", sort=True):
        hub = group["parent_topic"].iloc[0]
        spoke_df = group.sort_values("distance_to_centroid").head(max_spoke + 1)
        spoke_list = [kw for kw in spoke_df[keyword_col].tolist() if kw != hub][:max_spoke]
        rows.append(
            {
                "cluster": int(cluster_id),
                "hub_parent_topic": hub,
                "spoke_examples": " • ".join(spoke_list),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster")


# -----------------------
# Pipeline principale
# -----------------------
def run_pipeline(
    *,
    input_path: str,
    sheet_name: Optional[str],
    text_col: str,
    volume_col: Optional[str],
    k: Optional[int],
    auto_k: bool,
    out_path: str,
    k_min: int,
    k_max: int,
    random_state: int,
) -> None:
    if input_path.lower().endswith((".xls", ".xlsx")):
        df_raw = pd.read_excel(input_path, sheet_name=sheet_name)
    else:
        df_raw = pd.read_csv(input_path)

    if text_col not in df_raw.columns:
        raise ValueError(f"Colonna '{text_col}' non trovata.")

    extra_cols = [column for column in df_raw.columns if column != text_col]

    df = df_raw.copy()
    df["_kw_norm"] = df[text_col].astype(str).map(normalize_text)
    df = df[df["_kw_norm"].str.len() > 0].copy()
    df.drop_duplicates(subset=["_kw_norm"], inplace=True)
    texts = df["_kw_norm"].tolist()

    if not texts:
        raise ValueError("Nessuna keyword valida dopo la normalizzazione.")

    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(texts)

    k_table = None
    k_used = k
    if auto_k or k_used is None:
        k_selection = choose_k_auto(X, k_min=k_min, k_max=k_max, random_state=random_state)
        k_used = k_selection.best_k
        k_table = k_selection.table

    if k_used < 1:
        raise ValueError("Il numero di cluster deve essere almeno 1.")
    if k_used > X.shape[0]:
        raise ValueError("Il numero di cluster è maggiore del numero di keyword disponibili.")

    km = KMeans(n_clusters=k_used, init="k-means++", n_init=30, random_state=random_state)
    labels = km.fit_predict(X)

    cent_df, dist_df = centroid_and_distances(X, labels, texts)
    labels_df = label_clusters_kws(X, labels, vectorizer, topn=3)

    out = df[[text_col] + extra_cols].copy()
    out = out.merge(
        dist_df[["keyword_norm", "cluster", "distance_to_centroid"]],
        left_on="_kw_norm",
        right_on="keyword_norm",
        how="left",
    )
    out = out.merge(cent_df[["cluster", "parent_topic_norm"]], on="cluster", how="left")
    out.rename(columns={"parent_topic_norm": "parent_topic"}, inplace=True)

    out["rank_in_cluster"] = out.groupby("cluster")["distance_to_centroid"].rank(method="first")
    out.sort_values(["cluster", "rank_in_cluster"], inplace=True)
    out["distance_norm"] = MinMaxScaler().fit_transform(out[["distance_to_centroid"]])

    agg_dict: Dict[str, Tuple[str, str | callable]] = {
        "n_keywords": (text_col, "count"),
        "avg_distance": ("distance_to_centroid", "mean"),
        "p90_distance": ("distance_to_centroid", lambda s: np.percentile(s, 90)),
    }
    if volume_col and volume_col in out.columns:
        agg_dict[f"sum_{volume_col}"] = (volume_col, "sum")

    summary = (
        out.groupby(["cluster", "parent_topic"], as_index=False)
        .agg(**agg_dict)
        .sort_values(["n_keywords", "cluster"], ascending=[False, True])
    )

    if k_table is None:
        k_table = pd.DataFrame(
            {"k": [k_used], "WCSS": [float(km.inertia_)], "Silhouette": [np.nan], "ElbowGain": [np.nan]}
        )

    params = pd.DataFrame(
        [
            {
                "input_path": input_path,
                "sheet_name": sheet_name,
                "text_col": text_col,
                "volume_col": volume_col,
                "k_used": k_used,
                "k_min": k_min,
                "k_max": k_max,
                "random_state": random_state,
                "n_keywords": len(out),
                "vectorizer": "TF-IDF (it, ngram=1-2, min_df=2, max_df=0.9)",
                "note": "Metodo A&N – base Conforti; estensioni NVivo/Mappa Temi incluse",
            }
        ]
    )

    cluster_labels = labels_df.merge(summary[["cluster", "parent_topic"]], on="cluster", how="left")

    prompt_rows = []
    for _, row in summary.iterrows():
        hub = row["parent_topic"]
        prompt_rows.append({"cluster": int(row["cluster"]), "parent_topic": hub, **make_prompts_for_hub(hub)})
    prompt_df = pd.DataFrame(prompt_rows).sort_values("cluster")

    keyword_output_col = DEFAULT_KEYWORD_COLUMN
    out.rename(columns={text_col: keyword_output_col}, inplace=True)

    ilink_df = build_internal_link_map(out, keyword_col=keyword_output_col, max_spoke=10)

    nvivo_rows: List[Dict[str, object]] = []
    for _, cluster_row in cluster_labels.iterrows():
        cluster_id = int(cluster_row["cluster"])
        hub = cluster_row["parent_topic"]
        label = cluster_row["cluster_label"]
        examples = (
            out[out["cluster"] == cluster_id]
            .sort_values("distance_to_centroid")[keyword_output_col]
            .head(8)
            .tolist()
        )
        nvivo_rows.append(
            {
                "tema": hub,
                "sotto_tema": label,
                "esempi_keyword": " | ".join(examples),
                "insight_narrativo": (
                    f"Utenti interessati a {label} all’interno del topic {hub}: bisogni informativi da coprire con spoke mirati."
                ),
                "prompt_associato": (
                    f"Approfondisci il sotto-tema “{label}” come spoke del pillar “{hub}”, includendo FAQ reali e confronto alternative."
                ),
            }
        )
    nvivo_df = pd.DataFrame(nvivo_rows).sort_values(["tema", "sotto_tema"])

    total_kw = len(out)
    map_rows: List[Dict[str, object]] = []
    for _, row in summary.iterrows():
        cluster_id = int(row["cluster"])
        hub = row["parent_topic"]
        n_keyword = int(row["n_keywords"])
        entry: Dict[str, object] = {
            "cluster": cluster_id,
            "tema_hub": hub,
            "n_keyword": n_keyword,
            "perc_keyword": (n_keyword / total_kw) if total_kw else 0,
        }
        if volume_col and volume_col in out.columns:
            entry[f"sum_{volume_col}"] = row.get(f"sum_{volume_col}", np.nan)
        map_rows.append(entry)
    map_df = pd.DataFrame(map_rows).sort_values(["n_keyword", "cluster"], ascending=[False, True])

    kcols = [
        "cluster",
        "parent_topic",
        "rank_in_cluster",
        keyword_output_col,
        "distance_to_centroid",
        "distance_norm",
    ] + [col for col in extra_cols if col != text_col]

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        out[kcols].to_excel(writer, index=False, sheet_name="Keywords_Clustered")
        summary.to_excel(writer, index=False, sheet_name="Cluster_Summary")
        k_table.to_excel(writer, index=False, sheet_name="K_Selection")
        params.to_excel(writer, index=False, sheet_name="Run_Params")
        cluster_labels.to_excel(writer, index=False, sheet_name="Cluster_Labels")
        prompt_df.to_excel(writer, index=False, sheet_name="Prompt_Per_Cluster")
        ilink_df.to_excel(writer, index=False, sheet_name="Internal_Link_Map")
        nvivo_df.to_excel(writer, index=False, sheet_name="Analisi_Qualitativa_NVivo")
        map_df.to_excel(writer, index=False, sheet_name="Mappa_Temi")

        workbook = writer.book
        fmt_pct = workbook.add_format({"num_format": "0.00%"})
        fmt_2d = workbook.add_format({"num_format": "0.00"})
        fmt_int = workbook.add_format({"num_format": "0"})
        fmt_wrap = workbook.add_format({"text_wrap": True, "valign": "top"})

        def beautify(sheet_name: str, df_ref: pd.DataFrame) -> None:
            worksheet = writer.sheets[sheet_name]
            for idx, column in enumerate(df_ref.columns):
                width = min(60, max(10, int(df_ref[column].astype(str).str.len().quantile(0.90)) + 2))
                worksheet.set_column(idx, idx, width, fmt_wrap if df_ref[column].dtype == object else None)
            worksheet.autofilter(0, 0, len(df_ref), len(df_ref.columns) - 1)
            worksheet.freeze_panes(1, 0)

        beautify("Keywords_Clustered", out[kcols])
        beautify("Cluster_Summary", summary)
        beautify("K_Selection", k_table)
        beautify("Run_Params", params)
        beautify("Cluster_Labels", cluster_labels)
        beautify("Prompt_Per_Cluster", prompt_df)
        beautify("Internal_Link_Map", ilink_df)
        beautify("Analisi_Qualitativa_NVivo", nvivo_df)
        beautify("Mappa_Temi", map_df)

        ws_kw = writer.sheets["Keywords_Clustered"]
        dist_idx = kcols.index("distance_to_centroid")
        rank_idx = kcols.index("rank_in_cluster")
        norm_idx = kcols.index("distance_norm")
        ws_kw.conditional_format(1, norm_idx, len(out), norm_idx, {
            "type": "3_color_scale",
            "min_color": "#63BE7B",
            "mid_color": "#FFEB84",
            "max_color": "#F8696B",
        })
        ws_kw.set_column(dist_idx, dist_idx, None, fmt_2d)
        ws_kw.set_column(rank_idx, rank_idx, None, fmt_int)

        ws_map = writer.sheets["Mappa_Temi"]
        pct_col = list(map_df.columns).index("perc_keyword")
        ws_map.set_column(pct_col, pct_col, None, fmt_pct)

    print(f"✅ File creato: {out_path}")


# -----------------------
# CLI
# -----------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keyword Strategy – Metodo A&N (Conforti) con NVivo & Mappa Temi"
    )
    parser.add_argument("--input", required=True, help="Percorso file input (.xlsx o .csv)")
    parser.add_argument("--sheet", dest="sheet_name", default=None, help="Nome foglio (se Excel)")
    parser.add_argument("--text-col", default=DEFAULT_KEYWORD_COLUMN, help="Nome colonna testo")
    parser.add_argument("--volume-col", default=None, help="Nome colonna volume (opzionale)")
    parser.add_argument("--k", type=int, default=None, help="Numero cluster (se non passato, usa auto-k)")
    parser.add_argument("--auto-k", action="store_true", help="Abilita scelta automatica di k")
    parser.add_argument("--k-min", type=int, default=2, help="k minimo per auto-k (default: 2)")
    parser.add_argument("--k-max", type=int, default=20, help="k massimo per auto-k (default: 20)")
    parser.add_argument("--out", dest="out_path", default="keyword_clusters_AeN.xlsx", help="Percorso Excel output")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        input_path=args.input,
        sheet_name=args.sheet_name,
        text_col=args.text_col,
        volume_col=args.volume_col,
        k=args.k,
        auto_k=args.auto_k,
        out_path=args.out_path,
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

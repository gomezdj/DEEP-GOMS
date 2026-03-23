"""
DEEP-GOMS v2 — GOMS Heatmap Generator
========================================
Produces publication-quality heatmaps of GOMS taxa x cohorts for:
  - SHAP importance scores
  - CLR abundance
  - Log2 fold-change (responder vs non-responder)
  - Predicted immunotherapy response probability (ŷ)

Reads from:
    results/goms_signatures.tsv      — top-50 taxa + aggregated SHAP
    results/lodo_per_cohort.tsv      — per-cohort metrics
    data/processed/X_M.npy           — gut microbiome CLR matrix
    data/processed/y.npy             — response labels
    data/processed/metadata.tsv      — cohort, tumor_type, patient_id

User cohort overlay:
    Pass --cohort path/to/cohort.csv  (columns: patient_id, taxon, abundance,
                                       response, cohort, tumor_type)

Usage
-----
    python scripts/goms_heatmap.py --config config.yaml
    python scripts/goms_heatmap.py --config config.yaml --cohort my_cohort.csv
    python scripts/goms_heatmap.py --config config.yaml --scale shap --cancer MELA
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")

matplotlib.rcParams.update({
    "font.family":      "Arial",
    "font.size":        9,
    "axes.linewidth":   0.5,
    "xtick.major.width":0.5,
    "ytick.major.width":0.5,
    "pdf.fonttype":     42,   # editable text in Illustrator
    "svg.fonttype":     "none",
})

# ── GOMS literature priors (Thomas et al. Nat Rev Clin Oncol 2023) ───────────
GOMS_CLASS = {
    "Akkermansia muciniphila":       "beneficial",
    "Faecalibacterium prausnitzii":  "beneficial",
    "Bifidobacterium longum":        "beneficial",
    "Ruminococcaceae":               "beneficial",
    "Lachnospiraceae":               "beneficial",
    "Roseburia intestinalis":        "beneficial",
    "Eubacterium rectale":           "beneficial",
    "Blautia obeum":                 "beneficial",
    "Collinsella aerofaciens":       "beneficial",
    "Lactobacillus acidophilus":     "beneficial",
    "Fusobacterium nucleatum":       "suppressive",
    "Peptostreptococcus anaerobius": "suppressive",
    "Clostridium difficile":         "suppressive",
    "Bacteroides thetaiotaomicron":  "suppressive",
    "Prevotella copri":              "suppressive",
}


# ── Color palettes ────────────────────────────────────────────────────────────
CMAPS = {
    "shap":  sns.diverging_palette(145, 10, s=80, l=45, as_cmap=True),
    "clr":   sns.diverging_palette(220, 20, s=75, l=50, as_cmap=True),
    "fc":    sns.diverging_palette(145, 10, s=80, l=45, as_cmap=True),
    "ypred": sns.color_palette("RdYlGn", as_cmap=True),
}

RESPONSE_COLORS = {"R": "#1D9E75", "NR": "#A32D2D"}
CLASS_COLORS    = {"beneficial": "#085041", "suppressive": "#791F1F"}


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_goms(results_dir: Path, top_k: int = 30) -> pd.DataFrame:
    """Load aggregated SHAP GOMS signatures."""
    path = results_dir / "goms_signatures.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"GOMS signatures not found at {path}. "
            "Run train_model.py or lodo_cross_validation.py first."
        )
    df = pd.read_csv(path, sep="\t")
    df = df.nlargest(top_k, "importance")
    df["goms_class"] = df["feature"].map(
        lambda f: next((v for k,v in GOMS_CLASS.items() if k.lower() in f.lower()), "unknown")
    )
    return df.reset_index(drop=True)


def load_abundance(data_dir: Path, metadata: pd.DataFrame,
                   goms_features: list, feature_names: list) -> pd.DataFrame:
    """
    Extract CLR-normalised abundance for GOMS taxa from X_M.npy.
    Returns DataFrame (patients × goms_taxa).
    """
    X_M = np.load(data_dir / "X_M.npy")
    feat_lower = [f.lower() for f in feature_names]
    idx = []
    names = []
    for feat in goms_features:
        fl = feat.lower()
        matches = [i for i,f in enumerate(feat_lower) if fl[:12] in f]
        if matches:
            idx.append(matches[0])
            names.append(feat)
    if not idx:
        log.warning("No GOMS features matched X_M feature names — using random subset")
        idx   = list(range(min(len(goms_features), X_M.shape[1])))
        names = goms_features[:len(idx)]

    df = pd.DataFrame(X_M[:, idx], index=metadata.index, columns=names)
    return df


def compute_fold_change(abundance: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """Log2 fold-change: mean(R) - mean(NR) in CLR space."""
    R  = abundance[y == 1].mean()
    NR = abundance[y == 0].mean()
    return R - NR


def compute_pvalues(abundance: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """Mann-Whitney U test, BH-corrected."""
    pvals = {}
    for col in abundance.columns:
        r  = abundance.loc[y == 1, col].values
        nr = abundance.loc[y == 0, col].values
        if len(r) < 2 or len(nr) < 2:
            pvals[col] = 1.0
        else:
            _, p = mannwhitneyu(r, nr, alternative="two-sided")
            pvals[col] = p
    _, pvals_corr, _, _ = multipletests(list(pvals.values()), method="fdr_bh")
    return pd.Series(dict(zip(pvals.keys(), pvals_corr)))


# ── Per-cohort heatmap matrix builders ───────────────────────────────────────

def build_shap_matrix(goms_df: pd.DataFrame,
                      shap_dir: Path,
                      cohorts: list) -> pd.DataFrame:
    """
    Mean |SHAP| per taxon per cohort from per-cohort SHAP files.
    Falls back to global SHAP if per-cohort files are absent.
    """
    mat = {}
    for cohort in cohorts:
        path = shap_dir / f"cohort_{cohort}_shap_agg.npy"
        if path.exists():
            arr = np.load(path)
            n   = min(len(arr), len(goms_df))
            mat[cohort] = arr[:n]
        else:
            mat[cohort] = goms_df["importance"].values
    return pd.DataFrame(mat, index=goms_df["feature"].values)


def build_clr_matrix(abundance: pd.DataFrame,
                     metadata: pd.DataFrame,
                     goms_features: list) -> pd.DataFrame:
    """Mean CLR abundance per taxon per cohort."""
    mat = {}
    for cohort, sub in metadata.groupby("study_id"):
        idx  = sub.index
        vals = abundance.loc[abundance.index.isin(idx)]
        mat[str(cohort)] = vals.mean()
    df = pd.DataFrame(mat)
    df.index = [f for f in goms_features if f in df.index]
    return df


def build_fc_matrix(abundance: pd.DataFrame,
                    metadata: pd.DataFrame,
                    y: np.ndarray) -> pd.DataFrame:
    """Log2 fold-change per taxon per cohort."""
    mat = {}
    for cohort, sub in metadata.groupby("study_id"):
        idx   = sub.index.tolist()
        y_sub = y[idx] if max(idx) < len(y) else np.zeros(len(idx))
        ab    = abundance.loc[abundance.index.isin(idx)]
        if len(ab) > 0 and y_sub.sum() > 0:
            mat[str(cohort)] = compute_fold_change(ab, y_sub)
    return pd.DataFrame(mat)


# ── Main plotting function ────────────────────────────────────────────────────

def plot_goms_heatmap(
    matrix:      pd.DataFrame,
    goms_df:     pd.DataFrame,
    lodo_df:     pd.DataFrame,
    scale:       str,
    cancer_filter: str,
    output_path: Path,
    user_cohort: pd.DataFrame = None,
    figsize:     tuple = (14, 8),
):
    """
    Produce a publication-quality GOMS heatmap.

    Layout:
        Left col   : GOMS class bar (beneficial=teal, suppressive=red)
        Main panel : heatmap (taxa × cohorts) with SHAP/CLR/FC scale
        Top bar    : response label (R/NR) per cohort coloured strip
        Right col  : mean SHAP bar chart
        Bottom     : AUROC annotation per cohort
    """
    if cancer_filter != "all":
        cols = [c for c in matrix.columns if cancer_filter.upper() in c.upper()]
        if cols:
            matrix = matrix[cols]
        lodo_df = lodo_df[lodo_df["cohort_id"].str.upper().str.contains(cancer_filter.upper())]

    fig = plt.figure(figsize=figsize, dpi=150)
    gs  = fig.add_gridspec(
        3, 4,
        width_ratios=[0.04, 1, 0.18, 0.004],
        height_ratios=[0.04, 1, 0.08],
        hspace=0.04, wspace=0.03,
    )

    ax_class  = fig.add_subplot(gs[1, 0])
    ax_hm     = fig.add_subplot(gs[1, 1])
    ax_shap   = fig.add_subplot(gs[1, 2])
    ax_resp   = fig.add_subplot(gs[0, 1])
    ax_auroc  = fig.add_subplot(gs[2, 1])
    ax_cbar   = fig.add_subplot(gs[1, 3])

    n_taxa   = len(matrix)
    n_cohort = matrix.shape[1]
    taxa     = matrix.index.tolist()

    # ── Response strip (top) ─────────────────────────────────────────────────
    resp_row = []
    for col in matrix.columns:
        r = "R" if any(s in col for s in ["R","resp","Resp"]) else "NR"
        resp_row.append(RESPONSE_COLORS.get(r, "#888780"))

    ax_resp.imshow(
        np.array(resp_row).reshape(1, -1),
        aspect="auto", interpolation="none",
        extent=[-0.5, n_cohort-0.5, -0.5, 0.5]
    )
    for i, col in enumerate(matrix.columns):
        r_label = "R" if RESPONSE_COLORS.get("R") == resp_row[i] else "NR"
        ax_resp.text(i, 0, r_label, ha="center", va="center",
                     fontsize=6.5, color="white", fontweight="bold")
    ax_resp.set_xlim(-0.5, n_cohort-0.5)
    ax_resp.axis("off")

    # ── Main heatmap ──────────────────────────────────────────────────────────
    cmap  = CMAPS.get(scale, CMAPS["shap"])
    vmax  = matrix.abs().quantile(0.95).max()
    vmax  = max(vmax, 0.1)
    vcent = 0 if scale in ("shap","clr","fc") else vmax/2

    im = ax_hm.imshow(
        matrix.values,
        aspect="auto",
        cmap=cmap,
        vmin=-vmax if scale != "ypred" else 0,
        vmax=vmax,
        interpolation="none",
    )

    ax_hm.set_xticks(range(n_cohort))
    ax_hm.set_xticklabels(
        [c.replace("HGMT-","").replace("ONCOBIOME-","").replace("MCSPACE-","")
          .replace("NRCO-","")[:10]
         for c in matrix.columns],
        rotation=40, ha="right", fontsize=7
    )
    ax_hm.set_yticks(range(n_taxa))
    ax_hm.set_yticklabels(taxa, fontsize=7.5)
    ax_hm.tick_params(length=0)

    for spine in ax_hm.spines.values():
        spine.set_linewidth(0.3)

    if user_cohort is not None:
        ax_hm.axvline(n_cohort - len(user_cohort["cohort"].unique()) - 0.5,
                      color="#BA7517", lw=1.2, linestyle="--", label="user cohort")
        ax_hm.text(n_cohort - 0.5, -1.2, "user cohort",
                   color="#BA7517", fontsize=7, ha="right")

    # ── GOMS class bar (left) ─────────────────────────────────────────────────
    class_colors = [CLASS_COLORS.get(
        goms_df[goms_df["feature"]==t]["goms_class"].values[0]
        if t in goms_df["feature"].values else "unknown", "#B4B2A9"
    ) for t in taxa]

    ax_class.imshow(
        np.array([[1]]*n_taxa).reshape(n_taxa, 1),
        aspect="auto", interpolation="none",
        extent=[-0.5, 0.5, -0.5, n_taxa-0.5]
    )
    for i, col in enumerate(class_colors):
        ax_class.add_patch(mpatches.Rectangle((-0.5, i-0.5), 1, 1,
                                               color=col, linewidth=0))
    ax_class.set_xlim(-0.5, 0.5)
    ax_class.set_ylim(-0.5, n_taxa-0.5)
    ax_class.axis("off")

    # ── SHAP bar (right) ──────────────────────────────────────────────────────
    shap_vals  = goms_df.set_index("feature").reindex(taxa)["importance"].fillna(0)
    bar_colors = [CLASS_COLORS.get(
        goms_df[goms_df["feature"]==t]["goms_class"].values[0]
        if t in goms_df["feature"].values else "unknown", "#B4B2A9"
    ) for t in taxa]

    ax_shap.barh(range(n_taxa), shap_vals.values,
                 color=bar_colors, height=0.65, linewidth=0)
    ax_shap.set_ylim(-0.5, n_taxa-0.5)
    ax_shap.set_yticks([])
    ax_shap.set_xlabel("mean |SHAP|", fontsize=7)
    ax_shap.tick_params(axis="x", labelsize=6.5)
    ax_shap.spines["top"].set_visible(False)
    ax_shap.spines["right"].set_visible(False)
    ax_shap.spines["left"].set_visible(False)
    ax_shap.spines["bottom"].set_linewidth(0.3)

    # ── AUROC annotation (bottom) ─────────────────────────────────────────────
    aucs = []
    for col in matrix.columns:
        row = lodo_df[lodo_df["cohort_id"].astype(str) == str(col)]
        aucs.append(float(row["AUROC"].values[0]) if len(row) else 0.0)

    bar_cols = ["#1D9E75" if a >= 0.7 else "#BA7517" if a >= 0.6 else "#A32D2D" for a in aucs]
    ax_auroc.bar(range(n_cohort), aucs, color=bar_cols, width=0.7, linewidth=0)
    ax_auroc.axhline(0.7, color="#444441", lw=0.5, linestyle="--")
    ax_auroc.set_xlim(-0.5, n_cohort-0.5)
    ax_auroc.set_ylim(0, 1.05)
    ax_auroc.set_yticks([0.5, 0.7, 0.9])
    ax_auroc.set_yticklabels(["0.5","0.7","0.9"], fontsize=6)
    ax_auroc.set_xticks([])
    ax_auroc.set_ylabel("AUROC", fontsize=7)
    ax_auroc.spines["top"].set_visible(False)
    ax_auroc.spines["right"].set_visible(False)
    for spine in ["left","bottom"]:
        ax_auroc.spines[spine].set_linewidth(0.3)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cb = plt.colorbar(im, cax=ax_cbar)
    cb.ax.tick_params(labelsize=6.5)
    scale_labels = {"shap":"SHAP","clr":"CLR","fc":"log₂FC","ypred":"ŷ"}
    cb.set_label(scale_labels.get(scale, scale), fontsize=7)
    cb.outline.set_linewidth(0.3)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color=CLASS_COLORS["beneficial"], label="ICB-beneficial taxa"),
        mpatches.Patch(color=CLASS_COLORS["suppressive"],label="ICB-suppressive taxa"),
        mpatches.Patch(color=RESPONSE_COLORS["R"],       label="Responder (R)"),
        mpatches.Patch(color=RESPONSE_COLORS["NR"],      label="Non-responder (NR)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    scale_label = {"shap":"SHAP importance","clr":"CLR abundance",
                   "fc":"log₂ fold-change","ypred":"predicted ŷ"}
    fig.suptitle(
        f"DEEP-GOMS v2  ·  GOMS heatmap  ({scale_label.get(scale,scale)})"
        + (f"  ·  {cancer_filter}" if cancer_filter != "all" else ""),
        fontsize=10, y=1.01, fontweight="normal"
    )

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    log.info(f"Heatmap saved: {output_path}  +  .svg")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import yaml

    parser = argparse.ArgumentParser(description="DEEP-GOMS GOMS heatmap generator")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--scale",   default="shap",
                        choices=["shap","clr","fc","ypred"])
    parser.add_argument("--cancer",  default="all",
                        help="Tumor type filter e.g. MELA, NSCLC, all")
    parser.add_argument("--top_k",   default=30, type=int,
                        help="Top N GOMS features to show")
    parser.add_argument("--cohort",  default=None,
                        help="Path to user cohort CSV for overlay")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir    = Path(config["data"].get("processed_dir", "data/processed"))
    results_dir = Path(config.get("results_dir", "results"))
    shap_dir    = results_dir / "shap"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading GOMS signatures...")
    goms_df = load_goms(results_dir, top_k=args.top_k)
    taxa    = goms_df["feature"].tolist()

    log.info("Loading metadata and labels...")
    metadata = pd.read_csv(data_dir / "metadata.tsv", sep="\t", index_col="patient_id")
    y        = np.load(data_dir / "y.npy")

    log.info("Loading LODO results...")
    lodo_path = results_dir / "lodo_per_cohort.tsv"
    if lodo_path.exists():
        lodo_df = pd.read_csv(lodo_path, sep="\t")
    else:
        log.warning("lodo_per_cohort.tsv not found — AUROC bar will show zeros.")
        lodo_df = pd.DataFrame({"cohort_id": metadata["study_id"].unique(),
                                 "AUROC": [0.0]*metadata["study_id"].nunique()})

    log.info(f"Building {args.scale} matrix...")

    cohorts = sorted(metadata["study_id"].unique().tolist())

    if args.scale == "shap":
        mat = build_shap_matrix(goms_df, shap_dir, cohorts)

    elif args.scale == "clr":
        feat_names = [f"feat_{i}" for i in range(
            np.load(data_dir / "X_M.npy").shape[1])]
        abundance  = load_abundance(data_dir, metadata, taxa, feat_names)
        mat = build_clr_matrix(abundance, metadata, taxa)

    elif args.scale == "fc":
        feat_names = [f"feat_{i}" for i in range(
            np.load(data_dir / "X_M.npy").shape[1])]
        abundance  = load_abundance(data_dir, metadata, taxa, feat_names)
        mat = build_fc_matrix(abundance, metadata, y)
        mat = mat.reindex(taxa).fillna(0)

    else:
        mat = build_shap_matrix(goms_df, shap_dir, cohorts)

    log.info(f"Matrix shape: {mat.shape}")

    user_cohort = None
    if args.cohort:
        user_cohort = pd.read_csv(args.cohort)
        log.info(f"User cohort: {len(user_cohort)} rows from {args.cohort}")
        if "abundance" in user_cohort.columns and "taxon" in user_cohort.columns:
            uc_pivot = user_cohort.pivot_table(
                index="patient_id", columns="taxon", values="abundance", fill_value=0
            )
            for t in taxa:
                if t not in uc_pivot.columns:
                    uc_pivot[t] = 0.0
            uc_mat = uc_pivot[taxa].T
            uc_mat.columns = [f"User-{c}" for c in uc_mat.columns]
            mat = pd.concat([mat, uc_mat], axis=1).fillna(0)

    out = figures_dir / f"goms_heatmap_{args.scale}_{args.cancer}.pdf"
    log.info(f"Plotting → {out}")
    plot_goms_heatmap(
        matrix       = mat,
        goms_df      = goms_df,
        lodo_df      = lodo_df,
        scale        = args.scale,
        cancer_filter= args.cancer,
        output_path  = out,
        user_cohort  = user_cohort,
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
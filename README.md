🧬 DEEP-GOMS: Cohort-Driven Predictive Modeling of Immunotherapy Response

DEEP-GOMS (Deep Evolutionary Ensemble Predictor for Gut OncoMicrobiome Signatures) is an integrated, cohort-driven framework for predicting patient response to cancer immunotherapies and deriving mechanistic insights across the gut microbiome–immune system–tumor axis.

This version emphasizes harmonized, multi-cohort, multi-omic modeling, integrating bulk RNA-seq, single-cell RNA-seq, spatial transcriptomics (Visium, MERSCOPE, CosMx, etc.), tumor microenvironment (TME) deconvolution, and microbiome-derived features to enable robust, interpretable prediction of immunotherapy outcomes.

⸻

🎯 Project Objectives
	•	Predict response to immune checkpoint inhibitors (ICI), CAR-T/CAR-NK therapies, and intratumoral immunotherapy (ITIT).
	•	Identify cross-disease immune–microbiome patterns shared across cancer types.
	•	Model tumor microenvironment (TME) composition and spatial interactions.
	•	Derive patient-level GOMS linking gut dysbiosis, immune cell states, and tumor biology.

⸻

🧭 Action Plan: DEEP-GOMS Predictive Model

Phase 1 — Cohort Discovery & Acquisition
	1.	Identify relevant cohorts (TCGA, MCSPACE, ONCOBIOME, PRECISE, NRCO_GOMS).
	2.	Download bulk RNA-seq, clinical outcomes, and metadata (UCSC Xena, GEO).
	3.	Acquire scRNA-seq (Seurat/Scanpy objects) and CODEX Phenocycler spatial data when available.
	4.	Curate therapy annotations (ICI, CAR-T/NK, ITIT).

Output: Raw cohort-level datasets stored in data/raw/.

⸻

Phase 2 — Data Harmonization & Quality Control
	1.	Standardize gene identifiers (Ensembl ↔ HGNC).
	2.	Filter low-quality samples and low-expression genes.
	3.	Normalize expression (CPM/TPM/log-normalization).
	4.	Correct batch effects across cohorts (Harmony, ComBat, ComBat-seq).
	5.	Harmonize clinical variables and response labels.

Output: Harmonized expression and metadata matrices (data/processed/).

⸻

Phase 3 — Feature Engineering

Bulk & TME Features
	•	Immune deconvolution: CIBERSORT, EPIC, MCPcounter, xCell, TIMER, quanTIseq, ConsensusTME.
	•	Immune scores: ESTIMATE, IPS.

scRNA-seq Features
	•	Cell-type annotation and marker discovery.
	•	Pseudobulk signatures for immune and stromal compartments.

Spatial (CODEX Phenocycler)
	•	Cell phenotyping and neighborhood analysis.
	•	Spatial interaction graphs and proximity metrics.

Microbiome & Network Features
	•	Gut microbiome abundance and dysbiosis scores.
	•	ILRI (Immune–Ligand–Receptor Interaction) network construction.
	•	Graph-based features (centrality, modularity).

Output: Integrated feature matrix per patient.

⸻

Phase 4 — Model Training (DEEP-GOMS)
	1.	Assemble multi-omic feature matrix across cohorts.
	2.	Perform cohort-aware splits (LODO / leave-one-cohort-out CV).
	3.	Train deep ensemble models (PyTorch).
	4.	Optimize hyperparameters and evaluate robustness.

Output: Trained predictive models (models/*.pth).

⸻

Phase 5 — Prediction & Interpretation
	1.	Predict therapy response probabilities.
	2.	Generate patient-specific GOMS fingerprints.
	3.	Identify key microbiome–immune–tumor drivers.
	4.	Visualize networks, TME composition, and spatial interactions.

Output: Prediction scores, GOMS reports, figures.

⸻

💻 Setup and Installation

Python Environment (Modeling)

conda create -n deepgoms python=3.10
conda activate deepgoms
pip install torch scikit-learn pandas numpy scipy matplotlib seaborn networkx adjusttext

R Environment (Harmonization & TME)

Required R (≥ 4.2.0) packages:
	•	UCSCXenaTools
	•	immunedeconv
	•	EPIC, MCPcounter, quantiseqr, xCell
	•	ConsensusTME
	•	Harmony
	•	liana, OmnipathR
	•	Seurat, SingleCellExperiment
	•	codexr (or equivalent CODEX readers)

⸻

🚀 Quick Start

python src/data/download_cohort_data.py

from src.model.deepgoms import DEEPGOMS
model = DEEPGOMS(input_dim=512)


⸻

📂 Project Structure (Core)

DEEP-GOMS/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── microbiome/
│   ├── scRNAseq/
│   └── spatial/
├── src/
│   ├── data/
│   ├── features/
│   ├── model/
│   └── interpret/
├── models/
├── notebooks/
└── README.md


⸻

📊 Outputs
	•	Harmonized multi-cohort feature matrix
	•	DEEP-GOMS trained models
	•	Patient-level response predictions
	•	Interpretable GOMS signatures

⸻

📜 Citation

If you use DEEP-GOMS, please cite the associated manuscript and cohort data sources.

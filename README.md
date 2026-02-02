Here is the complete content for the `README.md` file, formatted correctly in Markdown, with the tables and code blocks properly enclosed.

````markdown
# 🧬 DEEP-GOMS: Cohort-Driven Predictive Modeling of Immunotherapy Response

DEEP-GOMS (Deep learning for Gut-Omni-Immunotherapy Modeling System) is an integrated framework designed to predict patient response to Immune Checkpoint Inhibitors (ICI) and provide mechanistic insights into the gut microbiome-immune system-tumor axis.

This version of DEEP-GOMS shifts focus to leveraging **integrated, harmonized multi-omic cohort datasets** to train and interpret predictive models. This streamlines the pipeline, improving accessibility and bypassing the need for raw sequencing data analysis (e.g., taxonomic profiling with Kraken2 or HUMAnN3).

---

## Key Features (Updated)

* **Cohort Data Integration:** Aggregate and harmonize processed microbiome, spatial, single-cell immunotherapy, immuno-oncology response data from multiple public and private cohort studies (e.g., NRCO_GOMS, MCSPACE, PRECISE-, etc.).
* **Multi-omic Feature Extraction:** Utilizes pre-calculated patient-level features, including gut microbiome profiles, intratumoral immune cell composition, and specialized **ILRI (Immunotherapy) network scores**.
* **Machine Learning:** Trains deep learning models to predict immunotherapy response (e.g., durable clinical benefit) using harmonized multi-omic features across cohorts.
* **Interpretability:** Derives patient-specific **GOMS** linking gut microbiome strains, immune cells, and tumor interactions for mechanistic insight.

---

## 💻 Setup and Installation

DEEP-GOMS requires a standard environment for data science and deep learning. No specialized bioinformatics tools are needed.

### 1. Environment Setup

We recommend using a virtual environment (e.g., `conda`) for dependency management.

```bash
# Clone the repository
git clone [https://github.com/gomezdj/deep-goms.git](https://github.com/gomezdj/deep-goms.git)
cd deep-goms

# Create and activate the conda environment
conda create -n deepgoms python=3.10
conda activate deepgoms
````

### 2\. Python Requirements (Model Training & Core Pipeline)

The primary machine learning, data handling, and network analysis components are run in **Python ($\ge 3.8$)**.

| Package | Version | Purpose |
| :--- | :--- | :--- |
| **PyTorch** | Latest Stable | Core Deep Learning Model (DEEP-GOMS) training. |
| **MetaPhlAn** | Latest Stable | **Taxonomic Profiling** of raw WGS/16S data. |
| **GraPhlAn** | Latest Stable | **Visualization** of taxonomic and phylogenic trees. |
| **scikit-learn** | Latest Stable | Model evaluation, splitting, and general ML utilities. |
| **Numpy** | 1.16.6+ | Numerical computing. |
| **Scipy** | 1.7.1+ | Scientific computing (includes `scipy.optimize` and kernel methods). |
| **Pandas** | 1.2.4+ | Data manipulation and handling. |
| **networkx** | Latest Stable | Core graph analysis for ILRI feature computation. |
| **Seaborn** | 0.11.2+ | Data visualization. |
| **Matplotlib** | 3.4.3+ | Plotting and data visualization. |
| **Adjusttext** | 0.7.3+ | Automated text placement in Matplotlib plots. |

To install the Python requirements:

```bash
# Install required Python packages
pip install torch scikit-learn pandas numpy scipy matplotlib seaborn networkx adjusttext
```

### 3\. R Requirements (Data Harmonization & Feature Engineering)

**R version ($\ge 4.2.0$)** is required for advanced steps like batch effect correction (e.g., Harmony) and the comprehensive immune cell deconvolution used in feature engineering.

| Package | Version | Purpose |
| :--- | :--- | :--- |
| **liana** | 0.1.10 | Ligand-Receptor Interaction (LRI) analysis. |
| **OmnipathR** | 3.7.0 | Accessing molecular networks and pathway data. |
| **immunedeconv** | 2.1.0 | Unified interface for multiple deconvolution methods. |
| **easier** | 1.4.0 | Ensemble immune signature analysis. |
| **EPIC** | 1.1.5 | Immune cell deconvolution method. |
| **MCPcounter** | 1.2.0 | Immune cell deconvolution method. |
| **quantiseqr** | 1.6.0 | Immune cell deconvolution method. |
| **xCell** | 1.1.0 | Immune cell deconvolution method. |
| **ConsensusTME** | 0.0.1.9000 | Consensus Tumor Microenvironment estimation. |
| **dplyr** | 1.0.10 | Data manipulation and structure. |
| **ggplot2** | 3.4.0 | Data visualization. |
| **corrplot** | 0.92 | Visualization of correlation matrices. |

-----

## 🚀 Quick Start: Running the Predictive Framework

This quick start guides you through downloading the integrated data and running a basic prediction/interpretation task using the combined cohort features.

### Step 1: Download Harmonized Cohort Features

The DEEP-GOMS pipeline relies on a pre-processed and harmonized feature matrix that integrates data from the specified cohort studies.

```bash
# This script downloads the pre-processed HDF5 file (or similar format) 
# containing the harmonized multi-omic features and clinical outcomes.
python src/data/download_cohort_data.py
# Output: data/harmonized_features_all_cohorts.h5
```

### Step 2: Load Data and Train a DEEP-GOMS Model

Use the integrated data to initialize and train the predictive model.

```python
import pandas as pd
from src.model.deepgoms import DEEPGOMS
from sklearn.model_selection import train_test_split

# 1. Load the harmonized feature matrix
data = pd.read_hdf('data/harmonized_features_all_cohorts.h5', key='features')
# Assuming features and labels are defined
X = data.drop(columns=['Response_Label', 'Cohort_ID'])
y = data['Response_Label']

# Split data for basic testing (Cross-validation across cohorts is recommended for robust evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize and Train the Model
input_dim = X_train.shape[1] 
model = DEEPGOMS(input_dim=input_dim, hidden_dim=64, num_layers=2)
model.train_model(X_train, y_train, epochs=50, learning_rate=0.001)

# 3. Evaluate and Predict
predictions = model.predict_proba(X_test)
print(f"Sample prediction probabilities (Non-Response vs. Response):\n{predictions[:5]}")
```

### Step 3: Interpret Patient Microbiome Signature 

Generate the core mechanistic insights that link specific microbiome and immune features to the model's prediction for a given patient.

```python
from src.interpret.fingerprint import generate_ilri_fingerprint

# Select a patient's features from the test set for interpretation
patient_features = X_test.iloc[0]

# Generate the patient-specific interpretation
GOMS = generate_goms_signatures(model, patient_features)

print("--- Patient-Specific GOMS ---")
print(f"Predicted Response Probability: {predictions[0][1]:.4f}")
print("Top 5 Positive Predictors (Microbe-Immune-Tumor Interactions):")
print(fingerprint['Positive_Drivers'].head())
```

-----

## ⚙️ Full Workflow (Cohort-Based)

The following steps detail the full DEEP-GOMS predictive pipeline. Users primarily interact with the outputs of **Data Harmonization** and proceed to **Model Training**.

| Step | Description | Dependencies/Tools | Output |
| :--- | :--- | :--- | :--- |
| **Data Acquisition** | Download processed multi-omic and clinical data from specified cohorts. | `src/data/download_cohort_data.py` | Raw Cohort Data Files |
| **Data Harmonization** | Standardize features, units, and metadata across cohorts. Correct batch effects using techniques like Harmony. | **R** (`Harmony`), **Python** (`pandas`) | `harmonized_features_all_cohorts.h5` |
| **Feature Engineering** | Construct **ILRI network features** (e.g., graph centrality) from harmonized immune and microbiome data. | `networkx`, `igraph`, `src/features/ilri_engineer.py` | Integrated Feature Matrix |
| **Model Training** | Train the DEEP-GOMS deep learning architecture using the integrated multi-omic and GOMS features. | `PyTorch`, `scikit-learn` | Trained `deepgoms_model.pth` |
| **Prediction & Interpretation** | Output patient-specific immunotherapy response probabilities and generate detailed GOMS. | `src/model/predict.py`, `src/interpret/signatures.py` | Predictions, Interpretive Reports |

```
```

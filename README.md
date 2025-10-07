# DEEP-GOMS Project
DEEP-GOMS (Deep Evolutionary Exercise Prediction Gut OncoMicrobiome Signatures)

## Overview
The DEEP-GOMS project aims to develop a reproducible and predictive model for gut oncomicrobiome signatures using machine learning algorithms. The pipeline integrates multi-omics data to predict responses to immunotherapy. This experimental design details the controls, experiments, variables, techniques, measurements, and data analysis plan to validate the DEEP-GOMS deep learning predictive model for gut oncomicrobiome signatures (GOMS), immune checkpoint inhibitors (ICI) response, CAR-T, CAR-NK, CAR-M immunotherapies, and intratumoral immunotherapy (ITIT) therapeutics.

### Potential mechanisms that may be employed to improve responses to immunotherapy via exercise

Immunotherapy: ICI
Mode of Exercise: Acute/training
Mechanism: Increase in trafficking and homing of T cells to tumors, Increase in T cell activation and proliferation, Reduce infiltration of immunosuppresive myeloid cells to the TME
Mode of Exercise: Chronic/long term
Mechanism: Diminish the presence of senescent T cells, Improve T cell function and metabolism

Immunotherapy: Adoptive, CAR, and gamma-delta T cell therapies
Mode of Exercise: Acute/training
Mechanism: Increase in T cell numbers including low frequency viral or antigen specific T cells for ex vivo expansion, increase in trafficking and homing of T cells to tumors, increase in T cell activation, proliferation, and cytotoxicity, enhanced persistence of T cells in vivo
Mode of Exercise: Chronic/training
Mechanism: Maintain homeostatic mechanisms for naive T cell survival via IL-7, enhance persistence of T cells in vivo, and decrease in dysfunctional senescent T cells

Immunotherapy: NK Cell Therapies
Mode of Exercise: Acute/training
Mechanism: Increase in cell numbers for ex vivo expansion, increase in trafficking and homing of NK cells to tumors, increase in NK cell activation, proliferation, and cytotoxicity, enhance persistence of NK cells in vivo
Mode of Exercise: Chronic/training
Mechanism: Prevent obesity-mediated NK cell dysfunction, enhance persistence of NK cells in vivo

Immunotherapy: Cancer vaccines: Dendritic cells and acellular
Mode of Exercise: Acute/training
Mechanism: Increase in cell yield from leukapheresis products, improve efficiency of DC maturation in vivo
Mode of Exercise: Chronic/training 
Mechanism: Improve maintenance of circulating DCs normally lost during aging, decreased age-related decline in phagocytic activity, antigen presentation, migratory capacity of DCs

## Features

- **Data Preprocessing**: Automated pipelines for raw data cleaning and transformation.
- **Taxonomic Profiling**: Tools like Kraken2 and Bracken for microbial classification.
- **Statistical Analysis**: Integration of results with HUMAnN3 for pathway analysis.
- **Machine Learning**: Scripts to train and validate predictive models.

## Requirements

- **R (≥ 4.4)**: For data preprocessing and analysis.
- **FASTQC (≥ 0.12.0)**: For quality control of sequencing reads.
- **Kraken2 (≥ 2.1.2)**: For taxonomic classification.
- **HUMAnN3 (≥ 3.6)**: For functional profiling.
- **MetaPhLAN**: For microbial communities from metagenomic shotgun sequencing.
- **Python (≥ 3.8)**: For Kraken2, HUMAnN3, and MetaphLAN integration.

## Use Cases

- **Microbiome Research**: Analyze metagenomic data to identify microbial communities.
- **Biomarker Discovery**: Uncover key taxa or pathways associated with conditions.
- **Predictive Modeling**: Train machine learning models using microbiome datasets.

## Quick Start

1. Clone the repository:
```
   bash
   git clone https://github.com/gomezdj/DEEPGOMS.git
   cd DEEPGOMS
```

2.	Install required dependencies:
```
  Rscript install_dependencies.R
```

## Setup and Installation

1. **Install Dependencies**:
```R
install.packages("fastqcr")
install.packages("curatedMetagenomicData")
install.packages("phyloseq")
install.packages("caret")
install.packages("randomForest")
install.packages("targets")
```
2. **Clone the Repository**:
```
git clone https://github.com/gomezdj/DEEP-GOMS.git
cd DEEP-GOMS
```

3. Running the Workflow
Activate Environment: Ensure you have the necessary environment set up with required dependencies.
Run the Pipeline:
```
library()
```

4. Customization for Personalization
Modify and config.yaml files according to your specific needs.

## Contact
For any questions or issues, please contact Daniel Gomez (danielphysiology@gmail.com).

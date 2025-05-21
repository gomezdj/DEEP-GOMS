# DEEP-GOMS Project

## Overview

The DEEP-GOMS project aims to develop a reproducible and predictive model for gut oncomicrobiome signatures using machine learning algorithms. The pipeline integrates multi-omics data to predict responses to immunotherapy.

## Features

- **Data Preprocessing**: Automated pipelines for raw data cleaning and transformation.
- **Taxonomic Profiling**: Tools like Kraken2 and Bracken for microbial classification.
- **Statistical Analysis**: Integration of results with HUMAnN3 for pathway analysis.
- **Machine Learning**: Scripts to train and validate predictive models.

## Requirements

- **R (≥ 4.4)**: For data preprocessing and analysis.
- **FASTQC (≥ 0.11.9)**: For quality control of sequencing reads.
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
   git clone https://github.com/gomez-dan/DEEP-GOMS.git
   cd DEEP-GOMS
```

2.	Install required dependencies:
```
  Rscript install_dependencies.R
```

## Setup and Installation

1. **Install Dependencies**:
```R
install.packages("fastqcr")
install.packages("dada2")
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
library(targets)
tar_make()
```

4. Customization for Personalization
Modify the _targets.R and config.yaml files according to your specific needs.

## Contact
For any questions or issues, please contact Daniel Gomez (gomezscientist0@gmail.com).

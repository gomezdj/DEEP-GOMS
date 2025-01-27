library(tidyr)
library(haia)
library(MMUPHin)
library(phyloseq)

# Load preprocessed data
# load("results/wide_bracken.RData")

# Abundance Profile Analysis
# abundance_data <- wide_bracken

# Load Bracken output for abundance profiling
abundance_data <- read.table("results/bracken_output.tsv", header = TRUE, sep = "\t")

# HAIIA for multi-resolution associations
# Assuming `haia` package is installed and haia() function works with required syntax
# haia_result <- haia(input_data, group_var = "condition")

# MMUPHin for phylogenetic tree construction
physeq <- phyloseq(otu_table(abundance_data, taxa_are_rows = TRUE), tax_table(tax_data), sample_data(meta_data))
tree <- multi_analysis(physeq, covariates = ~ condition, method = "UNIFRAC")
results <- phylogenetic_analysis(tree)
save(results, file = "results/phylogenetic_results.RData")
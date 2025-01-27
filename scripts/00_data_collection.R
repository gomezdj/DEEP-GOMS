# This R script ensures to preprocess the sequencing data including quality control and classification
library(data.table)
library(ggplot2)
library(dplyr)

# Data collection from multiple cohorts
list_of_files <- list.files(path = "datasets/", pattern = "*.csv", full.names = TRUE)
data_list <- lapply(list_of_files, fread)
all_data <- rbindlist(data_list)

# Quality Control with FastQC
system("fastqc -o results/fastqc raw_data/*.fastq")

# Trimming Reads Trimmomatic or fastp
system("trimmomatic PE -phred33 raw_data/sample_R1.fastq raw_data/sample_R2.fastq data/trimmed_R1.fastq data/unpaired_R1.fastq data/trimmed_R2.fastq data/unpaired_R2.fastq ILLUMINACLIP:adapters/TruSeq3-PE-2.fa:2:30:10 SLIDINGWINDOW:4:20 MINLEN:36")

# Kraken 2 Classification
system("kraken2 --db path_to_kraken_db --paired data/trimmed_R1.fastq data/trimmed_R2.fastq --output results/kraken2_output.tsv")

# Bracken Abundance Computation
system("bracken -d path_to_kraken_db -i results/kraken2_output.tsv -o results/bracken_output.tsv")

# Load Bracken results
bracken_results <- fread("results/bracken_output.tsv")

# Convert to wide format
wide_bracken <- dcast(bracken_results, SampleID ~ Taxon, value.var = "Fraction_total_reads")
save(wide_bracken, file = "results/wide_bracken.RData")
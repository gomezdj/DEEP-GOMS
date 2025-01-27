# Load necessary libraries
library(data.table)
library(ggplot2)
library(dplyr)

# Define paths for raw and processed data
input_dir <- "path/to/raw_data/"
output_dir <- "path/to/results/"
trimmed_dir <- file.path(output_dir, "trimmed")
kraken_output <- file.path(output_dir, "kraken2_output.tsv")
bracken_output <- file.path(output_dir, "bracken_output.tsv")
humann_output <- file.path(output_dir, "humann3_output")

# Create necessary directories
dir.create(trimmed_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(humann_output, recursive = TRUE, showWarnings = FALSE)

# Function to run system commands with error handling
run_command <- function(cmd) {
  result <- system(cmd, intern = TRUE)
  if (!is.null(attr(result, "status")) && attr(result, "status") != 0) {
    stop(paste("Error running command:", cmd, "\n", result))
  }
  return(result)
}

# Quality Control with FastQC
fastqc_cmd <- paste("fastqc -o", output_dir, "fastqc", input_dir, "*.fastq")
run_command(fastqc_cmd)

# Trim Reads with fastp
fastp_cmd <- paste(
  "fastp -i", file.path(input_dir, "sample_R1.fastq"),
  "-I", file.path(input_dir, "sample_R2.fastq"),
  "-o", file.path(trimmed_dir, "trimmed_R1.fastq"),
  "-O", file.path(trimmed_dir, "trimmed_R2.fastq")
)
run_command(fastp_cmd)

# Kraken 2 for DNA-seq Classification
kraken2_cmd <- paste(
  "kraken2 --db /path/to/kraken_db --paired", 
  file.path(trimmed_dir, "trimmed_R1.fastq"), 
  file.path(trimmed_dir, "trimmed_R2.fastq"), 
  "--output", kraken_output
)
run_command(kraken2_cmd)

# Bracken for Abundance Estimation
bracken_cmd <- paste(
  "bracken -d /path/to/kraken_db -i", kraken_output,
  "-o", bracken_output
)
run_command(bracken_cmd)

# HUMAnN 3.0 for Functional Profiling
humann_cmd <- paste(
  "humann --input", file.path(trimmed_dir, "trimmed_R1.fastq"), 
  paste(file.path(trimmed_dir, "trimmed_R2.fastq"), collapse=","),
  "--metaphlan", kraken_output,
  "--output", humann_output,
  "--threads 8"
)
run_command(humann_cmd)

# Load Bracken results
bracken_results <- fread(bracken_output)

# Convert to wide format
wide_bracken <- dcast(bracken_results, SampleID ~ Taxon, value.var = "Fraction_total_reads")
save(wide_bracken, file = file.path(output_dir, "wide_bracken.RData"))

# Function to process HUMAnN results
read_humann_result <- function(file_path) {
  humann_data <- fread(file_path, sep = "\t", header = TRUE)
  return(humann_data)
}

# Read HUMAnN results for pathways and gene families
pathways_result <- read_humann_result(file.path(humann_output, "pathways.tsv"))
genefamilies_result <- read_humann_result(file.path(humann_output, "gene_families.tsv"))

# Save HUMAnN results
save(pathways_result, file = file.path(output_dir, "humann_pathways.RData"))
save(genefamilies_result, file = file.path(output_dir, "humann_genefamilies.RData"))
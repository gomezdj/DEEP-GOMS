library(targets)

# Define target directory paths
input_dir <- "data/"
output_dir <- "results/"
trimmed_dir <- file.path(output_dir, "trimmed/")
kraken_output <- file.path(output_dir, "kraken2/")
bracken_output <- file.path(output_dir, "bracken/")
humann_output <- file.path(output_dir, "humann3/")

# Define your targets
list(
  tar_target(samples, list.files(input_dir, "_R1.fastq")),
  
  # Quality Control
  tar_target(
    fastqc_results,
    quality_control(samples, file.path(output_dir, "fastqc/")),
    format = "file"
  ),
  
  # Trim Reads
  tar_target(
    trimmed_files,
    trim_reads(samples, file.path(trimmed_dir, paste0(basename(samples), "_trimmed.fastq"))),
    format = "file"
  ),
  
  # Kraken 2 Classification
  tar_target(
    kraken_results,
    kraken2_classification(trimmed_files, "/path/to/kraken_db", kraken_output),
    format = "file"
  ),
  
  # Bracken Abundance Estimation
  tar_target(
    bracken_results,
    bracken_abundance(kraken_results, "/path/to/kraken_db", bracken_output),
    format = "file"
  ),
  
  # HUMAnN 3.0 Functional Profiling
  tar_target(
    humann_results,
    humann_functional(trimmed_files, "/path/to/humann_db", humann_output),
    format = "file"
  ),
  
  # Integrate Results
  tar_target(
    integrated_data,
    integrate_results(bracken_results, file.path(humann_output, "pathway_abundance.tsv"), file.path(humann_output, "genefamilies.tsv")),
    format = "file"
  ),
  
  # Model Training
  tar_target(
    model,
    train_model(integrated_data),
    format = "rds"
  )
)
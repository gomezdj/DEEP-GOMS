source("R/functions.R")

# Define file paths for results
bracken_file <- "results/bracken/bracken_report.tsv"
humann_pathways <- "results/humann3/pathway_abundance.tsv"
humann_genefamilies <- "results/humann3/genefamilies.tsv"

# Integrate results
integrated_data <- integrate_results(bracken_file, humann_pathways, humann_genefamilies)

# Save integrated data
write.csv(integrated_data, "results/integrated_data.csv", row.names = FALSE)

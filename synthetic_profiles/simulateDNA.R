library(simDNAmixtures)
library(dplyr)
library(xml2)

# File must be run from synthetic_profiles/ directory,
# otherwise adjust the path to sim_helpers.R accordingly.
source("sim_helpers.R")

# Repo root is the parent directory of synthetic_profiles
repo_root <- normalizePath("..")

# Allow caller to override the output directory suffix; default to timestamped folder
args <- commandArgs(trailingOnly = TRUE)
output_suffix <- if (length(args) >= 1 && nzchar(args[1])) args[1] else format(Sys.time(), "%Y%m%d_%H%M%S")
output_dir_name <- sprintf("generated_alleles_%s", output_suffix)
paths <- build_output_dirs(output_dir_name)

# Load allele frequencies
allele_freqs_file <- system.file("extdata","FBI_extended_Cauc_022024.csv", package = "simDNAmixtures")
allele_freqs <- read_allele_freqs(allele_freqs_file)

threshold_rfu <- 15
gf <- configure_global_filer(threshold_rfu)

dye_map <- kits$GlobalFiler[, c("Marker", "Color")] %>% distinct(Marker, .keep_all = TRUE)
size_standard_df <- gf_build_size_standard_df()
panel_lookup <- build_panel_lookup(file.path(repo_root, "resources", "data", "SGPanel_Globalfiler_Panel.xml"))
template_ratio_functions <- get_template_ratio_functions()

sim_params <- get_simulation_params()
base_template_amounts <- sim_params$base_template_amounts
degradation_settings <- sim_params$degradation_settings
contributors_list <- sim_params$contributors_list
replicates <- sim_params$replicates
n_per_config <- sim_params$n_per_config
config_id <- 1

# Persist run metadata alongside outputs
write_run_metadata(paths$output_dir, threshold_rfu, sim_params)


# Output directories
alleles_dir <- paths$alleles_dir
genotypes_dir <- paths$genotypes_dir

# Mapping list
alleles_to_genotypes <- data.frame(EPGFile=character(), GenotypeFile=character(), stringsAsFactors=FALSE)
genotype_id <- 1

for (n_contributors in contributors_list) {
  for (deg_idx in seq_along(degradation_settings)) {
    for (base_tmpl_idx in seq_along(base_template_amounts)) {
      for (ratio_fn_idx in seq_along(template_ratio_functions)) {
        contributors <- paste0("U", seq_len(n_contributors))
        set.seed(100000*config_id) # Unique seed per config

        # Compute exact template amounts for each contributor
        ratio_fn <- template_ratio_functions[[ratio_fn_idx]]
        base_template <- base_template_amounts[base_tmpl_idx]
        template_vec <- base_template * ratio_fn(n_contributors)

        # Set parameters for this config
        sampling_parameters <- list(
          min_template = template_vec,
          max_template = template_vec,
          degradation_shape = degradation_settings[[deg_idx]]$shape,
          degradation_scale = degradation_settings[[deg_idx]]$scale
        )

        mixtures <- sample_mixtures(
          n = n_per_config,
          contributors = contributors,
          freqs = allele_freqs,
          sampling_parameters = sampling_parameters,
          model_settings = gf$log_normal_settings,
          sample_model = sample_log_normal_model,
          number_of_replicates = replicates
        )

        # Print sample name and parameter summary in a readable format
        cat("\n==============================\n")
        cat(sprintf("Config ID: %d\n", config_id))
        print(mixtures$parameter_summary[1:18])
        cat("==============================\n\n")

        # Write each sample and replicate to file
        for (i in seq_along(mixtures$samples)) {
          sim_peaks <- mixtures$samples[[i]]$mixture
          sim_peaks_with_dye <- left_join(sim_peaks, dye_map, by = c("Locus" = "Marker"))

          # Adjust sizes to match panel sizes
          sim_peaks_with_panel <- left_join(sim_peaks_with_dye, panel_lookup, by = c("Locus", "Allele"))

          # If PanelSize is available, use it; otherwise, keep original Size
          peaks_fixed_size_df <- sim_peaks_with_panel %>%
            mutate(Size = ifelse(!is.na(PanelSize), PanelSize, Size)) %>%
            select(-PanelSize)

          combined_peaks_df <- rbind(peaks_fixed_size_df, size_standard_df)
          combined_peaks_df <- combined_peaks_df %>%
            mutate(Scan = round(Size * 11.2 + 3500))

          # Extract sample and replicate info from sample_name
          sample_name <- mixtures$samples[[i]]$sample_name
          
          # Save alleles file
          allele_file <- sprintf("%s/epg_configID%d_%s_deg%d.csv", alleles_dir, config_id, sample_name, deg_idx)
          write.csv(combined_peaks_df, allele_file, row.names = FALSE)

          # Save genotype file for each contributor
          genotype_file_list <- c()
          for (c_idx in seq_along(contributors)) {
            geno_df <- as.data.frame(mixtures$samples[[i]]$contributor_genotypes[[c_idx]])
            geno_df <- cbind(SampleName = genotype_id, geno_df)
            genotype_file <- sprintf("%s/genotype_%04d.csv", genotypes_dir, genotype_id)
            write.csv(geno_df, genotype_file, row.names = FALSE)
            genotype_file_list <- c(genotype_file_list, basename(genotype_file))
            genotype_id <- genotype_id + 1
          }
          # Add mapping row
          alleles_to_genotypes <- rbind(alleles_to_genotypes, data.frame(
            EPGFile = basename(allele_file),
            GenotypeFile = paste(genotype_file_list, collapse=","),
            stringsAsFactors=FALSE
          ))
        }
        config_id <- config_id + 1
      }
    }
  }
}
# Save mapping file
write.csv(alleles_to_genotypes, file.path(paths$output_dir, "alleles_to_genotypes_mapping.csv"), row.names = FALSE)

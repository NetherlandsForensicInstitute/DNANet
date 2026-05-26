library(simDNAmixtures)
library(dplyr)

# Output directory
output_dir <- "generated/generated_alleles_fixed_ratios_base_template_500_5000_improved/"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Load allele frequencies
allele_freqs_file <- system.file("extdata","FBI_extended_Cauc_022024.csv", package = "simDNAmixtures")
allele_freqs <- read_allele_freqs(allele_freqs_file)
gf <- gf_configuration()
dye_map <- kits$GlobalFiler[, c("Marker", "Color")] %>% distinct(Marker, .keep_all = TRUE)
size_standard_sizes <- c(20, 40, 60, 80, 100, 114, 120, 140, 160, 180, 200, 214,
                         220, 240, 250, 260, 280, 300, 314, 320, 340, 360, 380,
                         400, 414, 420, 440, 460, 480, 500, 514, 520, 540, 560,
                         580, 600)
size_standard_df <- data.frame(
  Locus = "LIZ",
  Allele = NA,
  Height = 800,
  Size = size_standard_sizes,
  Color = "orange"
)

# # Parameter grids
# template_ranges <- list(
#   c(50, 500),
#   c(5000, 10000),
#   c(50, 10000)
# )
# Template ratio functions
template_ratio_even <- function(n) rep(1/n, n)
template_ratio_increasing <- function(n) { ratios <- 1:n; ratios / sum(ratios) }
template_ratio_last_dominates <- function(n) { ratios <- rep(1/20, n); ratios[n] <- 1 - sum(ratios[-n]); ratios }
template_ratio_first_two_10 <- function(n) {
  ratios <- rep(0, n)
  if (n == 1) {
    ratios[1] <- 1
  } else if (n == 2) {
    ratios[1] <- 0.1
    ratios[2] <- 0.9
  } else {
    ratios[1:2] <- 0.1
    ratios[3:n] <- (1 - 0.2) / (n - 2)
  }
  ratios
}
template_ratio_functions <- list(
  template_ratio_even,
  template_ratio_increasing,
  template_ratio_last_dominates,
  template_ratio_first_two_10
)
# Varying parameters
base_template_amounts <- c(500, 5000)  # total template amount per sample

degradation_settings <- list(
  list(shape = 2.5, scale = 1e-3),
  list(shape = 3.5, scale = 2e-3)
)
contributors_list <- c(2, 3, 4, 5)
replicates <- 2

# Calculate number of genotypes per configuration
n_per_config <- 2
config_id <- 1


# Output directories
alleles_dir <- file.path(output_dir, "epgs")
genotypes_dir <- file.path(output_dir, "reference_genotypes")
dir.create(alleles_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(genotypes_dir, showWarnings = FALSE, recursive = TRUE)

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
          combined_peaks_df <- rbind(sim_peaks_with_dye, size_standard_df)
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
write.csv(alleles_to_genotypes, file.path(output_dir, "alleles_to_genotypes_mapping.csv"), row.names = FALSE)
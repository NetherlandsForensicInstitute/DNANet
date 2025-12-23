# Utility helpers for the synthetic profile generator.

build_output_dirs <- function(output_subdir) {
  output_dir <- file.path("generated", output_subdir)
  alleles_dir <- file.path(output_dir, "epgs")
  genotypes_dir <- file.path(output_dir, "reference_genotypes")
  dir.create(alleles_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(genotypes_dir, showWarnings = FALSE, recursive = TRUE)
  list(output_dir = output_dir, alleles_dir = alleles_dir, genotypes_dir = genotypes_dir)
}



configure_global_filer <- function(threshold_rfu = 15) {
  gf <- gf_configuration()
  gf$log_normal_settings$locus_names <- unique(c(gf$log_normal_settings$locus_names, "Yindel", "DYS391"))
  gf$log_normal_settings$detection_threshold <- setNames(
    rep(threshold_rfu, length(gf$log_normal_settings$locus_names)),
    gf$log_normal_settings$locus_names
  )
  gf
}

# Size standard used to append ladder peaks
gf_build_size_standard_df <- function() {
  size_standard_sizes <- c(20, 40, 60, 80, 100, 114, 120, 140, 160, 180, 200, 214,
                           220, 240, 250, 260, 280, 300, 314, 320, 340, 360, 380,
                           400, 414, 420, 440, 460, 480, 500, 514, 520, 540, 560,
                           580, 600)
  data.frame(
    Locus = "LIZ",
    Allele = NA,
    Height = 3000,
    Size = size_standard_sizes,
    Color = "orange"
  )
}

# This fixes the basepair mismatch that occurs in simDNAmixtures package
# Now we take the given basepairs from simDNAmixtures and match it to the nearest entry in the given panel file.
build_panel_lookup <- function(panel_xml_path) {
  panel_xml <- read_xml(panel_xml_path)
  panel_loci <- xml_find_all(panel_xml, ".//Locus")
  lapply(panel_loci, function(locus) {
    marker <- xml_text(xml_find_first(locus, "./MarkerTitle"))
    alleles <- xml_find_all(locus, ".//Allele")
    data.frame(
      Locus = marker,
      Allele = sapply(alleles, function(a) xml_attr(a, "Label")),
      PanelSize = as.numeric(sapply(alleles, function(a) xml_attr(a, "Size"))),
      stringsAsFactors = FALSE
    )
  }) %>% bind_rows()
}




get_template_ratio_functions <- function() {
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
  list(
    template_ratio_even,
    template_ratio_increasing,
    template_ratio_last_dominates,
    template_ratio_first_two_10
  )
}

get_simulation_params <- function() {
  list(
    base_template_amounts = c(300, 500, 1000, 5000),
    degradation_settings = list(
      list(shape = 2.5, scale = 1e-3),
      list(shape = 3.5, scale = 2e-3)
    ),
    contributors_list = c(2, 3, 4, 5),
    replicates = 2,
    n_per_config = 2
  )
}

# Persist the run configuration alongside outputs
write_run_metadata <- function(output_dir, threshold_rfu, sim_params) {
  metadata <- list(
    threshold_rfu = threshold_rfu,
    base_template_amounts = sim_params$base_template_amounts,
    degradation_settings = sim_params$degradation_settings,
    contributors_list = sim_params$contributors_list,
    replicates = sim_params$replicates,
    n_per_config = sim_params$n_per_config
  )
  metadata_path <- file.path(output_dir, "run_parameters.json")
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    json_txt <- jsonlite::toJSON(metadata, pretty = TRUE, auto_unbox = TRUE)
    writeLines(json_txt, metadata_path)
  } else {
    # Fallback to a simple R dump if jsonlite is unavailable
    dput(metadata, file = metadata_path)
  }
}



get_param_summary <- function() {
  # Varying parameters
  template_amounts <- c(100, 1000)  # total template amount per sample
  degradation_amounts <- c(0.0025, 0.007)  # total degradation amount per sample

  # Constants
  c2 <- 14.754
  k2BackStutter <- 14.48
  k2ForwardStutter <- 9.67
  k22bpBackStutter <- 3.13
  k2DoubleBackStutter <- 6.97

  # Create parameter summary for this sample (single row, all contributors as columns)
  parameter_summary <- data.frame(
    SampleName = paste0(
      "nC", n_contributors, "_tmp", template_amount, "_ratio", ratio_idx,
      "_deg", degradation_idx, "_s", sample_idx, "_rep", rep
    ),
    # Contributors
    setNames(as.list(contributors), paste0("contributors", seq_len(n_contributors))),
    model = "log_normal_model",
    # Templates
    setNames(as.list(templates), paste0("template", seq_len(n_contributors))),
    # Degradation
    setNames(as.list(rep(degradation_value, n_contributors)), paste0("degradation", seq_len(n_contributors))),
    # c2, k2BackStutter, k2ForwardStutter, etc. (single values)
    c2 = c2,
    k2BackStutter = k2BackStutter,
    k2ForwardStutter = k2ForwardStutter,
    k22bpBackStutter = k22bpBackStutter,
    k2DoubleBackStutter = k2DoubleBackStutter
  )
}
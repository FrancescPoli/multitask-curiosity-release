# analysis/Cognitive_analysis/demographics_processing.R
#
# Build a per-subject demographics table aligned with the `subj` indexing
# used in our infant datasets. For each infant dataset, the goal is to map
# every `subj` to its original participant ID and then to its demographic
# record (sex, age, parental info, etc.), reporting any unmatched cases.
#
# Currently handles:
#   - Roris_smiley
#
# Data sources are in the archived Habituation project (D:/Archived_projects/Habituation),
# which contains both the raw participant ID lists (markasgood_*.csv) and the
# demographics workbook (demo_smiley.csv). The script copies the relevant
# columns into a portable CSV in our project's Data/Infant_Data folder so
# downstream analyses do not depend on that drive.

suppressPackageStartupMessages({
    library(dplyr)
    library(readr)
})

# --- Configuration ---

HABITUATION_DIR <- "D:/Archived_projects/Habituation"
DATA_DIR        <- "analysis/Cognitive_analysis/Data/Infant_Data"
OUTPUT_DIR      <- DATA_DIR  # write demographics next to the per-trial data

# Excel date origin (Windows / Mac 2007+ default)
EXCEL_ORIGIN <- as.Date("1899-12-30")
excel_to_date <- function(x) {
    x <- suppressWarnings(as.numeric(x))
    out <- rep(as.Date(NA), length(x))
    valid <- !is.na(x) & x > 0 & x < 200000  # filter sentinels like 99999
    out[valid] <- EXCEL_ORIGIN + x[valid]
    out
}

# --- 1. Roris_smiley -----------------------------------------------------

cat("=== Roris_smiley ===\n")

# 1a. Read the `subj` -> original-ID mapping.
# markasgood_smiley.csv is a single-column file with no header; row N is the
# original participant ID for subj == N in Roris_smiley_final.csv.
markasgood_file <- file.path(HABITUATION_DIR, "no_sigma", "markasgood_smiley.csv")
if (!file.exists(markasgood_file)) {
    stop("Cannot find ", markasgood_file,
         ". Mount the D: drive or update HABITUATION_DIR.")
}
markasgood <- read_csv(markasgood_file, col_names = "original_id",
                       col_types = cols(original_id = col_integer()),
                       progress = FALSE)
subj_map <- tibble(
    subj        = seq_len(nrow(markasgood)),
    original_id = markasgood$original_id
)
cat("Loaded mapping for", nrow(subj_map), "subjects (subj 1..",
    max(subj_map$subj), ").\n", sep = "")

# 1b. Read demo_smiley.csv. It is semicolon-separated, contains 1,048,492 rows
# in total (Excel sheet padding), but only the rows with a numeric ID in
# column 1 are real. We filter on that.
demo_file <- file.path(HABITUATION_DIR, "demo_smiley.csv")
if (!file.exists(demo_file)) {
    stop("Cannot find ", demo_file)
}
demo_raw <- read_delim(demo_file, delim = ";",
                       show_col_types = FALSE,
                       progress = FALSE)
demo <- demo_raw %>%
    mutate(ID = suppressWarnings(as.integer(ID))) %>%
    filter(!is.na(ID))
cat("Loaded demographics for", nrow(demo), "infants.\n")

# Some demographic fields use sentinel values (99999) for missing data.
sentinel_to_na <- function(x, sentinels = c(99999)) {
    x[x %in% sentinels] <- NA
    x
}

demo_clean <- demo %>%
    transmute(
        original_id       = ID,
        visit_date        = excel_to_date(VisitDate),
        infant_sex_code   = sentinel_to_na(I_Sex),        # 1 / 2; 99999 = missing
        infant_dob        = excel_to_date(I_DoB),
        mother_dob        = excel_to_date(M_DoB),
        mother_age_years  = sentinel_to_na(M_Age),
        infant_age_days   = sentinel_to_na(I_Age_Days),
        infant_age_months = sentinel_to_na(I_Age_Months),
        mother_education  = sentinel_to_na(M_Education)
    )

# 1c. Join: subj -> original_id -> demographics
demo_smiley <- subj_map %>%
    left_join(demo_clean, by = "original_id")

# 1d. Report matches / mismatches
matched   <- demo_smiley %>% filter(!is.na(visit_date))
unmatched <- demo_smiley %>% filter(is.na(visit_date))

cat(sprintf("Matched   : %d / %d subjects\n", nrow(matched),   nrow(demo_smiley)))
cat(sprintf("Unmatched : %d / %d subjects\n", nrow(unmatched), nrow(demo_smiley)))
if (nrow(unmatched) > 0) {
    cat("Unmatched subj -> original_id:\n")
    print(unmatched %>% select(subj, original_id))
    cat("(These IDs are not present in demo_smiley.csv.",
        "Demographics for them are reported as NA.)\n")
}

# 1e. Quick summary of demographics for the matched subset (for sanity)
cat("\n--- Summary (matched subset) ---\n")
cat("Infant age (days)   : ",
    sprintf("median = %.0f, range = [%.0f, %.0f]\n",
            median(matched$infant_age_days,   na.rm = TRUE),
            min   (matched$infant_age_days,   na.rm = TRUE),
            max   (matched$infant_age_days,   na.rm = TRUE)))
cat("Infant age (months) : ",
    sprintf("median = %.1f, range = [%.1f, %.1f]\n",
            median(matched$infant_age_months, na.rm = TRUE),
            min   (matched$infant_age_months, na.rm = TRUE),
            max   (matched$infant_age_months, na.rm = TRUE)))
cat("Mother age (years)  : ",
    sprintf("median = %.1f, range = [%.0f, %.0f]\n",
            median(matched$mother_age_years,  na.rm = TRUE),
            min   (matched$mother_age_years,  na.rm = TRUE),
            max   (matched$mother_age_years,  na.rm = TRUE)))
cat("Infant sex          : ",
    paste(sprintf("code=%s: n=%d",
                  names(table(matched$infant_sex_code, useNA = "ifany")),
                  as.integer(table(matched$infant_sex_code, useNA = "ifany"))),
          collapse = "; "), "\n")

# 1f. Write output CSV
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)
out_file <- file.path(OUTPUT_DIR, "Roris_smiley_demographics.csv")
write_csv(demo_smiley, out_file)
cat("\nSaved:", out_file, "\n")

# ============================================================================
# Cross-dataset summary table
# ============================================================================
# One row per infant dataset, with the values needed to report the sample in
# the manuscript: initial N (number recruited), final N (number included in
# the analysed per-trial data), age (mean + SD), sex breakdown, and source.
# Unknown values are left as NA and computed across non-NA cells; they can be
# filled in once the underlying records are supplied.

cat("\n\n=== Cross-dataset demographics summary ===\n")

safe_mean <- function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
safe_sd   <- function(x) if (all(is.na(x))) NA_real_ else sd  (x, na.rm = TRUE)
count_sex <- function(codes, target) sum(codes == target, na.rm = TRUE)

# --- Roris_smiley ---
rs_trial_file <- file.path(DATA_DIR, "Roris_smiley_final.csv")
rs_trial <- suppressWarnings(read_csv(rs_trial_file, show_col_types = FALSE,
                                      progress = FALSE))
rs_final_n <- length(unique(rs_trial$subj))

roris_smiley_row <- tibble(
    dataset             = "Roris_smiley",
    initial_n           = 83L,            # infants who joined lab visit
    final_n             = as.integer(rs_final_n),
    n_with_demographics = as.integer(nrow(matched)),
    age_mean_months     = safe_mean(matched$infant_age_months),
    age_sd_months       = safe_sd  (matched$infant_age_months),
    age_mean_days       = safe_mean(matched$infant_age_days),
    age_sd_days         = safe_sd  (matched$infant_age_days),
    n_male              = as.integer(count_sex(matched$infant_sex_code, 1)),
    n_female            = as.integer(count_sex(matched$infant_sex_code, 2)),
    source              = NA_character_,
    notes               = sprintf("%d of %d infants in the per-trial data lack demographic records",
                                  nrow(unmatched), rs_final_n)
)

# --- Roris_nostd ---
# Numbers reported in Poli et al. (Open Mind, 2024):
# "Eight-Month-Old Infants Meta-Learn by Downweighting Irrelevant Evidence"
nostd_final_n <- 73L
nostd_n_female <- 34L
roris_nostd_row <- tibble(
    dataset             = "Roris_nostd",
    initial_n           = 90L,
    final_n             = nostd_final_n,
    n_with_demographics = nostd_final_n,                          # all reported in paper
    age_mean_months     = 8.02,                                   # paper, mean
    age_sd_months       = NA_real_,                               # paper reports SD in days only
    age_mean_days       = NA_real_,                               # paper reports mean in months only
    age_sd_days         = 11.37,                                  # paper, SD
    n_male              = nostd_final_n - nostd_n_female,         # 73 - 34 = 39
    n_female            = nostd_n_female,
    source              = "Poli et al. (2024), Open Mind: 'Eight-Month-Old Infants Meta-Learn by Downweighting Irrelevant Evidence'",
    notes               = "17 excluded for completing <20 trials in at least two sequences; 50 of 73 are re-analysed from Poli et al. (2020), 40 are new data"
)

# --- MPI ---
# Numbers not yet supplied; only the (provisional) final N is known.
mpi_row <- tibble(
    dataset             = "MPI",
    initial_n           = NA_integer_,
    final_n             = 130L,
    n_with_demographics = NA_integer_,
    age_mean_months     = NA_real_,
    age_sd_months       = NA_real_,
    age_mean_days       = NA_real_,
    age_sd_days         = NA_real_,
    n_male              = NA_integer_,
    n_female            = NA_integer_,
    source              = NA_character_,
    notes               = "Provisional N; demographics and source publication not yet supplied"
)

demographics_summary <- bind_rows(roris_smiley_row, roris_nostd_row, mpi_row)
print(demographics_summary, width = Inf)

# Pooled totals across the three datasets (NAs omitted)
cat("\n--- Pooled totals across datasets (NA-omitted) ---\n")
cat(sprintf("Total initial N : %s\n", sum(demographics_summary$initial_n, na.rm = TRUE)))
cat(sprintf("Total final N   : %s\n", sum(demographics_summary$final_n,   na.rm = TRUE)))
cat(sprintf("Total female    : %s\n", sum(demographics_summary$n_female,  na.rm = TRUE)))
cat(sprintf("Total male      : %s\n", sum(demographics_summary$n_male,    na.rm = TRUE)))

summary_file <- file.path(OUTPUT_DIR, "infant_datasets_demographics_summary.csv")
write_csv(demographics_summary, summary_file)
cat("\nSaved cross-dataset summary:", summary_file, "\n")

# ============================================================================
# Pooled statistics for the EXISTING (previously published) datasets
# ============================================================================
# For the methods write-up we report the existing data (Roris_smiley + Roris_nostd)
# as a single pooled sample with citations to the original publications, and the
# new data (MPI) as a separate, currently NA-only paragraph.
#
# Smiley provides per-subject day-level ages for the 63 infants with demographic
# records; Roris_nostd provides only group-level summaries (mean 8.02 months,
# SD 11.37 days). We therefore use the standard pooled-variance formula to
# combine the two groups in days.

DAYS_PER_MONTH <- 365.25 / 12                          # = 30.4375 days/month

# --- Per-group summaries in *days* ---
smiley_n     <- nrow(matched)                          # 63 with demographics
smiley_mean  <- safe_mean(matched$infant_age_days)
smiley_sd    <- safe_sd  (matched$infant_age_days)

nostd_n      <- 73L
nostd_mean   <- 8.02 * DAYS_PER_MONTH                  # convert paper's mean
nostd_sd     <- 11.37                                  # paper, days

pool_n     <- smiley_n + nostd_n
pool_mean  <- (smiley_n * smiley_mean + nostd_n * nostd_mean) / pool_n
pool_var   <- ((smiley_n - 1) * smiley_sd^2 +
               (nostd_n  - 1) * nostd_sd^2  +
                smiley_n      * (smiley_mean - pool_mean)^2 +
                nostd_n       * (nostd_mean  - pool_mean)^2) /
              (pool_n - 1)
pool_sd    <- sqrt(pool_var)

existing_initial <- sum(c(roris_smiley_row$initial_n, roris_nostd_row$initial_n),
                        na.rm = TRUE)
existing_final   <- sum(c(roris_smiley_row$final_n,   roris_nostd_row$final_n),
                        na.rm = TRUE)
existing_excl    <- existing_initial - existing_final
existing_demo_n  <- sum(c(roris_smiley_row$n_with_demographics,
                          roris_nostd_row$n_with_demographics), na.rm = TRUE)
existing_female  <- sum(c(roris_smiley_row$n_female, roris_nostd_row$n_female),
                        na.rm = TRUE)
existing_male    <- sum(c(roris_smiley_row$n_male,   roris_nostd_row$n_male),
                        na.rm = TRUE)

cat("\n=== Pooled existing datasets (Roris_smiley + Roris_nostd) ===\n")
cat(sprintf("Initial N recruited     : %d\n", existing_initial))
cat(sprintf("Final N (post-exclusion): %d (excluded: %d)\n",
            existing_final, existing_excl))
cat(sprintf("N with demographics     : %d / %d\n", existing_demo_n, existing_final))
cat(sprintf("Female / male           : %d / %d\n", existing_female, existing_male))
cat(sprintf("Mean age (days)         : %.1f (SD = %.1f)\n", pool_mean, pool_sd))
cat(sprintf("Mean age (months)       : %.2f (SD = %.2f)\n",
            pool_mean / DAYS_PER_MONTH, pool_sd / DAYS_PER_MONTH))

# Write a single-row summary for the existing pool as well, so it is easy to
# cite from the manuscript.
existing_pool_summary <- tibble(
    group                = "existing_pooled",
    datasets             = "Roris_smiley + Roris_nostd",
    citations            = "Poli et al. (2020); Poli et al. (2024); Scatolin et al. (2025)",
    initial_n            = as.integer(existing_initial),
    final_n              = as.integer(existing_final),
    n_excluded           = as.integer(existing_excl),
    n_with_demographics  = as.integer(existing_demo_n),
    n_female             = as.integer(existing_female),
    n_male               = as.integer(existing_male),
    age_mean_days        = pool_mean,
    age_sd_days          = pool_sd,
    age_mean_months      = pool_mean / DAYS_PER_MONTH,
    age_sd_months        = pool_sd   / DAYS_PER_MONTH
)

new_pool_summary <- tibble(
    group                = "new",
    datasets             = "MPI",
    citations            = NA_character_,
    initial_n            = NA_integer_,
    final_n              = mpi_row$final_n,
    n_excluded           = NA_integer_,
    n_with_demographics  = NA_integer_,
    n_female             = NA_integer_,
    n_male               = NA_integer_,
    age_mean_days        = NA_real_,
    age_sd_days          = NA_real_,
    age_mean_months      = NA_real_,
    age_sd_months        = NA_real_
)

pool_summary <- bind_rows(existing_pool_summary, new_pool_summary)
print(pool_summary, width = Inf)

pool_file <- file.path(OUTPUT_DIR, "infant_pool_summary.csv")
write_csv(pool_summary, pool_file)
cat("\nSaved pool summary:", pool_file, "\n")

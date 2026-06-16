# make_infant_release.R
# =====================
# Builds the SHAREABLE infant-foraging dataset from the private raw files.
#
# It performs the same harmonization as the analysis preprocessing, then:
#   - drops every identifier (raw subj / participant ids, demographics) and every
#     measure the analysis does not use (looking time, saccadic latency, etc.);
#   - replaces participant IDs with an anonymized integer `subj_id`. The id <-> raw
#     mapping is written to infant_id_key.csv, which is PRIVATE and must NEVER be
#     released (it is gitignored).
#
# Outputs:
#   Data/Infant_Data/infant_foraging_release.csv   <- shareable
#       columns: dataset, subj_id, nseq, ntrialseq, ntrialsubj, event, D, I, H
#   Data/Infant_Data/infant_id_key.csv             <- PRIVATE (do not release)
#
# The raw per-participant CSVs stay with us; downstream analysis
# (fit_infant_mvt.R) reads only the release file, so the public pipeline is
# reproducible from the shareable data alone.
suppressPackageStartupMessages({ library(dplyr) })

DATA_DIR <- "analysis/Cognitive_analysis/Data/Infant_Data"
RELEASE  <- file.path(DATA_DIR, "infant_foraging_release.csv")
KEY      <- file.path(DATA_DIR, "infant_id_key.csv")   # PRIVATE — do not release

# --- harmonize raw files (mirrors fit_infant_mvt.R Step 1; H/entropy added) ----
# Roris NoSTD (headerless): V2=subj, V3=nseq, V5=ntrialseq, V7=H, V8=I, V9=D, V10=event
d1 <- read.csv(file.path(DATA_DIR, "Roris_nostd_final.csv"), header = FALSE) %>%
    transmute(dataset = "Roris_nostd", subj = as.character(V2),
              subj_global = paste0(dataset, "_", subj),
              nseq = as.integer(V3), ntrialseq = as.integer(V5),
              event = V10, D = V9, I = V8, H = V7)

d2 <- read.csv(file.path(DATA_DIR, "Roris_smiley_final.csv")) %>%
    transmute(dataset = "Roris_smiley", subj = as.character(subj),
              subj_global = paste0(dataset, "_", subj),
              nseq = as.integer(nseq), ntrialseq = as.integer(ntrialseq),
              event = as.numeric(event), D = D, I = I, H = H)

d3 <- read.csv(file.path(DATA_DIR, "data_MPI.csv")) %>%
    transmute(dataset = "MPI", subj = as.character(participant),
              subj_global = paste0(dataset, "_", subj),
              nseq = as.integer(Sequence), ntrialseq = as.integer(Trial),
              event = LookAway, D = KL, I = Surprise, H = Entropy)

# data4R is currently excluded from the analysis pool; kept here (commented) for parity.
# d4 <- read.csv(file.path(DATA_DIR, "data4R.csv")) %>%
#     transmute(dataset = "data4R", subj = paste0(subject_id, "_age", subject_age),
#               subj_global = paste0(dataset, "_", subj),
#               nseq = as.integer(seq), ntrialseq = as.integer(seq_item),
#               event = last_item, D = D, I = I, H = H)

dat <- bind_rows(d1, d2, d3) %>%
    filter(!is.na(event), !is.na(D)) %>%
    arrange(subj_global, nseq, ntrialseq) %>%
    group_by(subj_global) %>%
    mutate(ntrialsubj = row_number()) %>%
    ungroup()

# --- anonymize: subj_global -> integer subj_id (deterministic; key kept private)
key <- dat %>%
    distinct(subj_global, dataset) %>%
    arrange(subj_global) %>%
    mutate(subj_id = row_number())
dat <- dat %>% left_join(key %>% select(subj_global, subj_id), by = "subj_global")

# --- shareable subset: only the analysis variables, no identifiers --------------
release <- dat %>%
    select(dataset, subj_id, nseq, ntrialseq, ntrialsubj, event, D, I, H) %>%
    arrange(subj_id, nseq, ntrialseq)

write.csv(release, RELEASE, row.names = FALSE)
write.csv(key,     KEY,     row.names = FALSE)   # PRIVATE
cat(sprintf("Release: %d trials, %d subjects -> %s\n",
            nrow(release), dplyr::n_distinct(release$subj_id), RELEASE))
cat(sprintf("PRIVATE id key -> %s  (do NOT release / is gitignored)\n", KEY))

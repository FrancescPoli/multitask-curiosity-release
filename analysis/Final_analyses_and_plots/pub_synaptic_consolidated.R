# =============================================================================
# pub_synaptic_consolidated.R
# -----------------------------------------------------------------------------
# ONE script consolidating every synaptic-density figure / statistic previously
# spread across:
#   Synaptic_analysis/Data/synapse_data.R          -> PART A  (human synapse density)
#   plotting/.../pub_huttenlocher_synapse_density.R -> PART A  (publication figure)
#   plotting/.../pub_population_trajectories.R       -> PART B  (RNN + human trajectories)
#   plotting/.../pub_age_landscape.R                 -> PART C  (fitted-age landscape)
#   Forest_analysis/chisquare_analysis.R            -> PART C  (chi-square age test)
#   plotting/.../pub_age_performance.R               -> PART D  (accuracy ~ fitted age)
#
# *** Alignment with pub_topology_consolidated.R ***
# Every "human-likeness" selection (the top-K most brain-like networks used by the
# fitted-age chi-square and the accuracy GAM) is done in the SAME Mahalanobis
# fingerprint over METRIC_COLS as pub_topology_consolidated.R — NOT the legacy
# norm_dist_synaptic / norm_dist_network columns. Change METRIC_COLS and the
# selection follows automatically.
#
# The fitted age itself (age_cap) is a synaptic-trajectory quantity, independent
# of the topology metrics. It is PREPROCESSING: derived upstream by
# analysis/Synaptic_analysis/analyze_synaptic_space.R and read here from
# grand_unified_metrics_v2.csv.
# =============================================================================

# -- Resolve project root so relative paths work from any working directory ----
local({
    sp <- NA_character_
    a <- commandArgs(trailingOnly = FALSE)
    m <- grep("^--file=", a)
    if (length(m)) {
        sp <- normalizePath(sub("^--file=", "", a[m[1]]), mustWork = FALSE)
    } else {
        for (i in seq_len(sys.nframe())) {
            of <- sys.frame(i)$ofile
            if (!is.null(of)) { sp <- normalizePath(of, mustWork = FALSE); break }
        }
    }
    if (!is.na(sp)) {
        d <- dirname(sp)
        while (!dir.exists(file.path(d, "analysis")) && dirname(d) != d) d <- dirname(d)
        if (dir.exists(file.path(d, "analysis"))) setwd(d)
    }
})

suppressPackageStartupMessages({
    library(dplyr); library(ggplot2); library(mgcv); library(scales)
    library(readr); library(tidyr)
})
options(scipen = 999)

# =============================================================================
# ####  TWEAKABLES  ###########################################################
# =============================================================================

# --- THE THREE METRICS that define topological human-likeness (must match
#     pub_topology_consolidated.R; the top-K selection is built from these) -----
METRIC_COLS <- c("modularity_leiden", "efficiency", "rich_club")

# --- which figure groups to render -------------------------------------------
RUN <- c(human_density   = TRUE,    # PART A  (Huttenlocher synapse density)
         trajectories    = TRUE,    # PART B  (RNN + human density trajectories)
         age_landscape   = TRUE,    # PART C  (fitted-age distribution + chi-square)
         age_performance = TRUE)    # PART D  (accuracy ~ fitted age GAM)

# --- analysis settings -------------------------------------------------------
MIN_SOLVED <- 0.30      # keep networks with fraction_solved >= this (matches topology)
TOP_K      <- 85       # top-K most human-like (smallest topological Mahalanobis)
N_BINS     <- 15        # bins for the fitted-age histogram (PART C)
N_TICKS    <- 3         # x-axis tick count (PART C)
DPI        <- 400

# --- output ------------------------------------------------------------------
OUT_DIR    <- "analysis/Final_analyses_and_plots/Figures"
CACHE_DIR  <- "analysis/Final_analyses_and_plots/cache"
OUT_SUFFIX <- ""        # set e.g. "_leiden_sw" when you change METRIC_COLS

# --- inputs ------------------------------------------------------------------
DATA        <- "analysis/grand_unified_metrics_v2.csv"                       # age_cap, accuracy, metrics
HCP_CSV     <- "analysis/Network_analysis/Results/human_topological_metrics.csv"
POP_CSV     <- "analysis/Synaptic_analysis/Data/population_weights.csv"      # weight trajectories (PART B/E)
SYN_METRICS <- "analysis/Synaptic_analysis/Results/synaptic_metrics.csv"     # fitted-age per run (PART B/E)

# --- palettes (consistent with pub_topology_consolidated.R) ------------------
POP_BLUE        <- "#A8C0DA"; TOP_BLUE <- "darkblue"   # population / top-K
HUMAN_GREEN     <- "#28b74e"; HUMAN_DARKGREEN <- "#044328"; WINNER_GOLD <- "#F7C548"
region_greens   <- c(auditory = "#a1d99b", visual = "#41ab5d", prefrontal = "#00441b")

# =============================================================================
# ####  END TWEAKABLES  #######################################################
# =============================================================================

if (!dir.exists(OUT_DIR))   dir.create(OUT_DIR, recursive = TRUE)
if (!dir.exists(CACHE_DIR)) dir.create(CACHE_DIR, recursive = TRUE)
outpng <- function(stem) file.path(OUT_DIR, paste0(stem, OUT_SUFFIX, ".png"))
stopifnot(length(METRIC_COLS) == 3)
rescale01 <- function(d) (max(d, na.rm = TRUE) - d) /
                         (max(d, na.rm = TRUE) - min(d, na.rm = TRUE))
birth_age <- 280 / 365                      # conception-to-birth offset (years)

cat(sprintf(">>> METRIC_COLS = %s\n\n", paste(METRIC_COLS, collapse = ", ")))


# =============================================================================
# SHARED — human synaptic reference (Huttenlocher 1982 / 1997)
# =============================================================================
# Used by PART A (figure + GAM stat) and PART B (human curve).
human_data_all <- local({
    h97 <- data.frame(
        region = rep(c("auditory", "prefrontal"), times = 14),
        CA_days = rep(c(192, 210, 280, 320, 360, 363, 700,
                        1620, 4700, 5000, 5700, 7300, 11700, 21500), each = 2),
        synapses_100um3 = c(12.2, 3.2, 7.5, 2.2, 29.4, 19.5, 21.0, 11.2,
                            40.7, 28.3, 54.3, 34.3, 53.0, 37.9, 55.7, 52.4,
                            24.7, 46.9, 26.0, 39.7, 38.9, 40.0, 24.0, 27.1,
                            35.0, 40.2, 34.7, 28.7),
        synapses_se = c(1.2, 0.5, 0.6, 0.5, 2.7, 2.2, 2.4, 1.1,
                        5.4, 0.9, 3.0, 4.7, 5.4, 4.9, 5.4, 6.0,
                        3.9, 3.1, 4.3, 6.1, 3.5, 3.9, 2.1, 2.5,
                        2.8, 3.7, 2.7, 2.0))
    h97$age_years <- h97$CA_days / 365
    v82 <- data.frame(
        region = "visual",
        age_years = c(0.53, 0.77, 0.94, 0.97, 1.1, 1.44, 1.69,
                      2.35, 3.77, 11.77, 26.77, 71.77),
        synapses_100um3 = c(12, 25.5, 26.5, 31.5, 51, 57.5, 56.5, 49, 44.5, 35, 35, 33),
        synapses_se = c(1.5, 1.8, 1.9, 2, 3.1, 2.7, 2.7, 1.8, 1.5, 1.7, 2, 1.8))
    out <- rbind(h97[, c("region", "age_years", "synapses_100um3", "synapses_se")],
                 v82[, c("region", "age_years", "synapses_100um3", "synapses_se")])
    out$norm_val <- (out$synapses_100um3 - min(out$synapses_100um3)) /
                    (max(out$synapses_100um3) - min(out$synapses_100um3))
    out$log_age  <- log10(out$age_years)
    out$w        <- 1 / (out$synapses_se^2)
    out
})

human_fit_data <- subset(human_data_all, age_years >= 0.4)
gam_human_full <- gam(norm_val ~ s(log_age, k = 5), data = human_fit_data, weights = w)
ref_grid_log   <- seq(log10(0.4), log10(70), length.out = 100)
ref_grid_age   <- 10^ref_grid_log
human_pred_ref <- as.numeric(predict(gam_human_full, newdata = data.frame(log_age = ref_grid_log)))
human_curve    <- data.frame(run_id = "Human", age = ref_grid_age, value = human_pred_ref)


# =============================================================================
# SHARED SETUP — Mahalanobis topological similarity on the sweep
# =============================================================================
# Identical fingerprint to pub_topology_consolidated.R: HCP-YA individual mean +
# covariance over METRIC_COLS; small Mahalanobis distance = most human-like.
need_sel <- any(unlist(RUN[c("trajectories", "age_landscape", "age_performance")]))
if (need_sel) {
    df_hcp <- read.csv(HCP_CSV)
    miss <- setdiff(METRIC_COLS, colnames(df_hcp))
    if (length(miss)) stop("METRIC_COLS missing from HCP CSV: ", paste(miss, collapse = ", "))
    HCP_IND <- df_hcp[!grepl("Consensus", df_hcp$full_id) & df_hcp$dataset == "HCPya", ]
    MU      <- sapply(METRIC_COLS, function(m) mean(HCP_IND[[m]], na.rm = TRUE))
    SIGMA   <- cov(as.matrix(HCP_IND[, METRIC_COLS]))
    SINV    <- solve(SIGMA)

    DF <- read.csv(DATA)
    miss <- setdiff(METRIC_COLS, colnames(DF))
    if (length(miss)) stop("METRIC_COLS missing from DATA: ", paste(miss, collapse = ", "))
    DF <- DF[!is.na(DF$mean_accuracy) & DF$fraction_solved >= MIN_SOLVED, ]
    DF <- DF[complete.cases(DF[, METRIC_COLS]), ]
    .dl <- sweep(as.matrix(DF[, METRIC_COLS]), 2, MU, "-")
    DF$mahal_dist <- sqrt(rowSums((.dl %*% SINV) * .dl))
    DF$similarity <- rescale01(DF$mahal_dist)
    cat(sprintf("Mahalanobis pool: N = %d (fraction_solved >= %.2f)\n\n", nrow(DF), MIN_SOLVED))
}


# =============================================================================
# PART A — HUMAN SYNAPSE DENSITY  (Huttenlocher; was pub_huttenlocher_*.R)
# =============================================================================
if (isTRUE(RUN["human_density"])) local({
    cat("=== PART A: human synapse density ===\n")
    gs <- summary(gam_human_full)
    cat(sprintf("Human reference GAM (norm_val ~ s(log10 age), weighted): edf=%.2f, p=%.2e, dev.expl=%.1f%%\n",
                gs$s.table[1, "edf"], gs$s.table[1, "p-value"], 100 * gs$dev.expl))

    y_label <- expression(atop("Synaptic density", "(synapses / 100 " * mu * "m"^3 * ")"))
    p <- ggplot(human_data_all, aes(x = age_years, y = synapses_100um3, colour = region)) +
        geom_point(size = 1.6) +
        geom_errorbar(aes(ymin = synapses_100um3 - synapses_se,
                          ymax = synapses_100um3 + synapses_se),
                      width = 0.05, alpha = 0.6, linewidth = 0.4) +
        geom_smooth(aes(weight = 1 / synapses_se^2),
                    method = "gam", formula = y ~ s(x, k = 7), se = FALSE, linewidth = 0.7) +
        scale_colour_manual(values = region_greens) +
        scale_x_log10(breaks = c(0.5, 1, 2, 5, 10, 20, 50, 80), labels = label_number(accuracy = 1)) +
        geom_vline(xintercept = birth_age, linetype = "dashed", colour = "black", linewidth = 0.4) +
        labs(x = "Age from conception (years)", y = y_label, colour = "Region") +
        theme_classic(base_size = 9) +
        theme(legend.position = "right", legend.key.size = unit(0.4, "cm"),
              legend.title = element_text(size = 8), legend.text = element_text(size = 7),
              axis.title.y = element_text(margin = margin(r = 4)),
              plot.background = element_rect(fill = "white", colour = NA),
              panel.background = element_rect(fill = "white", colour = NA))
    ggsave(outpng("pub_huttenlocher_synapse_density"), p, width = 3.3, height = 2.0, dpi = DPI)
    cat("Saved:", basename(outpng("pub_huttenlocher_synapse_density")), "\n\n")
})


# =============================================================================
# PART B — RNN + HUMAN DENSITY TRAJECTORIES  (was pub_population_trajectories.R)
# =============================================================================
# Highlighted "example model" is now the MOST human-like network by topological
# Mahalanobis distance (consistent with the rest of the analysis), not the
# legacy synaptic-distance winner.
if (isTRUE(RUN["trajectories"])) local({
    cat("=== PART B: density trajectories ===\n")
    if (!file.exists(SYN_METRICS)) {
        cat("  SKIP: missing", SYN_METRICS,
            "(run analyze_synaptic_space.R preprocessing first)\n\n"); return(invisible())
    }
    final_results <- read.csv(SYN_METRICS)
    curves_rds <- file.path(CACHE_DIR, paste0("all_curves", OUT_SUFFIX, ".rds"))

    if (file.exists(curves_rds)) {
        all_curves <- readRDS(curves_rds)
        cat("  loaded cached curves:", basename(curves_rds), "\n")
    } else if (file.exists(POP_CSV)) {
        cat("  building per-model trajectories from", basename(POP_CSV), "...\n")
        pop_data <- read.csv(POP_CSV); run_ids <- unique(pop_data$run_id)
        curves_list <- list()
        for (rid in run_ids) {
            row <- final_results[final_results$run_id == rid, ]
            if (nrow(row) == 0) next
            age_cap <- row$age_cap[1]; age_span <- age_cap - birth_age
            mr <- subset(pop_data, run_id == rid); mr <- mr[!is.na(mr$l1_sum), ]
            if (nrow(mr) < 5) next
            rng <- range(mr$l1_sum); if (diff(rng) == 0) next
            mr$norm_val  <- (mr$l1_sum - rng[1]) / diff(rng)
            mr$equiv_age <- birth_age + (mr$step / max(mr$step)) * age_span
            mr$log_age   <- log10(mr$equiv_age)
            gm <- tryCatch(gam(norm_val ~ s(log_age, k = 5), data = mr), error = function(e) NULL)
            if (is.null(gm)) next
            grid_log <- seq(log10(0.4), log10(age_cap), length.out = 100)
            curves_list[[rid]] <- data.frame(run_id = rid, age = 10^grid_log,
                value = as.numeric(predict(gm, newdata = data.frame(log_age = grid_log))))
        }
        all_curves <- bind_rows(curves_list)
        saveRDS(all_curves, curves_rds); cat("  cached:", basename(curves_rds), "\n")
    } else {
        cat("  SKIP: no cached curves and missing", POP_CSV, "\n\n"); return(invisible())
    }

    # example = most human-like (smallest topological Mahalanobis) run with a curve
    md <- DF[, c("run_id", "mahal_dist")]
    cand <- merge(data.frame(run_id = unique(all_curves$run_id)), md, by = "run_id")
    winner_id <- if (nrow(cand)) cand$run_id[which.min(cand$mahal_dist)] else
                 final_results$run_id[which.min(final_results$distance)]
    cat("  example (min Mahalanobis) run:", winner_id, "\n")

    p <- ggplot() +
        geom_line(data = subset(all_curves, run_id != winner_id),
                  aes(x = age, y = value, group = run_id, color = "All models"),
                  alpha = 0.03, size = 0.5) +
        geom_line(data = subset(all_curves, run_id == winner_id),
                  aes(x = age, y = value, color = "Example model"), alpha = 0.9, size = 1.5) +
        geom_line(data = human_curve, aes(x = age, y = value, color = "Human"),
                  alpha = 1.0, size = 1.5) +
        scale_color_manual(name = NULL,
            values = c("Human" = HUMAN_GREEN, "Example model" = TOP_BLUE, "All models" = POP_BLUE),
            breaks = c("Human", "Example model", "All models"),
            guide = guide_legend(override.aes = list(alpha = c(1, 1, 0.5), size = c(1.5, 1.5, 1)))) +
        scale_x_log10(breaks = c(0.5, 1, 5, 10, 20, 30, 50, 70), labels = label_number(accuracy = 1)) +
        labs(x = "Age (Years) [Log Scale]", y = "Normalized Density/Weights") +
        ylim(-0.8, 1.5) + theme_classic() + theme(legend.position = "right")
    ggsave(outpng("pub_population_trajectories"), p, width = 4.5, height = 2.5, dpi = DPI)
    cat("Saved:", basename(outpng("pub_population_trajectories")), "\n\n")
})


# =============================================================================
# PART C — FITTED-AGE LANDSCAPE + CHI-SQUARE  (was pub_age_landscape.R +
#          chisquare_analysis.R; top-K now from topological Mahalanobis)
# =============================================================================
if (isTRUE(RUN["age_landscape"])) local({
    cat("=== PART C: fitted-age landscape ===\n")
    dv <- DF[!is.na(DF$age_cap) & !is.na(DF$mahal_dist), ]
    dv <- dv[order(dv$mahal_dist), ]
    K_eff <- min(TOP_K, nrow(dv))
    top_k <- dv[1:K_eff, ]; rest <- dv[-(1:K_eff), ]

    # chi-square on raw age_cap levels (matches chisquare_analysis.R)
    all_ages <- sort(unique(dv$age_cap))
    obs  <- table(factor(top_k$age_cap, levels = all_ages))
    expf <- as.numeric(table(factor(dv$age_cap, levels = all_ages)) / nrow(dv))
    chi <- tryCatch(chisq.test(obs, p = expf, simulate.p.value = TRUE, B = 10000),
                    error = function(e) list(statistic = NA, p.value = NA))
    cat(sprintf("Pool %d | top-%d vs rest %d\n", nrow(dv), K_eff, nrow(rest)))
    cat(sprintf("fitted-age chi-square (top-%d vs pool): chi2 = %.1f, p = %.4g\n",
                K_eff, as.numeric(chi$statistic), chi$p.value))
    cat(sprintf("median fitted age: top-%d = %g y, rest = %g y\n",
                K_eff, median(top_k$age_cap), median(rest$age_cap)))

    # ---- split histogram (Population vs Top-K), counts, linear x -------------
    age_min <- min(dv$age_cap); age_max <- max(dv$age_cap)
    edges <- seq(age_min, age_max, length.out = N_BINS + 1)
    mids  <- (edges[-1] + edges[-(N_BINS + 1)]) / 2; widths <- diff(edges)
    bin_count <- function(x) as.numeric(table(factor(
        cut(x, breaks = edges, include.lowest = TRUE, right = TRUE, labels = FALSE),
        levels = seq_len(N_BINS))))
    wlab <- paste0("Top ", K_eff, " models")
    bd <- data.frame(AgeMid = mids, Width = widths,
                     Population = bin_count(dv$age_cap), Top = bin_count(top_k$age_cap))
    bd <- pivot_longer(bd, c("Population", "Top"), names_to = "Cohort", values_to = "Frequency")
    bd$Cohort <- factor(ifelse(bd$Cohort == "Population", "Population", wlab),
                        levels = c("Population", wlab))
    fill_pal <- setNames(c(POP_BLUE, TOP_BLUE), c("Population", wlab))
    line_pal <- setNames(c("#3568A0", TOP_BLUE), c("Population", wlab))
    p <- ggplot(bd, aes(AgeMid, Frequency, fill = Cohort, colour = Cohort, width = Width)) +
        geom_col(position = "identity", alpha = 0.7, colour = NA) +
        geom_smooth(method = "loess", se = FALSE, span = 0.5, linewidth = 1.0) +
        facet_wrap(~ Cohort, ncol = 1, scales = "free_y") +
        scale_fill_manual(values = fill_pal, guide = "none") +
        scale_colour_manual(values = line_pal, guide = "none") +
        scale_x_continuous(breaks = pretty(c(age_min, age_max), n = N_TICKS),
                           expand = expansion(mult = c(0, 0))) +
        scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
        labs(x = "Inferred age\n(years)", y = "Frequency") +
        theme_classic(base_size = 9) +
        theme(strip.background = element_blank(), strip.text = element_blank(),
              plot.background = element_rect(fill = "white", colour = NA),
              panel.background = element_rect(fill = "white", colour = NA))
    ggsave(outpng("pub_age_landscape_split"), p, width = 1.3, height = 2.86, dpi = DPI)
    cat("Saved:", basename(outpng("pub_age_landscape_split")), "\n\n")
})


# =============================================================================
# PART D — ACCURACY ~ FITTED AGE GAM  (was pub_age_performance.R; top-K from
#          topological Mahalanobis)
# =============================================================================
if (isTRUE(RUN["age_performance"])) local({
    cat("=== PART D: accuracy ~ fitted age ===\n")
    dv <- DF[!is.na(DF$age_cap) & !is.na(DF$mean_accuracy) & !is.na(DF$mahal_dist), ]
    dv <- dv[order(dv$mahal_dist), ]
    K_eff <- min(TOP_K, nrow(dv)); top_k <- dv[1:K_eff, ]

    fit <- gam(mean_accuracy ~ s(age_cap), data = top_k, method = "REML")
    gs <- summary(fit)
    cat(sprintf("GAM accuracy ~ s(age_cap) on top-%d: edf=%.2f, F=%.2f, p=%.4g, dev.expl=%.1f%%\n",
                K_eff, gs$s.table[1, "edf"], gs$s.table[1, "F"],
                gs$s.table[1, "p-value"], 100 * gs$dev.expl))

    ag <- data.frame(age_cap = seq(min(top_k$age_cap), max(top_k$age_cap), length.out = 200))
    pr <- predict(fit, newdata = ag, se.fit = TRUE)
    ag$fit <- pr$fit; ag$lower <- pr$fit - 1.96 * pr$se.fit; ag$upper <- pr$fit + 1.96 * pr$se.fit
    cat(sprintf("  predicted accuracy peaks at age_cap = %.1f y\n", ag$age_cap[which.max(ag$fit)]))
    p <- ggplot() +
        geom_point(data = top_k, aes(x = age_cap, y = mean_accuracy),
                   alpha = 0.1, size = 1.2, colour = TOP_BLUE) +
        geom_ribbon(data = ag, aes(x = age_cap, ymin = lower, ymax = upper),
                    fill = TOP_BLUE, alpha = 0.25) +
        geom_line(data = ag, aes(x = age_cap, y = fit), colour = TOP_BLUE, linewidth = 1.0) +
        labs(x = "Inferred Age (years)", y = "Mean accuracy") +
        theme_classic(base_size = 9) +
        theme(plot.background = element_rect(fill = "white", colour = NA),
              panel.background = element_rect(fill = "white", colour = NA))
    ggsave(outpng("pub_age_performance_network"), p, width = 2.0, height = 2.1, dpi = DPI)
    cat("Saved:", basename(outpng("pub_age_performance_network")), "\n\n")
})


# -----------------------------------------------------------------------------
# NOTE: the fitted age (age_cap) is PREPROCESSING, not analysis, and is NOT
# computed here. It is derived once from population_weights.csv by
# analysis/Synaptic_analysis/analyze_synaptic_space.R (startup_demo.md, Part 8,
# Step 5), which writes Synaptic_analysis/Results/synaptic_metrics.csv; that
# age_cap is then merged into grand_unified_metrics_v2.csv by
# aggregate_all_metrics.py and read above. The synaptic-space (peak / pruning /
# correlation) 3D characterization also lives in that preprocessing script.
# -----------------------------------------------------------------------------

cat(">>> Done.\n")

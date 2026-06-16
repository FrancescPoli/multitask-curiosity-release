# =============================================================================
# pub_topology_consolidated.R
# -----------------------------------------------------------------------------
# ONE script consolidating every topology figure previously spread across:
#   pub_structural_comparison_3d.R          -> PART A  (3D structural scatter)
#   pub_network_similarity.R                -> PART B  (Mahalanobis similarity)
#   pub_network_similarity_permetric.R      -> PART C  (per-metric similarity)
#   pub_network_similarity_distance_strongreg.R -> PART D (Distance+strong-reg EMMs)
#
# *** The three topological metrics used everywhere are selected ONCE, below ***
# Change `METRIC_COLS` to any three columns present in the topology CSVs, e.g.
#   c("modularity", "efficiency", "rich_club")              (original)
#   c("modularity_leiden", "efficiency", "small_worldness") (upgraded metrics)
#   c("modularity", "participation", "small_worldness")     (etc.)
# Every reference (HCP average, Mahalanobis covariance, per-metric similarity,
# 3D axes, facet rows) follows automatically.
#
# Available metric columns (after re-running the upgraded extraction):
#   modularity, modularity_louvain, modularity_leiden, efficiency,
#   rich_club, small_worldness, participation
# =============================================================================

# ── Resolve project root so relative paths work from any working directory ────
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
    library(dplyr); library(ggplot2); library(emmeans); library(car)
    library(readr); library(plot3D)
})
emm_options(msg.interaction = FALSE)
options(scipen = 999)

# =============================================================================
# ████  TWEAKABLES  ███████████████████████████████████████████████████████████
# =============================================================================

# --- THE THREE METRICS (the main knob) ---------------------------------------
METRIC_COLS <- c("modularity_leiden", "efficiency", "rich_club")

# --- which figure groups to render -------------------------------------------
RUN <- c(structural_3d = TRUE,   # PART A
         similarity     = TRUE,   # PART B
         permetric      = TRUE,   # PART C
         strongreg      = TRUE)   # PART D

# --- analysis settings -------------------------------------------------------
MIN_SOLVED <- 0.30      # keep networks with fraction_solved >= this
TEMP_CUT   <- 0.005     # curiosity (< cut) vs random (>= cut); labels derive from this
REG_STRONG <- 5e-4      # "strongest" regularisation strength (PART D)
TOP_K      <- 100       # top-K most human-like (PART A highlight, PART B enrichment)
DPI        <- 400

# --- output ------------------------------------------------------------------
OUT_DIR    <- "analysis/Final_analyses_and_plots/Figures"
# Suffix appended to every output filename. Leave "" to reproduce the canonical
# figures; set e.g. "_leiden_sw" when you change METRIC_COLS so you don't
# overwrite the originals.
OUT_SUFFIX <- ""

# --- inputs ------------------------------------------------------------------
DATA     <- "analysis/grand_unified_metrics_v2.csv"                       # PART B/C/D
HCP_CSV  <- "analysis/Network_analysis/Results/human_topological_metrics.csv"
RNN_CSV  <- "analysis/Network_analysis/Results/rnn_topological_metrics.csv"  # PART A

# --- palettes ----------------------------------------------------------------
line_palette <- c(curiosity = "#1abc9c", random = "#bcbcbc")   # B/C (teal = winners temp hue)
grp_pal      <- c(low = "#1abc9c", high = "#bcbcbc")            # D
POP_BLUE <- "#A8C0DA"; TOP_BLUE <- "darkblue"                   # A
POP_ALPHA <- 0.16; TOP_ALPHA <- 0.9

# =============================================================================
# ████  END TWEAKABLES  ███████████████████████████████████████████████████████
# =============================================================================

if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)
outpng <- function(stem) file.path(OUT_DIR, paste0(stem, OUT_SUFFIX, ".png"))

stopifnot(length(METRIC_COLS) == 3)

# --- metric metadata (label / 3D axis label / short tag); fallback = col name -
.metric_meta <- list(
    modularity         = c(label = "Modularity",            axis = "Segregation (Modularity)",            short = "mod"),
    modularity_louvain = c(label = "Modularity (Louvain)",  axis = "Segregation (Modularity, Louvain)",   short = "modL"),
    modularity_leiden  = c(label = "Modularity",   axis = "Segregation (Modularity)",    short = "modLe"),
    efficiency         = c(label = "Global\nEfficiency",            axis = "Integration (Efficiency)",            short = "eff"),
    rich_club          = c(label = "Rich Club",             axis = "Hierarchy (Rich Club)",               short = "rc"),
    small_worldness    = c(label = "Small-worldness (ω)", axis = "Small-worldness (ω)",         short = "sw"),
    participation      = c(label = "Participation",         axis = "Integration (Participation)",         short = "part")
)
.mget <- function(col, field) {
    if (!is.null(.metric_meta[[col]])) unname(.metric_meta[[col]][field]) else col
}
mlab   <- function(col) .mget(col, "label")
maxis  <- function(col) .mget(col, "axis")
mshort <- function(col) .mget(col, "short")
METRIC_LABS <- setNames(vapply(METRIC_COLS, mlab, ""), METRIC_COLS)

# --- temperature labels derived from TEMP_CUT (keeps text consistent) --------
.tc <- format(TEMP_CUT, scientific = FALSE, trim = TRUE)
TG_LABELS_INLINE <- c(curiosity = sprintf("τ < %s (curiosity)", .tc),
                      random     = sprintf("τ ≥ %s (random)", .tc))
TG_LABELS_2LINE  <- c(low  = sprintf("τ < %s\n(curiosity)", .tc),
                      high = sprintf("τ ≥ %s\n(random)", .tc))

# --- helper: validate that the chosen metrics exist in a data frame ----------
require_metrics <- function(data, where) {
    miss <- setdiff(METRIC_COLS, colnames(data))
    if (length(miss))
        stop(sprintf("METRIC_COLS not found in %s: %s\n  available numeric cols: %s",
                     where, paste(miss, collapse = ", "),
                     paste(intersect(names(.metric_meta), colnames(data)), collapse = ", ")),
             call. = FALSE)
}
# --- helper: linear rescale of a distance vector to [0,1] similarity ---------
rescale01 <- function(dst) (max(dst, na.rm = TRUE) - dst) /
                           (max(dst, na.rm = TRUE) - min(dst, na.rm = TRUE))

cat(sprintf(">>> METRIC_COLS = %s\n\n", paste(METRIC_COLS, collapse = ", ")))


# =============================================================================
# PART A — 3D STRUCTURAL SCATTER  (was pub_structural_comparison_3d.R)
# =============================================================================
if (isTRUE(RUN["structural_3d"])) local({
    cat("=== PART A: 3D structural scatter ===\n")
    hcp <- read_csv(HCP_CSV, show_col_types = FALSE)
    rnn <- read_csv(RNN_CSV, show_col_types = FALSE)
    require_metrics(hcp, basename(HCP_CSV)); require_metrics(rnn, basename(RNN_CSV))

    hcp <- hcp %>% mutate(
        type  = ifelse(grepl("Consensus", full_id, ignore.case = TRUE), "Consensus", "Individual"),
        group = "Human")
    rnn <- rnn %>% mutate(type = "RNN", group = "Model", full_id = run_id)

    hcp_ind <- hcp %>% filter(type == "Individual")
    hcp_con <- hcp %>% filter(type == "Consensus")
    # outlier removal: humans, 4 SD on any selected metric
    if (nrow(hcp_ind) > 0) {
        flag <- Reduce(`|`, lapply(METRIC_COLS, function(m) {
            mu <- mean(hcp_ind[[m]], na.rm = TRUE); s <- sd(hcp_ind[[m]], na.rm = TRUE)
            abs(hcp_ind[[m]] - mu) > 4 * s
        }))
        hcp_ind <- hcp_ind[!flag, , drop = FALSE]
    }

    keep <- c("full_id", "group", "type", METRIC_COLS)
    df <- bind_rows(hcp_ind, hcp_con, rnn) %>% select(all_of(keep))

    # top-K RNNs by Euclidean distance to the human-individual centroid
    centre <- sapply(METRIC_COLS, function(m) mean(df[[m]][df$type == "Individual"], na.rm = TRUE))
    rnn_all <- df %>% filter(type == "RNN")
    rnn_all$dist <- sqrt(rowSums(sweep(as.matrix(rnn_all[, METRIC_COLS]), 2,
                                       centre[METRIC_COLS], "-")^2))
    rnn_all <- rnn_all %>% arrange(dist)
    K_eff   <- min(TOP_K, nrow(rnn_all))
    rnn_top <- rnn_all %>% slice_head(n = K_eff)
    rnn_pop <- rnn_all %>% slice_tail(n = nrow(rnn_all) - K_eff)
    cat(sprintf("Population: %d  |  Top-%d\n", nrow(rnn_all), K_eff))

    h_ind <- df %>% filter(type == "Individual")
    h_con <- df %>% filter(type == "Consensus")

    render_panel <- function(x_var, y_var, z_var) {
        xlim <- range(df[[x_var]], na.rm = TRUE)
        ylim <- range(df[[y_var]], na.rm = TRUE)
        zlim <- range(df[[z_var]], na.rm = TRUE)
        set.seed(42)
        jit <- function(v, lim) v + rnorm(length(v), sd = diff(lim) * 0.05)

        stem <- sprintf("pub_structural_comparison_3d_x-%s_y-%s_z-%s",
                        mshort(x_var), mshort(y_var), mshort(z_var))
        png(outpng(stem), width = 2400, height = 1800, res = 300)
        on.exit(dev.off(), add = TRUE)
        par(mar = c(5, 4, 4, 2) + 0.1)

        scatter3D(jit(h_ind[[x_var]], xlim), jit(h_ind[[y_var]], ylim), jit(h_ind[[z_var]], zlim),
                  colvar = NULL, col = "forestgreen", pch = 16, cex = 1.0, alpha = 0.09,
                  bty = "g", ticktype = "detailed", nticks = 3, theta = 45, phi = 0,
                  xlab = maxis(x_var), ylab = maxis(y_var),
                  zlab = paste0("\n", maxis(z_var)),   # leading \n nudges z label left, off the ticks
                  xlim = xlim, ylim = ylim, zlim = zlim)
        pmat <- getplist()$mat
        if (nrow(rnn_top) > 0) {
            pt <- trans3D(jit(rnn_top[[x_var]], xlim), jit(rnn_top[[y_var]], ylim),
                          jit(rnn_top[[z_var]], zlim), pmat)
            points(pt$x, pt$y, pch = 16, cex = 1.0,
                   col = adjustcolor(TOP_BLUE, alpha.f = TOP_ALPHA))
        }
        if (nrow(rnn_pop) > 0)
            scatter3D(jit(rnn_pop[[x_var]], xlim), jit(rnn_pop[[y_var]], ylim),
                      jit(rnn_pop[[z_var]], zlim),
                      col = POP_BLUE, pch = 16, cex = 1.0, alpha = POP_ALPHA, add = TRUE)
        if (nrow(h_con) > 0) {
            pt <- trans3D(h_con[[x_var]], h_con[[y_var]], h_con[[z_var]], pmat)
            points(pt$x, pt$y, pch = 17, cex = 1.44, col = "#044328")
        }
        par(xpd = TRUE)
        legend("bottomleft", inset = c(0.24, -0.22),
               legend = c("Human (Individual)", "Human (Average)",
                          "Model (Population)", sprintf("Top %d Models", K_eff)),
               pch = c(16, 17, 16, 16),
               col = c("forestgreen", "#044328", POP_BLUE, TOP_BLUE),
               pt.cex = c(1.0, 1.44, 1.0, 1.0), ncol = 2, bty = "n")
        par(xpd = FALSE)
        cat("Saved:", basename(outpng(stem)), "\n")
    }

    perms <- list(c(1,2,3), c(1,3,2), c(2,1,3), c(2,3,1), c(3,1,2), c(3,2,1))
    for (p in perms) render_panel(METRIC_COLS[p[1]], METRIC_COLS[p[2]], METRIC_COLS[p[3]])
    cat("\n")
})


# =============================================================================
# SHARED SETUP for PARTS B/C/D — similarities on the sweep data
# =============================================================================
need_BCD <- any(unlist(RUN[c("similarity", "permetric", "strongreg")]))
if (need_BCD) {
    df_hcp  <- read.csv(HCP_CSV)
    require_metrics(df_hcp, basename(HCP_CSV))
    HCP_IND <- df_hcp[!grepl("Consensus", df_hcp$full_id) & df_hcp$dataset == "HCPya", ]

    MU        <- sapply(METRIC_COLS, function(m) mean(HCP_IND[[m]], na.rm = TRUE))
    SIGMA     <- cov(as.matrix(HCP_IND[, METRIC_COLS]))
    SIGMA_INV <- solve(SIGMA)
    SD_K      <- sapply(METRIC_COLS, function(m) sd(HCP_IND[[m]], na.rm = TRUE))

    # inter-individual correlations among the fingerprint metrics (Mahalanobis
    # rationale: metrics are mutually correlated, so use the covariance Sigma).
    cat("HCP-YA inter-individual correlations among METRIC_COLS:\n")
    print(round(cor(HCP_IND[, METRIC_COLS], use = "complete.obs"), 3))

    DF <- read.csv(DATA)
    require_metrics(DF, basename(DATA))
    DF <- DF[!is.na(DF$mean_accuracy) & DF$fraction_solved >= MIN_SOLVED, ]
    DF <- DF[complete.cases(DF[, METRIC_COLS]) & !is.na(DF$temp), ]

    # whole: Mahalanobis distance -> [0,1] similarity
    .dl <- sweep(as.matrix(DF[, METRIC_COLS]), 2, MU, "-")
    DF$mahal_dist <- sqrt(rowSums((.dl %*% SIGMA_INV) * .dl))
    DF$similarity <- rescale01(DF$mahal_dist)
    # per-metric: univariate standardized |deviation| -> [0,1] similarity
    for (m in METRIC_COLS) DF[[paste0("sim_", m)]] <- rescale01(abs(DF[[m]] - MU[m]) / SD_K[m])

    # interpretation aid
    .hmd <- sqrt(mahalanobis(as.matrix(HCP_IND[, METRIC_COLS]), center = MU, cov = SIGMA))
    cat(sprintf("Inter-human Mahalanobis to average: mean %.2f, 95%% %.2f, max %.2f\n",
                mean(.hmd), quantile(.hmd, 0.95), max(.hmd)))

    # factor frame shared by B and C (drop reg_value = 1e-6 L1-only edge level)
    D <- DF %>%
        filter(!is.na(reg_value) & !is.na(reg_type) & !is.na(alpha) &
               !is.na(beta) & !is.na(eps) & !is.na(travel)) %>%
        mutate(temp_group = factor(ifelse(temp < TEMP_CUT, "curiosity", "random"),
                                   levels = c("curiosity", "random")),
               reg_type_f = factor(reg_type, levels = c("L1", "Distance"))) %>%
        filter(reg_value > 1e-6) %>%
        mutate(reg_value_f = factor(formatC(reg_value, format = "f",
                                            digits = 7, drop0trailing = TRUE)),
               temp_f  = factor(temp),                 # full temperature (all levels)
               alpha_f = factor(alpha), beta_f = factor(beta),
               eps_f = factor(eps), travel_f = factor(travel),
               logtemp = log10(temp), logreg = log10(reg_value))
    cat(sprintf("N = %d functional networks (fraction_solved >= %.2f)\n\n", nrow(D), MIN_SOLVED))

    # x-axis helpers derived from the data (no hard-coded temp labels)
    TEMP_LEVELS <- sort(unique(D$temp))
    TEMP_LABS   <- formatC(TEMP_LEVELS, format = "g")
    REG_LEVELS  <- sort(unique(D$reg_value))
    REG_LABS    <- formatC(REG_LEVELS, format = "g")

    YLAB <- "Predicted similarity to\nhuman average (a.u.)"
    base_theme <- theme_classic(base_size = 9) +
        theme(strip.background = element_blank(),
              strip.text  = element_text(size = 9, face = "bold"),
              axis.title.y = element_text(lineheight = 0.95),
              legend.position = "top",
              legend.title = element_text(size = 8),
              legend.text  = element_text(size = 7),
              plot.title = element_text(size = 10, face = "bold"),
              plot.subtitle = element_text(size = 7.5, colour = "grey35"),
              plot.background  = element_rect(fill = "white", colour = NA),
              panel.background = element_rect(fill = "white", colour = NA))
}


# =============================================================================
# PART B — MAHALANOBIS SIMILARITY  (was pub_network_similarity.R)
# =============================================================================
if (isTRUE(RUN["similarity"])) local({
    d <- D
    # ---- 1. CORE REGRESSION ------------------------------------------------
    cat("=== PART B.1: CORE REGRESSION ===\n")

    # ---- FULL-TEMPERATURE MODEL (authoritative stats reported in Methods) ---
    # tau as a full categorical factor (all levels), NOT a curiosity/random split.
    Mft <- lm(similarity ~ temp_f * reg_value_f * reg_type_f, data = d)
    fsf <- summary(Mft)$fstatistic
    cat(sprintf("[full-temp] Overall: F(%d,%d) = %.1f, R2 = %.4f (adj %.4f)\n",
                fsf[2], fsf[3], fsf[1], summary(Mft)$r.squared, summary(Mft)$adj.r.squared))
    aft <- car::Anova(Mft, type = 2)
    for (trm in c("temp_f", "reg_value_f", "reg_type_f",
                  "temp_f:reg_value_f", "temp_f:reg_type_f", "reg_value_f:reg_type_f",
                  "temp_f:reg_value_f:reg_type_f"))
        cat(sprintf("   %-32s F=%8.2f  p=%.2e\n", trm, aft[trm, "F value"], aft[trm, "Pr(>F)"]))
    # within the optimal regime (Distance + strongest reg): temp profile + Tukey
    strong  <- levels(d$reg_value_f)[which.max(as.numeric(levels(d$reg_value_f)))]
    emm_opt <- emmeans(Mft, ~ temp_f, at = list(reg_value_f = strong, reg_type_f = "Distance"))
    eo      <- as.data.frame(emm_opt)
    cat(sprintf("[full-temp] optimal regime = Distance + reg=%s; peak tau=%s (s=%.3f)\n",
                strong, eo$temp_f[which.max(eo$emmean)], max(eo$emmean)))
    print(eo[, c("temp_f", "emmean", "lower.CL", "upper.CL")], row.names = FALSE)
    cat("[full-temp] Tukey contrasts within optimal regime:\n")
    print(as.data.frame(pairs(emm_opt))[, c("contrast", "estimate", "p.value")], row.names = FALSE)
    # contrast: temperature profile under L1 at the same (strongest) strength
    eo_l1 <- as.data.frame(emmeans(Mft, ~ temp_f, at = list(reg_value_f = strong, reg_type_f = "L1")))
    cat(sprintf("[full-temp] under L1 + reg=%s, similarity across temperatures spans s = %.2f - %.2f\n",
                strong, min(eo_l1$emmean), max(eo_l1$emmean)))

    # ---- ROBUSTNESS: 3-way kept; does each of the 4 MVT hyperparams modulate it? --
    cat("\n[full-temp] robustness (3-way temp:reg_value:reg_type vs each hyperparam):\n")
    for (hp in c("alpha_f", "beta_f", "travel_f", "eps_f")) {
        Mh <- lm(as.formula(sprintf("similarity ~ temp_f*reg_value_f*reg_type_f*%s", hp)), data = d)
        Ah <- car::Anova(Mh, type = 2)
        r3 <- "temp_f:reg_value_f:reg_type_f"
        r4 <- paste0("temp_f:reg_value_f:reg_type_f:", hp)
        cat(sprintf("   x %-8s  3-way F=%6.2f p=%.2e | 4-way F=%5.2f p=%.2e\n",
                    hp, Ah[r3, "F value"], Ah[r3, "Pr(>F)"], Ah[r4, "F value"], Ah[r4, "Pr(>F)"]))
    }
    # optimum-unchanged check: best (reg_type,reg_value,temp) cell within each of
    # the 64 hyperparameter combinations (does the optimum stay distance + strong?)
    combo <- d %>% group_by(alpha, beta, travel, eps) %>%
        group_modify(~{
            cm <- .x %>% group_by(reg_type, reg_value, temp) %>%
                summarise(s = mean(similarity), .groups = "drop")
            cm[which.max(cm$s), c("reg_type", "reg_value", "temp")]
        }) %>% ungroup()
    cat(sprintf("[full-temp] best cell across %d hyperparam combos: Distance %d/%d, strongest reg %d/%d\n",
                nrow(combo), sum(combo$reg_type == "Distance"), nrow(combo),
                sum(abs(combo$reg_value - REG_STRONG) < 1e-12), nrow(combo)))
    cat("   best-cell tau distribution:\n"); print(table(combo$temp))
    cat("\n")

    # ---- binarized model (kept only for the two-line predicted figure below) -
    M0 <- lm(similarity ~ temp_group * reg_value_f * reg_type_f, data = d)
    fs <- summary(M0)$fstatistic
    cat(sprintf("Overall: F(%d,%d) = %.1f, R2 = %.4f (adj %.4f)\n",
                fs[2], fs[3], fs[1], summary(M0)$r.squared, summary(M0)$adj.r.squared))
    a0 <- car::Anova(M0, type = 2)
    cat(sprintf("3-way temp:reg_value:reg_type  F=%.1f, p=%.2e\n",
                a0["temp_group:reg_value_f:reg_type_f", "F value"],
                a0["temp_group:reg_value_f:reg_type_f", "Pr(>F)"]))
    emm <- emmeans(M0, ~ temp_group * reg_value_f * reg_type_f)
    eb  <- as.data.frame(emm); eb <- eb[which.max(eb$emmean), ]
    cat(sprintf("Best cell: %s / reg=%s / %s  (s = %.3f)\n",
                eb$temp_group, eb$reg_value_f, eb$reg_type_f, eb$emmean))

    ep <- as.data.frame(emm)
    p1 <- ggplot(ep, aes(reg_value_f, emmean, colour = temp_group, group = temp_group)) +
        geom_line(linewidth = 0.7) +
        geom_pointrange(aes(ymin = lower.CL, ymax = upper.CL),
                        size = 0.32, fatten = 2.0, linewidth = 0.6) +
        facet_wrap(~ reg_type_f) +
        scale_colour_manual(values = line_palette, labels = TG_LABELS_INLINE, name = "Temperature") +
        labs(x = "Regularisation strength", y = YLAB) +
        base_theme + theme(axis.text.x = element_text(size = 8, angle = 30, hjust = 1))
    ggsave(outpng("pub_network_similarity_predicted"), p1, width = 4.6, height = 2.3, dpi = DPI)
    cat("Saved:", basename(outpng("pub_network_similarity_predicted")), "\n\n")

    # ---- 1a. PREDICTED, single temps τ = 0.003 vs τ = 1.0 (not the split) ---
    d2 <- d %>% filter(temp %in% c(0.003, 1.0)) %>%
        mutate(temp2 = factor(ifelse(temp == 0.003, "curiosity", "random"),
                              levels = c("curiosity", "random")))
    M0b <- lm(similarity ~ temp2 * reg_value_f * reg_type_f, data = d2)
    ep2 <- as.data.frame(emmeans(M0b, ~ temp2 * reg_value_f * reg_type_f))
    lab2 <- c(curiosity = "τ = 0.003 (curiosity)", random = "τ = 1.0 (random)")
    p1a <- ggplot(ep2, aes(reg_value_f, emmean, colour = temp2, group = temp2)) +
        geom_line(linewidth = 0.7) +
        geom_pointrange(aes(ymin = lower.CL, ymax = upper.CL),
                        size = 0.32, fatten = 2.0, linewidth = 0.6) +
        facet_wrap(~ reg_type_f) +
        scale_colour_manual(values = line_palette, labels = lab2, name = "Temperature") +
        labs(x = "Regularisation strength", y = YLAB) +
        base_theme + theme(axis.text.x = element_text(size = 8, angle = 30, hjust = 1))
    ggsave(outpng("pub_network_similarity_predicted_003v1"), p1a, width = 4.6, height = 2.3, dpi = DPI)
    cat("Saved:", basename(outpng("pub_network_similarity_predicted_003v1")), "\n\n")

    # ---- 1b. CONTINUOUS TEMPERATURE (quadratic in log10 temp) --------------
    cat("=== PART B.1b: CONTINUOUS TEMPERATURE ===\n")
    dc <- d %>% mutate(reg_grp = factor(reg_value))
    Mc <- lm(similarity ~ (logtemp + I(logtemp^2)) * logreg * reg_type_f, data = dc)
    ac <- car::Anova(Mc, type = 2)
    cat(sprintf("Quadratic I(logtemp^2) F=%.1f p=%.2e | logtemp:logreg F=%.1f p=%.2e | R2=%.4f\n",
                ac["I(logtemp^2)", "F value"], ac["I(logtemp^2)", "Pr(>F)"],
                ac["logtemp:logreg", "F value"], ac["logtemp:logreg", "Pr(>F)"],
                summary(Mc)$r.squared))
    gc <- expand.grid(logtemp = seq(min(dc$logtemp), max(dc$logtemp), length = 120),
                      reg_value = REG_LEVELS, reg_type_f = levels(dc$reg_type_f))
    gc$logreg <- log10(gc$reg_value)
    pr <- predict(Mc, gc, se.fit = TRUE)
    gc$pred <- pr$fit; gc$lo <- pr$fit - 1.96 * pr$se.fit; gc$hi <- pr$fit + 1.96 * pr$se.fit
    gc$temp <- 10^gc$logtemp; gc$reg_grp <- factor(gc$reg_value)
    p1b <- ggplot(gc, aes(temp, pred, colour = reg_value, fill = reg_value, group = reg_grp)) +
        geom_jitter(data = dc, aes(y = similarity), width = 0.04, height = 0,
                    size = 0.5, alpha = 0.18, show.legend = FALSE) +
        geom_ribbon(aes(ymin = lo, ymax = hi), colour = NA, alpha = 0.15, show.legend = FALSE) +
        geom_line(linewidth = 0.8) + facet_wrap(~ reg_type_f) +
        scale_x_log10(breaks = TEMP_LEVELS, labels = TEMP_LABS) +
        scale_colour_viridis_c(trans = "log10", option = "C", end = 0.92,
                               name = "Regularisation\nstrength", breaks = REG_LEVELS, labels = REG_LABS) +
        scale_fill_viridis_c(trans = "log10", option = "C", end = 0.92, guide = "none") +
        labs(x = "Temperature (τ)", y = YLAB) +
        base_theme + theme(legend.position = "right", legend.key.height = unit(0.9, "lines"),
                           axis.text.x = element_text(size = 7, angle = 30, hjust = 1))
    ggsave(outpng("pub_network_similarity_temp_continuous"), p1b, width = 5.4, height = 2.4, dpi = DPI)
    cat("Saved:", basename(outpng("pub_network_similarity_temp_continuous")), "\n\n")

    # ---- 1c. CONTINUOUS REGULARISATION (mirror) ----------------------------
    cat("=== PART B.1c: CONTINUOUS REGULARISATION ===\n")
    gr <- expand.grid(logreg = seq(min(dc$logreg), max(dc$logreg), length = 120),
                      temp = TEMP_LEVELS, reg_type_f = levels(dc$reg_type_f))
    gr$logtemp <- log10(gr$temp)
    prr <- predict(Mc, gr, se.fit = TRUE)
    gr$pred <- prr$fit; gr$lo <- prr$fit - 1.96 * prr$se.fit; gr$hi <- prr$fit + 1.96 * prr$se.fit
    gr$reg_value <- 10^gr$logreg; gr$temp_grp <- factor(gr$temp)
    p1c <- ggplot(gr, aes(reg_value, pred, colour = temp, fill = temp, group = temp_grp)) +
        geom_jitter(data = dc, aes(y = similarity, group = temp), width = 0.04, height = 0,
                    size = 0.5, alpha = 0.18, show.legend = FALSE) +
        geom_ribbon(aes(ymin = lo, ymax = hi), colour = NA, alpha = 0.12, show.legend = FALSE) +
        geom_line(linewidth = 0.8) + facet_wrap(~ reg_type_f) +
        scale_x_log10(breaks = REG_LEVELS, labels = REG_LABS) +
        scale_colour_viridis_c(trans = "log10", option = "D", end = 0.92, name = "Temperature\n(τ)",
                               breaks = TEMP_LEVELS, labels = TEMP_LABS) +
        scale_fill_viridis_c(trans = "log10", option = "D", end = 0.92, guide = "none") +
        labs(x = "Regularisation strength", y = YLAB) +
        base_theme + theme(legend.position = "right", legend.key.height = unit(0.8, "lines"),
                           axis.text.x = element_text(size = 7, angle = 30, hjust = 1))
    ggsave(outpng("pub_network_similarity_reg_continuous"), p1c, width = 5.4, height = 2.4, dpi = DPI)
    cat("Saved:", basename(outpng("pub_network_similarity_reg_continuous")), "\n\n")

    # ---- 2. ROBUSTNESS + 4-way plots ---------------------------------------
    cat("=== PART B.2: ROBUSTNESS ===\n")
    core3 <- "temp_group:reg_value_f:reg_type_f"
    for (P in c("travel_f", "eps_f", "alpha_f", "beta_f")) {
        M <- lm(as.formula(paste0("similarity ~ temp_group*reg_value_f*reg_type_f*", P)), data = d)
        A <- car::Anova(M, type = 2); hi <- paste0(core3, ":", P)
        cat(sprintf("core3 x %-8s | 3-way F=%.1f p=%.1e | 4+-way %s\n", P,
            A[core3, "F value"], A[core3, "Pr(>F)"],
            if (hi %in% rownames(A)) sprintf("F=%.2f p=%.3g", A[hi, "F value"], A[hi, "Pr(>F)"]) else "n/a"))
    }
    make_4way <- function(Pvar, Plab, lev_lab, stem) {
        M <- lm(as.formula(paste0("similarity ~ temp_group*reg_value_f*reg_type_f*", Pvar)), data = d)
        e <- as.data.frame(emmeans(M, as.formula(paste0("~ temp_group*reg_value_f*reg_type_f*", Pvar))))
        e$Plev <- factor(lev_lab[as.character(e[[Pvar]])], levels = unname(lev_lab))
        p <- ggplot(e, aes(reg_value_f, emmean, colour = temp_group, group = temp_group)) +
            geom_line(linewidth = 0.7) +
            geom_pointrange(aes(ymin = lower.CL, ymax = upper.CL), size = 0.32, fatten = 2.0, linewidth = 0.6) +
            facet_grid(Plev ~ reg_type_f) +
            scale_colour_manual(values = line_palette, labels = TG_LABELS_INLINE, name = "Temperature") +
            labs(x = "Regularisation strength", y = YLAB, title = Plab) +
            base_theme + theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
                               strip.text = element_text(size = 8.5, face = "bold"),
                               axis.text.x = element_text(size = 7, angle = 30, hjust = 1))
        ggsave(outpng(stem), p, width = 4.0, height = 3.4, dpi = DPI)
        cat("Saved:", basename(outpng(stem)), "\n")
    }
    make_4way("travel_f", "Travel cost", c("50" = "n_travel = 50", "500" = "n_travel = 500"),
              "pub_network_similarity_4way_travel")
    make_4way("eps_f", "Leave margin", c("-0.01" = "ε = -0.01", "0" = "ε = 0"),
              "pub_network_similarity_4way_eps")
    cat("\n")

    # ---- 3. TOP-K ENRICHMENT (chi-square, stacked) -------------------------
    cat("=== PART B.3: TOP-K ENRICHMENT ===\n")
    PARAM_ORDER <- c("reg_type", "reg_value", "temp")
    param_hues  <- c(reg_type = "#9b59b6", reg_value = "#e67e22", temp = "#1abc9c")
    param_labels<- c(reg_type = "Regularization\ntype", reg_value = "Regularization\nvalue",
                     temp = "Temperature\n(τ)")
    # horizontal (un-rotated) in-plot labels; everything else is rotated 90°.
    # temp: keep the small/curiosity temps horizontal, rotate 0.005/0.01/0.1/1.0
    NO_ROT <- list(reg_type = c("L1", "Distance"), reg_value = REG_STRONG,
                   temp = c(1e-4, 1e-3, 3e-3))
    dn <- DF[order(DF$mahal_dist), ]; top_k <- dn[1:min(TOP_K, nrow(dn)), ]; n_all <- nrow(DF)
    cat(sprintf("Top-%d by Mahalanobis distance (pool %d)\n", nrow(top_k), n_all))
    rows <- list()
    for (param in PARAM_ORDER) {
        lv <- sort(unique(DF[[param]]))
        obs <- table(factor(top_k[[param]], levels = lv))
        exf <- table(factor(DF[[param]], levels = lv)) / n_all
        chi <- tryCatch(chisq.test(obs, p = as.numeric(exf), simulate.p.value = TRUE, B = 10000),
                        error = function(e) list(p.value = 1))
        sig <- chi$p.value < 0.05
        cat(sprintf("  %s: chi2 = %s, p = %.4g\n", param,
                    ifelse(is.null(chi$statistic), "NA", sprintf("%.1f", chi$statistic)),
                    chi$p.value))
        for (l in lv) {
            fa <- sum(DF[[param]] == l) / n_all; ft <- sum(top_k[[param]] == l) / nrow(top_k)
            rows[[length(rows) + 1]] <- data.frame(parameter = param, level = as.character(l),
                enrichment = ifelse(fa > 0, ft / fa, 0),
                bright = sig & (ifelse(fa > 0, ft / fa, 0) > 1), stringsAsFactors = FALSE)
        }
    }
    pdf_ <- do.call(rbind, rows) %>%
        mutate(num = suppressWarnings(as.numeric(level)), parameter_f = factor(parameter, levels = PARAM_ORDER)) %>%
        arrange(parameter_f, num, level) %>% group_by(parameter) %>%
        mutate(ne = enrichment / sum(enrichment, na.rm = TRUE), vf = pmax(ne, 0.05), vf = vf / sum(vf)) %>%
        ungroup() %>%
        mutate(bar_id = factor(paste0(parameter, "=", level), levels = unique(paste0(parameter, "=", level))),
               block = ifelse(bright, paste0(level, "\n(", round(enrichment, 1), "x)"), level))
    fillv <- setNames(ifelse(pdf_$bright, param_hues[pdf_$parameter],
                      paste0(param_hues[pdf_$parameter], "59")), pdf_$bar_id)
    is_norot <- function(pp, ls) {
        sset <- NO_ROT[[pp]]; if (is.null(sset)) return(FALSE)
        if (is.character(sset)) return(ls %in% sset)
        n <- suppressWarnings(as.numeric(ls)); if (is.na(n)) return(FALSE)
        any(abs(n - sset) < 1e-12)
    }
    pdf_ <- pdf_ %>% mutate(parameter_f = factor(parameter, levels = rev(PARAM_ORDER)),
                            ang = ifelse(mapply(is_norot, parameter, level), 0, 90),
                            txt_size = ifelse(ang == 0, 3.3, 3.0))  # rotated labels keep the smaller size
    bl <- levels(pdf_$bar_id)
    pdf_$bar_id <- factor(as.character(pdf_$bar_id), levels = c(
        rev(bl[grepl("^reg_type=", bl)]), bl[grepl("^reg_value=", bl)], rev(bl[grepl("^temp=", bl)])))
    p3 <- ggplot(pdf_, aes(y = parameter_f, x = vf, fill = bar_id)) +
        geom_col(position = "stack", colour = "white", linewidth = 0.4, show.legend = FALSE) +
        geom_text(aes(label = block, angle = ang, size = txt_size), position = position_stack(vjust = 0.5),
                  colour = "black", lineheight = 0.95) +
        scale_size_identity() +
        scale_fill_manual(values = fillv) + scale_y_discrete(labels = param_labels) +
        scale_x_continuous(expand = expansion(mult = c(0, 0.02))) +
        labs(x = "Relative Frequency", y = NULL) +
        theme_minimal(base_size = 12.1) +
        theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
              panel.grid.major.x = element_line(colour = "grey90", linewidth = 0.3),
              axis.ticks = element_blank(), axis.title.x = element_text(size = 12.1, margin = margin(t = 6)),
              plot.background = element_rect(fill = "white", colour = NA),
              panel.background = element_rect(fill = "white", colour = NA))
    ggsave(outpng("pub_network_similarity_winners"), p3, width = 5.2, height = 2.4, dpi = DPI)
    cat("Saved:", basename(outpng("pub_network_similarity_winners")), "\n\n")
})


# =============================================================================
# PART C — PER-METRIC SIMILARITY  (was pub_network_similarity_permetric.R)
# =============================================================================
if (isTRUE(RUN["permetric"])) local({
    d <- D
    cat("=== PART C: PER-METRIC SIMILARITY ===\n")
    metric_levels <- unname(METRIC_LABS)

    # core regression per metric + combined predicted figure
    ep_list <- list()
    for (m in METRIC_COLS) {
        sub <- d; sub$similarity <- d[[paste0("sim_", m)]]
        M <- lm(similarity ~ temp_group * reg_value_f * reg_type_f, data = sub)
        a <- car::Anova(M, type = 2)
        eb <- as.data.frame(emmeans(M, ~ temp_group * reg_value_f * reg_type_f))
        best <- eb[which.max(eb$emmean), ]
        cat(sprintf("%-12s | R2=%.3f | 3-way F=%.1f p=%.2e | best: %s/%s/%s\n",
                    METRIC_LABS[m], summary(M)$r.squared,
                    a["temp_group:reg_value_f:reg_type_f", "F value"],
                    a["temp_group:reg_value_f:reg_type_f", "Pr(>F)"],
                    best$temp_group, best$reg_value_f, best$reg_type_f))
        eb$metric <- METRIC_LABS[m]; ep_list[[m]] <- eb
    }
    ep <- do.call(rbind, ep_list); ep$metric_f <- factor(ep$metric, levels = metric_levels)
    p1 <- ggplot(ep, aes(reg_value_f, emmean, colour = temp_group, group = temp_group)) +
        geom_line(linewidth = 0.7) +
        geom_pointrange(aes(ymin = lower.CL, ymax = upper.CL), size = 0.30, fatten = 1.9, linewidth = 0.55) +
        facet_grid(metric_f ~ reg_type_f) +
        scale_colour_manual(values = line_palette, labels = TG_LABELS_INLINE, name = "Temperature") +
        labs(x = "Regularisation strength", y = YLAB) +
        base_theme + theme(strip.text = element_text(size = 8.5, face = "bold"),
                           axis.text.x = element_text(size = 7, angle = 30, hjust = 1))
    ggsave(outpng("pub_network_similarity_permetric_predicted"), p1, width = 4.8, height = 5.0, dpi = DPI)
    cat("Saved:", basename(outpng("pub_network_similarity_permetric_predicted")), "\n")

    # continuous temperature, per metric
    gc_list <- list()
    for (m in METRIC_COLS) {
        sub <- d; sub$similarity <- d[[paste0("sim_", m)]]
        Mc <- lm(similarity ~ (logtemp + I(logtemp^2)) * logreg * reg_type_f, data = sub)
        g <- expand.grid(logtemp = seq(min(d$logtemp), max(d$logtemp), length = 120),
                         reg_value = REG_LEVELS, reg_type_f = levels(d$reg_type_f))
        g$logreg <- log10(g$reg_value); g$pred <- predict(Mc, g)
        g$temp <- 10^g$logtemp; g$metric <- METRIC_LABS[m]; gc_list[[m]] <- g
    }
    gc <- do.call(rbind, gc_list); gc$metric_f <- factor(gc$metric, levels = metric_levels)
    gc$reg_grp <- factor(gc$reg_value)
    p1b <- ggplot(gc, aes(temp, pred, colour = reg_value, group = reg_grp)) +
        geom_line(linewidth = 0.8) + facet_grid(metric_f ~ reg_type_f) +
        scale_x_log10(breaks = TEMP_LEVELS, labels = TEMP_LABS) +
        scale_colour_viridis_c(trans = "log10", option = "C", end = 0.92,
                               name = "Regularisation\nstrength", breaks = REG_LEVELS, labels = REG_LABS) +
        labs(x = "Temperature (τ)", y = YLAB) +
        base_theme + theme(legend.position = "right", strip.text = element_text(size = 8.5, face = "bold"),
                           legend.key.height = unit(0.9, "lines"),
                           axis.text.x = element_text(size = 7, angle = 30, hjust = 1))
    ggsave(outpng("pub_network_similarity_permetric_temp_continuous"), p1b, width = 5.6, height = 5.0, dpi = DPI)
    cat("Saved:", basename(outpng("pub_network_similarity_permetric_temp_continuous")), "\n")

    # continuous regularisation, per metric
    gr_list <- list()
    for (m in METRIC_COLS) {
        sub <- d; sub$similarity <- d[[paste0("sim_", m)]]
        Mc <- lm(similarity ~ (logtemp + I(logtemp^2)) * logreg * reg_type_f, data = sub)
        g <- expand.grid(logreg = seq(min(d$logreg), max(d$logreg), length = 120),
                         temp = TEMP_LEVELS, reg_type_f = levels(d$reg_type_f))
        g$logtemp <- log10(g$temp); g$pred <- predict(Mc, g)
        g$reg_value <- 10^g$logreg; g$metric <- METRIC_LABS[m]; gr_list[[m]] <- g
    }
    gr <- do.call(rbind, gr_list); gr$metric_f <- factor(gr$metric, levels = metric_levels)
    gr$temp_grp <- factor(gr$temp)
    p1c <- ggplot(gr, aes(reg_value, pred, colour = temp, group = temp_grp)) +
        geom_line(linewidth = 0.8) + facet_grid(metric_f ~ reg_type_f) +
        scale_x_log10(breaks = REG_LEVELS, labels = REG_LABS) +
        scale_colour_viridis_c(trans = "log10", option = "D", end = 0.92, name = "Temperature\n(τ)",
                               breaks = TEMP_LEVELS, labels = TEMP_LABS) +
        labs(x = "Regularisation strength", y = YLAB) +
        base_theme + theme(legend.position = "right", strip.text = element_text(size = 8.5, face = "bold"),
                           legend.key.height = unit(0.8, "lines"),
                           axis.text.x = element_text(size = 7, angle = 30, hjust = 1))
    ggsave(outpng("pub_network_similarity_permetric_reg_continuous"), p1c, width = 5.6, height = 5.0, dpi = DPI)
    cat("Saved:", basename(outpng("pub_network_similarity_permetric_reg_continuous")), "\n\n")
})


# =============================================================================
# PART D — DISTANCE + STRONG-REG EMMs  (was ..._distance_strongreg.R)
# =============================================================================
if (isTRUE(RUN["strongreg"])) local({
    cat("=== PART D: DISTANCE + STRONG-REG ===\n")
    s <- DF %>%
        filter(reg_type == "Distance", abs(reg_value - REG_STRONG) < 1e-12, !is.na(reg_value)) %>%
        mutate(tgrp = factor(ifelse(temp < TEMP_CUT, "low", "high"), levels = c("low", "high")),
               temp_f = factor(temp))
    cat(sprintf("Distance + reg=%.0e: N=%d (low τ: %d, high τ: %d)\n",
                REG_STRONG, nrow(s), sum(s$tgrp == "low"), sum(s$tgrp == "high")))
    metric_levels <- unname(METRIC_LABS)
    SUBTITLE <- sprintf("Distance regularisation · strongest strength (%s)", formatC(REG_STRONG, format = "g"))

    em_split <- function(simcol) {
        sub <- s; sub$y <- s[[simcol]]
        as.data.frame(emmeans(lm(y ~ tgrp, data = sub), ~ tgrp))
    }
    em_alltemp <- function(simcol) {
        sub <- s; sub$y <- s[[simcol]]
        e <- as.data.frame(emmeans(lm(y ~ temp_f, data = sub), ~ temp_f))
        e$temp <- as.numeric(as.character(e$temp_f))
        e$tgrp <- factor(ifelse(e$temp < TEMP_CUT, "low", "high"), levels = c("low", "high")); e
    }
    bind_metrics <- function(fun) {
        out <- do.call(rbind, lapply(METRIC_COLS, function(m) {
            e <- fun(paste0("sim_", m)); e$metric <- METRIC_LABS[m]; e }))
        out$metric_f <- factor(out$metric, levels = metric_levels); out
    }
    # console contrasts
    cat("Low vs High τ contrasts:\n")
    .report <- function(lbl, simcol) {
        sub <- s; sub$y <- s[[simcol]]
        ct <- as.data.frame(pairs(emmeans(lm(y ~ tgrp, data = sub), ~ tgrp)))
        cat(sprintf("  %-12s Δ(low-high)=%+.3f, t=%.2f, p=%.2e\n", lbl, ct$estimate, ct$t.ratio, ct$p.value))
    }
    .report("whole", "similarity")
    for (m in METRIC_COLS) .report(METRIC_LABS[m], paste0("sim_", m))

    # whole, split
    e <- as.data.frame(emmeans(lm(similarity ~ tgrp, data = s), ~ tgrp))
    p_ws <- ggplot(e, aes(tgrp, emmean, colour = tgrp)) +
        geom_line(aes(group = 1), colour = "grey75", linewidth = 0.5) +
        geom_pointrange(aes(ymin = lower.CL, ymax = upper.CL), size = 0.5, fatten = 2.6, linewidth = 0.8, show.legend = FALSE) +
        scale_colour_manual(values = grp_pal) + scale_x_discrete(labels = TG_LABELS_2LINE) +
        labs(x = NULL, y = YLAB, title = "Temperature contrast (whole)", subtitle = SUBTITLE) +
        base_theme + theme(axis.text.x = element_text(size = 8.5, lineheight = 0.9))
    ggsave(outpng("pub_network_similarity_distance_strongreg_whole_split"), p_ws, width = 3.1, height = 3.2, dpi = DPI)

    # whole, all temps
    e <- em_alltemp("similarity")
    p_wa <- ggplot(e, aes(temp_f, emmean)) +
        geom_line(aes(group = 1), colour = "grey75", linewidth = 0.5) +
        geom_pointrange(aes(ymin = lower.CL, ymax = upper.CL, colour = tgrp), size = 0.42, fatten = 2.3, linewidth = 0.7) +
        scale_colour_manual(values = grp_pal, labels = TG_LABELS_INLINE, name = "Temperature regime") +
        scale_x_discrete(labels = TEMP_LABS) +
        labs(x = "Temperature (τ)", y = YLAB, title = "Temperature curve (whole)", subtitle = SUBTITLE) +
        base_theme + theme(axis.text.x = element_text(size = 7.5, angle = 30, hjust = 1))
    ggsave(outpng("pub_network_similarity_distance_strongreg_whole_alltemp"), p_wa, width = 3.8, height = 3.2, dpi = DPI)

    # per-metric, split
    e <- bind_metrics(em_split)
    p_ps <- ggplot(e, aes(tgrp, emmean, colour = tgrp)) +
        geom_line(aes(group = 1), colour = "grey75", linewidth = 0.5) +
        geom_pointrange(aes(ymin = lower.CL, ymax = upper.CL), size = 0.42, fatten = 2.3, linewidth = 0.7, show.legend = FALSE) +
        facet_wrap(~ metric_f) +
        coord_cartesian(ylim = c(0.63, 0.92)) +
        scale_colour_manual(values = grp_pal) + scale_x_discrete(labels = TG_LABELS_2LINE) +
        labs(x = NULL, y = YLAB, title = "Temperature contrast per metric", subtitle = SUBTITLE) +
        base_theme + theme(axis.text.x = element_text(size = 7, lineheight = 0.9))
    ggsave(outpng("pub_network_similarity_distance_strongreg_permetric_split"), p_ps, width = 6.4, height = 2.8, dpi = DPI)

    # per-metric, all temps
    e <- bind_metrics(em_alltemp)
    p_pa <- ggplot(e, aes(temp_f, emmean)) +
        geom_line(aes(group = 1), colour = "grey75", linewidth = 0.5) +
        geom_pointrange(aes(ymin = lower.CL, ymax = upper.CL, colour = tgrp), size = 0.32, fatten = 2.0, linewidth = 0.6) +
        facet_wrap(~ metric_f, scales = "free_y") +
        scale_colour_manual(values = grp_pal, labels = TG_LABELS_INLINE, name = "Temperature regime") +
        scale_x_discrete(labels = TEMP_LABS) +
        labs(x = "Temperature (τ)", y = YLAB, title = "Temperature curve per metric", subtitle = SUBTITLE) +
        base_theme + theme(axis.text.x = element_text(size = 6.5, angle = 30, hjust = 1))
    ggsave(outpng("pub_network_similarity_distance_strongreg_permetric_alltemp"), p_pa, width = 6.6, height = 2.9, dpi = DPI)

    # per-metric, τ = 0.003 vs τ = 1.0, restricted to n_travel = 50
    s2 <- s %>% filter(travel == 50, 
        temp %in% c(0.003, 1.0)) %>%
        mutate(t2 = factor(ifelse(temp == 0.003, "c003", "r1"), levels = c("c003", "r1")))
    pal2 <- c(c003 = unname(grp_pal["low"]), r1 = unname(grp_pal["high"]))
    lab2 <- c(c003 = "curiosity\ndriven", r1 = "random")
    cat(sprintf("τ=0.003 vs τ=1.0 @ travel=50:  N=%d (0.003: %d, 1.0: %d)\n",
                nrow(s2), sum(s2$t2 == "c003"), sum(s2$t2 == "r1")))
    # contrasts behind the 003v1_trav50 figure (Δ = sim[τ=0.003] - sim[τ=1.0])
    cat("τ=0.003 vs τ=1.0 @ travel=50 per-metric contrasts (Δ = 0.003 - 1.0):\n")
    .report2 <- function(lbl, simcol) {
        sub <- s2; sub$y <- s2[[simcol]]
        ct <- as.data.frame(pairs(emmeans(lm(y ~ t2, data = sub), ~ t2)))
        cat(sprintf("  %-12s Δ=%+.3f, t(%d)=%.2f, p=%.2e\n",
                    lbl, ct$estimate, df.residual(lm(y ~ t2, data = sub)), ct$t.ratio, ct$p.value))
    }
    .report2("whole", "similarity")
    for (m in METRIC_COLS) .report2(METRIC_LABS[m], paste0("sim_", m))
    em_split2 <- function(simcol) {
        sub <- s2; sub$y <- s2[[simcol]]
        as.data.frame(emmeans(lm(y ~ t2, data = sub), ~ t2))
    }
    e <- do.call(rbind, lapply(METRIC_COLS, function(m) {
        d2 <- em_split2(paste0("sim_", m)); d2$metric <- METRIC_LABS[m]; d2 }))
    order2 <- METRIC_COLS[c(1, 3, 2)]   # this figure only: swap 2nd & 3rd metric (efficiency <-> rich-club)
    e$metric_f <- factor(e$metric, levels = METRIC_LABS[order2])
    p_p2 <- ggplot(e, aes(t2, emmean, colour = t2)) +
        geom_line(aes(group = 1), colour = "grey75", linewidth = 0.5) +
        geom_pointrange(aes(ymin = lower.CL, ymax = upper.CL), size = 0.42, fatten = 2.3, linewidth = 0.7, show.legend = FALSE) +
        facet_wrap(~ metric_f, labeller = as_labeller(function(x) sub(" \\((Leiden|Louvain)\\)", "", x))) +
        scale_colour_manual(values = pal2) + scale_x_discrete(labels = lab2) +
        labs(x = "Exploration", y = "Similarity to human average") +
        base_theme + theme(axis.text.x = element_text(size = 7, lineheight = 0.9),
                           strip.text = element_text(size = 9, face = "plain"))  # match axis-title size, no bold
    ggsave(outpng("pub_network_similarity_distance_strongreg_permetric_003v1_trav50"), p_p2, width = 3.1, height = 2.1, dpi = DPI)

    cat("Saved 5 distance_strongreg figures\n\n")
})

cat(">>> Done.\n")

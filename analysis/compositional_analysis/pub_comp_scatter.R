suppressPackageStartupMessages({
    library(dplyr)
    library(tidyr)
    library(ggplot2)
})

# Publication versions of
#   analysis/comparison_plots/compositional/scatter_2d_density_solved.png
#   analysis/comparison_plots/compositional/scatter_2d_win_score.png
#
# Differences vs source (analysis/compositional_analysis/analyse_compositionality.R):
#   - Cohort labels updated:
#       H-100   → "Human-like"
#       C-temp  → "Exploration Control"
#       C-reg   → "Regularization Control"
#       C-both  → "Both Controls"
#   - Human-like cohort recoloured to darkblue (matches the rest of the figure)
#   - Em-dashes (—) in axis labels replaced with short hyphens (-)
#   - No titles, no caption notes (legend already says it all)
#   - Sizes aligned with synaptic-figure plots; tweakable PLOT_W / PLOT_H

# ── Tweakables ───────────────────────────────────────────────────────────────
PLOT_W_DENS     <- 4.3     # density (2x2 faceted) plot
PLOT_H_DENS     <- 4.2
PLOT_W_DENS_ROW <- 8.5     # density (1-row, 4 panels) plot
PLOT_H_DENS_ROW <- 2.4
PLOT_W_WIN      <- 3.35     # win-score plot
PLOT_H_WIN      <- 2.0
DPI         <- 400

EXCLUDE_DM2 <- TRUE
MAX_STEPS   <- 5000

# Palette (Human-like aligned to darkblue used throughout the synaptic figure)
COHORT_LEVELS  <- c("H-100", "C-temp", "C-reg", "C-both")
COHORT_LABELS  <- c("Human-like", "High Temperature Control",
                    "Weak Regularization Control", "Both Controls")
# Multi-line variants used only in the density-plot LEGEND (one word per row)
COHORT_LABELS_LEGEND <- c("Human-like",
                          "High\nTemperature\nControl",
                          "Weak\nRegularization\nControl",
                          "Both\nControls")
COHORT_COLOURS <- c("H-100" = "darkblue",
                    "C-temp" = "#1abc9c",   # aquamarine — same temp colour used elsewhere
                    "C-reg"  = "#ff7f00",
                    "C-both" = "#6a3d9a")

OUT_DIR    <- "analysis/Final_analyses_and_plots/Figures"
CACHE_DIR  <- "analysis/Final_analyses_and_plots/cache"
OUT_DENS     <- file.path(OUT_DIR, "pub_comp_scatter_density.png")
OUT_DENS_ROW <- file.path(OUT_DIR, "pub_comp_scatter_density_row.png")
OUT_WIN      <- file.path(OUT_DIR, "pub_comp_scatter_winscore.png")
SD_RDS     <- file.path(CACHE_DIR, "comp_sd_space.rds")
COMP_CSV   <- "analysis/compositional_analysis/Data/compositional_dataset.csv"
TASK_CSV   <- "analysis/compositional_analysis/Data/task_metadata.csv"
if (!dir.exists(OUT_DIR))   dir.create(OUT_DIR,   recursive = TRUE)
if (!dir.exists(CACHE_DIR)) dir.create(CACHE_DIR, recursive = TRUE)

# ── HELDOUT_MATCHES (verbatim from source) ───────────────────────────────────
HELDOUT_MATCHES <- data.frame(
    held_out  = c(
        "poli.ctxdm1", "poli.ctxdm2", "poli.ctxdlydm1", "poli.ctxdlydm2",
        "poli.antictxdm1", "poli.antictxdm2",
        "poli.antictxdlydm1", "poli.antictxdlydm2",
        "poli.antidlyms", "poli.antidlynms",
        "poli.antictxdlyms", "poli.antictxcatdlyms"
    ),
    training  = c(
        "poli.ctxgo", "poli.ctxgo", "poli.dlyctxgo", "poli.dlyctxgo",
        "poli.antictxgo", "poli.antictxgo",
        "poli.dlyantictxgo", "poli.dlyantictxgo",
        "poli.dlyantigo", "poli.dlyantigo",
        "poli.dlyantictxgo", "poli.dlyantictxgo"
    ),
    comp_group = c(
        rep("ctx -> Decision", 8),
        rep("anti -> Match", 2),
        rep("ctx+anti -> Match", 2)
    ),
    stringsAsFactors = FALSE
)
if (EXCLUDE_DM2) HELDOUT_MATCHES <- HELDOUT_MATCHES %>% filter(!grepl("dm2", held_out))

# ── Build scatter data (cached) ──────────────────────────────────────────────
build_sd_space <- function() {
    cat("Building scatter data from CSVs...\n")
    df_wide   <- read.csv(COMP_CSV, stringsAsFactors = FALSE)
    task_meta <- read.csv(TASK_CSV, stringsAsFactors = FALSE)

    solved_cols <- grep("^probe_solved_at_poli_", names(df_wide), value = TRUE)
    df_sol <- df_wide[, c("run_id", "cohort", solved_cols)] %>%
        pivot_longer(cols = all_of(solved_cols),
                     names_to = "col", values_to = "solved_at") %>%
        mutate(task = sub("^probe_solved_at_poli_", "poli.", col)) %>%
        select(-col) %>%
        mutate(
            cohort     = factor(cohort, levels = COHORT_LEVELS),
            solved     = !is.na(solved_at),
            surv_time  = ifelse(is.na(solved_at), MAX_STEPS, solved_at)
        )
    if (EXCLUDE_DM2) df_sol <- df_sol %>% filter(!grepl("dm2", task))

    sd <- do.call(rbind, lapply(seq_len(nrow(HELDOUT_MATCHES)), function(i) {
        train_rows <- df_sol %>%
            filter(task == HELDOUT_MATCHES$training[i]) %>%
            select(run_id, cohort,
                   train_time = surv_time, train_solved = solved)
        held_rows <- df_sol %>%
            filter(task == HELDOUT_MATCHES$held_out[i]) %>%
            select(run_id, held_time = surv_time, held_solved = solved)
        inner_join(train_rows, held_rows, by = "run_id") %>%
            mutate(comp_group = HELDOUT_MATCHES$comp_group[i])
    })) %>%
        filter(!is.na(cohort)) %>%
        mutate(
            cohort   = factor(cohort, levels = COHORT_LEVELS),
            censored = !held_solved | !train_solved,
            lx       = log(train_time),
            ly       = log(held_time)
        )
    sd
}

if (file.exists(SD_RDS)) {
    cat("Loading cached:", SD_RDS, "\n")
    sd_space <- readRDS(SD_RDS)
} else {
    sd_space <- build_sd_space()
    saveRDS(sd_space, SD_RDS)
    cat("Cached:", SD_RDS, "\n")
}

sd_solved   <- sd_space %>% filter(!censored)
sd_censored <- sd_space %>% filter(censored)
cat(sprintf("sd_solved: %d  |  sd_censored: %d\n",
            nrow(sd_solved), nrow(sd_censored)))

# ── Common geometry / labels ─────────────────────────────────────────────────
xy_breaks <- log(c(100, 500, 1000, 5000))
xy_labels <- c("100", "500", "1k", "5k")
KDE_N     <- 80
FIXED_BW  <- c(0.3, 0.3)
x_lims     <- range(sd_solved$lx) + c(-0.3, 0.3)
y_lims     <- range(sd_solved$ly) + c(-0.3, 0.3)
x_lims_cen <- c(min(sd_space$lx) - 0.3, max(sd_space$lx) + 0.3)
y_lims_cen <- c(min(sd_space$ly) - 0.3, max(sd_space$ly) + 0.3)

X_LAB <- "Steps to solve - training task"
Y_LAB <- "Steps to solve - held-out task"

# ── KDEs (verbatim grid logic from source) ───────────────────────────────────
kde_df <- do.call(rbind, lapply(COHORT_LEVELS, function(coh) {
    sub <- sd_solved %>% filter(cohort == coh)
    k   <- MASS::kde2d(sub$lx, sub$ly, n = KDE_N, lims = c(x_lims, y_lims))
    expand.grid(lx = k$x, ly = k$y) %>%
        mutate(density = as.vector(k$z),
               density_norm = density / max(density),
               cohort = coh)
})) %>% mutate(cohort = factor(cohort, levels = COHORT_LEVELS))

kde_cen_df <- do.call(rbind, lapply(COHORT_LEVELS, function(coh) {
    sub <- sd_censored %>% filter(cohort == coh)
    if (nrow(sub) < 5) return(NULL)
    k <- tryCatch(
        MASS::kde2d(sub$lx, sub$ly, n = KDE_N, h = FIXED_BW,
                    lims = c(x_lims_cen, y_lims_cen)),
        error = function(e) NULL)
    if (is.null(k)) return(NULL)
    expand.grid(lx = k$x, ly = k$y) %>%
        mutate(density = as.vector(k$z),
               density_norm = density / max(density),
               cohort = coh)
})) %>% mutate(cohort = factor(cohort, levels = COHORT_LEVELS))

kde_solved_fill <- kde_df     %>% mutate(fill_group = as.character(cohort))
kde_cen_fill    <- kde_cen_df %>% mutate(fill_group = "Failure")

all_fill_colors <- c(COHORT_COLOURS, "Failure" = "grey58")
all_fill_labels <- c(COHORT_LABELS_LEGEND, "Fail")

# ── Plot 1: faceted density (per cohort) ─────────────────────────────────────
p_dens <- ggplot() +
    geom_point(data = sd_censored,
               aes(x = lx, y = ly),
               colour = "grey60", size = 0.7, alpha = 0.45) +
    geom_raster(data = kde_cen_fill,
                aes(x = lx, y = ly, fill = fill_group, alpha = density_norm),
                interpolate = TRUE) +
    geom_contour(data = kde_cen_df,
                 aes(x = lx, y = ly, z = density_norm),
                 colour = "grey35", breaks = c(0.2, 0.5, 0.8),
                 linewidth = 0.4) +
    geom_raster(data = kde_solved_fill,
                aes(x = lx, y = ly, fill = fill_group, alpha = density_norm),
                interpolate = TRUE) +
    geom_contour(data = kde_df,
                 aes(x = lx, y = ly, z = density_norm, colour = cohort),
                 breaks = c(0.2, 0.5, 0.8), linewidth = 0.5) +
    geom_jitter(data = sd_solved,
                aes(x = lx, y = ly, colour = cohort),
                width = 0.15, height = 0.09, alpha = 0.55, size = 0.9) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                colour = "grey25", linewidth = 0.4) +
    facet_wrap(~ cohort, ncol = 2,
               labeller = labeller(cohort = setNames(COHORT_LABELS, COHORT_LEVELS))) +
    scale_fill_manual(
        values = all_fill_colors,
        breaks = c(COHORT_LEVELS, "Failure"),
        labels = all_fill_labels,
        name   = NULL,
        guide  = guide_legend(override.aes = list(alpha = 0.7, colour = NA, size = 3))
    ) +
    scale_colour_manual(values = COHORT_COLOURS, guide = "none") +
    scale_alpha_continuous(range = c(0, 0.50), guide = "none") +
    scale_x_continuous(breaks = xy_breaks, labels = xy_labels) +
    scale_y_continuous(breaks = xy_breaks, labels = xy_labels) +
    coord_cartesian(xlim = x_lims_cen, ylim = y_lims_cen) +
    labs(x = X_LAB, y = Y_LAB) +
    theme_classic(base_size = 9) +
    theme(
        strip.background = element_blank(),
        strip.text       = element_text(face = "bold", size = 9),
        legend.position  = "bottom",
        legend.text      = element_text(size = 8),
        legend.key.size  = unit(0.4, "cm"),
        plot.background  = element_rect(fill = "white", colour = NA),
        panel.background = element_rect(fill = "white", colour = NA)
    )

ggsave(OUT_DENS, p_dens, width = PLOT_W_DENS, height = PLOT_H_DENS, dpi = DPI)
cat("Saved:", OUT_DENS, "\n")

# ── Plot 1b: same density, all 4 panels in a single horizontal row, no legend ─
p_dens_row <- p_dens +
    facet_wrap(~ cohort, nrow = 1,
               labeller = labeller(cohort = setNames(COHORT_LABELS, COHORT_LEVELS))) +
    guides(fill = "none", colour = "none", alpha = "none") +
    theme(legend.position = "none")

ggsave(OUT_DENS_ROW, p_dens_row,
       width = PLOT_W_DENS_ROW, height = PLOT_H_DENS_ROW, dpi = DPI)
cat("Saved:", OUT_DENS_ROW, "\n")

# ── Plot 2: win-score (Human-like density - controls density) ────────────────
kde_h100 <- MASS::kde2d(
    sd_solved %>% filter(cohort == "H-100") %>% pull(lx),
    sd_solved %>% filter(cohort == "H-100") %>% pull(ly),
    n = 100, lims = c(x_lims, y_lims))
kde_ctrl <- MASS::kde2d(
    sd_solved %>% filter(cohort != "H-100") %>% pull(lx),
    sd_solved %>% filter(cohort != "H-100") %>% pull(ly),
    n = 100, lims = c(x_lims, y_lims))

win_df <- expand.grid(lx = kde_h100$x, ly = kde_h100$y) %>%
    mutate(d_h100 = as.vector(kde_h100$z),
           d_ctrl = as.vector(kde_ctrl$z),
           win    = (d_h100 - d_ctrl) / (d_h100 + d_ctrl + 1e-10))

# Diverging fill: controls-enriched = warm, equal = white, Human-like-enriched = darkblue
p_win <- ggplot() +
    geom_raster(data = win_df,
                aes(x = lx, y = ly, fill = win), interpolate = TRUE) +
    geom_contour(data = win_df,
                 aes(x = lx, y = ly, z = win),
                 breaks = c(-0.3, 0, 0.3),
                 colour = "white", linewidth = 0.4, linetype = "dashed") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                colour = "grey20", linewidth = 0.4) +
    scale_fill_gradient2(low = "#f2857f", mid = "white", high = "darkblue",
                         midpoint = 0,
                         name = "Human-like\nvs controls",
                         limits = c(-1, 1)) +
    scale_x_continuous(breaks = xy_breaks, labels = xy_labels) +
    scale_y_continuous(breaks = xy_breaks, labels = xy_labels) +
    coord_cartesian(xlim = x_lims, ylim = y_lims) +
    labs(x = X_LAB, y = Y_LAB) +
    theme_classic(base_size = 9) +
    theme(
        legend.position  = "right",
        legend.text      = element_text(size = 8),
        legend.title     = element_text(size = 8),
        legend.key.size  = unit(0.4, "cm"),
        plot.background  = element_rect(fill = "white", colour = NA),
        panel.background = element_rect(fill = "white", colour = NA)
    )

ggsave(OUT_WIN, p_win, width = PLOT_W_WIN, height = PLOT_H_WIN, dpi = DPI)
cat("Saved:", OUT_WIN, "\n")

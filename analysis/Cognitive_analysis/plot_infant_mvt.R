# analysis/Cognitive_analysis/plot_infant_mvt.R
# Visualizations for the MVT infant fits produced by fit_infant_mvt.R
# Reads artifacts from Results/ and writes plots to Results/infant_plots/

library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)
library(lme4)
library(grid)

# Disable scientific notation on axes (1000 instead of 1e+3).
options(scipen = 999)

# --- Configuration ---
RESULTS_DIR <- "analysis/Cognitive_analysis/Results"
PLOT_DIR <- file.path(RESULTS_DIR, "infant_plots")
TRACE_DIR <- file.path(PLOT_DIR, "example_trace")
# Criteria to include an infant in the example-trace gallery
MIN_TRIALS <- 50
MIN_SEQS   <- 7

dir.create(TRACE_DIR, recursive = TRUE, showWarnings = FALSE)
# Clear stale traces from previous runs (ranking may shift between runs)
old_traces <- list.files(TRACE_DIR, pattern = "\\.png$", full.names = TRUE)
if (length(old_traces)) file.remove(old_traces)

# --- Helpers ---
alpha_from_hl <- function(hl) log(2) / hl

ema_vec <- function(r, alpha, init = 0) {
    out <- numeric(length(r))
    x <- init
    if (length(r) > 0) {
        for (i in seq_along(r)) {
            x <- (1 - alpha) * x + alpha * r[i]
            out[i] <- x
        }
    }
    out
}

# --- Load Artifacts ---
cat("Loading fit artifacts...\n")
dat_all      <- readRDS(file.path(RESULTS_DIR, "infant_dat_all.rds"))
indiv_df     <- read_csv(file.path(RESULTS_DIR, "infant_mvt_params.csv"), show_col_types = FALSE)
group_df     <- read_csv(file.path(RESULTS_DIR, "infant_group_mvt_params.csv"), show_col_types = FALSE)
group_model  <- readRDS(file.path(RESULTS_DIR, "infant_group_mvt_model.rds"))
control_model <- tryCatch(readRDS(file.path(RESULTS_DIR, "infant_control_model.rds")), error = function(e) NULL)

group_alpha <- group_df$alpha[1]
group_beta  <- group_df$beta[1]
cat(sprintf("Group alpha=%.4f (HL=%.2f), beta=%.4f (HL=%.2f)\n",
            group_alpha, log(2) / group_alpha, group_beta, log(2) / group_beta))

# --- 1. Example Traces for Infants with Enough Data ---
cat(sprintf("\nSelecting infants with >= %d trials AND >= %d unique sequences...\n",
            MIN_TRIALS, MIN_SEQS))

trial_counts <- dat_all %>%
    group_by(subj_global, dataset) %>%
    summarise(
        n_trials = n(),
        n_seqs   = n_distinct(nseq),
        .groups  = "drop"
    ) %>%
    filter(n_trials >= MIN_TRIALS, n_seqs >= MIN_SEQS) %>%
    arrange(desc(n_trials))

top_subjs <- trial_counts
cat(sprintf("Kept %d infants. Trials range: %d-%d; sequences range: %d-%d\n",
            nrow(top_subjs),
            min(top_subjs$n_trials), max(top_subjs$n_trials),
            min(top_subjs$n_seqs),   max(top_subjs$n_seqs)))

# Matched to pub_foraging_plots.R slice style (see ../plotting/publication_plots).
SEQ_PALETTE_10 <- c(
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
)
SWITCH_COL <- "#E26714"             # orange diamond for look-aways
PLOT_W     <- 4.0                   # panel width (in), matches slice plots
PLOT_H     <- 1.7                   # panel height (in)

base_theme <- theme_classic(base_size = 11) +
    theme(
        axis.line        = element_line(colour = "black", linewidth = 0.6),
        axis.ticks       = element_line(colour = "black", linewidth = 0.4),
        axis.text        = element_text(colour = "black", size = 9),
        axis.title       = element_text(colour = "black", size = 10, face = "bold"),
        legend.title     = element_blank(),
        legend.text      = element_text(size = 7),
        legend.key.size  = unit(0.35, "cm"),
        legend.position  = "right",
        plot.background  = element_rect(fill = "white", colour = NA),
        panel.background = element_rect(fill = "white", colour = NA)
    )

# Force the panel area to exactly (w × h) inches, let legend/axis add space.
save_fixed_panel <- function(plot, path, w_in = PLOT_W, h_in = PLOT_H, dpi = 300) {
    g <- ggplot2::ggplotGrob(plot)
    panel_cells <- g$layout[grepl("^panel", g$layout$name), ]
    g$widths [unique(panel_cells$l)] <- unit(w_in, "in")
    g$heights[unique(panel_cells$t)] <- unit(h_in, "in")

    tmp <- tempfile(fileext = ".png")
    grDevices::png(tmp, width = 20, height = 20, units = "in", res = 72)
    total_w <- grid::convertWidth (sum(g$widths ), "in", valueOnly = TRUE)
    total_h <- grid::convertHeight(sum(g$heights), "in", valueOnly = TRUE)
    grDevices::dev.off(); unlink(tmp)

    ggplot2::ggsave(path, g, width = total_w, height = total_h,
                    dpi = dpi, limitsize = FALSE)
}

plot_infant_trace <- function(sub_id, dataset_name, n_trials, rank_idx) {
    ex_data <- dat_all %>% filter(subj_global == sub_id)
    P_vec   <- ema_vec(ex_data$D, group_alpha)
    rho_vec <- ema_vec(ex_data$D, group_beta)

    plot_df <- ex_data %>%
        mutate(Trial = row_number(), P = P_vec, Rho = rho_vec)

    # Keep the first 10 sequences only
    seqs_kept    <- head(sort(unique(plot_df$nseq)), 10)
    plot_df_sub  <- plot_df %>%
        filter(nseq %in% seqs_kept) %>%
        mutate(seq_f = factor(nseq, levels = seqs_kept))
    n_seq_shown  <- length(seqs_kept)

    p <- ggplot(plot_df_sub, aes(x = Trial)) +
        geom_line(aes(y = Rho), colour = "black",
                  linetype = "dashed", linewidth = 0.7) +
        geom_line(aes(y = P, colour = seq_f, group = seq_f),
                  linewidth = 0.4) +
        geom_point(
            data = subset(plot_df_sub, event == 1),
            aes(y = P),
            colour = SWITCH_COL, fill = SWITCH_COL,
            shape = 18, size = 2.4
        ) +
        scale_colour_manual(values = SEQ_PALETTE_10[seq_len(n_seq_shown)],
                            name = NULL) +
        scale_x_continuous(name = "Trial",
                           expand = expansion(mult = c(0.01, 0.01))) +
        scale_y_continuous(name = "Reward (Learning Progress)") +
        base_theme

    safe_id  <- gsub("[^A-Za-z0-9_.-]", "_", sub_id)
    out_file <- file.path(TRACE_DIR, sprintf("%03d_%s.png", rank_idx, safe_id))
    save_fixed_panel(p, out_file)
}

for (i in seq_len(nrow(top_subjs))) {
    plot_infant_trace(top_subjs$subj_global[i],
                      top_subjs$dataset[i],
                      top_subjs$n_trials[i],
                      rank_idx = i)
}
cat(sprintf("Saved %d traces to %s\n", nrow(top_subjs), TRACE_DIR))

# --- 2. Effect Plot: Value (X = P - Rho) vs Look-Away ---
# What this shows:
#   X-axis = value = P_t - rho_t (local progress minus global rate, per trial)
#   Y-axis = P(look-away) (event == 1)
#   Grey ribbon/points: empirical proportion ± binomial SE within 15 quantile bins of X
#   Red line: MARGINAL (population-average) logistic curve from the group GLMER
#             using only the fixed effects, b0_raw + b1_raw * X, in raw X units.
#   Dotted vertical at X = 0 (local = global, MVT decision threshold).
# Interpretation: if MVT holds, the red curve should DECREASE with X (higher local
# value -> lower P(leave)). A flat or upward slope indicates the sign is off.
cat("\nGenerating effect plot (value vs look-away)...\n")

# Recompute X for every trial using group params, pooled across the full sample
# (all 275 infants), matching the group-level GLMER fit which uses the same pool.
df_pool <- dat_all
subj_list <- split(df_pool, df_pool$subj_global)
eff_rows  <- list()
for (i in seq_along(subj_list)) {
    s <- subj_list[[i]]
    eff_rows[[i]] <- data.frame(
        subj_global = s$subj_global,
        event       = s$event,
        X           = ema_vec(s$D, group_alpha) - ema_vec(s$D, group_beta)
    )
}
eff_df <- bind_rows(eff_rows)

# Empirical: bin X into deciles, get observed P(leave) ± SE
eff_binned <- eff_df %>%
    mutate(bin = ntile(X, 15)) %>%
    group_by(bin) %>%
    summarise(
        X_mid   = mean(X),
        p_leave = mean(event),
        se      = sqrt(p_leave * (1 - p_leave) / n()),
        .groups = "drop"
    )

# Model prediction: logistic curve from group GLMER fixed effects (marginal)
fix_ef <- fixef(group_model)
mu_X <- mean(eff_df$X)
sd_X <- sd(eff_df$X)
b0_raw <- fix_ef[1] - (fix_ef[2] * mu_X / sd_X)
b1_raw <- fix_ef[2] / sd_X

X_grid <- seq(min(eff_df$X), max(eff_df$X), length.out = 200)
pred_df <- data.frame(
    X       = X_grid,
    p_leave = plogis(b0_raw + b1_raw * X_grid)
)

p_effect <- ggplot() +
    geom_ribbon(data = eff_binned,
                aes(x = X_mid, ymin = pmax(0, p_leave - se),
                    ymax = pmin(1, p_leave + se)),
                fill = "grey80", alpha = 0.6) +
    geom_point(data = eff_binned,
               aes(x = X_mid, y = p_leave),
               color = "black", size = 2.5) +
    geom_line(data = pred_df,
              aes(x = X, y = p_leave),
              color = "red", size = 1) +
    geom_vline(xintercept = 0, linetype = "dotted", color = "grey40") +
    labs(
        title = "Effect of MVT Value on Look-Away",
        subtitle = sprintf("Value X = P - Rho | Group GLMER: b1=%.3f (T=%.2f, eps=%.2f)",
                           b1_raw, group_df$T[1], group_df$eps[1]),
        x = "Value  (Local P - Global Rho)",
        y = "P(look-away)"
    ) +
    theme_minimal()

ggsave(file.path(PLOT_DIR, "effect_value_vs_lookaway.png"),
       p_effect, width = 8, height = 5, dpi = 120)
cat("Saved effect plot to", file.path(PLOT_DIR, "effect_value_vs_lookaway.png"), "\n")

# --- 2b. jtools::effect_plot alternative ---
# Refit a tiny GLMER on the same pooled data but with the predictor literally
# named "value" so effect_plot can target `pred = value`.
cat("\nGenerating jtools effect_plot alternative...\n")

if (!requireNamespace("jtools", quietly = TRUE)) {
    cat("jtools not installed; skipping. Install with install.packages('jtools').\n")
} else {
    fit_df <- eff_df %>% rename(value = X)
    cat("Refitting glmer(event ~ value + (1|subj_global)) on pooled data...\n")
    alt_mod <- glmer(
        event ~ value + (1 | subj_global),
        data = fit_df,
        family = binomial,
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
    )

    p_jt <- jtools::effect_plot(alt_mod, pred = value, interval = TRUE) +
        labs(
            title = "jtools effect_plot — P(look-away) vs value",
            subtitle = sprintf("value = P - Rho | Alpha_hl=%.2f, Beta_hl=%.2f",
                               log(2) / group_alpha, log(2) / group_beta),
            x = "Value (P - Rho)",
            y = "P(look-away)"
        ) +
        theme_minimal()

    ggsave(file.path(PLOT_DIR, "effect_value_vs_lookaway_jtools.png"),
           p_jt, width = 7, height = 5, dpi = 120)
    cat("Saved jtools effect plot to",
        file.path(PLOT_DIR, "effect_value_vs_lookaway_jtools.png"), "\n")
}

# --- 2c. AIC Landscape Heatmap over (alpha_hl, beta_hl) ---
cat("\nGenerating AIC landscape heatmap...\n")
grid_file <- file.path(RESULTS_DIR, "infant_group_grid_aic.rds")
if (!file.exists(grid_file)) {
    cat("No grid AIC file found at", grid_file,
        "- rerun fit_infant_mvt.R to regenerate.\n")
} else {
    grid <- readRDS(grid_file)
    heat_df <- expand.grid(
        alpha_hl = grid$alpha_hl,
        beta_hl  = grid$beta_hl
    )
    heat_df$aic <- as.vector(grid$aic)
    heat_df$dAIC <- heat_df$aic - min(heat_df$aic, na.rm = TRUE)

    best_row <- heat_df[which.min(heat_df$aic), ]

    p_heat <- ggplot(heat_df, aes(x = alpha_hl, y = beta_hl, fill = dAIC)) +
        geom_tile() +
        geom_point(data = best_row, aes(x = alpha_hl, y = beta_hl),
                   color = "red", shape = 4, size = 4, stroke = 1.5,
                   inherit.aes = FALSE) +
        scale_fill_viridis_c(option = "magma", direction = -1,
                             name = "ΔAIC",
                             trans = "sqrt",
                             na.value = "grey90") +
        labs(
            title = "AIC Landscape — Group GLMER",
            subtitle = sprintf("Best: alpha_hl=%.1f, beta_hl=%.1f (AIC=%.1f) | red X marks optimum | grey = excluded by beta_hl < alpha_hl",
                               best_row$alpha_hl, best_row$beta_hl, best_row$aic),
            x = "Alpha half-life (trials) — local memory",
            y = "Beta half-life (trials) — global memory"
        ) +
        theme_minimal()

    ggsave(file.path(PLOT_DIR, "aic_landscape_heatmap.png"),
           p_heat, width = 7.5, height = 6, dpi = 120)
    cat("Saved AIC heatmap to",
        file.path(PLOT_DIR, "aic_landscape_heatmap.png"), "\n")

    # Zoomed-in version: alpha_hl <= 6, beta_hl <= 10
    zoom_df <- subset(heat_df, alpha_hl <= 6 & beta_hl <= 10)
    # Recompute dAIC relative to the zoom window so the colour scale is informative there
    zoom_df$dAIC_zoom <- zoom_df$aic - min(zoom_df$aic, na.rm = TRUE)
    best_zoom <- zoom_df[which.min(zoom_df$aic), ]

    p_heat_zoom <- ggplot(zoom_df, aes(x = alpha_hl, y = beta_hl, fill = dAIC_zoom)) +
        geom_tile() +
        geom_text(aes(label = ifelse(is.na(aic), "", sprintf("%.0f", dAIC_zoom))),
                  size = 3, color = "white") +
        geom_point(data = best_zoom, aes(x = alpha_hl, y = beta_hl),
                   color = "red", shape = 4, size = 5, stroke = 1.8,
                   inherit.aes = FALSE) +
        scale_fill_viridis_c(option = "magma", direction = -1,
                             name = "ΔAIC\n(zoom)",
                             trans = "sqrt",
                             na.value = "grey90") +
        scale_x_continuous(breaks = sort(unique(zoom_df$alpha_hl))) +
        scale_y_continuous(breaks = sort(unique(zoom_df$beta_hl))) +
        labs(
            title = "AIC Landscape — Zoom (alpha_hl <= 6, beta_hl <= 10)",
            subtitle = sprintf("Best in zoom: alpha_hl=%.1f, beta_hl=%.1f (AIC=%.1f) | cell labels = ΔAIC vs zoom-min",
                               best_zoom$alpha_hl, best_zoom$beta_hl, best_zoom$aic),
            x = "Alpha half-life (trials) — local memory",
            y = "Beta half-life (trials) — global memory"
        ) +
        theme_minimal()

    ggsave(file.path(PLOT_DIR, "aic_landscape_heatmap_zoom.png"),
           p_heat_zoom, width = 7.5, height = 6, dpi = 120)
    cat("Saved zoomed AIC heatmap to",
        file.path(PLOT_DIR, "aic_landscape_heatmap_zoom.png"), "\n")
}

# --- 3. Calibration Plot (Group GLMER) ---
cat("\nGenerating calibration plot...\n")

# Per-trial predicted P(leave) = plogis(b0_raw + b1_raw * X + u_subj)
ranef_tab <- ranef(group_model)$subj_all
u_df <- data.frame(
    subj_global = rownames(ranef_tab),
    u_subj      = ranef_tab[, 1]
)

cal_df <- eff_df %>%
    left_join(u_df, by = "subj_global") %>%
    mutate(
        u_subj  = ifelse(is.na(u_subj), 0, u_subj),  # new subj -> marginal
        p_pred  = plogis(b0_raw + b1_raw * X + u_subj)
    )

n_bins <- 10
cal_binned <- cal_df %>%
    mutate(bin = ntile(p_pred, n_bins)) %>%
    group_by(bin) %>%
    summarise(
        pred_mid = mean(p_pred),
        obs      = mean(event),
        se       = sqrt(obs * (1 - obs) / n()),
        n        = n(),
        .groups  = "drop"
    )

# Brier score as an overall fit diagnostic
brier <- mean((cal_df$p_pred - cal_df$event)^2)

p_cal <- ggplot(cal_binned, aes(x = pred_mid, y = obs)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
    geom_errorbar(aes(ymin = pmax(0, obs - se), ymax = pmin(1, obs + se)),
                  width = 0.015, color = "grey30") +
    geom_point(aes(size = n), color = "steelblue") +
    scale_size_continuous(range = c(2, 6), name = "n trials / bin") +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(
        title = "Calibration: Group GLMER Predicted vs Observed P(look-away)",
        subtitle = sprintf("%d deciles of predicted P | Brier=%.4f", n_bins, brier),
        x = "Predicted P(look-away)",
        y = "Observed P(look-away)"
    ) +
    theme_minimal()

ggsave(file.path(PLOT_DIR, "calibration_group_model.png"),
       p_cal, width = 6.5, height = 6, dpi = 120)
cat("Saved calibration plot to", file.path(PLOT_DIR, "calibration_group_model.png"), "\n")

cat("\nPlotting complete.\n")

suppressPackageStartupMessages({
    library(dplyr)
    library(readr)
    library(ggplot2)
    library(lme4)
})

# Publication-quality infant MVT effect plot, flipped to P(stay).
# Uses the same fit artifacts as analysis/Cognitive_analysis/plot_infant_mvt.R,
# but flips Y so the curve tracks P(stay on task) = 1 − P(look-away).

RESULTS_DIR <- "analysis/Cognitive_analysis/Results"
OUT_DIR     <- "analysis/Final_analyses_and_plots/Figures"

# Number of quantile bins for the empirical dots / SE ribbon
N_BINS <- 8

# ── Helpers ───────────────────────────────────────────────────────────────────
ema_vec <- function(r, alpha, init = 0) {
    out <- numeric(length(r))
    x <- init
    if (length(r) > 0) for (i in seq_along(r)) {
        x <- (1 - alpha) * x + alpha * r[i]
        out[i] <- x
    }
    out
}

# ── Load artifacts ────────────────────────────────────────────────────────────
cat("Loading infant MVT fit artifacts...\n")
dat_all     <- readRDS(file.path(RESULTS_DIR, "infant_dat_all.rds"))
indiv_df    <- read_csv(file.path(RESULTS_DIR, "infant_mvt_params.csv"), show_col_types = FALSE)
group_df    <- read_csv(file.path(RESULTS_DIR, "infant_group_mvt_params.csv"), show_col_types = FALSE)
group_model <- readRDS(file.path(RESULTS_DIR, "infant_group_mvt_model.rds"))

group_alpha <- group_df$alpha[1]
group_beta  <- group_df$beta[1]

# ── Per-trial value X = P_t − rho_t ───────────────────────────────────────────
# Use the full infant sample (all 275 subjects), matching the group-level GLMER
# fit which is pooled across the entire pool with no per-subject filtering.
df_pool     <- dat_all
subj_list   <- split(df_pool, df_pool$subj_global)

eff_df <- bind_rows(lapply(subj_list, function(s) {
    data.frame(
        subj_global = s$subj_global,
        event       = s$event,
        X           = ema_vec(s$D, group_alpha) - ema_vec(s$D, group_beta)
    )
}))

# ── Empirical bins (N_BINS quantiles), flipped to stay ───────────────────────
eff_binned <- eff_df %>%
    mutate(bin = cut(X, breaks = N_BINS)) %>%
    group_by(bin) %>%
    summarise(
        X_mid  = mean(X),
        p_look = mean(event),
        p_stay = 1 - p_look,
        se     = sqrt(p_look * (1 - p_look) / n()),
        .groups = "drop"
    )

# ── Marginal model curve with 95% CI, flipped ────────────────────────────────
# Model was fit as: event ~ X_sc + (1 | subj_all), where X_sc is standardized X.
# Delta-method CI: eta = b0 + b1*X_sc; SE(eta) from vcov(fixed); then plogis().
fix_ef <- fixef(group_model)
V      <- as.matrix(vcov(group_model))
mu_X   <- mean(eff_df$X)
sd_X   <- sd(eff_df$X)

X_grid     <- seq(min(eff_df$X), max(eff_df$X), length.out = 300)
X_grid_sc  <- (X_grid - mu_X) / sd_X

eta        <- fix_ef[1] + fix_ef[2] * X_grid_sc
se_eta     <- sqrt(V[1, 1] + X_grid_sc^2 * V[2, 2] + 2 * X_grid_sc * V[1, 2])
p_look_mid <- plogis(eta)
p_look_lo  <- plogis(eta - 1.96 * se_eta)
p_look_hi  <- plogis(eta + 1.96 * se_eta)

pred_df <- data.frame(
    X         = X_grid,
    p_stay    = 1 - p_look_mid,
    p_stay_lo = 1 - p_look_hi,   # flip: upper of p_look -> lower of p_stay
    p_stay_hi = 1 - p_look_lo
)

# ── Colours ───────────────────────────────────────────────────────────────────
curve_col <- rgb(226, 103,  20, maxColorValue = 255)   # #E26714
ribbon_col <- rgb(244, 177, 131, maxColorValue = 255)   # #F4B183

# ── Plot ──────────────────────────────────────────────────────────────────────
p_effect <- ggplot() +
    # Curve CI ribbon
    geom_ribbon(data = pred_df,
                aes(x = X,
                    ymin = pmax(0.5, p_stay_lo),
                    ymax = pmin(1,   p_stay_hi)),
                fill = ribbon_col, alpha = 0.55) +
    # Empirical SE ribbon connecting the binned points
    geom_ribbon(data = eff_binned,
                aes(x = X_mid,
                    ymin = pmax(0.5, p_stay - se),
                    ymax = pmin(1,   p_stay + se)),
                fill = "grey75", alpha = 0.6) +
    geom_point(data = eff_binned,
               aes(x = X_mid, y = p_stay),
               colour = "black", size = 1.6) +
    geom_line(data = pred_df,
              aes(x = X, y = p_stay),
              colour = curve_col, linewidth = 1.2) +
    scale_x_continuous(
        name   = "Relative Value",
        expand = expansion(mult = c(0.02, 0.02))
    ) +
    scale_y_continuous(
        name   = "P(stay on task)",
        limits = c(0.5, 1),
        breaks = c(0.5, 0.75, 1),
        expand = c(0.01, 0)
    ) +
    theme_classic(base_size = 13) +
    theme(
        axis.line        = element_line(colour = "black", linewidth = 0.7),
        axis.ticks       = element_line(colour = "black", linewidth = 0.5),
        axis.text        = element_text(colour = "black", size = 11),
        axis.title       = element_text(colour = "black", size = 12, face = "bold"),
        plot.background  = element_rect(fill = "white", colour = NA),
        panel.background = element_rect(fill = "white", colour = NA),
    )

ggsave(file.path(OUT_DIR, "pub_infant_mvt_stay.png"), p_effect,
       width = 2.0, height = 2.1, dpi = 300)
ggsave(file.path(OUT_DIR, "pub_infant_mvt_stay.pdf"), p_effect,
       width = 3.0, height = 2.6, device = cairo_pdf)
cat("Saved: pub_infant_mvt_stay.pdf / .png\n")

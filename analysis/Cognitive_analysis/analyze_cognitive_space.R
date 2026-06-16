# analysis/cognitive_space_plots.R
# Visualize the Cognitive Space Comparison: Infants vs Models
# Bounded [0,1] Metrics:
# 1. Baseline Stay = sigmoid(-b0) — default tendency to stay
# 2. Reward Modulation = P(stay|+1SD) - P(stay|-1SD) — reward sensitivity
# 3. Timescale Balance = alpha_hl / (alpha_hl + beta_hl) — relative learning speed

library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(scales)

# --- Configuration ---
INFANT_FILE <- "analysis/Cognitive_analysis/Results/infant_mvt_params.csv"
MODEL_FILE <- "analysis/Cognitive_analysis/Results/model_mvt_params_fitted.csv"
PLOT_DIR <- "analysis/comparison_plots/cognitive_space"

if (!dir.exists(PLOT_DIR)) dir.create(PLOT_DIR, recursive = TRUE)

# --- Helper: sigmoid ---
sigmoid <- function(x) 1 / (1 + exp(-pmax(pmin(x, 20), -20)))

# --- 1. Load Data ---
cat("Loading data...\n")

if (!file.exists(INFANT_FILE)) stop("Infant params not found")
if (!file.exists(MODEL_FILE)) stop("Model params not found")

# Infants: compute bounded metrics from raw GLM params
infants_raw <- read_csv(INFANT_FILE, show_col_types = FALSE)
infants <- infants_raw %>%
    filter(!is.na(beta0), !is.na(beta1), !is.na(signal_sd), beta1 < 0) %>%
    mutate(
        group = "Infant",
        type = "Biological",
        id = as.character(subj_global),
        baseline_stay = sigmoid(-beta0),
        reward_modulation = pmax(0, pmin(1,
            sigmoid(-(beta0 + beta1 * signal_sd)) -
            sigmoid(-(beta0 - beta1 * signal_sd))
        )),
        timescale_balance = alpha_hl / (alpha_hl + beta_hl)
    ) %>%
    select(id, group, type, baseline_stay, reward_modulation, timescale_balance)

# Models: bounded metrics already computed
models <- read_csv(MODEL_FILE, show_col_types = FALSE) %>%
    mutate(
        group = "Model",
        type = "Artificial",
        id = as.character(run_id)
    ) %>%
    select(id, group, type, baseline_stay, reward_modulation, timescale_balance)

# Combine
df <- bind_rows(infants, models)

# --- 2. Preprocessing & Cleaning ---
cat("Initial N:", nrow(df), "\n")
df <- df %>%
    filter(
        is.finite(baseline_stay),
        is.finite(reward_modulation),
        is.finite(timescale_balance)
    )

# Outlier removal (infants > 4SD)
cat("Checking for outliers (Infants > 4SD)...\n")
inf_subset <- df %>% filter(group == "Infant")

for (col in c("baseline_stay", "reward_modulation", "timescale_balance")) {
    mu <- mean(inf_subset[[col]], na.rm = TRUE)
    s <- sd(inf_subset[[col]], na.rm = TRUE)
    if (s > 0) {
        outlier_ids <- inf_subset %>%
            filter(abs(.data[[col]] - mu) > 4 * s) %>%
            pull(id)
        if (length(outlier_ids) > 0) {
            cat("Removing", length(outlier_ids), "outliers on", col, "\n")
            df <- df %>% filter(!id %in% outlier_ids)
            inf_subset <- df %>% filter(group == "Infant")
        }
    }
}

cat("Filtered N:", nrow(df), "\n")
print(table(df$group))

# --- 3. Visualization ---
theme_set(theme_minimal() + theme(text = element_text(size = 14)))
colors <- c("Infant" = "#E69F00", "Model" = "#56B4E9")

safe_save <- function(filename, plot_obj) {
    tryCatch(
        {
            ggsave(file.path(PLOT_DIR, filename), plot_obj, width = 8, height = 6)
            cat("Saved", filename, "\n")
        },
        error = function(e) cat("Error saving", filename, ":", e$message, "\n")
    )
}

# A. Pairwise Plots
plot_pairwise <- function(data, x_var, y_var) {
    p <- ggplot(data, aes(.data[[x_var]], .data[[y_var]], color = group)) +
        geom_point(alpha = 0.6, size = 2) +
        scale_color_manual(values = colors) +
        scale_fill_manual(values = colors)

    n_inf <- sum(data$group == "Infant")
    n_mod <- sum(data$group == "Model")
    if (n_inf > 10 && n_mod > 10) {
        p <- p + stat_density_2d(aes(fill = group), geom = "polygon", alpha = 0.1, color = NA)
    }
    return(p)
}

p1 <- plot_pairwise(df, "baseline_stay", "reward_modulation") +
    labs(x = "Baseline Stay", y = "Reward Modulation", title = "Reward Modulation vs Baseline Stay")
safe_save("1_modulation_vs_stay.png", p1)

p2 <- plot_pairwise(df, "baseline_stay", "timescale_balance") +
    labs(x = "Baseline Stay", y = "Timescale Balance", title = "Timescale Balance vs Baseline Stay")
safe_save("2_timescale_vs_stay.png", p2)

p3 <- plot_pairwise(df, "timescale_balance", "reward_modulation") +
    labs(x = "Timescale Balance", y = "Reward Modulation", title = "Reward Modulation vs Timescale Balance")
safe_save("3_modulation_vs_timescale.png", p3)

# B. Marginal Distributions
for (var in c("baseline_stay", "reward_modulation", "timescale_balance")) {
    d <- ggplot(df, aes(.data[[var]], fill = group)) +
        geom_density(alpha = 0.5) +
        scale_fill_manual(values = colors) +
        labs(title = paste("Distribution of", var), x = var)
    safe_save(paste0("dist_", var, ".png"), d)
}

# --- 4. Summary Stats ---
summary_stats <- df %>%
    group_by(group) %>%
    summarise(
        N = n(),
        baseline_stay_mean = mean(baseline_stay, na.rm = TRUE),
        baseline_stay_sd = sd(baseline_stay, na.rm = TRUE),
        reward_mod_mean = mean(reward_modulation, na.rm = TRUE),
        reward_mod_sd = sd(reward_modulation, na.rm = TRUE),
        timescale_bal_mean = mean(timescale_balance, na.rm = TRUE),
        timescale_bal_sd = sd(timescale_balance, na.rm = TRUE)
    )
print(summary_stats)

# --- 5. Distance & Winner Selection ---
cat("Calculating distances...\n")

infant_centroid <- df %>%
    filter(group == "Infant") %>%
    summarise(
        mu_bs = mean(baseline_stay, na.rm = TRUE),
        mu_rm = mean(reward_modulation, na.rm = TRUE),
        mu_tb = mean(timescale_balance, na.rm = TRUE)
    )

models_df <- df %>% filter(group == "Model")

if (nrow(models_df) > 0) {
    models_df <- models_df %>%
        mutate(
            distance = sqrt(
                (baseline_stay - infant_centroid$mu_bs)^2 +
                (reward_modulation - infant_centroid$mu_rm)^2 +
                (timescale_balance - infant_centroid$mu_tb)^2
            )
        )

    winner <- models_df[which.min(models_df$distance), ]
    cat(sprintf("\nGlobal Winner: %s (Dist=%.4f)\n", winner$id, winner$distance))
    write_csv(winner, file.path(PLOT_DIR, "cognitive_winner.csv"))
} else {
    cat("No models to evaluate distance.\n")
    winner <- NULL
}

# --- 6. 3D Visualization ---
if (!require(plot3D)) {
    warning("plot3D package not installed. Skipping 3D plot.")
} else {
    library(plot3D)

    models_clean <- models_df
    x_mod <- models_clean$baseline_stay
    y_mod <- models_clean$reward_modulation
    z_mod <- models_clean$timescale_balance

    inf_clean <- df %>% filter(group == "Infant")
    x_inf <- inf_clean$baseline_stay
    y_inf <- inf_clean$reward_modulation
    z_inf <- inf_clean$timescale_balance

    all_x <- c(x_mod, x_inf)
    all_y <- c(y_mod, y_inf)
    all_z <- c(z_mod, z_inf)

    xlim <- range(all_x, na.rm = TRUE)
    ylim <- range(all_y, na.rm = TRUE)
    zlim <- range(all_z, na.rm = TRUE)

    xlab <- "\nBaseline Stay"
    ylab <- "\n\nReward Modulation"
    zlab <- "\nTimescale Balance"

    set.seed(42)
    jit_x <- x_mod + rnorm(length(x_mod), sd = diff(xlim) * 0.03)
    jit_y <- y_mod + rnorm(length(y_mod), sd = diff(ylim) * 0.03)
    jit_z <- z_mod + rnorm(length(z_mod), sd = diff(zlim) * 0.03)

    plot_base_models <- function(a = 0.06) {
        scatter3D(
            x = jit_x, y = jit_y, z = jit_z,
            colvar = NULL, col = "darkblue",
            pch = 16, cex = 1.0, alpha = a,
            bty = "g", ticktype = "detailed",
            theta = 45, phi = 0,
            xlab = xlab, ylab = ylab, zlab = zlab,
            xlim = xlim, ylim = ylim, zlim = zlim
        )
    }

    # 6a. Models-Only
    png(file.path(PLOT_DIR, "cognitive_space_3d_models.png"), width = 2400, height = 1800, res = 300)
    op <- par(mar = c(5, 4, 4, 7) + 0.1)
    plot_base_models(a = 0.16)
    par(xpd = TRUE)
    legend("right", inset = c(-0.15, 0),
        legend = "Models", pch = 16, col = "darkblue", pt.cex = 1.0, bty = "n")
    par(xpd = FALSE)
    dev.off()
    cat("Saved cognitive_space_3d_models.png\n")

    # 6b. Models + Infants
    png(file.path(PLOT_DIR, "cognitive_space_3d_models_infants.png"), width = 2400, height = 1800, res = 300)
    op <- par(mar = c(5, 4, 4, 7) + 0.1)
    plot_base_models()
    if (length(x_inf) > 0) {
        scatter3D(x = x_inf, y = y_inf, z = z_inf,
            col = "forestgreen", pch = 16, cex = 1.0, alpha = 0.14, add = TRUE)
    }
    par(xpd = TRUE)
    legend("right", inset = c(-0.15, 0),
        legend = c("Infants", "Models"), pch = 16,
        col = c("forestgreen", "darkblue"), pt.cex = c(1.0, 1.0), bty = "n")
    par(xpd = FALSE)
    dev.off()
    cat("Saved cognitive_space_3d_models_infants.png\n")

    # 6c. Full: Models + Infants + Centroid + Winner
    png(file.path(PLOT_DIR, "cognitive_space_3d.png"), width = 2400, height = 1800, res = 300)
    op <- par(mar = c(5, 4, 4, 7) + 0.1)
    plot_base_models()
    if (length(x_inf) > 0) {
        scatter3D(x = x_inf, y = y_inf, z = z_inf,
            col = "forestgreen", pch = 16, cex = 1.0, alpha = 0.14, add = TRUE)
    }
    pmat <- getplist()$mat
    if (nrow(infant_centroid) > 0) {
        pt2d <- trans3D(infant_centroid$mu_bs, infant_centroid$mu_rm, infant_centroid$mu_tb, pmat)
        points(pt2d$x, pt2d$y, pch = 17, cex = 1.44, col = "#044328")
    }
    if (!is.null(winner)) {
        pt2d <- trans3D(winner$baseline_stay, winner$reward_modulation, winner$timescale_balance, pmat)
        points(pt2d$x, pt2d$y, pch = 17, cex = 1.44, col = "#F7C548")
    }
    par(xpd = TRUE)
    legend("right", inset = c(-0.15, 0),
        legend = c("Infants", "Average Infant", "Models", "Winner"),
        pch = c(16, 17, 16, 17),
        col = c("forestgreen", "#044328", "darkblue", "#F7C548"),
        pt.cex = c(1.0, 1.44, 1.0, 1.44), bty = "n")
    par(xpd = FALSE)
    dev.off()
    cat("Saved cognitive_space_3d.png\n")

    # 6d. Infant-Only
    png(file.path(PLOT_DIR, "cognitive_space_3d_infants.png"), width = 2400, height = 1800, res = 300)
    op <- par(mar = c(5, 4, 4, 7) + 0.1)
    if (length(x_inf) > 0) {
        scatter3D(x = x_inf, y = y_inf, z = z_inf,
            colvar = NULL, col = "forestgreen",
            pch = 16, cex = 1.0, alpha = 0.2,
            bty = "g", ticktype = "detailed", theta = 45, phi = 0,
            xlab = xlab, ylab = ylab, zlab = zlab,
            xlim = xlim, ylim = ylim, zlim = zlim)
    }
    if (nrow(infant_centroid) > 0) {
        pmat <- getplist()$mat
        pt2d <- trans3D(infant_centroid$mu_bs, infant_centroid$mu_rm, infant_centroid$mu_tb, pmat)
        points(pt2d$x, pt2d$y, pch = 17, cex = 1.44, col = "#044328")
    }
    par(xpd = TRUE)
    legend("right", inset = c(-0.15, 0),
        legend = c("Infants", "Average Infant"), pch = c(16, 17),
        col = c("forestgreen", "#044328"), pt.cex = c(1.0, 1.44), bty = "n")
    par(xpd = FALSE)
    dev.off()
    cat("Saved cognitive_space_3d_infants.png\n")
}

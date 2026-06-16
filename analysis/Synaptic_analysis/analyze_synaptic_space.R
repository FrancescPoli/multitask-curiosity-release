# Population-Level Developmental Analysis
# =======================================

# 0. Load Libraries
if (!require(ggplot2)) {
    install.packages("ggplot2", repos = "http://cran.us.r-project.org")
    library(ggplot2)
}
if (!require(scales)) {
    install.packages("scales", repos = "http://cran.us.r-project.org")
    library(scales)
}
if (!require(mgcv)) {
    install.packages("mgcv", repos = "http://cran.us.r-project.org")
    library(mgcv)
}
if (!require(dplyr)) {
    install.packages("dplyr", repos = "http://cran.us.r-project.org")
    library(dplyr)
}

has_plot3d <- require(plot3D, quietly = TRUE)

# -----------------------------------------------------------------------------
# 1. Load Data
# -----------------------------------------------------------------------------

# A. Human Data (Huttenlocher)
huttenlocher1997 <- data.frame(
    region = rep(c("auditory", "prefrontal"), times = 14),
    CA_days = rep(c(192, 210, 280, 320, 360, 363, 700, 1620, 4700, 5000, 5700, 7300, 11700, 21500), each = 2),
    synapses_100um3 = c(12.2, 3.2, 7.5, 2.2, 29.4, 19.5, 21.0, 11.2, 40.7, 28.3, 54.3, 34.3, 53.0, 37.9, 55.7, 52.4, 24.7, 46.9, 26.0, 39.7, 38.9, 40.0, 24.0, 27.1, 35.0, 40.2, 34.7, 28.7),
    synapses_se = c(1.2, 0.5, 0.6, 0.5, 2.7, 2.2, 2.4, 1.1, 5.4, 0.9, 3.0, 4.7, 5.4, 4.9, 5.4, 6.0, 3.9, 3.1, 4.3, 6.1, 3.5, 3.9, 2.1, 2.5, 2.8, 3.7, 2.7, 2.0)
)
huttenlocher1997$age_years <- huttenlocher1997$CA_days / 365

visual1982 <- data.frame(
    region = "visual",
    study = "Huttenlocher1982",
    age_years = c(0.53, 0.77, 0.94, 0.97, 1.1, 1.44, 1.69, 2.35, 3.77, 11.77, 26.77, 71.77),
    synapses_100um3 = c(12, 25.5, 26.5, 31.5, 51, 57.5, 56.5, 49, 44.5, 35, 35, 33),
    synapses_se = c(1.5, 1.8, 1.9, 2, 3.1, 2.7, 2.7, 1.8, 1.5, 1.7, 2, 1.8)
)

human_data_all <- rbind(
    huttenlocher1997[, c("region", "age_years", "synapses_100um3", "synapses_se")],
    visual1982[, c("region", "age_years", "synapses_100um3", "synapses_se")]
)

human_min <- min(human_data_all$synapses_100um3)
human_max <- max(human_data_all$synapses_100um3)
human_data_all$norm_val <- (human_data_all$synapses_100um3 - human_min) / (human_max - human_min)
human_data_all$log_age <- log10(human_data_all$age_years)
human_data_all$w <- 1 / (human_data_all$synapses_se^2)

human_fit_data <- subset(human_data_all, age_years >= 0.4)
gam_human_full <- gam(norm_val ~ s(log_age, k = 5), data = human_fit_data, weights = w)

ref_grid_log <- seq(log10(0.4), log10(70), length.out = 100)
ref_grid_age <- 10^ref_grid_log
human_pred_ref <- predict(gam_human_full, newdata = data.frame(log_age = ref_grid_log))

get_peak_age <- function(age, vals) age[which.max(vals)]
get_decay_rate <- function(age, vals) {
    peak_idx <- which.max(vals)
    if (peak_idx >= (length(vals) - 2)) {
        return(0)
    }
    slopes <- diff(vals[peak_idx:length(vals)]) / diff(age[peak_idx:length(age)])
    min_slope <- min(slopes, na.rm = TRUE)
    if (min_slope >= 0) {
        return(0)
    }
    return(abs(min_slope))
}
get_peak_fraction <- function(age, vals, age_cap) {
    peak_age <- age[which.max(vals)]
    return(peak_age / age_cap)
}
get_pruning_fraction <- function(age, vals) {
    peak_idx <- which.max(vals)
    peak_val <- vals[peak_idx]
    if (peak_val == 0) return(0)
    final_val <- vals[length(vals)]
    return(max(0, (peak_val - final_val) / peak_val))
}

human_peak_age <- get_peak_age(ref_grid_age, human_pred_ref)
human_decay_rate <- get_decay_rate(ref_grid_age, human_pred_ref)
human_peak_fraction <- human_peak_age / 75
human_pruning_fraction <- get_pruning_fraction(ref_grid_age, human_pred_ref)
cat(sprintf("Human Reference: Peak=%.2fy (%.1f%%), Pruning=%.1f%%, Decay=%.4f\n",
    human_peak_age, human_peak_fraction * 100, human_pruning_fraction * 100, human_decay_rate))

# B. Population Model Data
args <- commandArgs(trailingOnly = TRUE)
model_csv_path <- if (length(args) > 0) args[1] else "analysis/Synaptic_analysis/Data/population_weights.csv"

if (!file.exists(model_csv_path)) {
    if (file.exists(basename(model_csv_path))) {
        model_csv_path <- basename(model_csv_path)
    } else {
        stop(paste("Population CSV not found:", model_csv_path))
    }
}

cat("Loading population data from:", model_csv_path, "\n")
pop_data <- read.csv(model_csv_path)
run_ids <- unique(pop_data$run_id)
cat(sprintf("Found %d models.\n", length(run_ids)))

# -----------------------------------------------------------------------------
# 2. Optimization Loop (Per Model)
# -----------------------------------------------------------------------------
birth_age <- 280 / 365
candidates <- seq(3, 75, by = 3)
results_list <- list()
curves_list <- list()

cat("Processing models...\n")
pb <- txtProgressBar(min = 0, max = length(run_ids), style = 3)

for (i in seq_along(run_ids)) {
    rid <- run_ids[i]
    model_rows <- subset(pop_data, run_id == rid)
    model_rows <- model_rows[!is.na(model_rows$l1_sum), ]
    if (nrow(model_rows) < 5) next
    m_min <- min(model_rows$l1_sum)
    m_max <- max(model_rows$l1_sum)
    if (m_max == m_min) next
    model_rows$norm_val <- (model_rows$l1_sum - m_min) / (m_max - m_min)

    best_dist <- Inf
    best_res <- NULL
    best_curve_data <- NULL

    for (age_cap in candidates) {
        age_span <- age_cap - birth_age
        model_rows$equiv_age <- birth_age + (model_rows$step / max(model_rows$step)) * age_span
        model_rows$log_age <- log10(model_rows$equiv_age)

        tryCatch(
            {
                gam_model <- gam(norm_val ~ s(log_age, k = 5), data = model_rows)
                grid_log <- seq(log10(0.4), log10(age_cap), length.out = 100)
                grid_age <- 10^grid_log
                p_human <- predict(gam_human_full, newdata = data.frame(log_age = grid_log))
                p_model <- predict(gam_model, newdata = data.frame(log_age = grid_log))
                curr_cor <- cor(p_human, p_model)
                curr_peak <- get_peak_age(grid_age, p_model)
                curr_decay <- get_decay_rate(grid_age, p_model)
                curr_peak_frac <- get_peak_fraction(grid_age, p_model, age_cap)
                curr_pruning_frac <- get_pruning_fraction(grid_age, p_model)
                d_r <- (1 - curr_cor)^2
                d_p <- (human_peak_fraction - curr_peak_frac)^2
                d_d <- (human_pruning_fraction - curr_pruning_frac)^2
                dist <- sqrt(d_r + d_p + d_d)

                if (dist < best_dist) {
                    best_dist <- dist
                    params <- model_rows[1, c("l1", "beta", "alpha", "temp", "travel"), drop = FALSE]
                    best_res <- data.frame(
                        run_id = rid, age_cap = age_cap, distance = dist,
                        correlation = curr_cor, peak_age = curr_peak, decay_rate = curr_decay,
                        peak_fraction = curr_peak_frac, pruning_fraction = curr_pruning_frac,
                        l1 = params$l1, beta = params$beta, alpha = params$alpha,
                        temp = params$temp, travel = params$travel
                    )
                    best_curve_data <- data.frame(run_id = rid, age = grid_age, value = p_model)
                }
            },
            error = function(e) {
                return(NULL)
            }
        )
    }

    if (!is.null(best_res)) {
        results_list[[rid]] <- best_res
        curves_list[[rid]] <- best_curve_data
    }
    setTxtProgressBar(pb, i)
}
close(pb)

final_results <- bind_rows(results_list)
all_curves <- bind_rows(curves_list)
write.csv(final_results, "analysis/Synaptic_analysis/Results/synaptic_metrics.csv", row.names = FALSE)
cat("\nSaved synaptic_metrics.csv\n")

# -----------------------------------------------------------------------------
# 3. Visualization: Spaghetti Plot
# -----------------------------------------------------------------------------
cat("Generating population spaghetti plot...\n")
winner_id <- final_results$run_id[which.min(final_results$distance)]
cat("Winner ID:", winner_id, "\n")

human_curve <- data.frame(run_id = "Human", age = ref_grid_age, value = human_pred_ref)

output_dir <- "analysis/comparison_plots/synaptic"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

p_spag <- ggplot() +
    geom_line(
        data = subset(all_curves, run_id != winner_id),
        aes(x = age, y = value, group = run_id, color = "All other models"),
        alpha = 0.03, size = 0.5
    ) +
    geom_line(
        data = subset(all_curves, run_id == winner_id),
        aes(x = age, y = value, color = "Most similar model"),
        alpha = 0.9, size = 1.5
    ) +
    geom_line(
        data = human_curve,
        aes(x = age, y = value, color = "Human"),
        alpha = 1.0, size = 1.5, linetype = "solid"
    ) +
    scale_color_manual(
        name = NULL,
        values = c("Human" = "#28b74e", "Most similar model" = "#5ddcff", "All other models" = "darkblue"),
        breaks = c("Human", "Most similar model", "All other models"),
        guide = guide_legend(
            override.aes = list(alpha = c(1, 1, 0.5), size = c(1.5, 1.5, 1))
        )
    ) +
    scale_x_log10(breaks = c(0.5, 1, 5, 10, 20, 30, 50, 70), labels = label_number(accuracy = 1)) +
    labs(
        x = "Age (Years) [Log Scale]", y = "Normalized Density/Weights"
    ) +
    ylim(-0.8,1.5)+
    theme_classic() +
    theme(legend.position = "right")

ggsave(file.path(output_dir, "population_trajectories.png"), p_spag, width = 6, height = 4, dpi = 300)
cat("Saved population_trajectories.png\n")

# -----------------------------------------------------------------------------
# 4. 3D Plots (Unified Style)
# -----------------------------------------------------------------------------
if (has_plot3d) {
    winner <- final_results[which.min(final_results$distance), ]
    cat(sprintf("Global Winner: %s (Dist=%.4f, Age=%dy)\n", winner$run_id, winner$distance, winner$age_cap))

    xlab <- "\nPeak Age (%)"
    ylab <- "\n\nTotal Pruning (%)"
    zlab <- "\nCorrelation"

    # ---- FULL-SPACE limits (fractions * 100 for % display) ----
    full_xlim <- c(0, 100)
    full_ylim <- c(0, 100)
    full_zlim <- c(-1, 1)

    # Exclude models with pruning > 100% from full-space plot
    full_idx <- which(final_results$pruning_fraction <= 1.0)
    full_data <- final_results[full_idx, ]
    cat(sprintf("Full-space plot: %d of %d models (excluded %d with pruning > 100%%)\n",
        nrow(full_data), nrow(final_results), nrow(final_results) - nrow(full_data)))

    # Jitter all models (5%) for full space
    set.seed(42)
    full_jit_x <- full_data$peak_fraction * 100 + rnorm(nrow(full_data), sd = diff(full_xlim) * 0.05)
    full_jit_y <- full_data$pruning_fraction * 100 + rnorm(nrow(full_data), sd = diff(full_ylim) * 0.05)
    full_jit_z <- full_data$correlation + rnorm(nrow(full_data), sd = diff(full_zlim) * 0.05)

    # ---- ZOOMED limits ----
    zoom_xlim <- c(0, 50)
    zoom_ylim <- c(0, 80)
    zoom_zlim <- c(0.6, 1.0)

    zoom_idx <- which(
        final_results$peak_fraction * 100 >= zoom_xlim[1] & final_results$peak_fraction * 100 <= zoom_xlim[2] &
            final_results$pruning_fraction * 100 >= zoom_ylim[1] & final_results$pruning_fraction * 100 <= zoom_ylim[2] &
            final_results$correlation >= zoom_zlim[1] & final_results$correlation <= zoom_zlim[2]
    )
    zoom_data <- final_results[zoom_idx, ]
    cat(sprintf("Zoomed region: %d of %d models\n", nrow(zoom_data), nrow(final_results)))

    set.seed(42)
    if (nrow(zoom_data) > 0) {
        zoom_jit_x <- zoom_data$peak_fraction * 100 + rnorm(nrow(zoom_data), sd = diff(zoom_xlim) * 0.05)
        zoom_jit_y <- zoom_data$pruning_fraction * 100 + rnorm(nrow(zoom_data), sd = diff(zoom_ylim) * 0.05)
        zoom_jit_z <- zoom_data$correlation + rnorm(nrow(zoom_data), sd = diff(zoom_zlim) * 0.05)
    }

    # Helper: zoomed base scatter (models only)
    plot_base_zoomed <- function() {
        if (nrow(zoom_data) > 0) {
            scatter3D(x = zoom_jit_x, y = zoom_jit_y, z = zoom_jit_z,
                colvar = NULL, col = "darkblue",
                pch = 16, cex = 1.0, alpha = 0.16,
                bty = "g", ticktype = "detailed", theta = 45, phi = 0,
                xlab = xlab, ylab = ylab, zlab = zlab,
                xlim = zoom_xlim, ylim = zoom_ylim, zlim = zoom_zlim)
        } else {
            scatter3D(x = zoom_xlim[1], y = zoom_ylim[1], z = zoom_zlim[1],
                colvar = NULL, col = "white", pch = ".",
                bty = "g", ticktype = "detailed", theta = 45, phi = 0,
                xlab = xlab, ylab = ylab, zlab = zlab,
                xlim = zoom_xlim, ylim = zoom_ylim, zlim = zoom_zlim)
        }
    }

    # ==================================================================
    # 4a. population_3d.png â€” FULL SPACE (all models + human + winner)
    # ==================================================================
    png(file.path(output_dir, "population_3d.png"), width = 2400, height = 1800, res = 300)
    op <- par(mar = c(5, 4, 4, 7) + 0.1)
    scatter3D(x = full_jit_x, y = full_jit_y, z = full_jit_z,
        colvar = NULL, col = "darkblue",
        pch = 16, cex = 1.0, alpha = 0.16,
        bty = "g", ticktype = "detailed", theta = 45, phi = 0,
        xlab = xlab, ylab = ylab, zlab = zlab,
        xlim = full_xlim, ylim = full_ylim, zlim = full_zlim)
    pmat <- getplist()$mat
    pt2d <- trans3D(human_peak_fraction * 100, human_pruning_fraction * 100, 1.0, pmat)
    points(pt2d$x, pt2d$y, pch = 17, cex = 1.44, col = "#044328")
    pt2d <- trans3D(winner$peak_fraction * 100, winner$pruning_fraction * 100, winner$correlation, pmat)
    points(pt2d$x, pt2d$y, pch = 17, cex = 1.44, col = "#F7C548")
    par(xpd = TRUE)
    legend("right", inset = c(-0.15, 0),
        legend = c("Human", "Model (Population)", "Winning Model"),
        pch = c(17, 16, 17),
        col = c("#044328", "darkblue", "#F7C548"),
        pt.cex = c(1.44, 1.0, 1.44), bty = "n")
    par(xpd = FALSE)
    dev.off()
    cat("Saved population_3d.png\n")

    # ==================================================================
    # 4b. population_3d_zoomed.png â€” ZOOMED SPACE (models + human + winner)
    # ==================================================================
    png(file.path(output_dir, "population_3d_zoomed.png"), width = 2400, height = 1800, res = 300)
    op <- par(mar = c(5, 4, 4, 7) + 0.1)
    plot_base_zoomed()
    pmat <- getplist()$mat
    pt2d <- trans3D(human_peak_fraction * 100, human_pruning_fraction * 100, 1.0, pmat)
    points(pt2d$x, pt2d$y, pch = 17, cex = 1.44, col = "#044328")
    pt2d <- trans3D(winner$peak_fraction * 100, winner$pruning_fraction * 100, winner$correlation, pmat)
    points(pt2d$x, pt2d$y, pch = 17, cex = 1.44, col = "#F7C548")
    par(xpd = TRUE)
    legend("right", inset = c(-0.15, 0),
        legend = c("Human", "Model (Population)", "Winning Model"),
        pch = c(17, 16, 17),
        col = c("#044328", "darkblue", "#F7C548"),
        pt.cex = c(1.44, 1.0, 1.44), bty = "n")
    par(xpd = FALSE)
    dev.off()
    cat("Saved population_3d_zoomed.png\n")

    # ==================================================================
    # 4c. Models-Only (ZOOMED)
    # ==================================================================
    png(file.path(output_dir, "population_3d_models.png"), width = 2400, height = 1800, res = 300)
    op <- par(mar = c(5, 4, 4, 7) + 0.1)
    plot_base_zoomed()
    par(xpd = TRUE)
    legend("right", inset = c(-0.15, 0),
        legend = "Model (Population)", pch = 16, col = "darkblue", pt.cex = 1.0, bty = "n")
    par(xpd = FALSE)
    dev.off()
    cat("Saved population_3d_models.png\n")

    # ==================================================================
    # 4d. Models + Human (ZOOMED)
    # ==================================================================
    png(file.path(output_dir, "population_3d_models_human.png"), width = 2400, height = 1800, res = 300)
    op <- par(mar = c(5, 4, 4, 7) + 0.1)
    plot_base_zoomed()
    pmat <- getplist()$mat
    pt2d <- trans3D(human_peak_fraction * 100, human_pruning_fraction * 100, 1.0, pmat)
    points(pt2d$x, pt2d$y, pch = 17, cex = 1.44, col = "#044328")
    par(xpd = TRUE)
    legend("right", inset = c(-0.15, 0),
        legend = c("Human", "Model (Population)"), pch = c(17, 16),
        col = c("#044328", "darkblue"), pt.cex = c(1.44, 1.0), bty = "n")
    par(xpd = FALSE)
    dev.off()
    cat("Saved population_3d_models_human.png\n")

    # ==================================================================
    # 4e. Human-Only (ZOOMED)
    # ==================================================================
    png(file.path(output_dir, "population_3d_human.png"), width = 2400, height = 1800, res = 300)
    op <- par(mar = c(5, 4, 4, 7) + 0.1)
    scatter3D(x = human_peak_fraction * 100, y = human_pruning_fraction * 100, z = 1.0,
        colvar = NULL, col = "white", pch = ".",
        bty = "g", ticktype = "detailed", theta = 45, phi = 0,
        xlab = xlab, ylab = ylab, zlab = zlab,
        xlim = zoom_xlim, ylim = zoom_ylim, zlim = zoom_zlim)
    pmat <- getplist()$mat
    pt2d <- trans3D(human_peak_fraction * 100, human_pruning_fraction * 100, 1.0, pmat)
    points(pt2d$x, pt2d$y, pch = 17, cex = 1.44, col = "#044328")
    par(xpd = TRUE)
    legend("right", inset = c(-0.15, 0),
        legend = "Human", pch = 17, col = "#044328", pt.cex = 1.44, bty = "n")
    par(xpd = FALSE)
    dev.off()
    cat("Saved population_3d_human.png\n")

    # ==================================================================
    # 4f. Empty (ZOOMED)
    # ==================================================================
    png(file.path(output_dir, "population_3d_empty.png"), width = 2400, height = 1800, res = 300)
    op <- par(mar = c(5, 4, 4, 7) + 0.1)
    scatter3D(x = zoom_xlim[1], y = zoom_ylim[1], z = zoom_zlim[1],
        colvar = NULL, col = "white", pch = ".",
        bty = "g", ticktype = "detailed", theta = 45, phi = 0,
        xlab = xlab, ylab = ylab, zlab = zlab,
        xlim = zoom_xlim, ylim = zoom_ylim, zlim = zoom_zlim)
    dev.off()
    cat("Saved population_3d_empty.png\n")
}


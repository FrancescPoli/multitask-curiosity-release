suppressPackageStartupMessages({
    library(ggplot2)
    library(dplyr)
    library(tidyr)
})

# ── Data ──────────────────────────────────────────────────────────────────────
# x-axis: raw reward r = Δloss (objective learning progress per step).
# The value function v(r) is applied inside the sigmoid, which is the only
# source of asymmetry between gains and losses:
#   v(r) = tanh(r/scale)                if r ≥ 0   (gain)
#   v(r) = lambda * tanh(r/scale)       if r < 0   (loss, weighted 2.25×)
# P(stay) = σ(v(r)/τ). Loss side of each curve drops ~λ times steeper than
# the gain side rises.
scale       <- 0.1
loss_weight <- 2.25
value_fn    <- function(r) ifelse(r >= 0, tanh(r / scale), loss_weight * tanh(r / scale))

x        <- seq(-0.3, 0.3, length.out = 800)
tau_vals <- c(0.005, 0.3, 1.0)
tau_lab  <- c("τ = 0.005", "τ = 0.3", "τ = 1.0")

df <- do.call(rbind, lapply(seq_along(tau_vals), function(i) {
    # Highest tau -> visually flat (random-switching regime)
    p_vals <- if (tau_vals[i] >= 1.0) rep(0.5, length(x))
              else 1 / (1 + exp(-value_fn(x) / tau_vals[i]))
    data.frame(
        x   = x,
        p   = p_vals,
        tau = factor(tau_lab[i], levels = tau_lab)
    )
}))

# Coral-green palette: dark → light (deterministic → random)
tau_colors <- setNames(c("#1f5a3e", "#4ea37a", "#a8d8b9"), tau_lab)

# ── Plot ──────────────────────────────────────────────────────────────────────
p <- ggplot(df, aes(x = x, y = p, colour = tau)) +
    geom_hline(yintercept = 0.5, colour = "grey80", linewidth = 0.4, linetype = "dashed") +
    geom_vline(xintercept = 0,   colour = "grey80", linewidth = 0.4, linetype = "dashed") +
    geom_line(linewidth = 1.1) +
    scale_colour_manual(values = tau_colors, name = NULL) +
    scale_x_continuous(
        name   = "Relative Value",
        breaks = c(-0.3, -0.15, 0, 0.15, 0.3),
        labels = c("-0.3", "-0.15", "0", "+0.15", "+0.3"),
        limits = c(-0.33, 0.33),
        expand = c(0, 0)
    ) +
    scale_y_continuous(
        name   = "P(stay on task)",
        limits = c(0, 1),
        breaks = c(0, 0.5, 1),
        expand = c(0.02, 0)
    ) +
    theme_classic(base_size = 13) +
    theme(
        axis.line        = element_line(colour = "black", linewidth = 0.7),
        axis.ticks       = element_line(colour = "black", linewidth = 0.5),
        axis.text        = element_text(colour = "black", size = 11),
        axis.title       = element_text(colour = "black", size = 12, face = "bold"),
        legend.position  = "none",
        plot.background  = element_rect(fill = "white", colour = NA),
        panel.background = element_rect(fill = "white", colour = NA),
    )

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir <- file.path(getwd(), "analysis", "Final_analyses_and_plots", "Figures")

ggsave(file.path(out_dir, "pub_tau_plot.png"), p, width = 3.0, height = 2.6,
       dpi = 300)
cat("Saved: pub_tau_plot.png\n")

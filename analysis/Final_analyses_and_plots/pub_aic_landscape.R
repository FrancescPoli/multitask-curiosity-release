suppressPackageStartupMessages({
    library(ggplot2)
})

# Publication version of the zoomed AIC landscape heatmap.
# Same data as analysis/Cognitive_analysis/plot_infant_mvt.R, but stripped
# to figure-ready styling (no title, compact axis labels, compact size).

RESULTS_DIR <- "analysis/Cognitive_analysis/Results"
OUT_DIR     <- "analysis/Final_analyses_and_plots/Figures"

grid_file <- file.path(RESULTS_DIR, "infant_group_grid_aic.rds")
grid      <- readRDS(grid_file)

heat_df <- expand.grid(
    alpha_hl = grid$alpha_hl,
    beta_hl  = grid$beta_hl
)
heat_df$aic  <- as.vector(grid$aic)
heat_df$dAIC <- heat_df$aic - min(heat_df$aic, na.rm = TRUE)

# Zoom window: alpha_hl ≤ 6, beta_hl ≤ 10
zoom_df <- subset(heat_df, alpha_hl <= 6 & beta_hl <= 10)
zoom_df$dAIC_zoom <- zoom_df$aic - min(zoom_df$aic, na.rm = TRUE)
best_zoom <- zoom_df[which.min(zoom_df$aic), ]

p_heat <- ggplot(zoom_df, aes(x = alpha_hl, y = beta_hl, fill = dAIC_zoom)) +
    geom_tile() +
    geom_point(data = best_zoom,
               aes(x = alpha_hl, y = beta_hl),
               colour = "red", shape = 4, size = 2.6, stroke = 1.1,
               inherit.aes = FALSE) +
    scale_fill_viridis_c(option = "magma", direction = -1,
                         name = "ΔAIC",
                         trans = "sqrt",
                         na.value = "grey90") +
    scale_x_continuous(breaks = sort(unique(zoom_df$alpha_hl)),
                       expand = c(0, 0)) +
    scale_y_continuous(breaks = sort(unique(zoom_df$beta_hl)),
                       expand = c(0, 0)) +
    labs(
        x = "Alpha half-life (trials)",
        y = "Beta half-life (trials)"
    ) +
    theme_classic(base_size = 9) +
    theme(
        axis.line        = element_line(colour = "black", linewidth = 0.5),
        axis.ticks       = element_line(colour = "black", linewidth = 0.4),
        axis.text        = element_text(colour = "black", size = 7),
        axis.title       = element_text(colour = "black", size = 8, face = "bold"),
        legend.title     = element_text(size = 7),
        legend.text      = element_text(size = 6),
        legend.key.width = unit(0.25, "cm"),
        legend.key.height = unit(0.35, "cm"),
        legend.margin    = margin(0, 0, 0, 0),
        legend.box.margin = margin(0, 0, 0, -4),
        plot.background  = element_rect(fill = "white", colour = NA),
        panel.background = element_rect(fill = "white", colour = NA),
    )

ggsave(file.path(OUT_DIR, "pub_aic_landscape.png"), p_heat,
       width = 1.7, height = 2.1, dpi = 400)
cat("Saved: pub_aic_landscape.pdf / .png\n")

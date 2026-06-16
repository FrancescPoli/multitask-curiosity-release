suppressPackageStartupMessages({
    library(dplyr)
    library(readr)
    library(ggplot2)
    library(tidyr)
    library(grid)
})

# Disable scientific notation on axes (1000 instead of 1e+3).
options(scipen = 999)

# Publication-quality foraging traces (rho/P and per-task EMA of negative loss)
# for the top-N H-100 cohort models (by norm_dist_cognitive).
#
# Outputs (PNG only) per run:
#   rho_P_slice_*   – P per task + rho baseline over SLICE window, TRAVEL
#                     rows removed; x is compact trial index; orange diamond
#                     at the last step of every task block before a switch.
#   perf_full_*     – whole-timeline per-task EMA of (−loss).
#   perf_30pct_*    – first 30% of steps.
#   perf_slice_*    – same SLICE window, TRAVEL-compacted x-axis.

# ── Tweakables ───────────────────────────────────────────────────────────────
SLICE      <- c(1000, 1800)     # [min_step, max_step] for slice plots
EMA_ALPHA  <- 0.1
TOP_N      <- 1
PCT_30     <- 0.30              # fraction of timeline for perf_30pct
SWITCH_COL <- "#E26714"         # orange diamond colour

PLOT_W <- 3.6                  # plot panel width (inches, excluding legend)
PLOT_H <- 1.8                  # plot panel height (inches)

SWEEP_DIR <- "Z:/fp02/logs/sweep/forage_v5.1/forage_v6"
OUT_DIR   <- "analysis/Final_analyses_and_plots/Figures/traceplots"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Palette: 20 colours. First 10 = infant example_trace (matplotlib tab10),
# then 10 more from tab20b so slice plots (typically ≤10 tasks) stay inside
# the first-10 block naturally. ───────────────────────────────────────────────
SEQ_PALETTE_20 <- c(
    # tab10 — same as analysis/Cognitive_analysis/plot_infant_mvt.R
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    # extra 10 from tab20b (darker/saturated, distinct from tab10)
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#5254a3", "#8ca252", "#bd9e39", "#ad494a", "#a55194"
)
task_palette <- function(tasks) {
    setNames(rep_len(SEQ_PALETTE_20, length(tasks)), tasks)
}

# ── Top-N H-100 runs by norm_dist_cognitive ──────────────────────────────────
cat(sprintf("Selecting top-%d H-100 runs...\n", TOP_N))
comp <- read_csv("analysis/compositional_dataset.csv",
                 show_col_types = FALSE, guess_max = 50000)
top_runs <- comp %>%
    filter(cohort == "H-100", !is.na(norm_dist_cognitive)) %>%
    arrange(norm_dist_cognitive) %>%
    slice_head(n = TOP_N) %>%
    select(run_id, norm_dist_cognitive)

# ── Helpers ──────────────────────────────────────────────────────────────────
strip_poli <- function(x) sub("^poli\\.", "", x)

add_segment_id <- function(df, key_col) {
    df <- df %>% arrange(step)
    key <- df[[key_col]]
    df$segment_id <- cumsum(key != lag(key, default = "___NA___"))
    df
}

ema_by_task <- function(d, alpha) {
    seen <- list(); out <- numeric(nrow(d))
    task <- d$task; raw <- d$perf_raw
    for (i in seq_len(nrow(d))) {
        tk <- task[i]
        prev <- seen[[tk]]; if (is.null(prev)) prev <- raw[i]
        new  <- (1 - alpha) * prev + alpha * raw[i]
        seen[[tk]] <- new
        out[i] <- new
    }
    out
}

# Force the panel area to exactly (w × h) inches, let legend/axis add space.
save_fixed_panel <- function(plot, path, w_in = PLOT_W, h_in = PLOT_H, dpi = 300) {
    g <- ggplot2::ggplotGrob(plot)
    panel_cells <- g$layout[grepl("^panel", g$layout$name), ]
    g$widths [unique(panel_cells$l)] <- unit(w_in, "in")
    g$heights[unique(panel_cells$t)] <- unit(h_in, "in")

    # Open a throwaway png device so grid can resolve null/grob units to inches.
    # png() avoids the pdf-encoding ('WinAnsi.enc') issue seen on some Windows setups.
    tmp <- tempfile(fileext = ".png")
    grDevices::png(tmp, width = 20, height = 20, units = "in", res = 72)
    total_w <- grid::convertWidth (sum(g$widths ), "in", valueOnly = TRUE)
    total_h <- grid::convertHeight(sum(g$heights), "in", valueOnly = TRUE)
    grDevices::dev.off(); unlink(tmp)

    ggplot2::ggsave(path, g, width = total_w, height = total_h,
                    dpi = dpi, limitsize = FALSE)
}

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

# ── rho/P slice plot ─────────────────────────────────────────────────────────
# TRAVEL rows dropped, x recomputed as compact trial index.
# Orange diamond at the last step of every task block before a switch.
make_rho_P_slice <- function(trace_df) {
    d <- trace_df %>%
        filter(step >= SLICE[1], step <= SLICE[2]) %>%
        arrange(step) %>%
        filter(task_plot != "TRAVEL") %>%
        mutate(step_compact = row_number())
    if (nrow(d) == 0) return(ggplot())

    d <- add_segment_id(d, "task_plot")

    n  <- nrow(d)
    sw <- d[c(d$task_plot[-n] != d$task_plot[-1], FALSE), ]

    tasks_here <- sort(unique(d$task_plot))
    pal        <- task_palette(tasks_here)

    ggplot(d, aes(x = step_compact)) +
        geom_line(aes(y = P, colour = task_plot, group = segment_id),
                  linewidth = 0.4, na.rm = TRUE) +
        geom_line(aes(y = rho), colour = "black",
                  linetype = "dashed", linewidth = 0.7, na.rm = TRUE) +
        geom_point(data = sw, aes(x = step_compact, y = P),
                   colour = SWITCH_COL, fill = SWITCH_COL,
                   shape = 18, size = 2.4) +
        scale_colour_manual(values = pal, name = NULL,
                            breaks = tasks_here, labels = strip_poli(tasks_here)) +
        scale_x_continuous(name = "Trial (step)",
                           expand = expansion(mult = c(0.01, 0.01))) +
        scale_y_continuous(name = "Reward (Learning Progress)") +
        base_theme
}

# ── Performance plots (full / 30% / slice) ───────────────────────────────────
# Signal plotted = EMA of (−loss), which is what the reward Δ-operates on.
# When `compact = TRUE` x collapses to a gap-free trial index.
make_perf_plot <- function(perf_df, step_range = NULL, compact = FALSE) {
    if (!is.null(step_range))
        perf_df <- perf_df %>% filter(step >= step_range[1], step <= step_range[2])
    perf_df <- perf_df %>% arrange(step)
    if (compact) perf_df <- perf_df %>% mutate(step = row_number())
    d <- add_segment_id(perf_df, "task")
    tasks_here <- sort(unique(d$task))
    pal        <- task_palette(tasks_here)

    p <- ggplot(d, aes(x = step, y = ema, colour = task, group = segment_id)) +
        geom_line(linewidth = 0.4, na.rm = TRUE)

    if (compact && nrow(d) >= 2) {
        n  <- nrow(d)
        sw <- d[c(d$task[-n] != d$task[-1], FALSE), ]
        if (nrow(sw) > 0) {
            p <- p + geom_point(data = sw, aes(x = step, y = ema),
                                colour = SWITCH_COL, fill = SWITCH_COL,
                                shape = 18, size = 2.4,
                                inherit.aes = FALSE)
        }
    }

    p + scale_colour_manual(values = pal, name = NULL,
                            breaks = tasks_here, labels = strip_poli(tasks_here)) +
        scale_x_continuous(name = "Trial (step)",
                           expand = expansion(mult = c(0.01, 0.01))) +
        scale_y_continuous(name = "Negative CE Loss") +
        base_theme
}

# ── Per-run pipeline ─────────────────────────────────────────────────────────
process_run <- function(rank_idx, run_id) {
    csv <- file.path(SWEEP_DIR, run_id, "run_curriculum.csv")
    if (!file.exists(csv)) {
        cat(sprintf("[%02d] MISSING %s\n", rank_idx, csv)); return(invisible())
    }
    cat(sprintf("[%02d] %s\n", rank_idx, run_id))

    df <- suppressWarnings(read_csv(csv, show_col_types = FALSE, guess_max = 50000)) %>%
        filter(regime == "foraging") %>%
        mutate(
            step      = as.integer(step),
            rho       = suppressWarnings(as.numeric(rho)),
            P         = suppressWarnings(as.numeric(P)),
            loss      = suppressWarnings(as.numeric(loss)),
            task      = as.character(task),
            mode      = as.character(mode),
            task_plot = ifelse(mode == "travel", "TRAVEL", task)
        )

    trace_df <- df %>% filter(!is.na(rho), !is.na(P)) %>%
        select(step, rho, P, task_plot)

    # Perf signal = −loss (what the reward is computed from; see foraging.py).
    perf_df <- df %>% filter(mode != "travel", !is.na(loss)) %>%
        mutate(perf_raw = -loss) %>% arrange(step)
    perf_df$ema <- ema_by_task(perf_df, EMA_ALPHA)

    step_max <- max(perf_df$step, na.rm = TRUE)
    step_30  <- c(min(perf_df$step, na.rm = TRUE), round(step_max * PCT_30))

    p_rho_slice  <- make_rho_P_slice(trace_df)
    p_perf_full  <- make_perf_plot(perf_df)
    p_perf_30    <- make_perf_plot(perf_df, step_range = step_30)
    p_perf_slice <- make_perf_plot(perf_df, step_range = SLICE, compact = TRUE)

    tag <- sprintf("%02d_%s", rank_idx, run_id)

    save_fixed_panel(p_rho_slice,  file.path(OUT_DIR, paste0("rho_P_slice_",  tag, ".png")))
    save_fixed_panel(p_perf_full,  file.path(OUT_DIR, paste0("perf_full_",    tag, ".png")))
    save_fixed_panel(p_perf_30,    file.path(OUT_DIR, paste0("perf_30pct_",   tag, ".png")))
    save_fixed_panel(p_perf_slice, file.path(OUT_DIR, paste0("perf_slice_",   tag, ".png")))
}

for (i in seq_len(nrow(top_runs))) process_run(i, top_runs$run_id[i])
cat("Done. Saved plots to:", OUT_DIR, "\n")

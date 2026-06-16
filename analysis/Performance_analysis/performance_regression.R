# Performance ↔ Human-Likeness Regression Analysis
# ================================================
#
# Two sections:
#   Section A: Predict distances (human-likeness) FROM performance metrics only
#   Section B: Predict performance metrics FROM model parameters
#
# Uses same stepwise AIC approach as Forest_analysis/interaction_analysis.R

suppressPackageStartupMessages({
    library(parameters)
    library(modelbased)
    library(see)
    library(ggplot2)
    library(dplyr)
})

# ── Paths ────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = FALSE)
project_root <- getwd()
data_path <- file.path(project_root, "analysis", "grand_unified_metrics_v2.csv")

if (!file.exists(data_path)) {
    script_path <- normalizePath(dirname(sub("^--file=", "", args[grep("^--file=", args)])))
    if (length(script_path) > 0) {
        project_root <- normalizePath(file.path(script_path, "..", ".."))
        data_path <- file.path(project_root, "analysis", "grand_unified_metrics_v2.csv")
    }
}
if (!file.exists(data_path)) stop("Data file not found at: ", data_path)

output_dir <- file.path(project_root, "analysis", "Performance_analysis", "Results")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# ── Load & Clean ─────────────────────────────────────────────────────
cat("Loading data from:", data_path, "\n")
df <- read.csv(data_path)

# Drop rows missing performance data
df <- df[!is.na(df$mean_accuracy), ]
cat("Rows with performance data:", nrow(df), "\n")

# Preprocessing: Factors for model parameters (Section B)
df$beta_f <- as.factor(df$beta)
df$alpha_f <- as.factor(df$alpha)
df$temp_f <- as.factor(df$temp)
df$reg_value_f <- as.factor(format(df$reg_value, scientific = FALSE))
df$reg_type_f <- as.factor(df$reg_type)
df$travel_f <- as.factor(df$travel)

# ── Helper: Run stepwise AIC + plot top terms ────────────────────────
run_stepwise <- function(df, target, predictors_formula, output_dir, prefix) {

    cat("\n=============================================\n")
    cat("Analyzing Target:", target, "\n")
    cat("Predictors:", deparse(predictors_formula), "\n")
    cat("=============================================\n")

    f_full <- as.formula(paste(target, "~", predictors_formula))

    # Fit and step
    m_full <- lm(f_full, data = df)
    m_best <- step(m_full, direction = "both", trace = 0)

    cat("Best Model Formula:\n")
    print(formula(m_best))

    s <- summary(m_best)
    cat("R-squared:", round(s$r.squared, 4), "\n")
    cat("Adj R-squared:", round(s$adj.r.squared, 4), "\n")

    # Save summary to text
    txt_file <- file.path(output_dir, paste0(prefix, "_", target, "_summary.txt"))
    sink(txt_file)
    cat("Target:", target, "\n")
    cat("Prefix:", prefix, "\n\n")
    cat("Best Model Formula:\n")
    print(formula(m_best))
    cat("\n")
    print(s)
    sink()
    cat("Saved summary:", txt_file, "\n")

    # Check null
    coeffs <- coef(m_best)
    if (length(coeffs) == 1 && names(coeffs)[1] == "(Intercept)") {
        cat("Winning model is Null — no predictors retained.\n")
        return(invisible(NULL))
    }

    # Extract terms and filter to significant ones (ANOVA p < 0.05)
    terms_in_model <- attr(terms(m_best), "term.labels")
    anova_table <- anova(m_best)
    sig_terms <- rownames(anova_table)[anova_table[["Pr(>F)"]] < 0.05]
    sig_terms <- sig_terms[sig_terms %in% terms_in_model]

    term_orders <- sapply(sig_terms, function(x) length(unlist(strsplit(x, ":"))))
    sorted_terms <- sig_terms[order(term_orders, decreasing = TRUE)]

    cat("All terms in Best Model:\n")
    print(terms_in_model)
    cat("Significant terms (p < 0.05):\n")
    print(sorted_terms)

    # Plot significant terms (max 5)
    plotted <- 0
    for (term in sorted_terms) {
        if (plotted >= 5) break

        vars <- unlist(strsplit(term, ":"))
        order <- length(vars)

        tryCatch({
            trends <- modelbased::estimate_means(m_best, by = vars)
            p <- plot(trends) +
                ggplot2::labs(
                    title = paste0(order, "-Way Effect: ", term),
                    subtitle = paste0("Target: ", target,
                                     " | R²=", round(s$r.squared, 3))
                ) +
                ggplot2::theme_minimal() +
                ggplot2::theme(axis.text.x = element_text(angle = 45, hjust = 1))

            safe_term <- gsub(":", "_", term)
            fname <- file.path(output_dir, paste0(prefix, "_", target, "_", safe_term, ".png"))
            ggplot2::ggsave(fname, plot = p, width = 10, height = 7)
            cat("Saved plot:", fname, "\n")
            plotted <- plotted + 1
        }, error = function(e) {
            cat("Error plotting", term, ":", e$message, "\n")
        })
    }

    return(invisible(m_best))
}

# =====================================================================
# SECTION A: Predict distances FROM performance metrics
# =====================================================================
cat("\n\n########################################################\n")
cat("# SECTION A: Distance ~ Performance Metrics            #\n")
cat("########################################################\n")

dist_targets <- c("norm_dist_synaptic", "norm_dist_network",
                   "norm_dist_cognitive", "grand_distance")

# Performance predictors: continuous, so use 2-way interactions
perf_rhs <- "(mean_accuracy + fraction_solved + mean_speed + switch_rate + entropy)^2"

for (target in dist_targets) {
    run_stepwise(df, target, perf_rhs, output_dir, "perf_predicts_dist")
}

# =====================================================================
# SECTION B: Predict performance FROM model parameters
# =====================================================================
cat("\n\n########################################################\n")
cat("# SECTION B: Performance ~ Model Parameters            #\n")
cat("########################################################\n")

perf_targets <- c("mean_accuracy", "fraction_solved", "mean_speed",
                   "switch_rate", "entropy")

param_rhs <- "(beta_f + alpha_f + temp_f + reg_type_f + reg_value_f + travel_f)^2"

for (target in perf_targets) {
    run_stepwise(df, target, param_rhs, output_dir, "params_predict_perf")
}

# =====================================================================
# SECTION C: Predict distances FROM model parameters (including travel)
# =====================================================================
cat("\n\n########################################################\n")
cat("# SECTION C: Distance ~ Model Parameters (w/ travel)   #\n")
cat("########################################################\n")

param_dist_rhs <- "(beta_f + alpha_f + temp_f + reg_type_f + reg_value_f + travel_f)^2"

for (target in dist_targets) {
    run_stepwise(df, target, param_dist_rhs, output_dir, "params_predict_dist")
}

cat("\n\nPerformance Regression Analysis Complete.\n")

# analysis/fit_mvt_infant.R
# Infer latent cognitive parameters (MVT) from infant foraging data
# Method: Grid Search MLE for memory parameters (alpha, beta) + GLMER for decision parameters (T, eps)

library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)
library(lme4)

# --- Configuration ---
DATA_DIR <- "analysis/Cognitive_analysis/Data/Infant_Data"
OUTPUT_FILE <- "analysis/Cognitive_analysis/Results/infant_mvt_params.csv"

# Grid for memory half-lives (in trials)
# alpha_hl in 1..10 (local), beta_hl in 1..20 (global).
# Constraint: beta_hl >= alpha_hl (MIN_RATIO=1) — includes alpha==beta diagonal.
HALF_LIVES_P   <- seq(1, 10, by = 1)
HALF_LIVES_RHO <- seq(1, 20, by = 1)

# Constrain global rate to be at least MIN_RATIO x slower than local memory
MIN_RATIO <- 1

# Helper: Convert half-life to alpha
alpha_from_hl <- function(hl) {
    log(2) / hl
}

# Helper: EMA Vectorized
ema_vec <- function(r, alpha, init = 0) {
    out <- numeric(length(r))
    x <- init
    if (length(r) > 0) {
        for (i in seq_along(r)) {
            x <- (1 - alpha) * x + alpha * r[i]
            out[i] <- x
        }
    }
    return(out)
}

# --- 1. Load harmonized, anonymized data ---
# The raw per-participant files are private. make_infant_release.R harmonizes
# them, strips identifiers and unused measures, and writes the shareable
# infant_foraging_release.csv loaded here, so this analysis is reproducible from
# the released data alone. The anonymized integer subj_id is used as the
# subject grouping variable (subj_global).
cat("Loading harmonized infant data...\n")
RELEASE_FILE <- file.path(DATA_DIR, "infant_foraging_release.csv")
dat_all <- read.csv(RELEASE_FILE) %>%
    mutate(subj_global = as.character(subj_id)) %>%
    arrange(subj_global, nseq, ntrialseq)

cat("Total Data Points:", nrow(dat_all), "\n")
cat("Total Subjects:", n_distinct(dat_all$subj_global), "\n")

# Persist harmonized data for downstream plotting
saveRDS(dat_all, "analysis/Cognitive_analysis/Results/infant_dat_all.rds")


# --- 2. Grid Search MLE Function (Individual Fit still uses GLM for speed) ---

fit_subject_mvt <- function(df_subj) {
    best_aic <- Inf
    best_params <- list(alpha = NA, beta = NA, T = NA, eps = NA, beta0 = NA, beta1 = NA)

    # Pre-compute alphas
    alphas <- alpha_from_hl(HALF_LIVES_P)
    betas <- alpha_from_hl(HALF_LIVES_RHO)

    # Grid Search — constrain beta_hl >= MIN_RATIO * alpha_hl (global slower than local)
    for (i_a in seq_along(alphas)) {
        a <- alphas[i_a]
        a_hl <- HALF_LIVES_P[i_a]
        for (i_b in seq_along(betas)) {
            b <- betas[i_b]
            b_hl <- HALF_LIVES_RHO[i_b]
            if (b_hl < MIN_RATIO * a_hl) next
            # Compute Signals
            P_t <- ema_vec(df_subj$D, a)
            rho_t <- ema_vec(df_subj$D, b)

            # Predictor: X = P - rho (Positive X means Stay, Negative X means Leave)
            X <- P_t - rho_t

            # Fit GLM
            model <- tryCatch(
                {
                    glm(df_subj$event ~ X, family = binomial(link = "logit"))
                },
                error = function(e) NULL
            )

            if (!is.null(model)) {
                current_aic <- AIC(model)

                if (current_aic < best_aic) {
                    best_aic <- current_aic
                    coefs <- coef(model)
                    b0 <- coefs[1]
                    b1 <- coefs[2]

                    # Recover T, eps
                    T_val <- -1.0 / b1
                    eps_val <- b0 * T_val

                    best_params <- list(
                        alpha_hl = log(2) / a,
                        beta_hl = log(2) / b,
                        T = T_val,
                        eps = eps_val,
                        beta0 = b0,
                        beta1 = b1,
                        aic = current_aic
                    )
                }
            }
        }
    }
    return(best_params)
}

# --- 3. Run Fitting Loop ---
if (file.exists(OUTPUT_FILE)) {
    cat("Subject-level params already exist. Loading from:", OUTPUT_FILE, "\n")
    final_df <- read_csv(OUTPUT_FILE, show_col_types = FALSE)
} else {
    cat("Fitting MVT parameters per subject...\n")

    results_list <- list()
    subjects <- unique(dat_all$subj_global)

    # Loop with progress
    pb <- txtProgressBar(min = 0, max = length(subjects), style = 3)

    for (i in seq_along(subjects)) {
        sub_id <- subjects[i]
        df_s <- dat_all %>% filter(subj_global == sub_id)

        # Skip subjects with too few events
        if (sum(df_s$event) < 5 || nrow(df_s) < 10) {
            next
        }

        params <- fit_subject_mvt(df_s)

        # also compute global stats for subject to get relative metrics
        # Re-compute best signals
        a_best <- alpha_from_hl(params$alpha_hl)
        b_best <- alpha_from_hl(params$beta_hl)
        P_opt <- ema_vec(df_s$D, a_best)
        rho_opt <- ema_vec(df_s$D, b_best)
        signal_sd <- sd(P_opt - rho_opt, na.rm = TRUE)

        results_list[[i]] <- data.frame(
            subj_global = sub_id,
            dataset = df_s$dataset[1],
            alpha_hl = params$alpha_hl,
            beta_hl = params$beta_hl,
            T = params$T,
            eps = params$eps,
            beta0 = params$beta0,
            beta1 = params$beta1,
            aic = params$aic,
            signal_sd = signal_sd
        )

        setTxtProgressBar(pb, i)
    }
    close(pb)

    final_df <- bind_rows(results_list)
}

# --- 4. Post-Processing & Saving ---

# Calculate Dimensionless Metrics
final_df <- final_df %>%
    mutate(
        # 1. Timescale Ratio
        tau_ratio = beta_hl / alpha_hl,

        # 2. Relative Noise (Inverse SNR)
        # R_noise = T / sigma
        R_noise = abs(T) / signal_sd,

        # 3. Stickiness (Beta0)
        # Just beta0
        stickiness = beta0,

        # Flag invalid models (where P-gain leads to Leaving)
        valid_mvt = beta1 < 0
    )

write.csv(final_df, OUTPUT_FILE, row.names = FALSE)
cat("\nResults saved to:", OUTPUT_FILE, "\n")
cat("Valid MVT fits (beta1 < 0):", mean(final_df$valid_mvt, na.rm = TRUE) * 100, "%\n")

# Summary
print(summary(final_df))

# --- 5. Group Level Fit (Pooled GLMER with Random Intercepts) ---
cat("\nFitting Group-Level Model (Pooled GLMER, all infants)...\n")
# Find the single (alpha, beta) that maximizes likelihood across all participants
# accounting for random subject intercepts.
#
# Note on inclusion: the per-subject GLMs above exclude infants with <5 events
# or <10 trials because individual GLMs are unstable in that regime. The
# group-level GLMER does not need that exclusion: the subject random intercept
# absorbs idiosyncratic baselines (including very low event counts) and the
# shared fixed effect is estimated jointly across the full pool. We therefore
# fit the group model on ALL infants in dat_all (which has already been filtered
# to non-NA event/D rows).

cat("Pre-computing signal vectors for grid...\n")

df_pool <- dat_all
cat(sprintf("Group-level pool: %d infants, %d trials.\n",
            n_distinct(df_pool$subj_global), nrow(df_pool)))

# Define grid values
alphas <- alpha_from_hl(HALF_LIVES_P)
betas <- alpha_from_hl(HALF_LIVES_RHO)

# Helper to compute matrix of signals for a vector
compute_signals_matrix <- function(D_vec, rate_vec) {
    mat <- matrix(NA, nrow = length(D_vec), ncol = length(rate_vec))
    for (j in seq_along(rate_vec)) {
        mat[, j] <- ema_vec(D_vec, rate_vec[j])
    }
    return(mat)
}

# Iterate subjects and stack matrices
subj_list <- split(df_pool, df_pool$subj_global)
P_list <- list()
rho_list <- list()
event_list <- list()
subj_id_list <- list() # Need subj ID for random effects
ntrialseq_list <- list() # Need for control
ntrialsubj_list <- list() # Need for control

# Progress bar for pre-computation
pb_pre <- txtProgressBar(min = 0, max = length(subj_list), style = 3)
for (i in seq_along(subj_list)) {
    s_dat <- subj_list[[i]]
    D <- s_dat$D

    # Compute P matrix (N_trials x N_alphas)
    P_mat <- compute_signals_matrix(D, alphas)

    # Compute rho matrix (N_trials x N_betas)
    rho_mat <- compute_signals_matrix(D, betas)

    P_list[[i]] <- P_mat
    rho_list[[i]] <- rho_mat
    event_list[[i]] <- s_dat$event
    subj_id_list[[i]] <- s_dat$subj_global
    ntrialseq_list[[i]] <- s_dat$ntrialseq
    ntrialsubj_list[[i]] <- s_dat$ntrialsubj

    setTxtProgressBar(pb_pre, i)
}
close(pb_pre)

# Stack everything
P_all <- do.call(rbind, P_list) # [N_total, N_alphas]
rho_all <- do.call(rbind, rho_list) # [N_total, N_betas]
y_all <- unlist(event_list) # [N_total]
subj_all <- unlist(subj_id_list)
ntrialseq_all <- unlist(ntrialseq_list)
ntrialsubj_all <- unlist(ntrialsubj_list)

cat("Total pooled observations:", length(y_all), "\n")

# Grid Search with GLMER
best_group_aic <- Inf
best_group_params <- list(alpha = NA, beta = NA, T = NA, eps = NA)
best_group_model <- NULL

# Full AIC landscape for downstream heatmap plotting
grid_aic <- matrix(NA_real_, nrow = length(alphas), ncol = length(betas))

cat("Running vectorized group grid search (GLMER)...\n")
cat("Note: This may take significant time per iteration.\n")

pb_grp <- txtProgressBar(min = 0, max = length(alphas) * length(betas), style = 3)
cnt <- 0

for (idx_a in seq_along(alphas)) {
    P_col <- P_all[, idx_a]
    a_hl <- HALF_LIVES_P[idx_a]

    for (idx_b in seq_along(betas)) {
        cnt <- cnt + 1
        b_hl <- HALF_LIVES_RHO[idx_b]

        # Enforce global half-life >= MIN_RATIO * local half-life
        if (b_hl < MIN_RATIO * a_hl) {
            setTxtProgressBar(pb_grp, cnt)
            next
        }

        rho_col <- rho_all[, idx_b]
        X <- P_col - rho_col
        # Need standardized X for convergence stability
        X_sc <- scale(X)

        # Fit GLMER: y ~ X + (1|subj)
        # We model event (1=Leave), predictor X (P-rho).
        # Expected: X high (P>rho) -> Prob(Leave) LOW. So coef should be NEGATIVE.

        try(
            {
                # Use nAGQ=0 for speed if needed, but default is safer for accuracy
                mod <- glmer(y_all ~ X_sc + (1 | subj_all),
                    family = binomial,
                    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
                )

                aic <- AIC(mod)
                grid_aic[idx_a, idx_b] <- aic

                if (aic < best_group_aic) {
                    best_group_aic <- aic

                    # Extract fixed effects
                    fix_ef <- fixef(mod)
                    b0_sc <- fix_ef[1] # Intercept
                    b1_sc <- fix_ef[2] # Slope on SCALED X

                    # Unscale coefficients to get raw biological parameters
                    # X_sc = (X - mu) / sd
                    # logit = b0_sc + b1_sc * (X - mu)/sd
                    #       = (b0_sc - b1_sc*mu/sd) + (b1_sc/sd) * X
                    # So: b0_raw = b0_sc - b1_sc*mu/sd
                    #     b1_raw = b1_sc / sd

                    mu_X <- mean(X)
                    sd_X <- sd(X)

                    b1_raw <- b1_sc / sd_X
                    b0_raw <- b0_sc - (b1_sc * mu_X / sd_X)

                    # Recover T, eps
                    T_est <- -1.0 / b1_raw
                    eps_est <- b0_raw * T_est

                    best_group_params <- list(
                        alpha = alphas[idx_a],
                        beta = betas[idx_b],
                        T = T_est,
                        eps = eps_est,
                        b0 = b0_raw,
                        b1 = b1_raw
                    )
                    best_group_model <- mod
                }
            },
            silent = TRUE
        )

        setTxtProgressBar(pb_grp, cnt)
    }
}
close(pb_grp)

# Save the full AIC landscape over (alpha, beta) for heatmap plotting
saveRDS(
    list(
        alphas   = alphas,
        betas    = betas,
        alpha_hl = log(2) / alphas,
        beta_hl  = log(2) / betas,
        aic      = grid_aic
    ),
    "analysis/Cognitive_analysis/Results/infant_group_grid_aic.rds"
)

group_alpha <- best_group_params$alpha
group_beta <- best_group_params$beta

cat("\nGroup Level Estimates (Pooled GLMER):\n")
cat(sprintf("Alpha (Local): %.4f (HL=%.2f)\n", group_alpha, log(2) / group_alpha))
cat(sprintf("Beta (Global): %.4f (HL=%.2f)\n", group_beta, log(2) / group_beta))
cat(sprintf("Temperature: %.4f\n", best_group_params$T))
cat(sprintf("Epsilon: %.4f\n", best_group_params$eps))

cat("\n--- Best Group MVT Model Summary ---\n")
if (!is.null(best_group_model)) {
    print(summary(best_group_model))
    saveRDS(best_group_model, "analysis/Cognitive_analysis/Results/infant_group_mvt_model.rds")
}

# --- 6. Control Model (MVT + Covariates) ---
cat("\n--- Fitting Control Model (MVT + Time Covariates) ---\n")
# Using the BEST alpha/beta found above

if (!is.null(best_group_model)) {
    # Re-compute best X
    # Note: We need to find the correct columns again or just recompute
    # Easier to recompute for clarity
    P_best <- ema_vec(dat_all$D, group_alpha) # Wait, need to use P_all if ordered correctly
    # Use P_all/rho_all from grid using indices
    # We didn't save indices. Let's find index.

    idx_a_best <- which(alphas == group_alpha)
    idx_b_best <- which(betas == group_beta)

    P_best <- P_all[, idx_a_best]
    rho_best <- rho_all[, idx_b_best]
    X_best <- P_best - rho_best

    # Fit Control: event ~ scale(X) + scale(ntrialsubj) + scale(ntrialseq) + (1|subj)
    # Using glmer
    cat("Fitting control model with ntrialseq & ntrialsubj...\n")

    control_mod <- glmer(
        y_all ~ scale(X_best) + scale(ntrialseq_all) + scale(ntrialsubj_all) + (1 | subj_all),
        family = binomial,
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
    )

    print(summary(control_mod))
    saveRDS(control_mod, "analysis/Cognitive_analysis/Results/infant_control_model.rds")
    cat("Saved control model to: analysis/Cognitive_analysis/Results/infant_control_model.rds\n")
}

# Save Group Result
group_res <- data.frame(
    type = "Group_Pooled_GLMER",
    alpha = group_alpha,
    beta = group_beta,
    alpha_hl = log(2) / group_alpha,
    beta_hl = log(2) / group_beta,
    T = best_group_params$T,
    eps = best_group_params$eps,
    aic = best_group_aic
)
write.csv(group_res, "analysis/Cognitive_analysis/Results/infant_group_mvt_params.csv", row.names = FALSE)

# --- 7. Model Comparison: MVT vs. Alternative Theoretical Models ---
# Three competing accounts of disengagement, all fit on the same pool
# (all infants) as mixed-effects logistic regressions with a random
# intercept by subject:
#   (a) MVT:                event ~ scale(X)      + (1 | subj)
#       Infants leave when local progress falls below the global baseline.
#       Uses X = P_t - rho_t at the group-best (alpha, beta).
#   (b) Learning progress:  event ~ scale(D)      + (1 | subj)
#       Infants leave when the current trial offers little learning progress
#       (information-gain account, no comparison to a global baseline).
#   (c) Quadratic surprise: event ~ poly(I, 2)    + (1 | subj)
#       Infants engage with moderately surprising stimuli, with an
#       inverted-U on surprise I ("Goldilocks effect", Kidd et al., 2012).

cat("\n\n=== Model Comparison: MVT vs Alternative Models ===\n")

# Build a complete-case data frame so all three models are fit on identical rows.
mc_df <- dat_all %>%
    mutate(X_best = P_all[, idx_a_best] - rho_all[, idx_b_best]) %>%
    filter(!is.na(event), !is.na(D), !is.na(I), !is.na(X_best))

cat(sprintf("Comparison pool: %d trials across %d infants.\n",
            nrow(mc_df), n_distinct(mc_df$subj_global)))

cat("Fitting (a) MVT model: event ~ scale(X) + (1|subj)\n")
mod_mvt <- glmer(
    event ~ scale(X_best) + (1 | subj_global),
    data    = mc_df,
    family  = binomial,
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

cat("Fitting (b) learning-progress only model: event ~ scale(D) + (1|subj)\n")
mod_D <- glmer(
    event ~ scale(D) + (1 | subj_global),
    data    = mc_df,
    family  = binomial,
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

cat("Fitting (c) quadratic surprise model: event ~ poly(I, 2) + (1|subj)\n")
mod_I2 <- glmer(
    event ~ poly(I, 2) + (1 | subj_global),
    data    = mc_df,
    family  = binomial,
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

# Effective parameter counts. For MVT, the regression itself has 2 fixed
# effects (intercept + slope on X), but we also grid-searched (alpha_hl, beta_hl)
# from a 10x20 grid. To compare fairly against the alternative models — which
# are fit at fixed hyperparameters — we count alpha_hl and beta_hl as two
# additional parameters of the MVT model and report both raw and adjusted
# AIC / BIC.
MVT_GRID_PARAMS <- 2L  # alpha_hl + beta_hl selected by grid search
n_obs           <- nrow(mc_df)

comparison_df <- data.frame(
    model           = c("MVT", "Learning_progress_D", "Quadratic_surprise"),
    formula         = c("event ~ scale(X) + (1|subj)",
                        "event ~ scale(D) + (1|subj)",
                        "event ~ poly(I, 2) + (1|subj)"),
    n_fixef         = c(2L, 2L, 3L),
    n_grid_searched = c(MVT_GRID_PARAMS, 0L, 0L),
    aic             = c(AIC(mod_mvt), AIC(mod_D), AIC(mod_I2)),
    bic             = c(BIC(mod_mvt), BIC(mod_D), BIC(mod_I2)),
    loglik          = c(as.numeric(logLik(mod_mvt)),
                        as.numeric(logLik(mod_D)),
                        as.numeric(logLik(mod_I2))),
    n_obs           = rep(n_obs, 3),
    n_subjects      = rep(n_distinct(mc_df$subj_global), 3)
)
# Adjusted criteria: add 2*k_grid to AIC and log(n)*k_grid to BIC for the
# grid-searched hyperparameters (no-op for the non-grid-searched models).
comparison_df$aic_adj   <- comparison_df$aic + 2          * comparison_df$n_grid_searched
comparison_df$bic_adj   <- comparison_df$bic + log(n_obs) * comparison_df$n_grid_searched
comparison_df$delta_aic <- comparison_df$aic     - min(comparison_df$aic)
comparison_df$delta_bic <- comparison_df$bic     - min(comparison_df$bic)
comparison_df$delta_aic_adj <- comparison_df$aic_adj - min(comparison_df$aic_adj)
comparison_df$delta_bic_adj <- comparison_df$bic_adj - min(comparison_df$bic_adj)

cat("\n--- Raw AIC / BIC ---\n")
print(comparison_df[, c("model", "n_fixef", "n_grid_searched",
                        "aic", "delta_aic", "bic", "delta_bic")],
      row.names = FALSE)

cat("\n--- Adjusted AIC / BIC (grid-searched parameters penalised) ---\n")
print(comparison_df[, c("model", "n_fixef", "n_grid_searched",
                        "aic_adj", "delta_aic_adj",
                        "bic_adj", "delta_bic_adj")],
      row.names = FALSE)

# Save outputs
write.csv(
    comparison_df,
    "analysis/Cognitive_analysis/Results/infant_model_comparison.csv",
    row.names = FALSE
)
saveRDS(mod_D,  "analysis/Cognitive_analysis/Results/infant_mod_D.rds")
saveRDS(mod_I2, "analysis/Cognitive_analysis/Results/infant_mod_I2.rds")
cat("\nSaved model-comparison artifacts to analysis/Cognitive_analysis/Results/\n")

cat("\nFitting complete. Run plot_infant_mvt.R to generate figures.\n")

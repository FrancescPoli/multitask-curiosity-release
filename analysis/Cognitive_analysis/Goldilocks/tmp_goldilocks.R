InsuppressPackageStartupMessages({
    library(dplyr); library(tidyr); library(ggplot2); library(lme4); library(mgcv)
})

# Goldilocks: does curiosity spend most TIME-ON-TASK on INTERMEDIATE-difficulty tasks?
#   time-on-task : curiosity cohorts' sampling fraction
#   difficulty   : (1) steps-to-criterion under RANDOM cohorts (intrinsic, uniform exposure)
#                  (2) compositional depth n_mods (a-priori structural)
# Two independent plots.

ALL  <- read.csv("analysis/Cognitive_analysis/Goldilocks/tmp_alltask_out.csv", stringsAsFactors = FALSE)
META <- read.csv("analysis/compositional_analysis/Data/task_metadata.csv", stringsAsFactors = FALSE)

CURIOSITY <- c("H-100", "C-reg")    # low temperature
RANDOM    <- c("C-temp", "C-both")  # high temperature
MAXSTEP   <- 600000

train_tasks <- META %>%
    mutate(is_heldout = as.logical(is_heldout)) %>%
    filter(!is_heldout) %>%
    select(task, n_mods, family)
cat("Training tasks:", nrow(train_tasks), "| n_mods dist:",
    paste(names(table(train_tasks$n_mods)), table(train_tasks$n_mods), sep=":", collapse=" "), "\n")

ALLt <- ALL %>% inner_join(train_tasks, by = "task")

# ── Difficulty (1): steps-to-criterion under RANDOM cohorts ──────────────────
# never reaching criterion -> censored at MAXSTEP (hardest); average across random nets
difficulty <- ALLt %>% filter(cohort %in% RANDOM) %>%
    mutate(stc = ifelse(is.na(steps_to_crit), MAXSTEP, steps_to_crit)) %>%
    group_by(task) %>%
    summarise(difficulty = mean(stc),
              pct_solved_random = round(100 * mean(!is.na(steps_to_crit)), 0),
              .groups = "drop")

# ── Time-on-task under CURIOSITY: complete the grid so unsampled task = 0 ─────
runs_cur <- ALLt %>% filter(cohort %in% CURIOSITY) %>% distinct(run_id)
cur <- expand.grid(run_id = runs_cur$run_id, task = train_tasks$task,
                   stringsAsFactors = FALSE) %>%
    left_join(ALLt %>% filter(cohort %in% CURIOSITY) %>% select(run_id, task, samp_frac),
              by = c("run_id", "task")) %>%
    mutate(samp_frac = ifelse(is.na(samp_frac), 0, samp_frac) * 100) %>%   # -> percent
    left_join(train_tasks, by = "task") %>%
    left_join(difficulty, by = "task")

# ════════════════════════════════════════════════════════════════════════════
# PLOT 1 — time-on-task vs steps-to-criterion difficulty
# ════════════════════════════════════════════════════════════════════════════
cur1 <- cur %>% mutate(diff_k = difficulty / 1000)   # difficulty in thousands of steps
m1 <- lmer(samp_frac ~ poly(scale(diff_k), 2) + (1 | run_id), data = cur1)
m1_lin <- lmer(samp_frac ~ scale(diff_k) + (1 | run_id), data = cur1)
cat("\n=== Plot 1: time-on-task ~ poly(steps-to-crit, 2) ===\n")
print(round(summary(m1)$coefficients, 4))
cat("Quadratic LRT (vs linear): ",
    {a <- anova(m1_lin, m1); sprintf("chi2 = %.2f, df = %d, p = %.3g",
        a$Chisq[2], a$Df[2], a$`Pr(>Chisq)`[2])}, "\n")

task1 <- cur1 %>% group_by(task, family, difficulty, diff_k) %>%
    summarise(tot = mean(samp_frac), se = sd(samp_frac)/sqrt(n()), .groups = "drop")

p1 <- ggplot(cur1, aes(diff_k, samp_frac)) +
    geom_smooth(method = "lm", formula = y ~ poly(x, 2), colour = "darkblue",
                fill = "darkblue", alpha = 0.15, linewidth = 0.9) +
    geom_point(data = task1, aes(diff_k, tot), colour = "grey30", size = 2) +
    geom_errorbar(data = task1, aes(diff_k, tot, ymin = tot-se, ymax = tot+se),
                  width = 0, colour = "grey30", inherit.aes = FALSE) +
    labs(x = "Task difficulty\n(steps to criterion under uniform exposure, ×1000)",
         y = "Time on task under curiosity\n(% of training steps)") +
    theme_classic(base_size = 10)
ggsave("analysis/Final_analyses_and_plots/Figures/tmp_goldilocks_steps.png", p1,
       width = 5, height = 4, dpi = 300)

# ════════════════════════════════════════════════════════════════════════════
# PLOT 2 — time-on-task vs compositional depth (n_mods)
# ════════════════════════════════════════════════════════════════════════════
m2 <- lmer(samp_frac ~ poly(n_mods, 2) + (1 | run_id), data = cur)
m2_lin <- lmer(samp_frac ~ n_mods + (1 | run_id), data = cur)
cat("\n=== Plot 2: time-on-task ~ poly(compositional depth, 2) ===\n")
print(round(summary(m2)$coefficients, 4))
cat("Quadratic LRT (vs linear): ",
    {a <- anova(m2_lin, m2); sprintf("chi2 = %.2f, df = %d, p = %.3g",
        a$Chisq[2], a$Df[2], a$`Pr(>Chisq)`[2])}, "\n")

depth_means <- cur %>% group_by(n_mods) %>%
    summarise(tot = mean(samp_frac), se = sd(samp_frac)/sqrt(n()),
              n_tasks = n_distinct(task), .groups = "drop")
cat("\nTime-on-task by depth:\n"); print(as.data.frame(depth_means))

p2 <- ggplot(cur, aes(factor(n_mods), samp_frac)) +
    geom_violin(fill = "grey85", colour = NA, scale = "width") +
    stat_summary(fun = mean, geom = "point", colour = "darkblue", size = 3) +
    stat_summary(fun.data = mean_se, geom = "errorbar", colour = "darkblue", width = 0.1) +
    geom_smooth(aes(x = n_mods + 1, group = 1), method = "lm",
                formula = y ~ poly(x, 2), colour = "darkblue", se = FALSE, linewidth = 0.8) +
    labs(x = "Compositional depth (number of primitives composed)",
         y = "Time on task under curiosity\n(% of training steps)") +
    theme_classic(base_size = 10)
ggsave("analysis/Final_analyses_and_plots/Figures/tmp_goldilocks_depth.png", p2,
       width = 5, height = 4, dpi = 300)

cat("\nSaved tmp_goldilocks_steps.png and tmp_goldilocks_depth.png\n")

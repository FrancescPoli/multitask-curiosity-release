# =============================================================================
# Compositional generalisation analysis (final).
#
# For each selected network, a frozen-weight probe (see Methods) measures how
# many adaptation steps it needs to solve each held-out task and its familiar
# reference task. Each pair is summarised by the geometric-mean solve time
#   G = sqrt(t_reference * t_held-out)
# (small only in the "bottom-left corner": fast on BOTH), modelled with a
# log-normal accelerated-failure-time model (right-censored at the 5,000-step
# budget). Positive coefficients => longer (slower) solving.
#
# Model:  Surv(G, both_solved) ~ temperature * regularisation * recombination
#                                + end-of-training accuracy on the base ability
#
# Outputs (printed): interaction LRTs; temperature / regularisation contrasts;
#   acquisition slope. Figure: temperature x regularisation x recombination at
#   mean base-ability accuracy -> pub_compositional_generalisation.png
#
# Run from project root:
#   "C:/Program Files/R/R-4.2.2/bin/Rscript.exe" \
#       analysis/compositional_analysis/pub_compositional_generalisation.R
# =============================================================================
suppressPackageStartupMessages({
    library(dplyr); library(tidyr); library(survival); library(emmeans); library(ggplot2)
})

# ── Config ───────────────────────────────────────────────────────────────────
COMP_CSV    <- "analysis/compositional_analysis/Data/compositional_dataset.csv"        # probe solve times
ACC_CSV     <- "analysis/compositional_analysis/Data/compositional_base_accuracy.csv"  # end-of-training accuracy per base task
OUT_PNG     <- "analysis/Final_analyses_and_plots/Figures/pub_compositional_generalisation.png"
EXCLUDE_DM2 <- TRUE     # dm2 held-out tasks excluded (degenerate ring-2-only variant)
MAX_STEPS   <- 5000     # probe budget; unsolved tasks right-censored here

# ── Pairing table ────────────────────────────────────────────────────────────
# held_out  : the novel recombination, never trained
# training  : the familiar reference (same modifiers, family where it was trained)
# base_task : the trained task supplying the ability the held-out task redeploys
#             (covariate = network's end-of-training accuracy on this task)
# comp_group: type of recombination tested
HELDOUT_MATCHES <- data.frame(
    held_out  = c("poli.ctxdm1","poli.ctxdm2","poli.ctxdlydm1","poli.ctxdlydm2",
                  "poli.antictxdm1","poli.antictxdm2","poli.antictxdlydm1","poli.antictxdlydm2",
                  "poli.antidlyms","poli.antidlynms","poli.antictxdlyms","poli.antictxcatdlyms"),
    training  = c("poli.ctxgo","poli.ctxgo","poli.dlyctxgo","poli.dlyctxgo",
                  "poli.antictxgo","poli.antictxgo","poli.dlyantictxgo","poli.dlyantictxgo",
                  "poli.dlyantigo","poli.dlyantigo","poli.dlyantictxgo","poli.dlyantictxgo"),
    base_task = c("poli.dm1","poli.dm2","poli.dlydm1","poli.dlydm2",
                  "poli.antidm1","poli.antidm2","poli.antidlydm1","poli.antidlydm2",
                  "poli.dlyms","poli.dlynms","poli.dlyms","poli.dlyms"),
    comp_group = c(rep("ctx -> Decision", 8), rep("anti -> Match", 2), rep("ctx+anti -> Match", 2)),
    stringsAsFactors = FALSE)
if (EXCLUDE_DM2) HELDOUT_MATCHES <- HELDOUT_MATCHES %>% filter(!grepl("dm2", held_out))

COMP_LEVELS <- c("ctx -> Decision", "anti -> Match", "ctx+anti -> Match")

# ── Build per-pair data ──────────────────────────────────────────────────────
df  <- read.csv(COMP_CSV, stringsAsFactors = FALSE)
sc  <- grep("^probe_solved_at_poli_", names(df), value = TRUE)
sol <- df[, c("run_id", "cohort", sc)] %>%
    pivot_longer(all_of(sc), names_to = "col", values_to = "solved_at") %>%
    mutate(task      = sub("^probe_solved_at_poli_", "poli.", col),
           solved    = !is.na(solved_at),
           surv_time = ifelse(is.na(solved_at), MAX_STEPS, solved_at)) %>%
    select(-col)

sd <- do.call(rbind, lapply(seq_len(nrow(HELDOUT_MATCHES)), function(i) {
    ref <- sol %>% filter(task == HELDOUT_MATCHES$training[i]) %>%
           select(run_id, cohort, train_time = surv_time, train_solved = solved)
    hel <- sol %>% filter(task == HELDOUT_MATCHES$held_out[i]) %>%
           select(run_id, held_time = surv_time, held_solved = solved)
    inner_join(ref, hel, by = "run_id") %>%
        mutate(comp_group = HELDOUT_MATCHES$comp_group[i],
               base_task  = HELDOUT_MATCHES$base_task[i])
})) %>% filter(!is.na(cohort))

# ── Covariate: end-of-training accuracy on the transported base ability ──────
acc <- read.csv(ACC_CSV, check.names = FALSE) %>% select(-cohort) %>%
    pivot_longer(-run_id, names_to = "base_task", values_to = "base_acc")

sd <- sd %>%
    left_join(acc, by = c("run_id", "base_task")) %>%
    mutate(
        both_solved = train_solved & held_solved,
        temp = factor(ifelse(cohort %in% c("H-100", "C-reg"),  "low",    "high"), levels = c("low", "high")),
        reg  = factor(ifelse(cohort %in% c("H-100", "C-temp"), "strong", "weak"), levels = c("strong", "weak")),
        grp  = factor(comp_group, levels = COMP_LEVELS),
        G        = sqrt(train_time * held_time),
        z_acc    = as.numeric(scale(base_acc)))
cat(sprintf("N pairs = %d (acc missing = %d)\n", nrow(sd), sum(is.na(sd$base_acc))))

# ── Model: acquisition-adjusted AFT ──────────────────────────────────────────
full  <- survreg(Surv(G, both_solved) ~ temp*reg*grp + z_acc,            data = sd, dist = "lognormal")
no3   <- survreg(Surv(G, both_solved) ~ temp*reg + temp*grp + reg*grp + z_acc, data = sd, dist = "lognormal")
no_tg <- survreg(Surv(G, both_solved) ~ temp*reg + reg*grp + z_acc,      data = sd, dist = "lognormal")
no_rg <- survreg(Surv(G, both_solved) ~ temp*reg + temp*grp + z_acc,     data = sd, dist = "lognormal")
no_tr <- survreg(Surv(G, both_solved) ~ temp*grp + reg*grp + z_acc,      data = sd, dist = "lognormal")
lr <- function(a, b, nm) {
    L <- 2*(logLik(a) - logLik(b)); d <- length(coef(a)) - length(coef(b))
    cat(sprintf("  %-14s chi2 = %6.2f, df = %d, p = %.3g\n", nm, L, d, pchisq(L, d, lower.tail = FALSE)))
}
cat("\n=== Interaction LRTs ===\n")
lr(full, no3, "temp:reg:grp"); lr(no3, no_tg, "temp:grp"); lr(no3, no_rg, "reg:grp"); lr(no3, no_tr, "temp:reg")

cat("\n=== Temperature contrast (curiosity[low] - random[high]) within recombination type ===\n")
print(summary(contrast(emmeans(full, ~ temp | grp), "pairwise")))
cat("\n=== Regularisation contrast (strong - minimal[weak]), across recombination types ===\n")
print(summary(contrast(emmeans(full, ~ reg), "pairwise")))
cat("\n=== Acquisition slope (z_acc, per SD of base-ability accuracy) ===\n")
print(round(summary(full)$table["z_acc", ], 4))

# ── Figure: temp x reg x recombination at mean base-ability accuracy ─────────
nd <- expand.grid(temp = factor(c("low","high"), levels = c("low","high")),
                  reg  = factor(c("strong","weak"), levels = c("strong","weak")),
                  grp  = factor(COMP_LEVELS, levels = COMP_LEVELS),
                  z_acc = 0)                      # z = 0 => mean acquisition
pr <- predict(full, newdata = nd, type = "lp", se.fit = TRUE)
nd$logG <- pr$fit; nd$se <- pr$se.fit
nd <- nd %>% mutate(
    temp_lab = factor(ifelse(temp == "low", "Curiosity\n(low τ)", "Random\n(high τ)"),
                      levels = c("Curiosity\n(low τ)", "Random\n(high τ)")),
    reg_lab  = factor(ifelse(reg == "strong", "Strong reg", "Weak reg"),
                      levels = c("Strong reg", "Weak reg")))

p <- ggplot(nd, aes(temp_lab, logG, colour = reg_lab, group = reg_lab)) +
    geom_line(linewidth = 0.7, position = position_dodge(0.12)) +
    geom_pointrange(aes(ymin = logG - se, ymax = logG + se),
                    size = 0.5, fatten = 3, linewidth = 0.8, position = position_dodge(0.12)) +
    facet_wrap(~ grp, nrow = 1) +
    scale_colour_manual(values = c("Strong reg" = "darkblue", "Weak reg" = "#ff7f00"), name = NULL) +
    labs(x = NULL, y = "Joint solve time at mean ability acquisition\nlog √(reference × held-out) steps") +
    theme_classic(base_size = 9) +
    theme(legend.position = "top", strip.background = element_blank(),
          strip.text = element_text(face = "bold", size = 9),
          axis.text.x = element_text(size = 8, lineheight = 0.9))
ggsave(OUT_PNG, p, width = 7.5, height = 3.2, dpi = 300)
cat("\nSaved:", OUT_PNG, "\n")

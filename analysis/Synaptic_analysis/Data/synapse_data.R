# existing data
huttenlocher1997 <- data.frame(
  case = rep(1:14, each = 2),
  region = rep(c("auditory", "prefrontal"), times = 14),
  CA_days = rep(c(192, 210, 280, 320, 360, 363, 700,
                  1620, 4700, 5000, 5700, 7300, 11700, 21500),
                each = 2),
  age_label = rep(c("fetus", "fetus", "NB", "1/12",
                    "3/12", "3/12", "12/12", "3.5",
                    "12", "13", "15", "19", "32", "59"),
                  each = 2),
  synapses_100um3 = c(
    12.2,  3.2,
     7.5,  2.2,
    29.4, 19.5,
    21.0, 11.2,
    40.7, 28.3,
    54.3, 34.3,
    53.0, 37.9,
    55.7, 52.4,
    24.7, 46.9,
    26.0, 39.7,
    38.9, 40.0,
    24.0, 27.1,
    35.0, 40.2,
    34.7, 28.7
  ),
  synapses_se = c(
    1.2, 0.5,
    0.6, 0.5,
    2.7, 2.2,
    2.4, 1.1,
    5.4, 0.9,
    3.0, 4.7,
    5.4, 4.9,
    5.4, 6.0,
    3.9, 3.1,
    4.3, 6.1,
    3.5, 3.9,
    2.1, 2.5,
    2.8, 3.7,
    2.7, 2.0
  )
)

# add conceptual age in years
huttenlocher1997$age_years <- huttenlocher1997$CA_days / 365
huttenlocher1997$layer     <- "all_layers"
huttenlocher1997$study     <- "Huttenlocher1997"


# helper: conceptual age in years assuming 40w term
gest_weeks_to_years <- function(gest_weeks) gest_weeks / 52
postnatal_months_to_years <- function(months) months / 12

visual1982 <- data.frame(
  region = "visual",
  layer  = "all_layers",                # note: Fig 2 is all layers, not layer I only
  study  = "Huttenlocher1982",
  age_label = c("28w", "birth", "2m", "2.4m", "4m", "8m", "11m",
                "19m", "3y", "11y", "26y", "71y"),
  # conceptual age in years
  age_years = c(
    gest_weeks_to_years(28),                               # 28w gestation
    gest_weeks_to_years(40),                               # birth ~40w
    gest_weeks_to_years(40) + postnatal_months_to_years(2),
    gest_weeks_to_years(40) + postnatal_months_to_years(2.4),
    gest_weeks_to_years(40) + postnatal_months_to_years(4),
    gest_weeks_to_years(40) + postnatal_months_to_years(8),
    gest_weeks_to_years(40) + postnatal_months_to_years(11),
    gest_weeks_to_years(40) + postnatal_months_to_years(19),
    gest_weeks_to_years(40) + 3,
    gest_weeks_to_years(40) + 11,
    gest_weeks_to_years(40) + 26,
    gest_weeks_to_years(40) + 71
  ),
  # Fig 2 units ≈ 10^8 synapses/mm^3 -> multiply by 10 for synapses/100 µm^3
  synapses_100um3 = c(
    1.2, 2.55, 2.65, 3.15, 5.1, 5.75,
    5.65, 4.9, 4.45, 3.5, 3.5, 3.3
  ) * 10,
  synapses_se = c(
    0.15, 0.18, 0.19, 0.2, 0.31, 0.27,
    0.27, 0.18, 0.15, 0.17, 0.2, 0.18
  ) * 10
)

# derive CA_days to keep your older code working (if you still use CA_days-based plots)
visual1982$CA_days <- round(visual1982$age_years * 365)

# merge everything
syn_data <- rbind(
  huttenlocher1997[, c("region", "layer", "study",
                       "age_label", "age_years",
                       "CA_days", "synapses_100um3", "synapses_se")],
  visual1982
)



library(ggplot2)
library(scales)

region_greens <- c(
  auditory   = "#a1d99b",
  visual     = "#41ab5d",
  prefrontal = "#00441b"
)

p_synapse <- ggplot(syn_data,
       aes(x = age_years, y = synapses_100um3,
           colour = region)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = synapses_100um3 - synapses_se,
                    ymax = synapses_100um3 + synapses_se),
                width = 0.05, alpha = 0.6) +
  geom_smooth(
    aes(weight = 1 / synapses_se^2),
    method  = "gam",
    formula = y ~ s(x, k = 7),
    se      = F
  ) +
  scale_colour_manual(values = region_greens) +
  scale_x_log10(
    breaks = c(0.5, 1, 2, 5, 10, 20, 50, 80),
    labels = label_number(accuracy = 1)
  ) +
  labs(x = "Age from conception (years)",
       y = "Synaptic density (synapses / 100 µm³)",
       colour = "Region") +
  geom_vline(xintercept = 280/365,
             linetype = "dashed",
             colour = "black")+
  theme_classic()

out_dir <- "analysis/comparison_plots/synaptic"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
ggsave(file.path(out_dir, "huttenlocher_synapse_density.png"),
       plot = p_synapse, width = 6, height = 4, dpi = 300)




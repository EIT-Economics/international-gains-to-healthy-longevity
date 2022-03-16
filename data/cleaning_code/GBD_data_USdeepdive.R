### 
#Global Burden of Disease data
# If using the more granular data, can merge with: 
#GBD_hierarchy <- read_xlsx("data/GBD/GBD_2019_CODEBOOK/IHME_GBD_2019_CAUSE_HIERARCHY_Y2020M11D25.XLSX")
###


setwd("/Users/julianashwin/Documents/GitHub/TargetingAging")
rm(list=ls())

require(ggplot2)
require(ggpubr)
require(reshape)
require(stringr)
require(stargazer)
require(purrr)
require(dplyr)
require(readxl)
require(tidyr)









###
# More detailed, granular look at the US 
###
GBD_full_df <- read.csv("data/GBD/US_GBD_1990-2019_DATA.csv",stringsAsFactors = FALSE)
names(GBD_full_df)

GBD_full_df <- GBD_full_df[,c("measure_id","measure_name", "age_id", "age_name", "cause_id",
                              "cause_name", "metric_name", "year", "val")]
# Clean up age variable
table(GBD_full_df$age_name)
GBD_full_df$age_name[which(GBD_full_df$age_name == "<1 year")] <- "0 to 0"
GBD_full_df$age_name[which(GBD_full_df$age_name == "95 plus")] <- "95 to 99"
# Shorten the measure names
GBD_full_df$measure_name[which(GBD_full_df$measure_name == "YLLs (Years of Life Lost)")] <- "YLL"
GBD_full_df$measure_name[which(GBD_full_df$measure_name == "DALYs (Disability-Adjusted Life Years)")] <- "DALY"
GBD_full_df$measure_name[which(GBD_full_df$measure_name == "YLDs (Years Lived with Disability)")] <- "YLD"


GBD_full_df[,c("age_low", "age_high")] <-  
  do.call(rbind, str_split(as.character(GBD_full_df$age_name), " to "))
GBD_full_df$age_low <- as.numeric(GBD_full_df$age_low)
GBD_full_df$age_high <- as.numeric(GBD_full_df$age_high)
GBD_full_df$age_mid <- round(GBD_full_df$age_low + (GBD_full_df$age_high - GBD_full_df$age_low)/2)



###
# Most useful is probably the mortality rate and YLD
###

"
Mortality
"
mort_df <- GBD_full_df[which(GBD_full_df$measure_name == "Deaths" &
                               GBD_full_df$metric_name == "Rate"),]
mort_df$val <- mort_df$val/100000
mort_causes <- unique(mort_df$cause_name)
# Transform
mort_df <- mort_df %>% 
  select(age_name, age_mid, year, cause_name, val) %>%
  pivot_wider(names_from = cause_name, values_from = val) %>%
  arrange(year, age_mid) %>%
  replace(is.na(.), 0)
mort_df$Total <- rowSums(mort_df[,mort_causes])
# Plot
ggplot(mort_df, aes(x = age_mid)) + 
  geom_line(aes(y = Total, color = as.factor(year)))
# Export
write.csv(mort_df, "data/GBD/mortality_data.csv", row.names = FALSE)

### Plot difference in total mortality from 2010 to 2019
mort_1990_df <- mort_df[which(mort_df$year == 1990),]
mort_2000_df <- mort_df[which(mort_df$year == 2000),]
mort_2010_df <- mort_df[which(mort_df$year == 2010),]
mort_2019_df <- mort_df[which(mort_df$year == 2019),]


###  Compare 2010 and 2019
mort_comp_df <- mort_2010_df[,c("age_name", "age_mid")]
mort_comp_df[,mort_causes] <- mort_2010_df[,mort_causes] - mort_1990_df[,mort_causes] 
mort_comp_df <- melt(as.data.frame(mort_comp_df), id = c("age_name", "age_mid"),
                     variable_name = "Cause")
# Plot changes
ggplot(mort_comp_df, aes(x = age_mid, y = value)) + theme_bw() + 
  geom_bar(aes(fill = Cause), position="stack", stat="identity") + 
  geom_hline(yintercept=0, linetype="dashed", color = "black") + 
  xlab("Age") + ylab("Change in Death rate (1990 to 2010)")
ggsave("figures/data/GBD/mort_comp_1990_2010.pdf", width = 15, height = 5)


###  Compare 2010 and 2019
mort_comp_df <- mort_2010_df[,c("age_name", "age_mid")]
mort_comp_df[,mort_causes] <- mort_2019_df[,mort_causes] - mort_2010_df[,mort_causes] 
mort_comp_df <- melt(as.data.frame(mort_comp_df), id = c("age_name", "age_mid"),
                     variable_name = "Cause")
# Plot changes
ggplot(mort_comp_df, aes(x = age_mid, y = value)) + theme_bw() + 
  geom_bar(aes(fill = Cause), position="stack", stat="identity") + 
  geom_hline(yintercept=0, linetype="dashed", color = "black") + 
  xlab("Age") + ylab("Change in Death rate (2010 to 2019)")
ggsave("figures/data/GBD/mort_comp_2010_2019.pdf", width = 15, height = 5)


# Which categories had the biggest change from 2010 to 2019
mort_pop_df <- GBD_full_df[which(GBD_full_df$measure_name == "Deaths" &
                                   GBD_full_df$metric_name == "Number"),]
mort_comp_tab <- aggregate(list(val = mort_pop_df$val), FUN = sum,
                           by = list(Cause = mort_pop_df$cause_name, Year = mort_pop_df$year))
# Pivot to wide in years
mort_comp_tab <- mort_comp_tab %>% 
  mutate(Value = val/1000) %>%
  select(Cause, Year, Value) %>%
  pivot_wider(names_from = Year, values_from = Value)

pop_2010 <- 309.3*1000 
pop_2019 <- 328.3*1000

mort_comp_tab$Change <- 100*((mort_comp_tab$`2019`/pop_2019) - (mort_comp_tab$`2010`/pop_2010))
mort_comp_tab[,c("1990", "2000", "2010", "2019", "Change")] <- 
  round(mort_comp_tab[,c("1990", "2000", "2010", "2019", "Change")], 3)

mort_comp_tab <- mort_comp_tab[order(-mort_comp_tab$`2019`),]
stargazer(as.matrix(mort_comp_tab[,c("Cause","1990", "2000", "2010", "2019")]))




" 
Health 
"
health_df <- GBD_full_df[which(GBD_full_df$measure_name == "YLD" &
                                 GBD_full_df$metric_name == "Rate"),]
health_df$val <- health_df$val/100000
health_causes <- unique(health_df$cause_name)
# Transform
health_df <- as.data.frame(health_df %>% 
                             select(age_name, age_mid, year, cause_name, val) %>%
                             pivot_wider(names_from = cause_name, values_from = val) %>%
                             arrange(year, age_mid) %>%
                             replace(is.na(.), 0))
health_df$Total <- rowSums(health_df[,health_causes])
# Plot
ggplot(health_df, aes(x = age_mid)) + 
  geom_line(aes(y = Total, color = as.factor(year)))
# Export
write.csv(health_df, "data/GBD/health_data.csv", row.names = FALSE)


### Plot difference in total YLD lost from 2010 to 2019
health_1990_df <- health_df[which(health_df$year == 1990),]
health_2000_df <- health_df[which(health_df$year == 2000),]
health_2010_df <- health_df[which(health_df$year == 2010),]
health_2019_df <- health_df[which(health_df$year == 2019),]


## 1990 to 2010
health_comp_df <- health_2010_df[,c("age_name", "age_mid")]
health_comp_df[,health_causes] <- health_2010_df[,health_causes] - health_1990_df[,health_causes] 

health_comp_df <- melt(as.data.frame(health_comp_df), id = c("age_name", "age_mid"),
                       variable_name = "Cause")
# Plot changes
ggplot(health_comp_df, aes(x = age_mid, y = value)) + theme_bw() + 
  geom_bar(aes(fill = Cause), position="stack", stat="identity") + 
  geom_hline(yintercept=0, linetype="dashed", color = "black") + 
  xlab("Age") + ylab("Change in YLD rate (1990 to 2010)")
ggsave("figures/data/GBD/health_comp_1990_2010.pdf", width = 15, height = 5)



## 2010 to 2019
health_comp_df <- health_2010_df[,c("age_name", "age_mid")]
health_comp_df[,health_causes] <- health_2019_df[,health_causes] - health_2010_df[,health_causes] 

health_comp_df <- melt(as.data.frame(health_comp_df), id = c("age_name", "age_mid"),
                       variable_name = "Cause")
# Plot changes
ggplot(health_comp_df, aes(x = age_mid, y = value)) + theme_bw() + 
  geom_bar(aes(fill = Cause), position="stack", stat="identity") + 
  geom_hline(yintercept=0, linetype="dashed", color = "black") + 
  xlab("Age") + ylab("Change in YLD rate (2010 to 2019)")
ggsave("figures/data/GBD/health_comp_2010_2019.pdf", width = 15, height = 5)




# Which categories had the biggest change from 2010 to 2019
health_pop_df <- GBD_full_df[which(GBD_full_df$measure_name == "YLD" &
                                     GBD_full_df$metric_name == "Number"),]
health_comp_tab <- aggregate(list(val = health_pop_df$val), FUN = sum,
                             by = list(Cause = health_pop_df$cause_name, Year = health_pop_df$year))
# Pivot to wide in years
health_comp_tab <- health_comp_tab %>% 
  mutate(Value = val/1000000) %>%
  select(Cause, Year, Value) %>%
  pivot_wider(names_from = Year, values_from = Value)

pop_2010 <- 309.3 
pop_2019 <- 328.3

health_comp_tab$Change <- 100*((health_comp_tab$`2019`/pop_2019) - (health_comp_tab$`2010`/pop_2010))
health_comp_tab[,c("2010", "2019", "Change")] <- round(health_comp_tab[,c("2010", "2019", "Change")], 3)
health_comp_tab <- health_comp_tab[order(health_comp_tab$Change),]

stargazer(as.matrix(health_comp_tab[,c("Cause", "2010", "2019", "Change")]))




###
# Some more detailed exploratory plots
###

# Plot death rates
plot_df <- GBD_full_df[which(GBD_full_df$year == 2019 & GBD_full_df$measure_name == "Deaths" &
                               GBD_full_df$metric_name == "Rate"),]
ggplot(plot_df, aes(x = age_mid, y = val)) + theme_bw() + 
  geom_bar(aes(fill = cause_name), position="stack", stat="identity") + 
  xlab("Age") + ylab("Deaths per 100,000")
ggsave("figures/data/GBD/death_rates_2019.pdf", width = 12, height = 4)

# Plot years life lost
plot_df <- GBD_full_df[which(GBD_full_df$year == 2019 & GBD_full_df$measure_name == "YLL" &
                               GBD_full_df$metric_name == "Rate"),]
ggplot(plot_df, aes(x = age_mid, y = val)) + theme_bw() + 
  geom_bar(aes(fill = cause_name), position="stack", stat="identity") + 
  xlab("Age") + ylab("YLL per 100,000")
ggsave("figures/data/GBD/yll_rates_2019.pdf", width = 12, height = 4)

# Plot incidence
plot_df <- GBD_full_df[which(GBD_full_df$year == 2019 & GBD_full_df$measure_name == "Incidence" &
                               GBD_full_df$metric_name == "Percent"),]
ggplot(plot_df, aes(x = age_mid, y = val)) + theme_bw() + 
  geom_bar(aes(fill = cause_name), position="fill", stat="identity") + 
  xlab("Age") + ylab("Incidence relative to all diseases")
ggsave("figures/data/GBD/incidence_percentage_2019.pdf", width = 12, height = 4)

# Plot prevalence
plot_df <- GBD_full_df[which(GBD_full_df$year == 2019 & GBD_full_df$measure_name == "Prevalence" &
                               GBD_full_df$metric_name == "Percent"),]
ggplot(plot_df, aes(x = age_mid, y = val)) + theme_bw() + 
  geom_bar(aes(fill = cause_name), position="fill", stat="identity") + 
  xlab("Age") + ylab("Prevalence relative to all diseases")
ggsave("figures/data/GBD/prevalence_percentage_2019.pdf", width = 12, height = 4)

plot_df <- GBD_full_df[which(GBD_full_df$year == 2019 & GBD_full_df$measure_name == "Prevalence" &
                               GBD_full_df$metric_name == "Rate"),]
total_val <- aggregate(plot_df$val, by = list(age_mid = plot_df$age_mid), FUN = sum)
ggplot(plot_df, aes(x = age_mid, y = val)) + theme_bw() + 
  geom_bar(aes(fill = cause_name), position="stack", stat="identity") + 
  xlab("Age") + ylab("Prevalence (current cases per 100,000)")
ggsave("figures/data/GBD/prevalence_rate_2019.pdf", width = 12, height = 4)

ggplot(total_val, aes()) + theme_bw() + ylim(0, max(total_val$x)) +
  geom_line(aes(x = age_mid, y = x), color = "black") + 
  xlab("Age") + ylab("Prevalence (current cases per 100,000)")
ggsave("figures/data/GBD/prevalence_total_2019.pdf", width = 4, height = 3.2)


# YLD
plot_df <- GBD_full_df[which(GBD_full_df$year == 2019 & GBD_full_df$measure_name == "YLD" &
                               GBD_full_df$metric_name == "Rate"),]
total_val <- aggregate(plot_df$val, by = list(age_mid = plot_df$age_mid), FUN = sum)
ggplot(plot_df, aes(x = age_mid, y = val)) + theme_bw() + 
  geom_bar(aes(fill = cause_name), position="stack", stat="identity") + 
  xlab("Age") + ylab("YLD per 100,000")
ggsave("figures/data/GBD/yld_rate_2019.pdf", width = 12, height = 4)

ggplot(total_val, aes()) + theme_bw() + ylim(0, max(total_val$x)) +
  geom_line(aes(x = age_mid, y = x), color = "black") + 
  xlab("Age") + ylab("YLD per 100,000")
ggsave("figures/data/GBD/yld_total_2019.pdf", width = 4, height = 3.2)

# YLD based health
plot_df <- GBD_full_df[which(GBD_full_df$year == 2019 & GBD_full_df$measure_name == "YLD" &
                               GBD_full_df$metric_name == "Rate"),]
plot_df$prop <- plot_df$val/100000
total_val <- aggregate(plot_df$prop, by = list(age_mid = plot_df$age_mid), FUN = sum)
total_val$health <- 1 - total_val$x
ggplot(plot_df, aes(x = age_mid, y = - prop)) + theme_bw() + theme(axis.text.y=element_blank()) +
  geom_line(data = total_val, aes(x = age_mid, y = health-1), color = "black") + 
  geom_bar(aes(fill = cause_name), position="stack", stat="identity") + 
  xlab("Age") + ylab("Health") + ylim(-1,0)
ggsave("figures/data/GBD/health_2019.pdf", width = 12, height = 4)


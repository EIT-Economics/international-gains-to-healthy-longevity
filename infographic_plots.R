setwd("/Users/julianashwin/Documents/GitHub/international-gains-to-healthy-longevity/")
rm(list = ls())

library(tidyverse)
library(readxl)


table_data <- read_csv("data/Table3.csv")
countries <- unique(table_data$country)

raw_data <- read_xlsx("figures/Olshansky_plots/Andrew_international.xlsx", sheet = "All Data")  %>%
  mutate(country = str_replace(str_replace(country, "United Kingdom", "UK"), "United States of America", "USA")) %>%
  select(country, year, real_gdp_pc) %>%
  filter(country %in% countries, year %in% c("1990", "1999", "2009", "2018")) %>%
  group_by(country) %>%
  mutate(gdp_growth_pc = real_gdp_pc - lag(real_gdp_pc), 
         period = str_c(lag(year), "-", year)) %>%
  drop_na(period)

table_data %>%
  mutate(period = str_replace(str_replace(period, "2000", "1999"), "2010", "2009")) %>%
  left_join(raw_data) %>%
  mutate(both = gdp_growth_pc + wtp_pc) %>%
  pivot_longer(cols = c(gdp_growth_pc, both)) %>%
  mutate(value_clean = sprintf(fmt = "%01.1f",value)) %>%
  mutate(name_clean = factor(case_when(name == "gdp_growth_pc" ~ "GDP growth per capita", 
                                name == "both" ~ "GDP + longevity gains per capita"))) %>%
  mutate(country = factor(country)) %>%
  ggplot(aes(x = (period), y = fct_rev(country), fill = value)) + theme_bw() +  
  facet_wrap(~fct_rev(name_clean)) + 
  geom_tile(color = "white") + 
  geom_text(aes(label = value_clean), color = "black", size = 3) + 
  scale_fill_gradient2(low = "red", high = "green", mid = "white", 
                       midpoint = 0, limit = c(-10,40), guide = "none") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1)) + 
  labs(x = "Period", y = "Country")
ggsave(paste0("figures/infographic/historical.pdf"), width = 5, height = 3.5)


table2_data <- read_csv("data/Table2.csv")

table2_data %>%
  pivot_longer(cols = -country) %>%
  mutate(value_clean = sprintf(fmt = "%01.1f",value)) %>%
  mutate(country = factor(country)) %>%
  mutate(name = factor(name, ordered = T, levels = c("Life Expectancy", "Healthy Life Expectancy", 
                                                     "Population (million people)", "Value of one extra year (trillion US$)"))) %>%
  ggplot(aes(x = (name), y = fct_rev(country))) + theme_bw() +  
  geom_tile(color = "grey", fill = "white") + 
  geom_text(aes(label = value_clean), color = "black", size = 3) + 
  scale_fill_gradient2(low = "red", high = "green", mid = "white", 
                       midpoint = 0, limit = c(-10,400), guide = "none") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1)) + 
  labs(x = "", y = "")
ggsave(paste0("figures/infographic/oneyear.pdf"), width = 3, height = 4.5)


  
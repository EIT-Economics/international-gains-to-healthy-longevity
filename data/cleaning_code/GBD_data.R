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
require(xlsx)





"
Import and clean up data
"

# Import and combine the international GBD data
GBD_df1 <- read.csv("data/GBD/International_GBD_1990-2019_DATA_1.csv",stringsAsFactors = FALSE)
GBD_df2 <- read.csv("data/GBD/International_GBD_1990-2019_DATA_2.csv",stringsAsFactors = FALSE)
GBD_df3 <- read.csv("data/GBD/International_GBD_1990-2019_DATA_3.csv",stringsAsFactors = FALSE)
GBD_full_df <- rbind(GBD_df1, GBD_df2, GBD_df3)
rm(GBD_df1, GBD_df2, GBD_df3)
# Drop some columns
names(GBD_full_df)
GBD_full_df <- GBD_full_df[,c("location_name", "measure_name", "age_name", "metric_name", 
                              "cause_name", "year", "val")]
# Clean up age name variable
table(GBD_full_df$age_name)
GBD_full_df$age_name[which(GBD_full_df$age_name == "<1 year")] <- "0 to 0"
GBD_full_df$age_name[which(GBD_full_df$age_name == "95 plus")] <- "95 to 99"
# Shorten the measure names
GBD_full_df$measure_name[which(GBD_full_df$measure_name == "YLDs (Years Lived with Disability)")] <- "YLD"
# Get numerical ages
GBD_full_df[,c("age_low", "age_high")] <-  
  do.call(rbind, str_split(as.character(GBD_full_df$age_name), " to "))
GBD_full_df$age_low <- as.numeric(GBD_full_df$age_low)
GBD_full_df$age_high <- as.numeric(GBD_full_df$age_high)
GBD_full_df$age <- round(GBD_full_df$age_low + (GBD_full_df$age_high - GBD_full_df$age_low)/2)
GBD_full_df <- GBD_full_df[,c("location_name", "measure_name", "year", "age_name", "age", "cause_name", "val")]

# Save list of countries and ages to help select population data
write.csv(table(GBD_full_df$location_name), "data/GBD/GBD_countries.csv", row.names = FALSE) 
write.csv(table(GBD_full_df$age_nam), "data/GBD/GBD_ages.csv", row.names = FALSE) 


# Import and merge the total population data (made by ~/Documents/Research/Targeting_Aging/data/GBD/pop_data/clean_combine.R)
GBD_pop_df <- read.csv("data/GBD/GBD_population.csv",stringsAsFactors = FALSE)
GBD_df <- merge(GBD_full_df, GBD_pop_df, by = c("location_name", "year", "age_name"))

rm(GBD_full_df, GBD_pop_df)



"
Plot population distribution
"
# Look only at US and add up the population for 9 and 1-4 so that buckets are evenly split
plot_df <- GBD_df[which(GBD_df$location_name == "United States of America" &
                          GBD_df$cause_name == "All causes" & GBD_df$measure_name == "YLD"),]
plot_df$population[which(plot_df$age == 2)] <- plot_df$population[which(plot_df$age == 0)] + 
  plot_df$population[which(plot_df$age == 2)]
plot_df <- plot_df[which(plot_df$age > 0),]
# Plot population over time
ggplot(plot_df, aes(x = age, y = population)) +
  geom_line(aes(group = year, color = year))


plot_df <- GBD_df[which(GBD_df$location_name == "Global" &
                          GBD_df$cause_name == "All causes" & GBD_df$measure_name == "YLD"),]
plot_df$population[which(plot_df$age == 2)] <- plot_df$population[which(plot_df$age == 0)] + 
  plot_df$population[which(plot_df$age == 2)]
plot_df <- plot_df[which(plot_df$age > 0),]
# Plot population over time
ggplot(plot_df, aes(x = age, y = population)) +
  geom_line(aes(group = year, color = year))

rm(plot_df)


"
Compute and export mortality rates for each country over time
"
# Just deaths
mort_df <- GBD_df[which(GBD_df$measure_name == "Deaths"),]
# Convert to a probability
mort_df$val <- mort_df$val/100000
# Mortality causes (not the same as disability causes)
mort_causes <- unique(mort_df$cause_name)
mort_causes <- mort_causes[which(mort_causes != "All causes")]

# Transform to wide
mort_df <- mort_df %>% 
  select(location_name, age_name, age, year, cause_name, population, val) %>%
  pivot_wider(names_from = cause_name, values_from = val) %>%
  arrange(location_name, year, age) %>%
  replace(is.na(.), 0)

# Check that "All causes" is roughly correct
mort_df$Total <- rowSums(mort_df[,mort_causes])
max(abs(mort_df$Total - mort_df$`All causes`))
mort_df <- mort_df[ , -which(names(mort_df) %in% c("All causes"))]

# Plot 2019 and 1990 curves
plt_mort1990 <- ggplot(mort_df[which(mort_df$year == 1990),], aes(x = age)) + 
  geom_line(aes(y = Total, color = as.factor(location_name))) + ylim(0,0.5) +
  ggtitle("1990 Death rates")
plt_mort2019 <- ggplot(mort_df[which(mort_df$year == 2019),], aes(x = age)) + 
  geom_line(aes(y = Total, color = as.factor(location_name))) + ylim(0,0.5) +
  ggtitle("2019 Death rates")
ggarrange(plt_mort1990,plt_mort2019, nrow = 2, ncol=1, common.legend = TRUE, 
          legend = "right")
rm(plt_mort1990, plt_mort2019)

# Export
write.csv(mort_df, "data/GBD/mortality_data.csv", row.names = FALSE)



" 
Compute and export disability rates for each country over time
"
# Just YLD
health_df <- GBD_df[which(GBD_df$measure_name == "YLD"),]
# Convert to a probability
health_df$val <- health_df$val/100000
# Health causes (not the same as disability causes)
health_causes <- unique(health_df$cause_name)
health_causes <- health_causes[which(health_causes != "All causes")]

# Transform to wide
health_df <- health_df %>% 
  select(location_name, age_name, age, year, cause_name, population, val) %>%
  pivot_wider(names_from = cause_name, values_from = val) %>%
  arrange(location_name, year, age) %>%
  replace(is.na(.), 0)

# Check that "All causes" is roughly correct
health_df$Total <- rowSums(health_df[,health_causes])
max(abs(health_df$Total - health_df$`All causes`))
health_df <- health_df[,-which(names(health_df) %in% c("All causes"))]

# Plot 2019 and 1990 curves
plt_health1990 <- ggplot(health_df[which(health_df$year == 1990),], aes(x = age)) + 
  geom_line(aes(y = Total, color = as.factor(location_name))) + ylim(0,0.5) +
  ggtitle("1990 YLD rates")
plt_health2019 <- ggplot(health_df[which(health_df$year == 2019),], aes(x = age)) + 
  geom_line(aes(y = Total, color = as.factor(location_name))) + ylim(0,0.5) +
  ggtitle("2019 YLD rates")
ggarrange(plt_health1990,plt_health2019, nrow = 2, ncol=1, common.legend = TRUE, 
          legend = "right")
rm(plt_health1990, plt_health2019)

# Export
write.csv(health_df, "data/GBD/health_data.csv", row.names = FALSE)

health_df$Total[which(health_df$location_name == "United States of America" & 
                        health_df$year == 2010)]



"
Real GDP data (in constant USD and LC) for each country from https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
"
# Import data
USD_GDP_df <- read.csv("data/WB_GDP_data/GDP_constant_USD.csv", stringsAsFactors = FALSE, 
                   skip = 4)
LC_GDP_df <- read.csv("data/WB_GDP_data/GDP_constant_PPP.csv", stringsAsFactors = FALSE, 
                       skip = 4)


# Remove some columns
LC_GDP_df$location_name <- LC_GDP_df$Country.Name
LC_GDP_df <- LC_GDP_df[,-which(names(LC_GDP_df) %in% c("Country.Name", "Country.Code", 
                                              "Indicator.Name", "Indicator.Code"))]
USD_GDP_df$location_name <- USD_GDP_df$Country.Name
USD_GDP_df <- USD_GDP_df[,-which(names(USD_GDP_df) %in% c("Country.Name", "Country.Code", 
                                                       "Indicator.Name", "Indicator.Code"))]




# Convert to long
LC_GDP_df <- melt(LC_GDP_df, id = c("location_name"), variable_name = "year")
LC_GDP_df$year <- as.numeric(str_remove(as.character(LC_GDP_df$year), "X"))
LC_GDP_df <- LC_GDP_df[which(LC_GDP_df$year >= 1990 & LC_GDP_df$year < 2020),]
LC_GDP_df$real_gdp_lc <- LC_GDP_df$value
LC_GDP_df <- LC_GDP_df[,c("location_name", "year", "real_gdp_lc")]
LC_GDP_df <- LC_GDP_df[order(LC_GDP_df$location_name, LC_GDP_df$year),]

USD_GDP_df <- melt(USD_GDP_df, id = c("location_name"), variable_name = "year")
USD_GDP_df$year <- as.numeric(str_remove(as.character(USD_GDP_df$year), "X"))
USD_GDP_df <- USD_GDP_df[which(USD_GDP_df$year >= 1990 & USD_GDP_df$year < 2020),]
USD_GDP_df$real_gdp_usd <- USD_GDP_df$value
USD_GDP_df <- USD_GDP_df[,c("location_name", "year", "real_gdp_usd")]
USD_GDP_df <- USD_GDP_df[order(USD_GDP_df$location_name, USD_GDP_df$year),]



# Rename some locations to match the GBD data
LC_GDP_df$location_name[which(LC_GDP_df$location_name == "United States")] <- "United States of America"
LC_GDP_df$location_name[which(LC_GDP_df$location_name == "World")] <- "Global"
LC_GDP_df$location_name[which(LC_GDP_df$location_name == "Czech Republic")] <- "Czechia"
LC_GDP_df$location_name[which(LC_GDP_df$location_name == "Iran, Islamic Rep.")] <- "Iran (Islamic Republic of)"

USD_GDP_df$location_name[which(USD_GDP_df$location_name == "United States")] <- "United States of America"
USD_GDP_df$location_name[which(USD_GDP_df$location_name == "World")] <- "Global"
USD_GDP_df$location_name[which(USD_GDP_df$location_name == "Czech Republic")] <- "Czechia"
USD_GDP_df$location_name[which(USD_GDP_df$location_name == "Iran, Islamic Rep.")] <- "Iran (Islamic Republic of)"


 
# Check which ones won't match and only keep those that will 
GBD_countries <- unique(GBD_df$location_name)
GBD_countries[which(GBD_countries %in% unique(LC_GDP_df$location_name))]
GBD_countries[which(!(GBD_countries %in% unique(LC_GDP_df$location_name)))]

LC_GDP_df <- LC_GDP_df[which(LC_GDP_df$location_name %in% GBD_countries),]
USD_GDP_df <- USD_GDP_df[which(USD_GDP_df$location_name %in% GBD_countries),]

LC_GDP_df$population <- NA
LC_GDP_df$real_gdp_usd <- NA

# Merge the total population in from the GBD data
for (loc_name in unique(LC_GDP_df$location_name)){
  # Calculate population
  loc_df <- mort_df[which(mort_df$location_name == loc_name),]
  loc_df <- aggregate(loc_df$population, by = list(year = loc_df$year), FUN = sum)
  # Combine LC_GDP_df and population
  stopifnot(all(LC_GDP_df[which(LC_GDP_df$location_name == loc_name),"year"] == loc_df$year))
  LC_GDP_df[which(LC_GDP_df$location_name == loc_name),"population"] = loc_df$x
  # Combine LC_GDP_df and USD_GDP_df
  us_df <- USD_GDP_df[which(USD_GDP_df$location_name == loc_name),]
  stopifnot(all(LC_GDP_df[which(LC_GDP_df$location_name == loc_name),"year"] == us_df$year))
  LC_GDP_df[which(LC_GDP_df$location_name == loc_name),"real_gdp_usd"] = us_df$real_gdp_usd
}


# Plot real GDP pc over time for each location
ggplot(LC_GDP_df, aes(x = year)) + 
  geom_line(aes(y = real_gdp_lc/population, color = as.factor(location_name))) +
  ylab("Real GDP p.c.") + ggtitle("PPP") + scale_fill_discrete(name = "Country")
ggplot(LC_GDP_df, aes(x = year)) + theme_bw() + 
  geom_line(aes(y = real_gdp_usd/population, color = as.factor(location_name))) +
  ylab("Real GDP p.c.  (2015 US$)") + ggtitle("") + scale_color_discrete(name = "Country")
ggsave("figures/Olshansky_plots/GDP_pc_all.pdf", width = 9, height = 6)



# Export
write.csv(LC_GDP_df, "data/GBD/real_gdp_data.csv", row.names = FALSE)







"
Quick analysis of results
"
results_df <- read.csv("figures/Olshansky_plots/international_comp.csv", stringsAsFactors = FALSE)
# Replace zeros for 2019 with NA
results_df[which(results_df$year == 2019),
           which(str_detect(names(results_df), "wtp"))] <- NA



###
# Reshape into Excel WB as Andrew wants it
###
new_vars <- c("Real GDP p.c.", "Real GDP + WTP p.c.", "Share Health Capital", 
              "Real GDP growth", "Real GDP + WTP growth")
results_df$real_gdp_pc_plus_wtp_pc <- results_df$real_gdp_pc + results_df$wtp_pc
results_df[,new_vars] <- NA
for (country in unique(results_df$country)){
  obs <- which(results_df$country == country)
  results_df[obs, new_vars[1]] <- results_df$real_gdp_pc[obs]
  results_df[obs, new_vars[2]] <- results_df$real_gdp_pc_plus_wtp_pc[obs]
  results_df[obs, new_vars[3]] <- results_df$wtp_pc[obs]/results_df$real_gdp_pc_plus_wtp_pc[obs]
  results_df[obs, new_vars[4]] <- 100*c(NA, diff(results_df$real_gdp_pc[obs]))/
    c(NA, results_df$real_gdp_pc[obs[1:(length(obs)-1)]])
  results_df[obs, new_vars[5]] <- 100*c(NA, diff(results_df$real_gdp_pc_plus_wtp_pc[obs]))/
    c(NA, results_df$real_gdp_pc_plus_wtp_pc[obs[1:(length(obs)-1)]])
}




reshape_df <- function(results_df, var_name){
  # Keep only this variable
  export_df <- results_df[,c("country", "year", var_name)]
  # Pivot to wide
  export_df <- as.data.frame(pivot_wider(export_df, names_from = country, values_from = var_name))
  return(export_df)
} 

# Create workbook and include all data
wb = createWorkbook()
sheet = createSheet(wb, "All Data")
addDataFrame(results_df, sheet=sheet, startColumn=1, row.names=FALSE)

for (var_name in new_vars){
  # Find and reshape data
  export_df <- reshape_df(results_df, var_name)
  # Add to wb as new sheet
  sheet = createSheet(wb, var_name)
  addDataFrame(export_df, sheet=sheet, startColumn=1, row.names=FALSE)
} 







saveWorkbook(wb, "figures/Olshansky_plots/Andrew_international.xlsx")




###
# US only 
###
plot_df <- results_df[which(results_df$country %in% c("Global", "United States of America")),]
wtp_plt <- ggplot(plot_df, aes(x = year)) + theme_bw() + xlab("Year") + 
  ylab("Trillion US$") + #ylim(c(-10,10)) +
  #geom_line(aes(y = wtp_pc/real_gdp_pc, color = country)) +
  geom_point(aes(y = wtp, color = country)) +
  geom_smooth(aes(y = wtp, color = country), method = "loess") +
  ggtitle("Total WTP for next year's distribution")
wtp_plt
wtp_gdp_plt <- ggplot(plot_df, aes(x = year)) + theme_bw() + xlab("Year") + ylab("WTP/GDP") +
  #ylim(c(-10,10)) +
  #geom_line(aes(y = wtp_pc/real_gdp_pc, color = country)) +
  geom_point(aes(y = wtp_pc/real_gdp_pc, color = country)) +
  geom_smooth(aes(y = wtp_pc/real_gdp_pc, color = country), method = "loess") +
  ggtitle("WTP/GDP for next year's distribution")
wtp_gdp_plt
ggarrange(wtp_plt,wtp_gdp_plt, nrow = 1, ncol=2, common.legend = TRUE)
ggsave("figures/Olshansky_plots/US_emp_WTP.pdf", width = 12, height = 4)

# Health versus Survival
all_numbers <- c((plot_df$wtp_s/plot_df$population)/plot_df$real_gdp_pc, 
                 (plot_df$wtp_h/plot_df$population)/plot_df$real_gdp_pc)
all_numbers[which(is.na(all_numbers))] <- 0
wtps_gdp_plt <- ggplot(plot_df, aes(x = year)) + theme_bw() + xlab("Year") + ylab("WTP_S/GDP") +
  ylim(c(min(all_numbers), max(all_numbers))) +
  geom_point(aes(y = (wtp_s/population)/real_gdp_pc, color = country)) +
  geom_smooth(aes(y = (wtp_s/population)/real_gdp_pc, color = country), method = "loess") +
  ggtitle("Survival")
wtps_gdp_plt
wtph_gdp_plt <- ggplot(plot_df, aes(x = year)) + theme_bw() + xlab("Year") + ylab("WTP_H/GDP") +
  ylim(c(min(all_numbers), max(all_numbers))) +
  geom_point(aes(y = (wtp_h/population)/real_gdp_pc, color = country)) +
  geom_smooth(aes(y = (wtp_h/population)/real_gdp_pc, color = country), method = "loess") +
  ggtitle("Health")
wtph_gdp_plt
ggarrange(wtps_gdp_plt,wtph_gdp_plt, nrow = 1, ncol=2, common.legend = TRUE)
ggsave("figures/Olshansky_plots/US_emp_WTP_SandH.pdf", width = 12, height = 4)



###
# A few selected countries
###
plot_df <- results_df[which(results_df$country %in% c("United Kingdom", "France", "Germany", 
                                                      "Italy", "Spain", "Netherlands", 
                                                      "Denmark", "Sweden", "European Union")),]
wtp_gdp_plt1 <- ggplot(plot_df, aes(x = year)) + theme_bw() + xlab("Year") +
  ylab("WTP/GDP") + #ylim(c(-10,10)) +
  geom_smooth(aes(y = wtp_pc/real_gdp_pc, color = country), method = "loess", se = FALSE) +
  ggtitle("Western Europe")
wtp_gdp_plt1 
plot_df <- results_df[which(results_df$country %in% c("Bangladesh", "China", "India", "Indonesia", "Japan",
                                                      "Iran (Islamic Republic of)")),]
wtp_gdp_plt2 <- ggplot(plot_df, aes(x = year)) + theme_bw() + xlab("Year") +
  ylab("WTP/GDP") + #ylim(c(-10,10)) +
  geom_smooth(aes(y = wtp_pc/real_gdp_pc, color = country), method = "loess", se = FALSE) +
  ggtitle("Asia")
wtp_gdp_plt2
plot_df <- results_df[which(results_df$country %in% c("Argentina","Brazil", "Canada", "Mexico", "Peru", 
                                                      "United States of America", "Uruguay", "Chile")),]
wtp_gdp_plt3 <- ggplot(plot_df, aes(x = year)) + theme_bw() + xlab("Year") +
  ylab("WTP/GDP") + #ylim(c(-10,10)) +
  geom_smooth(aes(y = wtp_pc/real_gdp_pc, color = country), method = "loess", se = FALSE) +
  ggtitle("Americas")
wtp_gdp_plt3
plot_df <- results_df[which(results_df$country %in% c("Kenya", "Nigeria", "South Africa")),]
wtp_gdp_plt4 <- ggplot(plot_df, aes(x = year)) + theme_bw() + xlab("Year") +
  ylab("WTP/GDP") + #ylim(c(-10,10)) +
  geom_smooth(aes(y = wtp_pc/real_gdp_pc, color = country), method = "loess", se = FALSE) +
  ggtitle("Africa")
wtp_gdp_plt4
plot_df <- results_df[which(results_df$country %in% c("Australia","New Zealand", "Russian Federation", 
                                                      "Czechia", "Poland", "Israel", "Turkey")),]
wtp_gdp_plt5 <- ggplot(plot_df, aes(x = year)) + theme_bw() + xlab("Year") +
  ylab("WTP/GDP") + #ylim(c(-10,10)) +
  geom_smooth(aes(y = wtp_pc/real_gdp_pc, color = country), method = "loess", se = FALSE) +
  ggtitle("Other")
wtp_gdp_plt5
# Group together
ggarrange(wtp_gdp_plt1, wtp_gdp_plt2, wtp_gdp_plt3, wtp_gdp_plt4, wtp_gdp_plt5, 
          nrow = 3, ncol=2)
ggsave("figures/Olshansky_plots/grouped_emp_WTP.pdf", width = 12, height = 10)


###
# All countries
###
plot_df <- results_df
ggplot(plot_df, aes(x = year)) + theme_bw() + xlab("Year") + ylab("WTP/GDP") +
  ylim(c(-10,10)) +
  #geom_line(aes(y = wtp_pc/real_gdp_pc, color = country)) +
  geom_point(aes(y = wtp_pc/real_gdp_pc, color = country), size = 0.2) +
  geom_smooth(aes(y = wtp_pc/real_gdp_pc), method = "loess") +
  ggtitle("All countries/regions")
ggsave("figures/Olshansky_plots/all_WTP.pdf", width = 10, height = 6)



le_plt <- ggplot(plot_df, aes(x = year)) + theme_bw() + xlab("Year") + ylab("LE") +
  geom_line(aes(y = le, color = country)) +
  geom_smooth(aes(y = le), method = "loess", color = "black") +
  ggtitle("Life Expectancy")
hle_plt <- ggplot(plot_df, aes(x = year)) + theme_bw() + xlab("Year") + ylab("HLE") +
  geom_line(aes(y = hle, color = country)) +
  geom_smooth(aes(y = hle), method = "loess", color = "black") +
  ggtitle("Healthy Life Expectancy")

ggarrange(le_plt, hle_plt, common.legend = TRUE,
          nrow = 1, ncol=2)
ggsave("figures/Olshansky_plots/all_LE_HLE.pdf", width = 10, height = 6)




ggplot(plot_df, aes(x = year)) + theme_bw() + xlab("Year") + ylab("Thousand US$") +
  geom_line(aes(y = wtp_pc, color = country)) +
  ggtitle("WTP per capita for next year's distribution")



plot_df <- results_df[which(results_df$country %in% c("Italy", "United States of America",
                                                      "Japan", "Global")),]
ggplot(plot_df, aes(x = year)) + theme_bw() + xlab("Year") + ylab("Thousand US$") +
  geom_line(aes(y = le, color = country)) +
  geom_line(aes(y = hle, color = country), linetype = "dashed") +
  ggtitle("WTP per capita for next year's distribution")



























"
# Some more exploratory plots
"
# Select country and year
country <- "United States of America"
country <- "Netherlands"
country <- "Global"
year <- 2019
GBD_full_df <- GBD_df[which(GBD_df$location_name == country & 
                              GBD_df$cause_name != "All causes"),]

# Plot death rates














"
End of script
"

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


"
Population
"
# Estimates of historical populations
estimates_df <- as.data.frame(read_xlsx("data/WPP/UN_WPP2019_popbyage.xlsx", 
                                        sheet = "ESTIMATES", skip = 16))
estimates_df <- estimates_df[which(estimates_df$Type == "Country/Area"),]
estimates_df$Country <- estimates_df[,"Region, subregion, country or area *"]
estimates_df$Year <- estimates_df[,"Reference date (as of 1 July)"]
estimates_df$`100-104` <- estimates_df$`100+`
estimates_df <- estimates_df[,c("Country", "Variant", "Year", "0-4", "5-9", "10-14",
                                "15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
                                "45-49", "50-54", "55-59", "60-64", "65-69", "70-74",
                                "75-79", "80-84", "85-89", "90-94", "95-99", "100-104")]
age_cols <- which(str_detect(names(estimates_df), "[0-9]+"))
for (aa in age_cols){
  estimates_df[,aa] <- as.numeric(estimates_df[,aa])
}
rownames(estimates_df) <- NULL
# Convert to long format
estimates_df <- melt(estimates_df, id = c("Country", "Variant", "Year"),
                     variable_name = "Age")
# Clean up age variable
estimates_df[,c("Age_low", "Age_high")] <- do.call(rbind, str_split(as.character(estimates_df$Age), "-"))
estimates_df$Age_low <- as.numeric(estimates_df$Age_low)
estimates_df$Age_high <- as.numeric(estimates_df$Age_high)
estimates_df$Age_mid <- estimates_df$Age_low + (estimates_df$Age_high - estimates_df$Age_low)/2
# Export
write.csv(estimates_df, "data/WPP/WPP_estimates.csv", row.names = FALSE)



# Projections of future populations
proj_med_df <- as.data.frame(read_xlsx("data/WPP/UN_WPP2019_popbyage.xlsx", 
                                       sheet = "MEDIUM VARIANT", skip = 16))
proj_med_df <- proj_med_df[which(proj_med_df$Type == "Country/Area"),]
proj_med_df$Country <- proj_med_df[,"Region, subregion, country or area *"]
proj_med_df$Year <- proj_med_df[,"Reference date (as of 1 July)"]
proj_med_df$`100-104` <- proj_med_df$`100+`
proj_med_df <- proj_med_df[,c("Country", "Variant", "Year", "0-4", "5-9", "10-14",
                                "15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
                                "45-49", "50-54", "55-59", "60-64", "65-69", "70-74",
                                "75-79", "80-84", "85-89", "90-94", "95-99", "100-104")]
age_cols <- which(str_detect(names(proj_med_df), "[0-9]+"))
for (aa in age_cols){
  proj_med_df[,aa] <- as.numeric(proj_med_df[,aa])
}
rownames(proj_med_df) <- NULL
# Convert to long format
proj_med_df <- melt(proj_med_df, id = c("Country", "Variant", "Year"),
                    variable_name = "Age")
# Clean up age variable
proj_med_df[,c("Age_low", "Age_high")] <- do.call(rbind, str_split(as.character(proj_med_df$Age), "-"))
proj_med_df$Age_low <- as.numeric(proj_med_df$Age_low)
proj_med_df$Age_high <- as.numeric(proj_med_df$Age_high)
proj_med_df$Age_mid <- proj_med_df$Age_low + (proj_med_df$Age_high - proj_med_df$Age_low)/2
# Export
write.csv(proj_med_df, "data/WPP/WPP_projections.csv", row.names = FALSE)





"
Life expectancy
"
# Life expectancy estimates
le_est_df <- as.data.frame(read_xlsx("data/WPP/UN_WPP2019_lifeexpectancy.xlsx", 
                                       sheet = "ESTIMATES", skip = 16))
le_est_df <- le_est_df[which(le_est_df$Type == "Country/Area"),]
le_est_df$Country <- le_est_df[,"Region, subregion, country or area *"]
le_est_df <- le_est_df[,c("Country", "Variant", "1950-1955", "1955-1960", "1960-1965",
                          "1965-1970", "1970-1975","1975-1980", "1980-1985",
                          "1985-1990", "1990-1995", "1995-2000", "2000-2005", 
                          "2005-2010", "2010-2015", "2015-2020")]
year_cols <- which(str_detect(names(le_est_df), "[0-9]+"))
for (yy in year_cols){
  le_est_df[,yy] <- as.numeric(le_est_df[,yy])
}
# Convert to long format
le_est_df <- melt(le_est_df, id = c("Country", "Variant"),
                    variable_name = "Year")
# Clean up year variable
le_est_df[,c("Year_low", "Year_high")] <- do.call(rbind, str_split(as.character(le_est_df$Year), "-"))
le_est_df$Year_low <- as.numeric(le_est_df$Year_low)
le_est_df$Year_high <- as.numeric(le_est_df$Year_high)
le_est_df$Year_mid <- le_est_df$Year_low + (le_est_df$Year_high - le_est_df$Year_low)/2
# Export
write.csv(le_est_df, "data/WPP/WPP_LE_estimates.csv", row.names = FALSE)




# Life expectancy projections
le_proj_df <- as.data.frame(read_xlsx("data/WPP/UN_WPP2019_lifeexpectancy.xlsx", 
                                     sheet = "MEDIUM VARIANT", skip = 16))
le_proj_df <- le_proj_df[which(le_proj_df$Type == "Country/Area"),]
le_proj_df$Country <- le_proj_df[,"Region, subregion, country or area *"]
le_proj_df <- le_proj_df[,c("Country", "Variant", "2020-2025", "2025-2030", "2030-2035", 
                            "2035-2040", "2040-2045", "2045-2050", "2050-2055", "2055-2060", 
                            "2060-2065", "2065-2070", "2070-2075", "2075-2080", "2080-2085", 
                            "2085-2090", "2090-2095", "2095-2100" )]
year_cols <- which(str_detect(names(le_proj_df), "[0-9]+"))
for (yy in year_cols){
  le_proj_df[,yy] <- as.numeric(le_proj_df[,yy])
}
# Convert to long format
le_proj_df <- melt(le_proj_df, id = c("Country", "Variant"),
                  variable_name = "Year")
# Clean up year variable
le_proj_df[,c("Year_low", "Year_high")] <- do.call(rbind, str_split(as.character(le_proj_df$Year), "-"))
le_proj_df$Year_low <- as.numeric(le_proj_df$Year_low)
le_proj_df$Year_high <- as.numeric(le_proj_df$Year_high)
le_proj_df$Year_mid <- le_proj_df$Year_low + (le_proj_df$Year_high - le_proj_df$Year_low)/2
# Export
write.csv(le_proj_df, "data/WPP/WPP_LE_projections.csv", row.names = FALSE)





"
Fertility
"
# Life expectancy estimates
fert_est_df <- as.data.frame(read_xlsx("data/WPP/UN_WPP2019_fertility.xlsx", 
                                     sheet = "ESTIMATES", skip = 16))
fert_est_df <- fert_est_df[which(fert_est_df$Type == "Country/Area"),]
fert_est_df$Country <- fert_est_df[,"Region, subregion, country or area *"]
fert_est_df <- fert_est_df[,c("Country", "Variant", "1950-1955", "1955-1960", "1960-1965",
                          "1965-1970", "1970-1975","1975-1980", "1980-1985",
                          "1985-1990", "1990-1995", "1995-2000", "2000-2005", 
                          "2005-2010", "2010-2015", "2015-2020")]
year_cols <- which(str_detect(names(fert_est_df), "[0-9]+"))
for (yy in year_cols){
  fert_est_df[,yy] <- as.numeric(fert_est_df[,yy])
}
# Convert to long format
fert_est_df <- melt(fert_est_df, id = c("Country", "Variant"),
                  variable_name = "Year")
# Clean up year variable
fert_est_df[,c("Year_low", "Year_high")] <- do.call(rbind, str_split(as.character(fert_est_df$Year), "-"))
fert_est_df$Year_low <- as.numeric(fert_est_df$Year_low)
fert_est_df$Year_high <- as.numeric(fert_est_df$Year_high)
fert_est_df$Year_mid <- fert_est_df$Year_low + (fert_est_df$Year_high - fert_est_df$Year_low)/2
# Export
write.csv(fert_est_df, "data/WPP/WPP_fertility_estimates.csv", row.names = FALSE)



# Life expectancy projections
fert_proj_df <- as.data.frame(read_xlsx("data/WPP/UN_WPP2019_fertility.xlsx", 
                                      sheet = "MEDIUM VARIANT", skip = 16))
fert_proj_df <- fert_proj_df[which(fert_proj_df$Type == "Country/Area"),]
fert_proj_df$Country <- fert_proj_df[,"Region, subregion, country or area *"]
fert_proj_df <- fert_proj_df[,c("Country", "Variant", "2020-2025", "2025-2030", "2030-2035", 
                            "2035-2040", "2040-2045", "2045-2050", "2050-2055", "2055-2060", 
                            "2060-2065", "2065-2070", "2070-2075", "2075-2080", "2080-2085", 
                            "2085-2090", "2090-2095", "2095-2100" )]
year_cols <- which(str_detect(names(fert_proj_df), "[0-9]+"))
for (yy in year_cols){
  fert_proj_df[,yy] <- as.numeric(fert_proj_df[,yy])
}
# Convert to long format
fert_proj_df <- melt(fert_proj_df, id = c("Country", "Variant"),
                   variable_name = "Year")
# Clean up year variable
fert_proj_df[,c("Year_low", "Year_high")] <- do.call(rbind, str_split(as.character(fert_proj_df$Year), "-"))
fert_proj_df$Year_low <- as.numeric(fert_proj_df$Year_low)
fert_proj_df$Year_high <- as.numeric(fert_proj_df$Year_high)
fert_proj_df$Year_mid <- fert_proj_df$Year_low + (fert_proj_df$Year_high - fert_proj_df$Year_low)/2
# Export
write.csv(fert_proj_df, "data/WPP/WPP_fertility_projections.csv", row.names = FALSE)


#ggplot(fert_proj_df[which(fert_proj_df$Country =="United States of America" ),])+
#  geom_line(aes(x = Year_low, y = value))















# GDP data and projections from OECD 
GDP_df <- read.csv("data/OECD_GDPLT.csv", stringsAsFactors = FALSE)
GDP_df$Code <- GDP_df$LOCATION
GDP_df <- GDP_df[which(!(GDP_df$Code %in% c("EA17", "G20", "G7M", "OECD"))),]
GDP_df$Year <- GDP_df$TIME
GDP_df$GDP <- GDP_df$Value
GDP_df <- GDP_df[,c("Code", "Year", "GDP")]
# Import WB country codes in order to add names to OECD data
WB_country_codes <- read.csv("data/WB_GDP_data/Metadata_Country_API_NY.GDP.MKTP.CD_DS2_en_csv_v2_3469429.csv",
                             stringsAsFactors = FALSE)
WB_country_codes$Code <- WB_country_codes$Country.Code
WB_country_codes$Country <- WB_country_codes$TableName
WB_country_codes$Country[which(WB_country_codes$Country == "United States")] <- "United States of America"
WB_country_codes <- WB_country_codes[,c("Code", "Country")]
# Merge into GDP data to give country names
GDP_df <- merge(WB_country_codes, GDP_df, by = "Code", all.y = TRUE)
# Export
write.csv(GDP_df, "data/WB_GDP_data/GDP_data.csv", row.names = FALSE)









# Inspect Martin's matlab files
require(R.matlab)




WTP_1LE <- as.data.frame(readMat("/Users/julianashwin/Downloads/WTP_1st_1LE.mat"))
WTP_1LE$WTP.H[which(is.infinite(WTP_1LE$WTP.H))] <- NA
WTP_1LE$WTP.S[which(is.infinite(WTP_1LE$WTP.S))] <- NA
WTP_1LE$WTP <- WTP_1LE$WTP.H + WTP_1LE$WTP.S
WTP_1LE$age <- 0:240

ggplot(WTP_1LE) + xlab("age") + ylab("WTP") + theme_bw() + 
  geom_line(aes(x = age, y = WTP, color = "WTP")) +
  geom_line(aes(x = age, y = WTP.H, color = "WTP_H")) +
  geom_line(aes(x = age, y = WTP.S, color = "WTP_S")) 




Pdata <- as.data.frame(readMat("/Users/julianashwin/Downloads/Pdata_0.mat"))




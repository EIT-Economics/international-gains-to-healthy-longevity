# Notes

## Input/outputs we need

### Tables 

`Table2.csv` - Summary Statistics by Country
- Content: Life Expectancy, Healthy Life Expectancy, Population (million people), Value of one extra year (trillion US$)
- Structure: One row per country with key summary metrics
- Current usage: Consumed by infographic_plots.R for the summary statistics visualization

`Table3.csv` - Time Series Data with GDP and Longevity Gains
- Content: period, country, gdp_pc, wtp_pc (willingness-to-pay per capita)
- Structure: Time series data showing GDP growth vs. GDP + longevity gains across periods
- Current usage: Consumed by infographic_plots.R for the historical heatmap visualization

These tables are most likely created by:
`international_empirical.jl` - This script:
- Processes international data across countries and years
- Calculates WTP (willingness-to-pay) values
- Outputs international_comp.csv which likely gets processed into `Table3.csv` ?

`social_WTP.jl` - This script:
- Creates comprehensive WTP tables with country-level statistics
- Outputs social_wtp_table.csv which could be processed into Table2.csv
- Contains the social WTP calculations that would generate the summary metrics

Potential Link:
- international_empirical.jl → processes it into Table3.csv
- social_WTP.jl → processes it into Table2.csv


### Data

`Andrew_international.xlsx`

Excel File Structure (Sheet 1: "All Data"):
- Contains the full results dataframe with all variables
- One row per country-year combination
- Includes original variables plus calculated ones
- Current usage: consumed by infographic_plots.R - Reads the "All Data" sheet to create visualizations

Potential link:
- international_empirical.jl → international_comp.csv → GBD_data.R → Andrew_international.xlsx




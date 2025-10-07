# R-to-Python Data Processing Script

## Overview

We now use a Python script (`data_processing.py`) that efficiently emulates all the R data processing functionality from the original scripts.

## Functionality Comparison

| R Script                  | Python Equivalent                  | Status      |
| ------------------------- | ---------------------------------- | ----------- |
| `GBD_data.R`            | `process_gbd_data()`             | ✅ Complete |
| `GBD_data_USdeepdive.R` | `process_us_detailed_analysis()` | Eliminated  |
| `WPP_data.R`            | `process_wpp_data()`             | ✅ Complete |
| GDP processing            | `process_gdp_data()`             | ✅ Complete |
| Exploratory plotting      | `create_exploratory_plots()`     | ✅ Complete |

### Usage

### Basic Usage

```python
from data_processing_python import DataProcessor

# Create processor instance
processor = DataProcessor()

# Run all processing
results = processor.run_all_processing()
```

### Individual Processing

```python
# Process specific data sources
gbd_results = processor.process_gbd_data()
wpp_results = processor.process_wpp_data()
gdp_results = processor.process_gdp_data()
```

## Data Sources Processed

### 1. **Global Burden of Disease (GBD)**

- **Input**: Raw CSV files in `./data/GBD/` (International_GBD_1990-2019_DATA_*.csv, US_GBD_1990-2019_DATA.csv, GBD_population.csv)
- **Processing**:
  - Combines multiple GBD files
  - Cleans age groups and country names
  - Processes mortality and disability rates
  - Creates wide-format datasets
- **Output**: Processed files saved to `./intermediate/GBD`
  - `mortality_rates.csv` - Mortality rates by country, age, year, cause
  - `morbidity_rates.csv` - Health/Disability rates by country, age, year, cause

### 2. **UN World Population Prospects (WPP)**

- **Input**: Raw Excel files in `./data/WPP/` (UN_WPP2019_*.xlsx)
- **Processing**:
  - Historical population estimates and projections
  - Life expectancy data (historical and projected)
  - Fertility data (historical and projected)
  - Age group standardization
- **Output**: Processed files saved to `./intermediate/WPP/`
  - `population.csv` - Historical + future population data/estimates
  - `life_expectancy.csv` - Historical + future LE data/estimates
  - `fertility.csv` - Historical + future fertility data/estimates

### 3. **World Bank GDP Data**

- **Input**: Raw CSV files in `./data/WB/` (GDP_constant_*.csv)
- **Processing**:
  - USD and PPP GDP data
  - Country name standardization
  - Time series data (1990-2019)
- **Output**: Processed file saved to `./intermediate/WB/`
  - `real_gdp_data.csv` - GDP data with population

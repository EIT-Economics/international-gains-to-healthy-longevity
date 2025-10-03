# Python Data Processing Script

## Overview

This directory now contains a comprehensive Python script (`data_processing.py`) that efficiently emulates all the R data processing functionality from the original R scripts. The Python script provides the same data cleaning, processing, and output capabilities as the R scripts but with improved performance and modern Python data science tools.

## Files

### `data_processing_python.py`

**Purpose**: Complete Python equivalent of all R data processing scripts

**Key Features**:

- **Object-Oriented Design**: Clean, modular `DataProcessor` class
- **Comprehensive Processing**: Handles all data sources (GBD, WPP, GDP, US-specific)
- **Error Handling**: Robust error handling and data validation
- **Performance**: Efficient pandas operations for large datasets
- **Visualization**: Built-in exploratory plotting capabilities

## Functionality Comparison

| R Script                  | Python Equivalent                  | Status      |
| ------------------------- | ---------------------------------- | ----------- |
| `GBD_data.R`            | `process_gbd_data()`             | ✅ Complete |
| `GBD_data_USdeepdive.R` | `process_us_detailed_analysis()` | ✅ Complete |
| `WPP_data.R`            | `process_wpp_data()`             | ✅ Complete |
| GDP processing            | `process_gdp_data()`             | ✅ Complete |
| Exploratory plotting      | `create_exploratory_plots()`     | ✅ Complete |

## Key Differences with R

### 1. **Performance**

- **Pandas**: Optimized data operations for large datasets
- **Vectorized Operations**: Faster than R loops for data manipulation

### 2. **Code Organization**

- **Class-Based**: Clean, maintainable object-oriented design
- **Modular Functions**: Each data source has dedicated processing method
- **Error Handling**: Comprehensive error handling and validation

### 3. **Visualization**

- **Matplotlib/Seaborn**: Modern, publication-ready plots
- **Automatic Plotting**: Generates exploratory plots automatically
- **High-Quality Output**: PDF plots with customizable styling

## Usage

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
us_results = processor.process_us_detailed_analysis()
```

## Data Sources Processed

### 1. **Global Burden of Disease (GBD)**

- **Input**: Raw CSV files in `./data/GBD/` (International_GBD_1990-2019_DATA_*.csv, US_GBD_1990-2019_DATA.csv, GBD_population.csv)
- **Processing**:
  - Combines multiple GBD files
  - Cleans age groups and country names
  - Processes mortality and disability rates
  - Creates wide-format datasets
- **Output**: Processed files saved to `./intermediate/`
  - `GBD_mortality_data.csv` - Mortality rates by country, age, year, cause
  - `GBD_health_data.csv` - Disability rates by country, age, year, cause
  - `GBD_countries.csv` - Country list
  - `GBD_ages.csv` - Age group list

### 2. **UN World Population Prospects (WPP)**

- **Input**: Raw Excel files in `./data/WPP/` (UN_WPP2019_*.xlsx)
- **Processing**:
  - Historical population estimates and projections
  - Life expectancy data (historical and projected)
  - Fertility data (historical and projected)
  - Age group standardization
- **Output**: Processed files saved to `./intermediate/`
  - `WPP_estimates.csv` - Historical population data
  - `WPP_projections.csv` - Future population projections
  - `WPP_LE_estimates.csv` - Historical life expectancy
  - `WPP_LE_projections.csv` - Future life expectancy
  - `WPP_fertility_estimates.csv` - Historical fertility
  - `WPP_fertility_projections.csv` - Future fertility

### 3. **World Bank GDP Data**

- **Input**: Raw CSV files in `./data/WB/` (GDP_constant_*.csv)
- **Processing**:
  - USD and PPP GDP data
  - Country name standardization
  - Time series data (1990-2019)
- **Output**: Processed file saved to `./intermediate/`
  - `WB_real_gdp_data.csv` - GDP data with population

### 4. **US-Specific Analysis**

- **Input**: Raw CSV file in `./data/GDB/` (US_GBD_1990-2019_DATA.csv)
- **Processing**:
  - Detailed cause-specific analysis
  - Mortality and disability trends
  - Temporal comparisons
- **Output**: Enhanced mortality and health datasets saved to `./intermediate/GDB_*_US.csv`

## Directory Structure

The repository has been refactored to separate raw input data from processed data and figures:

```
├── data/                    # Raw input data files
│   ├── GBD/
│   ├── WB_GDP/
│   ├── WPP/
├── intermediate/            # Processed data files
│   ├── GBD_mortality_data.csv
│   ├── GBD_health_data.csv
│   ├── WB_real_gdp_data.csv
│   ├── GBD_countries.csv
│   ├── GBD_ages.csv
│   ├── WPP_estimates.csv
│   ├── WPP_projections.csv
│   ├── WPP_LE_estimates.csv
│   ├── WPP_LE_projections.csv
│   ├── WPP_fertility_estimates.csv
│   └── WPP_fertility_projections.csv
└── figures/                 # Generated plots and visualizations
```

## Future Enhancements

### Planned Features

- **Parallel Processing**: Multi-core processing for large datasets?
- **Data Validation**: Enhanced data quality checks?
- **Interactive Plots**: Jupyter notebook integration?
- **API Integration**: Direct data source APIs?

---

*For detailed usage examples and advanced features, see the docstrings in `data_processing.py`.*

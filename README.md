# International Gains to Healthy Longevity

Code for *"International Gains to Achieving Healthy Longevity" (2023)* in Cold Spring Harbor Perspectives in Medicine, by Andrew Scott, Julian Ashwin, Martin Ellison, and David Sinclair.

## Project Overview

This repository contains a Python implementation for quantifying the **economic value of healthy longevity gains** across countries and time periods. The codebase consolidates all functionality from the original Julia and R scripts into a more maintainable and extensible Python framework.

## Research Objectives

### Primary Goals

1. **Quantify Economic Value**: Calculate the willingness-to-pay (WTP) for health improvements and longevity gains across multiple countries
2. **International Comparison**: Analyze health and economic outcomes across diverse countries and time periods (1990-2019)
3. **Methodological Innovation**: Develop a comprehensive economic model that integrates health, mortality, and economic factors
4. **Policy Implications**: Demonstrate the economic case for investing in healthy aging interventions

## Methodology

### Economic Modeling Framework

The project employs a **life-cycle economic model** that integrates:

#### **Biological Model** (`models/biological_model.py`)

- **Survival Functions**: Age-specific mortality rates and survival probabilities
- **Health Functions**: Disability rates and healthy life expectancy calculations
- **Frailty Dynamics**: Mathematical modeling of biological aging processes
- **Disease-Specific Analysis**: Cause-specific mortality and disability patterns

#### **Economic Model** (`models/economic_model.py`)

- **Life-Cycle Optimization**: Intertemporal consumption and leisure choices
- **Wage Functions**: Health-dependent productivity and earnings
- **Value of Statistical Life (VSL)**: Economic valuation of mortality risk reduction
- **Willingness-to-Pay Calculations**: Economic value of health improvements

### Data Sources

- **Global Burden of Disease (GBD)**: Health, mortality, and disability data (1990-2019)
- **UN World Population Prospects (WPP)**: Population, life expectancy, and fertility data
- **World Bank GDP Data**: Economic indicators in USD and local currency
- **WHO Health Data**: Life expectancy and healthy life expectancy metrics

## Analysis Pipeline

### 1. **Data Processing** (`analysis/data_processing.py`)

- Clean and standardize international health and economic data
- Merge multiple data sources with consistent country/year classifications
- Validate data consistency and handle missing values

### 2. **Economic Modeling** (`models/`, `analysis/`)

- **Country-Specific Analysis**: Tailored modeling for different economic contexts
- **Scenario Analysis**: Multiple intervention scenarios and parameter sensitivity

### 3. **Visualization** (`visualization/`)

- **Comparative Analysis**: International comparisons of health and economic outcomes
- **Interactive Plots**: 3D visualizations and exploratory data analysis

## Repository Structure

```
├── models/                          # Core model classes
│   ├── biological_model.py          # Biological aging models
│   ├── economic_model.py           # Economic optimization models
│   └── health_longevity_model.py   # Unified model framework
├── analysis/                        # Analysis modules
│   ├── data_processing.py          # Data processing pipeline
│   ├── international_analysis.py   # Cross-country analysis
│   ├── scenario_analysis.py       # Intervention scenarios
│   └── social_welfare.py          # Social welfare analysis
├── visualization/                   # Plotting modules
│   ├── plot_3d_surfaces.py         # 3D surface plots
│   ├── plot_comparisons.py         # Comparison plots
│   └── plotting.py    		     # Publication figures
├── main.py                          # Main execution script
├── data/                            # Raw input data
│   ├── GBD/                         # Global Burden of Disease data
│   ├── WPP/                         # UN Population Prospects data
│   └── WB/                          # World Bank economic data
├── intermediate/                    # Generated intermediate data
├── figures/                         # Generated visualizations
├── output/                          # Generated outputs
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Getting Started

### Prerequisites

- **Python** (≥3.10) with required packages
- **Data files** (see data/ directory structure)

### Installation

To install Python dependencies, simply run

```bash
pip install -r requirements.txt
```

### Running the Analysis

#### Command Line Interface

```bash
# Process raw data and run all analyses
python main.py --process_data --analysis all

# Run specific analysis
python main.py --analysis international --countries "United States of America" "United Kingdom" --years 2010 2015 2019

# Run 3D surface analysis
python main.py --analysis 3d_surfaces

# Run scenario analysis
python main.py --analysis scenarios

# Create infographic plots
python main.py --create_infographic
```

#### Programmatic Interface

```python
from main import HealthLongevityAnalysis

# Initialize analysis
analysis = HealthLongevityAnalysis()

# Run specific analysis
results = analysis.run_international_analysis(
    countries=["United States of America", "United Kingdom"],
    years=[2010, 2015, 2019]
)

# Run all analyses
all_results = analysis.run_all_analyses()
```

## TODO

### Continued Workflow Integration

To create the final figures, the old Julia-based workflow looks something like this:
```
TargetingAging.jl + GBD data + WPP data → social WTP.jl → social wtp table.csv → Table2.csv → Publication figures

TargetingAging.jl + GBD data → international_empirical.jl → international_comp.csv + GBD data → Andrewinternational.xlsx → Table3.csv → Publication figures 
```

Our new Python-based workflow now looks like this:
Julia Economic Modeling → Results Tables → Python Visualization → Publication Figures
```
international_analysis.py + ./intermediate data → international_analysis.csv → Publication figures

social_welfare.py + ./intermediate data → social_welfare_analysis.csv → Publication figures
```

See @Sanj [Overleaf](https://www.overleaf.com/project/68e3b39f08accb2be4d82d51) for more details.

What we need to do is go through `international_analysis.py`, `social_welfare.py`, and `models/*` to make sure we can do the analysis appropriately.
# International Gains to Healthy Longevity

**Code for "International Gains to Achieving Healthy Longevity" (2023)**
**Authors:** Andrew Scott, Julian Ashwin, Martin Ellison, and David Sinclair

## Project Overview

This repository contains the complete computational pipeline for quantifying the **economic value of healthy longevity gains** across countries and time periods...

## Research Objectives

### Primary Goals

1. **Quantify Economic Value**: Calculate the willingness-to-pay (WTP) for health improvements and longevity gains across multiple countries
2. **International Comparison**: Analyze health and economic outcomes across diverse countries and time periods (1990-2019)
3. **Methodological Innovation**: Develop a comprehensive economic model that integrates health, mortality, and economic factors
4. **Policy Implications**: Demonstrate the economic case for investing in healthy aging interventions
   ...

## Methodology

### Economic Modeling Framework

The project employs a **life-cycle economic model** that integrates:

#### **Biological Model** (`src/biology.jl`)

- **Survival Functions**: Age-specific mortality rates and survival probabilities
- **Health Functions**: Disability rates and healthy life expectancy calculations
- **Frailty Dynamics**: Mathematical modeling of biological aging processes
- **Disease-Specific Analysis**: Cause-specific mortality and disability patterns

#### **Economic Model** (`src/economics.jl`)

- **Life-Cycle Optimization**: Intertemporal consumption and leisure choices
- **Wage Functions**: Health-dependent productivity and earnings
- **Value of Statistical Life (VSL)**: Economic valuation of mortality risk reduction
- **Willingness-to-Pay Calculations**: Economic value of health improvements

#### **Key Economic Parameters**

- **Discount Rates**: Time preference for future health and consumption
- **Elasticity Parameters**: Responsiveness of wages to health improvements
- **Utility Functions**: Individual preferences for consumption, leisure, and health
- **Budget Constraints**: Lifetime income and expenditure optimization

### Data Sources

- **Global Burden of Disease (GBD)**: Health, mortality, and disability data (1990-2019)
- **UN World Population Prospects (WPP)**: Population, life expectancy, and fertility data
- **World Bank GDP Data**: Economic indicators in USD and local currency
- **WHO Health Data**: Life expectancy and healthy life expectancy metrics

## Analysis Pipeline

### 1. **Data Processing** (`code/data_processing.py`)

- Clean and standardize international health and economic data
- Merge multiple data sources with consistent country/year classifications
- Validate data consistency and handle missing values

### 2. **Economic Modeling** (`src/`, `jla/*_empirical.jl`)

- **Julia Implementation**: High-performance numerical optimization
- **Country-Specific Analysis**: Tailored modeling for different economic contexts
- **Scenario Analysis**: Multiple intervention scenarios and parameter sensitivity

### 3. **Visualization** (`code/infographic_plotting.py`, `jla/3d_plots.jl`)

- **Publication-Ready Figures**: Professional visualizations for research publication
- **Comparative Analysis**: International comparisons of health and economic outcomes
- **Interactive Plots**: 3D visualizations and exploratory data analysis

## Repository Structure

```
├── _src/                        # Core Julia modeling code
│   ├── biology.jl               # Biological aging models
│   ├── economics.jl             # Economic optimization models
│   ├── data_functions.jl        # Data processing utilities
│   └── objects.jl               # Data structures and parameters
├── _jlr/                        # More Julia modeling code
│   ├── *_empirical.jl           # Country-specific analysis scripts
│   ├── scenarios_wolverine.jl   # Intervention scenario analysis
│   ├── 3d_plots.jl   
│   ├── social_wtp.jl  
├── code/                        # Python code
│   ├── data_processing.py       # Data processing
│   ├── infographic_plotting.py  # Visualization
├── data/                        # Data processing and storage
│   ├── GBD/                     # Global Burden of Disease data
│   ├── WPP/                     # UN Population Prospects data
│   └── WB/                      # World Bank economic data
├── intermediate/                # Generated intermediate data
├── figures/                     # Generated visualizations
├── results/                     # Necessary (Julia?) results
│   ├── Table2.csv              
│   ├── Table3.csv              
│   └── Andrew_international.xlsx            

```

## Getting Started

### Prerequisites

- **Julia** (≥1.6) with required packages
- **Python** (≥3.10) with required packages
- **Data files** (see data/ directory structure)

### Installation

To install Python dependencies, simply run

```bash
pip install -r requirements.txt
```

### Running the Analysis

1. **Data Preparation**: Run Python scripts in `code/data_processing.py` (see `code/PROCESSING.md` for details)
2. **Economic Modeling**: Execute Julia scripts for country-specific analysis
3. **Visualization**: Generate publication-ready figures with Python scripts (see `code/PLOTTING.md` for details)

### Key Scripts

- `international_empirical.jl`: Main international analysis
- `US_empirical_WTP.jl`: US-specific detailed analysis
- `social_WTP.jl`: Social willingness-to-pay calculations
- `scenarios_wolverine.jl`: Intervention scenario modeling

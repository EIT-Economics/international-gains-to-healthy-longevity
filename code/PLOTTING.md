# Simple Infographic Plotting

## Overview

We describe the Python script (`infographic_plotting.py`) that creates two visualizations for the "International Gains to Healthy Longevity" research project. This script creates publication-ready figures with minimal code complexity, translated from the original R code across various files.

## Key Features

- **Historical GDP vs GDP + Longevity Gains Heatmap**: Compares traditional GDP growth with combined GDP + longevity gains
- **Country Summary Table**: Clean visualization of population and economic value metrics by country

## Script Details

### `infographic_plotting.py` - Main Visualization Script

**Key Functions**:

- **`create_historical_plot()`**: Creates the GDP vs longevity gains heatmap
- **`create_oneyear_plot()`**: Creates the country summary table
- **`main()`**: Orchestrates both plot creation

## Visualizations Created

### 1. **Historical GDP vs GDP + Longevity Gains Heatmap**

**Data Sources**:

- `results/Table3.csv` (longevity gains data)
- `results/Andrew_international.xlsx` (GDP data)

**Output**: `figures/historical.pdf` (10" × 3.5")

### 2. **Country Summary Table**

**Data Source**: `results/Table2.csv`

**Output**: `figures/oneyear.pdf` (8" × 6")

## Usage

### **Command Line Usage**

```bash
# Run complete visualization pipeline
python infographic_plotting.py
```

### **Individual Functions**

```python
from infographic_plotting import create_historical_plot, create_oneyear_plot

# Create specific visualizations
create_historical_plot()  # Creates historical.pdf
create_oneyear_plot()     # Creates oneyear.pdf
```

## Data Dependencies

### **Required Input Files**

As per original code, we require three (intermediate) input files:

- **`results/Table3.csv`** - Longevity gains data by country and period
- **`results/Andrew_international.xlsx`** - Raw economic data with GDP per capita
- **`results/Table2.csv`** - Summary statistics by country

## Output Files

### **Generated Visualizations**

- **`figures/historical.pdf`** - Main heatmap comparing GDP vs GDP+longevity gains
- **`figures/oneyear.pdf`** - Summary statistics table visualization

## Dependencies

### **Required Packages**

- **pandas** (≥1.5.0) - Data manipulation and analysis
- **numpy** (≥1.21.0) - Numerical computing
- **matplotlib** (≥3.5.0) - Core plotting functionality
- **openpyxl** (≥3.0.0) - Excel file reading

## Workflow Integration

```
Julia Economic Modeling → Results Tables → Python Visualization → Publication Figures
```

The plotting script serves as the final step in the research pipeline, creating publication-ready figures that communicate the economic value of healthy longevity gains to a broad audience.

## Future Enhancements

### **Planned Features**

- **Interactive Plots**: Jupyter notebook integration
- **Animation**: Time-series animations showing trends over time
- **Custom Themes**: Publication-specific styling options
- **Export Options**: Multiple output formats (PNG, SVG, etc.)

---

*For detailed usage examples and advanced features, see the docstrings in `infographic_plotting.py`.*

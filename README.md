# International Gains to Healthy Longevity

**Economic Valuation of Health Improvements Across Countries and Time**

This repository contains parallel **Python** and **Julia** implementations of a lifecycle economic model for valuing changes in survival and health. Both implementations compute the Value of Statistical Life (VSL) and willingness-to-pay (WTP) for health improvements using the Murphy & Topel framework as developed for *"International Gains to Achieving Healthy Longevity" (2023)* in Cold Spring Harbor Perspectives in Medicine, by Andrew Scott, Julian Ashwin, Martin Ellison, and David Sinclair.

**Last Updated:** October 11, 2025

---

## Table of Contents

- [Quick Start](#quick-start)
- [Repository Overview](#repository-overview)
- [Python Implementation](#python-implementation)
- [Julia Implementation](#julia-implementation)
- [Implementation Comparison](#implementation-comparison)
- [Testing &amp; Validation](#testing--validation)
- [Documentation](#documentation)

---

## Quick Start

### Prerequisites

**For Python:**

- Python 3.9+
- Dependencies in `requirements.txt`

**For Julia:**

- Julia 1.9+
- Dependencies managed via `Project.toml`

### Run Both Implementations

```bash
# 0. Clone project and move into project repository
git clone https://github.com/EIT-Economics/international-gains-to-healthy-longevity.git longevity
cd longevity

# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run Python preprocessing (creates intermediate data files)
python code/preprocess.py

# 3. Run Python analysis
python code/analysis.py

# 4. Set up Julia environment
./julia/setup_julia.sh

# 5. Run Julia analysis (analyzes all available data by default)
julia --project=julia julia/international_empirical.jl

# Or analyze specific countries/years
julia --project=julia julia/international_empirical.jl --default  # 10 countries, 1990-1999

# 6. Compare results
python julia_python_comparison.py
```

**Outputs:**

- Python: `output/international_analysis.csv`
- Julia: `output/international_comp.csv`
- Comparison: `output/comparison_report.md`

---

## Repository Overview

### Project Structure

```
.
├── code/                           # == Python implementation ==
│   ├── analysis.py                 # Main analysis orchestration
│   ├── model.py                    # Lifecycle economic model
│   ├── preprocess.py               # Data preprocessing
│   ├── plot.py                     # Visualization
│   ├── config.py                   # Parameter documentation
│   └── paths.py                    # Path management
│
├── julia/                          # == Julia implementation ==
│   ├── international_empirical.jl  # Main analysis script
│   ├── TargetingAging.jl           # Parameters & module loader
│   ├── economics.jl                # Economic model
│   ├── biology.jl                  # Biological functions
│   ├── data_functions.jl           # Data processing
│   └── ...                         # Other helper files
│
├── data/                           # == Required raw data ==
│   ├── GBD/                        # Global Burden of Disease
│   ├── WPP/                        # UN World Population Prospects
│   ├── WB/                         # World Bank GDP data
│   └── USCData2018.xlsx            # Consumption data (US, 2018)
│
├── intermediate/                   # == Preprocessed data ==
│   ├── GBD/                        # Mortality & morbidity rates
│   ├── WB/                         # Real GDP data
│   ├── WPP/                        # Population & life expectancy
│   └── usc_data.csv                # Consumption data
│
├── figures/                        # == Analysis plots ==
├── output/                         # == Analysis results ==
│
├── julia_python_comparison.py      # Comparison test suite
├── README.md                       # This file
└── DEVELOPMENT_NOTES.md            # Development notes
```

### Data Sources

- **GBD (Global Burden of Disease):** Mortality and morbidity data by age, cause, country, year
- **UN WPP (World Population Prospects):** Population structure and projections
- **World Bank:** Real GDP data (constant PPP and USD)
- **XXX**: US consumption data by age

### Analysis Scope

**Countries:** Argentina, Australia, Bangladesh, Brazil, Canada, Chile, China, Czechia, Denmark, European Union, France, Germany, Global, India, Indonesia, Iran (Islamic Republic of), Israel, Italy, Japan, Kenya, Mexico, Netherlands, New Zealand, Nigeria, Peru, Poland, Russian Federation, South Africa, Spain, Sweden, Turkey, United Kingdom, United States of America, Uruguay

- **Python:** 10 developed countries by default (Australia, France, Germany, Italy, Japan, Netherlands, Spain, Sweden, United Kingdom, United States)
- **Julia:** All available countries by default, or filter with `--countries` flag to personalize (or `--default` flag to match Python sample)

**Time Period:** 1990-2020

- **Python:** 1990-2000 by default (configurable via command-line)
- **Julia:** All available years by default, or filter with `--years` flag to personalize (or `--default` flag to match Python sample 1990-1999)

**Outputs:**

- Life expectancy (*LE*) and healthy life expectancy (*HLE*)
- Value of Statistical Life (*VSL*)
- Willingness-to-pay for survival improvements (*WTP_S*)
- Willingness-to-pay for health improvements (*WTP_H*)
- Total *WTP* and WTP per capita (*WTP_PC*)

---

## Python Implementation

### Overview

The Python implementation provides a comprehensive, heavily-documented, and optimized lifecycle economic model with:

- ✅ Vectorized operations (O(n) complexity for VSL and WTP)
- ✅ Type hints and input validation
- ✅ Centralized configuration with Pydantic
- ✅ Robust VSL calibration with fallback tolerance

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run preprocessing (one-time, creates intermediate data)
python code/preprocess.py
```

**Dependencies:**

- `numpy`, `pandas` - Data manipulation
- `scipy` - Optimization and numerical methods
- `matplotlib`, `seaborn` - Visualization
- `pydantic` - Configuration validation

### Running the Analysis

```bash
# Run full analysis 
python code/analysis.py

# Run default analysis (ten developed countries, ten years 1990-2000)
python code/analysis.py --default

# Customize countries and years
python code/analysis.py --countries "United States of America,Japan" --years "2010,2011,2012"
```

**Output:** `output/international_analysis.csv`

### Architecture

#### Core Workflow (`analysis.py`)

1. **Initialize model** with parameters from `config.py`
2. **Load and interpolate data** via `preprocess.py`
3. **Set up biological variables** (health *H*, survival *S*, population)
4. **Compute wage profiles** (age-dependent with health effects)
5. **Calibrate WageChild** to match target VSL (using root-finding)
6. **Solve lifecycle optimization** (consumption, leisure paths via Euler equations)
7. **Compute WTP** for survival and health improvements
8. **Aggregate** to population-level outcomes

#### Key Features

**Lifecycle Model (`model.py`):**

- CES utility: $U(C, L) = \left[\gamma C^\rho + (1-\gamma) L^\rho\right]^{1/\rho}$
- Backward induction with Euler equations
- Vectorized VSL and WTP calculations (10-50x speedup over loops)

**Data Processing (`preprocess.py`):**

- Piecewise constant interpolation for mortality/morbidity rates
- Ogive interpolation for population
- Health extrapolation for ages >100 with maintained trend

**Configuration (`config.py`):**

- Pydantic-based parameter management
- Extensive documentation with sources and literature citations
- Validation rules and units

**VSL Calibration:**

- Uses `scipy.optimize.brentq` with adaptive grid search fallback
- Target: $11.5M (US 2019 value)
- Strict 1% tolerance with 30% fallback
- Returns fully solved DataFrame (no redundant re-solve)

### Performance

- **Vectorized operations:** WTP and VSL are O(n) instead of O(n²)
- **Array indexing:** Direct array access instead of `df.loc` (2-3x faster)
- **Efficient calibration:** No redundant model solves
- **Typical runtime:** 1-5 minutes for 100 country-years

### Parameters

See `code/config.py` for full parameter documentation. Key parameters:

| Parameter     | Value | Description                   |
| ------------- | ----- | ----------------------------- |
| `rho`       | 0.02  | Pure time preference rate     |
| `r`         | 0.02  | Real interest rate            |
| `gamma`     | 0.5   | Consumption weight in utility |
| `sigma`     | 1.5   | Elasticity of substitution    |
| `MaxAge`    | 110   | Maximum age in model          |
| `AgeRetire` | 65    | Retirement age                |

---

## Julia Implementation

### Overview

The Julia implementation groups the original code [here](https://github.com/julianashwin/international-gains-to-healthy-longevity.git) with minor editing, producing ~2-5x faster execution (after compilation).

### Setup

```bash
# One-time setup (installs all packages)
./julia/setup_julia.sh
```

**Manual setup alternative:**

```bash
cd julia
julia --project=.

# In Julia REPL:
using Pkg
Pkg.instantiate()
Pkg.precompile()
```

**Dependencies:**

- DataFrames, CSV - Data manipulation
- Interpolations - Data interpolation
- NLsolve, Roots - Optimization
- QuadGK - Numerical integration
- ProgressMeter - Progress tracking
- Plots - Visualization

### Running the Analysis

**Default (all available data):**

```bash
# From project root
julia --project=julia julia/international_empirical.jl

# From julia directory
cd julia
julia --project=. international_empirical.jl
```

**Filter specific countries/years:**

```bash
# Analyze only USA and Japan for 2010-2020
julia --project=julia julia/international_empirical.jl --countries "United States of America,Japan" --years 2010:2020

# Use default 10 developed countries (1990-1999)
julia --project=julia julia/international_empirical.jl --default

# Analyze specific years only
julia --project=julia julia/international_empirical.jl --years 1990,2000,2010

# Show help
julia --project=julia julia/international_empirical.jl --help
```

**With optimization (faster, production runs):**

```bash
julia --project=julia --optimize=3 julia/international_empirical.jl
```

**Output:** `output/international_comp.csv`

**Command-Line Options:**

- `--countries "country1,country2"` - Filter for specific countries (comma-separated)
- `--years 1990:2020` - Filter for year range (or `1990,1995,2000` for specific years)
- `--default` - Use default 10 developed countries and years 1990-1999
- `--help` or `-h` - Show help message

### Architecture

#### Main Script (`international_empirical.jl`)

1. **Load data** (GDP, mortality, morbidity, population)
2. **Filter** for target countries (10 developed) and years (1990-2020)
3. **For each country-year:**
   - Compute target VSL from GDP per capita
   - Calibrate economic model
   - Solve lifecycle optimization
   - Compute WTP for survival and health improvements
4. **Output results** to CSV

#### Parameter Definitions (`TargetingAging.jl`)

**Biological Parameters:**

- Life span: T, TH, TS
- Frailty: δ1, δ2, F0, F1
- Mortality: γ, M0, M1
- Disability: ψ, D0, D1, α

**Economic Parameters:**

- Time preference: ρ, r, β
- Productivity: A, ζ1, ζ2
- Utility: σ, η, ϕ, z0
- Lifecycle: AgeGrad, AgeRetire
- Wage: WageChild, MaxHours

### Performance

- **First run:** 5-10 minutes (includes JIT compilation)
- **Subsequent runs:** 30-60 seconds (compiled code)
- **Typical analysis:** 310 country-years (10 countries × 31 years)
- **Memory usage:** 2-4 GB

**Optimization tip:** Use `--optimize=3` for production runs

### File Structure

| File                           | Purpose                                          |
| ------------------------------ | ------------------------------------------------ |
| `international_empirical.jl` | Main analysis entry point                        |
| `TargetingAging.jl`          | Parameter definitions & module loader            |
| `economics.jl`               | Economic model (utility, optimization, VSL, WTP) |
| `biology.jl`                 | Biological functions (survival, health)          |
| `data_functions.jl`          | Data loading & processing                        |
| `aux_functions.jl`           | Helper utilities                                 |
| `plotting.jl`                | Visualization functions                          |

---

## Implementation Comparison

### Similarities

Both implementations:

- Use Murphy & Topel framework for VSL and WTP
- Calibrate to match $11.5M US VSL (2019)
- Solve lifecycle optimization via Euler equations
- Use same data sources (GBD, UN, World Bank)
- Compute same output variables
- Handle same countries and time periods

### Key Differences

| Aspect                        | Python                      | Julia             | Impact                        |
| ----------------------------- | --------------------------- | ----------------- | ----------------------------- |
| **Speed**               | Baseline                    | 2-5x faster       | Julia faster for large runs   |
| **Compilation**         | No compilation              | JIT on first run  | Julia slower first run        |
| **Interpolation**       | Piecewise constant          | Linear            | Minor numerical differences   |
| **VSL Calibration**     | Multiple guesses + fallback | Single attempt    | Python more robust            |
| **WTP/VSL Computation** | Vectorized O(n)             | Loop-based O(n²) | Python faster for these       |
| **Biological Model**    | Empirical data              | Parametric model  | Different but compatible      |
| **Configuration**       | Pydantic-based              | Struct-based      | Different syntax, same values |
| **Documentation**       | Extensive docstrings        | Function comments | Python more detailed          |

### Expected Results

- **Correlation:** >0.99 for all variables
- **Mean differences:** <1% for most variables
- **VSL differences:** <5% (due to calibration methods)
- **Coverage:** ~95%+ overlap in country-years

### Language-Specific Details

**Data Structures:**

| Aspect         | Python           | Julia                |
| -------------- | ---------------- | -------------------- |
| DataFrames     | pandas.DataFrame | DataFrames.DataFrame |
| Arrays         | numpy.ndarray    | Array{T,N}           |
| Missing values | np.nan           | Missing or NaN       |
| Index          | 0-based          | 1-based              |

**Performance Characteristics:**

| Aspect           | Python                      | Julia                         |
| ---------------- | --------------------------- | ----------------------------- |
| Execution model  | Interpreted + JIT (NumPy C) | JIT compiled                  |
| First run        | Faster (no compilation)     | Slower (compilation overhead) |
| Subsequent runs  | Consistent speed            | Much faster after compilation |
| Expected speedup | Baseline                    | 2-10x faster                  |

### Critical Parameters to Verify

When comparing implementations, verify these parameters match:

- `rho` (time preference rate): 0.02
- `r` (interest rate): 0.02
- `gamma` (consumption weight): 0.5
- `sigma` (elasticity of substitution): 1.5
- `MaxAge`: 110
- `AgeRetire`: 65
- `AgeGrad`: 20

**Parameter Access:**

- Python: `config.economic.rho`
- Julia: `econ_pars.rho`

### Acceptable vs Warning Differences

**Acceptable (<1% threshold):**

| Metric      | Expected Range | Cause                                |
| ----------- | -------------- | ------------------------------------ |
| LE, HLE     | <0.1 years     | Floating-point precision             |
| VSL         | <5%            | Calibration tolerance differences    |
| WTP         | <5%            | Accumulated optimization differences |
| Correlation | >0.99          | Should be very high for all          |

**Warning Thresholds (Investigate):**

| Metric      | Warning If | Likely Cause                                |
| ----------- | ---------- | ------------------------------------------- |
| LE, HLE     | >1 year    | Data processing or interpolation difference |
| VSL         | >10%       | Calibration method difference               |
| WTP         | >10%       | Model parameter difference                  |
| Correlation | <0.95      | Fundamental implementation difference       |

### Coverage Differences

- **0-5% difference:** Normal due to edge cases in missing data handling
- **5-10% difference:** Check data filtering logic
- **>10% difference:** Likely systematic difference in data availability checks

**Common causes:**

1. GDP data availability (different handling of missing values)
2. Health data availability (different completeness requirements)
3. Convergence failures (one implementation skips, other fills with NaN)
4. Year bounds verification (both should use 1990-2020)
5. Country list verification (both should use DEFAULT_COUNTRIES)

### Debugging Large Differences

When mean relative difference >5% for any variable:

**Step 1: Verify Parameters**

- Compare `config.py` economic parameters with Julia `EconomicParameters`
- Compare `config.py` lifecycle parameters with Julia `BiologicalParameters`
- Check VSL reference values and income elasticity

**Step 2: Check Data Processing**

- Verify age calculation is identical
- Check missing value handling
- Compare data sorting
- Verify same countries and years processed

**Step 3: Compare Calibration**

- Check VSL calibration target and method
- Compare calibration tolerance
- Verify convergence criteria
- Check fallback behavior

**Step 4: Review Optimization**

- Compare backward induction approach
- Check Euler equation convergence tolerance
- Verify wage calculation
- Compare utility function implementation

**Step 5: Debug Single Case**

- Pick one country-year (e.g., USA 2000)
- Add debug prints to both implementations
- Compare intermediate values step-by-step
- Identify where divergence occurs

---

## Testing & Validation

### Automated Comparison

The repository includes a comprehensive test suite for comparing Python and Julia outputs.

```bash
# Run full comparison (executes both + compares)
python julia_python_comparison.py

# Compare existing outputs only
python julia_python_comparison.py --skip-runs
```

**Output:** `output/comparison_report.md`

### What Gets Compared

1. **Performance:**

- Execution time
- Speed ratio (Julia vs Python)
- Success/failure status

2. **Data Coverage:**

- Country-year combinations in each output
- Overlap and differences

3. **Numerical Values:**

- Mean, std, min, max for each variable
- Absolute and relative differences
- Correlation coefficients

4. **Variables Compared:**

- Life expectancy (LE)
- Healthy life expectancy (HLE)
- Value of Statistical Life (VSL)
- WTP survival, health, total, per capita
- Population, GDP, GDP per capita

### Interpretation

✅ **Good (Expected):**

- Correlation >0.99
- Mean differences <1%
- Max differences <5%

⚠️ **Warning (Investigate):**

- Correlation 0.95-0.99
- Mean differences 1-5%

❌ **Error (Fix Required):**

- Correlation <0.95
- Mean differences >5%

### Continuous Integration

To set up automated testing:

**GitHub Actions Example:**

```yaml
name: Julia vs Python Comparison

on: [push, pull_request]

jobs:
  compare:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
  
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
  
      - name: Set up Julia
        uses: julia-actions/setup-julia@v1
        with:
          version: '1.8'
  
      - name: Install Python dependencies
        run: pip install -r requirements.txt
  
      - name: Install Julia dependencies
        run: julia --project=julia -e 'using Pkg; Pkg.instantiate()'
  
      - name: Run comparison
        run: python julia_python_comparison.py
  
      - name: Upload report
        uses: actions/upload-artifact@v2
        with:
          name: comparison-report
          path: output/comparison_report.md
```

### Manual Comparison Checklist

When manually comparing implementations, verify:

- [ ] **Parameters match exactly**

  - Economic parameters (rho, r, gamma, etc.)
  - Lifecycle parameters (MaxAge, AgeRetire, etc.)
  - Calibration tolerances
- [ ] **Data processing is identical**

  - Age calculation (midpoint of age_low/age_high)
  - Missing value handling
  - Data sorting before interpolation
- [ ] **Model logic matches**

  - Wage calculation
  - Utility function
  - Euler equations
  - VSL calculation
  - WTP calculation
- [ ] **Calibration approach**

  - Target VSL value and reference
  - Root-finding method
  - Tolerance and convergence criteria
  - Fallback behavior
- [ ] **Output format**

  - Same variables computed
  - Same units (e.g., WTP in trillions, VSL in millions)
  - Same precision/rounding

### Advanced Debugging

**Compare Individual Country-Years:**

```python
import pandas as pd

py = pd.read_csv('output/international_analysis.csv')
jl = pd.read_csv('output/international_comp.csv')

# Example: Compare USA 2000
py_usa = py[(py.country == 'United States of America') & (py.year == 2000)]
jl_usa = jl[(jl.country == 'United States of America') & (jl.year == 2000)]

print("Python VSL:", py_usa.vsl.values)
print("Julia VSL:", jl_usa.vsl.values)
print("Difference:", py_usa.vsl.values - jl_usa.vsl.values)
```

**Profile Performance:**

Python:

```bash
python -m cProfile -o python_profile.stats code/analysis.py
```

Julia:

```julia
using Profile
@profile include("julia/international_empirical.jl")
```

**Check Intermediate Outputs:**

Add debug prints to compare intermediate calculations:

- Survival curves
- Health curves
- Wage profiles
- Consumption/leisure paths
- Value functions

---

## Documentation

### Main Documentation

- **`README.md`** (this file) - Complete repository documentation
  - Setup and installation
  - Implementation comparison with detailed differences
  - Testing and validation guide
  - Troubleshooting
  - Development guidelines
- **`DEVELOPMENT_NOTES.md`** - Development history and optimizations
- **`julia_python_comparison.py`** - Automated comparison test suite

### Python-Specific

- **Docstrings:** All functions have comprehensive Google-style docstrings
- **`code/config.py`:** Extensive parameter documentation with sources
- **Type hints:** Complete type annotations throughout

### Julia-Specific

- **`julia/Project.toml`:** Dependency manifest
- **`julia/setup_julia.sh`:** Automated setup script
- **Function comments:** Inline documentation throughout
- **Type annotations:** Full type stability

### Data Sources

- **GBD:** http://ghdx.healthdata.org/gbd-results-tool
- **UN WPP:** https://population.un.org/wpp/
- **World Bank:** https://data.worldbank.org/
- **Consumption** **data**: ???

---

## Development Notes

### Python Optimizations (Completed)

- Vectorized WTP calculation (10-50x speedup)
- Vectorized VSL calculation (O(n²) → O(n))
- Array indexing instead of `df.loc` (2-3x speedup)
- Eliminated redundant re-solve after calibration
- Pydantic-based configuration
- Robust VSL calibration with adaptive methods
- See `DEVELOPMENT_NOTES.md`

### General TODOs

- Where does USC2018 come from?
- Assert parameters correct (across implementations) and reflecting Murphy Topel choices
- Investigate <5% differences in numbers from Python vs Julia international analysis
- **Implement social welfare analysis!**

### Python TODOs (Optional)

- Unit tests for core functions
- Integration tests with known outputs
- Parallel processing for country-years (joblib)
- Performance profiling dashboard

### Julia Optimizations (Potential)

- Type stability checks (`@code_warntype`)
- In-place operations (functions with `!`)
- Pre-allocation of arrays
- `@inbounds` and `@simd` in hot loops
- Vectorize WTP and VSL computations

### Julia Development

**Adding New Packages:**

Edit `julia/Project.toml` to add dependencies:

```toml
[deps]
NewPackage = "uuid-here"
```

Then install:

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

**Testing Changes:**

```bash
cd julia
julia --project=.

# In Julia REPL:
include("TargetingAging.jl")
# ... test your changes
```

**Performance Analysis:**

```julia
# Type stability
@code_warntype function_name(args)

# Profiling
using Profile
@profile include("international_empirical.jl")
Profile.print()

# Benchmarking
using BenchmarkTools
@benchmark function_name(args)
```

---

## Troubleshooting

### Python Issues

**Import errors:**

```bash
pip install -r requirements.txt
```

**Missing data:**

```bash
python code/preprocess.py
```

**VSL calibration failures:**

- Check `code/config.py` for parameter values
- Review `DEVELOPMENT_NOTES.md` Section 7 for calibration details
- Adjust `wage_child_bounds` or `fallback_tolerance`

### Julia Issues

**Julia not found:**

```bash
brew install julia
julia --version
```

**Package installation fails:**

```bash
# Clear cache and re-run setup
julia -e 'using Pkg; Pkg.gc()'
./julia/setup_julia.sh
```

**Data files missing:**

```bash
# Run Python preprocessing first
python code/preprocess.py
```

**Out of memory:**

```bash
# Increase heap size
julia --project=julia --heap-size-hint=16G julia/international_empirical.jl
```

**Slow first run:**

- This is normal! Julia compiles on first run (~5-10 min)
- Subsequent runs are much faster (~30-60 sec)

---

## Contributing

When adding features or fixing bugs:

1. **Update both implementations** for feature parity
2. **Run comparison test** to verify consistency
3. **Update documentation** (docstrings, README, DEVELOPMENT_NOTES.md)
4. **Check for performance** regressions
5. **Add tests** where appropriate

---

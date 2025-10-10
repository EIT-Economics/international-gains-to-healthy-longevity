# Julia Analysis Code

This directory contains the Julia implementation of the health and longevity analysis, originally used to compute international comparisons of gains from changes in survival and health.

## Quick Start

### 1. Install Julia

**macOS:**

```bash
brew install julia
```

**Or download from:** https://julialang.org/downloads/

**Recommended:** Julia 1.9 or later

### 2. Run Setup Script

From the project root:

```bash
./julia/setup_julia.sh
```

This will:

- Check Julia installation
- Install all required packages
- Precompile dependencies
- Test that all imports work

**Note:** First-time setup takes 5-10 minutes to download and compile packages.

### 3. Prepare Data

Ensure the Python preprocessing has been run to create:

- `intermediate/WB/real_gdp_data.csv`
- `intermediate/GBD/mortality_rates.csv`
- `intermediate/GBD/morbidity_rates.csv`

**Run Python preprocessing if needed:**

```bash
python3 -m code.preprocess
```

### 4. Run the Analysis

**From julia directory:**

```bash
cd julia
julia --project=. international_empirical.jl
```

**From project root:**

```bash
julia --project=julia julia/international_empirical.jl
```

**Output:** Results saved to `output/international_comp.csv`

---

## Manual Setup (Alternative)

If you prefer to set up manually:

### Install Julia Packages

```bash
cd julia
julia --project=.
```

Then in Julia REPL:

```julia
using Pkg
Pkg.instantiate()  # Reads Project.toml and installs everything
Pkg.precompile()   # Speeds up first run
```

---

## File Structure

```
julia/
├── Project.toml              # Dependency management
├── setup_julia.sh            # Automated setup script
├── README.md                 # This file
│
├── international_empirical.jl  # Main analysis script
├── TargetingAging.jl          # Parameter definitions & module loader
│
├── aux_functions.jl           # Helper functions
├── biology.jl                 # Biological model functions
├── economics.jl               # Economic model functions
├── data_functions.jl          # Data processing functions
├── objects.jl                 # Object/structure management
└── plotting.jl                # Visualization functions
```

---

## Key Dependencies

### Core Computation

- **Parameters** - Parameter management (`@with_kw` macro)
- **DataFrames** - Data manipulation
- **QuadGK** - Numerical integration
- **NLsolve** - Nonlinear equation solving
- **Roots** - Root finding
- **FiniteDifferences** - Numerical differentiation
- **Interpolations** - Data interpolation

### Data I/O

- **CSV** - CSV file reading/writing
- **XLSX** - Excel file support

### Visualization & Output

- **Plots** - Plotting (GR backend)
- **ProgressMeter** - Progress bars
- **Formatting** - Number formatting
- **Latexify** - LaTeX export
- **LaTeXStrings** - LaTeX in plots

---

## What the Code Does

### Main Analysis (`international_empirical.jl`)

1. **Load Data**

   - GDP data from World Bank
   - Mortality rates from GBD
   - Morbidity (health) rates from GBD
   - Population structure
2. **For Each Country-Year:**

   - Compute target VSL based on GDP per capita
   - Calibrate economic model to match VSL
   - Solve lifecycle optimization
   - Compute willingness-to-pay (WTP) for:
     - Survival improvements (WTP_S)
     - Health improvements (WTP_H)
     - Total WTP
3. **Output Results:**

   - Population
   - Real GDP and GDP per capita
   - Life expectancy (LE)
   - Healthy life expectancy (HLE)
   - VSL
   - WTP metrics (total and per capita)

### Parameter Definitions (`TargetingAging.jl`)

**Biological Parameters:**

- Life span parameters (T, TH, TS)
- Frailty coefficients (δ1, δ2, F0, F1)
- Mortality parameters (γ, M0, M1)
- Disability parameters (ψ, D0, D1, α)

**Economic Parameters:**

- Time preference (ρ, r, β)
- Productivity (A, ζ1, ζ2)
- Utility function (σ, η, ϕ, z0)
- Lifecycle stages (AgeGrad, AgeRetire)
- Wage parameters (WageChild, MaxHours)

---

## Comparison with Python Implementation

| Feature                    | Julia                      | Python                         |
| -------------------------- | -------------------------- | ------------------------------ |
| **Speed**            | ~2-5x faster               | Baseline                       |
| **Biological model** | Full parametric model      | Uses empirical data            |
| **Interpolation**    | Linear                     | Piecewise constant             |
| **Calibration**      | Single VSL target          | Multiple initial guesses       |
| **Output**           | `international_comp.csv` | `international_analysis.csv` |

**Core logic is equivalent** - both compute VSL and WTP using Murphy & Topel framework.

---

## Troubleshooting

### Julia Not Found

```bash
# Install Julia
brew install julia

# Verify installation
julia --version
```

### Package Installation Fails

```bash
# Clear package cache
julia -e 'using Pkg; Pkg.gc()'

# Re-run setup
./julia/setup_julia.sh
```

### "File not found" Errors

Ensure you're running from correct directory:

```bash
# Should see intermediate/ and julia/ directories
ls -la
```

If data files missing:

```bash
# Run Python preprocessing
python3 -m code.preprocess
```

### Out of Memory

Julia can be memory-intensive. For large datasets:

```bash
# Increase heap size (example: 16GB)
julia --project=. --heap-size-hint=16G international_empirical.jl
```

### Slow First Run

**Normal!** Julia compiles code on first run. Subsequent runs are much faster (~100x speedup).

---

## Development

### Adding New Packages

Edit `Project.toml`:

```toml
[deps]
NewPackage = "uuid-here"
```

Then:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Testing Changes

```bash
# Run with project environment
julia --project=.

# In Julia REPL:
include("TargetingAging.jl")
# ... test your changes
```

---

## Performance Notes

- **First run:** ~5-10 minutes (compilation)
- **Subsequent runs:** ~30-60 seconds
- **Full analysis:** ~360 country-years
- **Memory usage:** ~2-4 GB

**Optimization tip:** Use `--optimize=3` for production runs:

```bash
julia --project=. --optimize=3 international_empirical.jl
```

---

**Last Updated:** October 10, 2025

# Julia Quick Start

## One-Time Setup

```bash
# 1. Install Julia (if not installed)
brew install julia

# 2. Run setup script
./julia/setup_julia.sh
```

**That's it!** The setup script handles all package installation.

---

## Running the Analysis

```bash
# Option 1: From julia directory
cd julia
julia --project=. international_empirical.jl

# Option 2: From project root
julia --project=julia julia/international_empirical.jl
```

**Output:** `output/international_comp.csv`

---

## Common Commands

```bash
# Check Julia version
julia --version

# Open Julia REPL with project
cd julia
julia --project=.

# Test a single country-year manually
julia --project=. -e 'include("TargetingAging.jl"); println("✓ Loaded")'

# Run with optimization (faster)
julia --project=. --optimize=3 international_empirical.jl
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Julia not found | `brew install julia` |
| Package errors | Re-run `./julia/setup_julia.sh` |
| Data files missing | Run Python preprocessing first |
| Slow first run | Normal - Julia compiles on first run |
| Out of memory | Add `--heap-size-hint=16G` |

---

## Expected Runtime

- **First run:** 5-10 min (compilation + analysis)
- **Subsequent runs:** 30-60 seconds (analysis only)

---

## Need More Help?

See [`julia/README.md`](README.md) for full documentation.


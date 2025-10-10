#!/bin/bash
# Setup script for Julia environment

set -e  # Exit on error

echo "======================================================================="
echo "Julia Environment Setup for International Health & Longevity Analysis"
echo "======================================================================="
echo ""

# Check if Julia is installed
if ! command -v julia &> /dev/null
then
    echo "❌ Julia is not installed!"
    echo ""
    echo "Please install Julia first:"
    echo "  macOS:   brew install julia"
    echo "  Or download from: https://julialang.org/downloads/"
    echo ""
    echo "Recommended: Julia 1.9 or later"
    exit 1
fi

echo "✓ Julia found: $(julia --version)"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Installing Julia packages from Project.toml..."
echo "This may take 5-10 minutes on first run..."
echo ""

julia --project=. -e '
    using Pkg
    
    println("Activating project environment...")
    Pkg.activate(".")
    
    println("\nInstantiating dependencies...")
    Pkg.instantiate()
    
    println("\nPrecompiling packages (this may take a while)...")
    Pkg.precompile()
    
    println("\n" * "="^70)
    println("✓ Julia environment setup complete!")
    println("="^70)
    
    # Test imports
    println("\nTesting package imports...")
    packages = [
        "Statistics", "Parameters", "DataFrames", 
        "QuadGK", "NLsolve", "Roots", "FiniteDifferences", 
        "Interpolations", "SpecialFunctions", "Plots", 
        "XLSX", "CSV", "ProgressMeter", "Formatting", 
        "Latexify", "LaTeXStrings"
    ]
    
    all_ok = true
    for pkg in packages
        try
            eval(Meta.parse("using $pkg"))
            println("  ✓ $pkg")
        catch e
            println("  ✗ $pkg - ERROR: $e")
            all_ok = false
        end
    end
    
    if all_ok
        println("\n✓ All packages loaded successfully!")
    else
        println("\n⚠ Some packages failed to load. Check errors above.")
        exit(1)
    end
'

echo ""
echo "======================================================================="
echo "Setup Complete!"
echo "======================================================================="
echo ""
echo "To run the analysis:"
echo "  cd julia"
echo "  julia --project=. international_empirical.jl"
echo ""
echo "Or from project root:"
echo "  julia --project=julia julia/international_empirical.jl"
echo ""


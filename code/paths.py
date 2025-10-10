#!/usr/bin/env python3
"""Centralized Path Management for Health and Longevity Analysis.

This module provides a single source of truth for all file paths used across
the project. Using centralized path management ensures:
- Code works regardless of working directory
- Easy to relocate project
- Clear documentation of expected directory structure
- Automatic directory creation where needed

Directory Structure:
    project_root/
    ├── code/           # Python source code
    ├── data/           # Raw input data
    │   ├── GBD/        # Global Burden of Disease data
    │   ├── WPP/        # UN World Population Prospects data
    │   └── WB/         # World Bank economic data
    ├── intermediate/   # Processed/intermediate data
    │   ├── GBD/        # Processed health data
    │   ├── WPP/        # Processed population data
    │   └── WB/         # Processed economic data
    ├── output/         # Final analysis outputs
    ├── figures/        # Generated plots and visualizations
    └── julia/          # Julia reference implementation

Usage:
    >>> from paths import OUTPUT_DIR, INTERMEDIATE_DIR
    >>> df.to_csv(OUTPUT_DIR / "results.csv")
    >>> mort_df = pd.read_csv(INTERMEDIATE_DIR / "GBD" / "mortality_rates.csv")
"""

from pathlib import Path

# ============================================================================
# Root Directory
# ============================================================================

# Project root is parent of code/ directory
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# Input Data Directories (Raw Data)
# ============================================================================

DATA_DIR = PROJECT_ROOT / "data"
"""Root directory for all raw input data."""

GBD_DATA_DIR = DATA_DIR / "GBD"
"""Global Burden of Disease (GBD) raw data directory."""

WPP_DATA_DIR = DATA_DIR / "WPP"
"""UN World Population Prospects (WPP) raw data directory."""

WB_DATA_DIR = DATA_DIR / "WB"
"""World Bank economic data directory."""

# ============================================================================
# Intermediate Data Directories (Processed Data)
# ============================================================================

INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate"
"""Root directory for processed/intermediate data."""

INTERMEDIATE_GBD_DIR = INTERMEDIATE_DIR / "GBD"
"""Processed GBD data (mortality and morbidity rates)."""

INTERMEDIATE_WPP_DIR = INTERMEDIATE_DIR / "WPP"
"""Processed WPP data (population projections)."""

INTERMEDIATE_WB_DIR = INTERMEDIATE_DIR / "WB"
"""Processed World Bank data (GDP series)."""

# ============================================================================
# Output Directories
# ============================================================================

OUTPUT_DIR = PROJECT_ROOT / "output"
"""Final analysis outputs and results."""

FIGURES_DIR = PROJECT_ROOT / "figures"
"""Generated plots, charts, and visualizations."""

# ============================================================================
# Reference Implementation
# ============================================================================

JULIA_DIR = PROJECT_ROOT / "julia"
"""Julia reference implementation scripts."""

# ============================================================================
# Automatic Directory Creation
# ============================================================================

def ensure_directories_exist() -> None:
    """Create all output and intermediate directories if they don't exist.
    
    This function is called automatically on import to ensure all required
    directories exist. It's safe to call multiple times.
    
    Creates:
        - intermediate/ and subdirectories (GBD, WPP, WB)
        - output/
        - figures/
    
    Note:
        Does NOT create data/ directories, as these should contain
        user-provided raw data.
    """
    # Intermediate data directories
    INTERMEDIATE_GBD_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_WPP_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_WB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Automatically ensure directories exist on import
ensure_directories_exist()

# ============================================================================
# Path Validation
# ============================================================================

def validate_data_directories() -> bool:
    """Check if all required raw data directories exist.
    
    Returns:
        True if all data directories exist, False otherwise.
        
    Example:
        >>> if not validate_data_directories():
        ...     print("Error: Missing data directories. Please download data.")
    """
    required_dirs = [
        (GBD_DATA_DIR, "GBD"),
        (WPP_DATA_DIR, "WPP"),
        (WB_DATA_DIR, "WB"),
    ]
    
    all_exist = True
    for dir_path, name in required_dirs:
        if not dir_path.exists():
            print(f"Warning: {name} data directory not found: {dir_path}")
            all_exist = False
    
    return all_exist


# ============================================================================
# Common File Paths
# ============================================================================

class CommonPaths:
    """Commonly used file paths for quick access.
    
    This class provides convenient access to frequently used files.
    All paths are properties so they're computed on access (ensuring
    they reflect current directory structure).
    """
    
    # === GBD Processed Data ===
    @property
    def mortality_rates(self) -> Path:
        """Processed mortality rates by country, year, and age band."""
        return INTERMEDIATE_GBD_DIR / "mortality_rates.csv"
    
    @property
    def morbidity_rates(self) -> Path:
        """Processed morbidity (YLD) rates by country, year, and age band."""
        return INTERMEDIATE_GBD_DIR / "morbidity_rates.csv"
    
    # === World Bank Processed Data ===
    @property
    def real_gdp_data(self) -> Path:
        """Processed real GDP data by country and year."""
        return INTERMEDIATE_WB_DIR / "real_gdp_data.csv"
    
    # === Merged Data ===
    @property
    def merged_health_gdp(self) -> Path:
        """Merged health and GDP data at country-year-age level."""
        return INTERMEDIATE_DIR / "merged_health_gdp.csv"
    
    @property
    def merged_health_gdp_summary(self) -> Path:
        """Summary statistics of merged health and GDP data."""
        return INTERMEDIATE_DIR / "merged_health_gdp_summaries.csv"
    
    # === Analysis Outputs ===
    @property
    def international_analysis(self) -> Path:
        """International analysis results (main output)."""
        return OUTPUT_DIR / "international_analysis.csv"
    
    @property
    def international_comp(self) -> Path:
        """International comparison results."""
        return OUTPUT_DIR / "international_comp.csv"
    
    @property
    def test_output(self) -> Path:
        """Test analysis output."""
        return OUTPUT_DIR / "test.csv"
    
    @property
    def debug_df(self) -> Path:
        """Detailed DataFrame for debugging (last country-year processed)."""
        return OUTPUT_DIR / "df.csv"
    
    # === Figures ===
    @property
    def mortality_trends(self) -> Path:
        """Mortality trends plot."""
        return FIGURES_DIR / "mortality_trends_US.pdf"
    
    @property
    def health_trends(self) -> Path:
        """Health/disability trends plot."""
        return FIGURES_DIR / "health_trends_US.pdf"


# Create singleton instance for easy imports
paths = CommonPaths()





if __name__ == "__main__":
    """Print all configured paths for debugging if run as script."""
    print("=" * 80)
    print("Configured Paths for Health & Longevity Analysis")
    print("=" * 80)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"\n  Data Directories:")
    print(f"    GBD:  {GBD_DATA_DIR}")
    print(f"    WPP:  {WPP_DATA_DIR}")
    print(f"    WB:   {WB_DATA_DIR}")
    print(f"\n  Intermediate Directories:")
    print(f"    Root: {INTERMEDIATE_DIR}")
    print(f"    GBD:  {INTERMEDIATE_GBD_DIR}")
    print(f"    WPP:  {INTERMEDIATE_WPP_DIR}")
    print(f"    WB:   {INTERMEDIATE_WB_DIR}")
    print(f"\n  Output Directories:")
    print(f"    Results: {OUTPUT_DIR}")
    print(f"    Figures: {FIGURES_DIR}")
    print(f"\n  Julia Code: {JULIA_DIR}")
    
    print(f"\n{'=' * 80}")
    print("Directory Validation")
    print("=" * 80)
    
    all_valid = validate_data_directories()
    if all_valid:
        print("✓ All required data directories found")
    else:
        print("✗ Some data directories are missing")
    


#!/usr/bin/env python3
"""
Comprehensive Test Suite: Julia vs Python Implementation Comparison

This script:
1. Runs both Julia and Python implementations
2. Compares performance (execution time)
3. Compares results (numerical outputs)
4. Documents all differences
5. Generates a detailed comparison report

Usage:
    python test_julia_python_comparison.py [--skip-runs] [--output REPORT_PATH]
    
Options:
    --skip-runs: Skip running the scripts, only compare existing outputs
    --output: Path for comparison report (default: output/comparison_report.md)
"""

import argparse
import subprocess
import time
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime


class ImplementationComparison:
    """Compare Julia and Python implementations of the health-longevity model."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.results = {}
        self.differences = []
        self.warnings = []
        
    def run_python_analysis(self) -> Tuple[float, bool]:
        """Run Python analysis and measure execution time.
        
        Returns:
            (execution_time, success)
        """
        print("\n" + "="*70)
        print("RUNNING PYTHON ANALYSIS")
        print("="*70)
        
        cmd = [sys.executable, "code/analysis.py"]
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                print(f"ERROR: Python analysis failed with return code {result.returncode}")
                print(f"STDERR:\n{result.stderr}")
                return elapsed, False
                
            print(f"✓ Python analysis completed in {elapsed:.2f} seconds")
            return elapsed, True
            
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"ERROR: Python analysis timed out after {elapsed:.2f} seconds")
            return elapsed, False
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"ERROR: Python analysis failed: {e}")
            return elapsed, False
    
    def run_julia_analysis(self) -> Tuple[float, bool]:
        """Run Julia analysis and measure execution time.
        
        Returns:
            (execution_time, success)
        """
        print("\n" + "="*70)
        print("RUNNING JULIA ANALYSIS")
        print("="*70)
        
        cmd = ["julia", "--project=julia", "julia/international_empirical.jl"]
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                print(f"ERROR: Julia analysis failed with return code {result.returncode}")
                print(f"STDERR:\n{result.stderr}")
                return elapsed, False
                
            print(f"✓ Julia analysis completed in {elapsed:.2f} seconds")
            return elapsed, True
            
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"ERROR: Julia analysis timed out after {elapsed:.2f} seconds")
            return elapsed, False
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"ERROR: Julia analysis failed: {e}")
            return elapsed, False
    
    def load_outputs(self, 
                     py_filename: str = "analysis.csv", 
                     jl_filename: str = "international_comp.csv",
                     skip_2019: bool = True
        ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load output CSV files from both implementations.
        
        Returns:
            (python_df, julia_df)
        """
        print("\n" + "="*70)
        print("LOADING OUTPUT FILES")
        print("="*70)
        
        python_output = self.repo_root / "output" / py_filename
        julia_output = self.repo_root / "output" / jl_filename
        
        python_df = None
        julia_df = None
        
        # Load Python output
        if python_output.exists():
            try:
                python_df = pd.read_csv(python_output)
                print(f"✓ Loaded Python output: {len(python_df)} rows, {len(python_df.columns)} columns")
            except Exception as e:
                print(f"ERROR: Failed to load Python output: {e}")
        else:
            print(f"ERROR: Python output not found at {python_output}")
        
        # Load Julia output
        if julia_output.exists():
            try:
                julia_df = pd.read_csv(julia_output)
                print(f"✓ Loaded Julia output: {len(julia_df)} rows, {len(julia_df.columns)} columns")
            except Exception as e:
                print(f"ERROR: Failed to load Julia output: {e}")
        else:
            print(f"ERROR: Julia output not found at {julia_output}")
        
        if python_df is not None and julia_df is not None:
            assert set(julia_df.columns).issubset(set(python_df.columns)), "Columns do not match"
            print(f"✓ Columns match: {list(julia_df.columns)}")        
        
        if skip_2019:
            print("Skipping 2019 due to synthetic data generation difference...")
            python_df = python_df[python_df['year'] != 2019]
            julia_df = julia_df[julia_df['year'] != 2019]

        return python_df, julia_df
    
    def compare_data_coverage(self, py_df: pd.DataFrame, jl_df: pd.DataFrame) -> Dict:
        """Compare which country-year combinations are present in each dataset.
        
        Returns:
            Dictionary with coverage statistics
        """
        print("\n" + "="*70)
        print("COMPARING DATA COVERAGE")
        print("="*70)

        # Create sets of (country, year) tuples
        py_keys = set(zip(py_df['country'], py_df['year']))
        jl_keys = set(zip(jl_df['country'], jl_df['year']))
        
        only_python = py_keys - jl_keys
        only_julia = jl_keys - py_keys
        both = py_keys & jl_keys
        
        coverage = {
            'python_total': len(py_keys),
            'julia_total': len(jl_keys),
            'common': len(both),
            'only_python': len(only_python),
            'only_julia': len(only_julia),
            'only_python_samples': list(only_python)[:5],
            'only_julia_samples': list(only_julia)[:5]
        }
        
        print(f"Python total country-years: {coverage['python_total']}")
        print(f"Julia total country-years:  {coverage['julia_total']}")
        print(f"Common country-years:       {coverage['common']}")
        print(f"Only in Python:             {coverage['only_python']}")
        print(f"Only in Julia:              {coverage['only_julia']}")
        
        if only_python:
            print(f"\nSample Python-only: {coverage['only_python_samples']}")
        if only_julia:
            print(f"Sample Julia-only:  {coverage['only_julia_samples']}")
        
        return coverage
    
    def align_dataframes(self, py_df: pd.DataFrame, jl_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align dataframes on common country-year combinations.
        
        Returns:
            (aligned_python_df, aligned_julia_df)
        """
        print("\n" + "="*70)
        print("ALIGNING DATAFRAMES")
        print("="*70)
        
        # Merge on country and year
        merged = pd.merge(
            py_df,
            jl_df,
            on=['country', 'year'],
            how='inner',
            suffixes=('_py', '_jl')
        )
        
        print(f"✓ Aligned {len(merged)} common country-year combinations")
        
        return merged
    
    def compare_numerical_values(self, merged_df: pd.DataFrame) -> Dict:
        """Compare numerical values between Python and Julia outputs.
        
        Returns:
            Dictionary with comparison statistics for each variable
        """
        print("\n" + "="*70)
        print("COMPARING NUMERICAL VALUES")
        print("="*70)
        
        # Variables to compare (common between both)
        # Map: human-readable name -> column name
        variable_mapping = {
            'Life Expectancy (LE)': 'le',
            'Healthy Life Expectancy (HLE)': 'hle',
            'Value of Statistical Life (VSL)': 'vsl',
            'WTP Survival (WTP_S)': 'wtp_s',
            'WTP Health (WTP_H)': 'wtp_h',
            'Total WTP': 'wtp',
            'WTP per capita': 'wtp_pc',
            'Population': 'population',
            'Real GDP': 'real_gdp',
            'Real GDP per capita': 'real_gdp_pc'
        }
        
        comparisons = {}
        
        for var_name, col_name in variable_mapping.items():
            # Check if columns exist
            col_py_name = col_name + "_py"
            col_jl_name = col_name + "_jl"
            assert col_py_name in merged_df.columns, f"Column {col_name} not found in {merged_df.columns.to_list()}"
            assert col_jl_name in merged_df.columns, f"Column {col_name} not found in {merged_df.columns.to_list()}"
            
            # Get values
            py_vals = merged_df[col_py_name].values
            jl_vals = merged_df[col_jl_name].values
            
            # Remove NaNs for comparison
            valid_mask = ~(np.isnan(py_vals) | np.isnan(jl_vals))
            py_vals_valid = py_vals[valid_mask]
            jl_vals_valid = jl_vals[valid_mask]
            
            if len(py_vals_valid) == 0:
                print(f"⚠ No valid values for {var_name}")
                continue
            
            # Calculate differences
            abs_diff = py_vals_valid - jl_vals_valid
            rel_diff = (py_vals_valid - jl_vals_valid) / np.where(jl_vals_valid != 0, jl_vals_valid, 1)
            pct_diff = rel_diff * 100
            
            comparisons[var_name] = {
                'n_valid': len(py_vals_valid),
                'n_total': len(py_vals),
                'python_mean': float(np.mean(py_vals_valid)),
                'julia_mean': float(np.mean(jl_vals_valid)),
                'python_std': float(np.std(py_vals_valid)),
                'julia_std': float(np.std(jl_vals_valid)),
                'abs_diff_mean': float(np.mean(abs_diff)),
                'abs_diff_std': float(np.std(abs_diff)),
                'abs_diff_max': float(np.max(np.abs(abs_diff))),
                'rel_diff_mean': float(np.mean(pct_diff)),
                'rel_diff_std': float(np.std(pct_diff)),
                'rel_diff_max': float(np.max(np.abs(pct_diff))),
                'correlation': float(np.corrcoef(py_vals_valid, jl_vals_valid)[0, 1])
            }
            
            print(f"\n{var_name}:")
            print(f"  Valid observations: {comparisons[var_name]['n_valid']}/{comparisons[var_name]['n_total']}")
            print(f"  Python mean: {comparisons[var_name]['python_mean']:.4f}")
            print(f"  Julia mean:  {comparisons[var_name]['julia_mean']:.4f}")
            print(f"  Mean diff:   {comparisons[var_name]['abs_diff_mean']:.4f} ({comparisons[var_name]['rel_diff_mean']:.2f}%)")
            print(f"  Max diff:    {comparisons[var_name]['abs_diff_max']:.4f} ({comparisons[var_name]['rel_diff_max']:.2f}%)")
            print(f"  Correlation: {comparisons[var_name]['correlation']:.6f}")
        
        return comparisons
    
    def generate_report(self, output_path: Path, performance: Dict, coverage: Dict, 
                       comparisons: Dict, skip_runs: bool) -> None:
        """Generate comprehensive markdown report.
        
        Args:
            output_path: Path to save report
            performance: Performance comparison results
            coverage: Data coverage comparison
            comparisons: Numerical value comparisons
            skip_runs: If True, skip reporting on performance 
        """
        print("\n" + "="*70)
        print("GENERATING COMPARISON REPORT")
        print("="*70)
        
        with open(output_path, 'w') as f:
            f.write("# Julia vs Python Implementation Comparison Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("This report compares the Julia and Python implementations of the ")
            f.write("international health-longevity analysis model.\n\n")
            
            # Performance
            if not skip_runs:
                f.write("## Performance Comparison\n\n")
                f.write("| Implementation | Execution Time | Status |\n")
                f.write("|----------------|----------------|--------|\n")
                
                if 'python' in performance:
                    py_time = performance['python']['time']
                    py_status = "✓ Success" if performance['python']['success'] else "✗ Failed"
                    f.write(f"| Python | {py_time:.2f}s | {py_status} |\n")
                
                if 'julia' in performance:
                    jl_time = performance['julia']['time']
                    jl_status = "✓ Success" if performance['julia']['success'] else "✗ Failed"
                    f.write(f"| Julia | {jl_time:.2f}s | {jl_status} |\n")
                
                if 'python' in performance and 'julia' in performance:
                    if performance['python']['success'] and performance['julia']['success']:
                        speedup = py_time / jl_time
                        f.write(f"\n**Speed Ratio:** Julia is {speedup:.2f}x ")
                        f.write("faster" if speedup > 1 else "slower")
                        f.write(" than Python\n")
            
            # Data Coverage
            f.write("\n## Data Coverage\n\n")
            f.write("| Metric | Count |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Python total country-years | {coverage['python_total']} |\n")
            f.write(f"| Julia total country-years | {coverage['julia_total']} |\n")
            f.write(f"| Common country-years | {coverage['common']} |\n")
            f.write(f"| Only in Python | {coverage['only_python']} |\n")
            f.write(f"| Only in Julia | {coverage['only_julia']} |\n")
            
            if coverage['only_python'] > 0:
                f.write(f"\n**Python-only samples:** {coverage['only_python_samples']}\n")
            if coverage['only_julia'] > 0:
                f.write(f"\n**Julia-only samples:** {coverage['only_julia_samples']}\n")
            
            
            # Numerical Comparisons
            f.write("\n## Numerical Value Comparison\n\n")
            f.write("For common country-year combinations:\n\n")
            
            f.write("| Variable | Valid N | Python Mean | Julia Mean | Mean Diff (%) | Max Diff (%) | Correlation |\n")
            f.write("|----------|---------|-------------|------------|---------------|--------------|-------------|\n")
            
            for var_name, stats in comparisons.items():
                f.write(f"| {var_name} | {stats['n_valid']} | ")
                f.write(f"{stats['python_mean']:.4f} | ")
                f.write(f"{stats['julia_mean']:.4f} | ")
                f.write(f"{stats['rel_diff_mean']:.2f} | ")
                f.write(f"{stats['rel_diff_max']:.2f} | ")
                f.write(f"{stats['correlation']:.6f} |\n")
            
            # Detailed Statistics
            f.write("\n### Detailed Statistics\n\n")
            for var_name, stats in comparisons.items():
                f.write(f"#### {var_name}\n\n")
                f.write(f"- **Valid observations:** {stats['n_valid']} / {stats['n_total']}\n")
                f.write(f"- **Python:** mean={stats['python_mean']:.4f}, std={stats['python_std']:.4f}\n")
                f.write(f"- **Julia:** mean={stats['julia_mean']:.4f}, std={stats['julia_std']:.4f}\n")
                f.write(f"- **Absolute difference:** mean={stats['abs_diff_mean']:.4f}, ")
                f.write(f"std={stats['abs_diff_std']:.4f}, max={stats['abs_diff_max']:.4f}\n")
                f.write(f"- **Relative difference:** mean={stats['rel_diff_mean']:.2f}%, ")
                f.write(f"std={stats['rel_diff_std']:.2f}%, max={stats['rel_diff_max']:.2f}%\n")
                f.write(f"- **Correlation:** {stats['correlation']:.6f}\n\n")
            
            # Known Differences
            f.write("\n## Known Implementation Differences\n\n")
            f.write("### Data Processing\n")
            f.write("- **Age calculation:** Both use midpoint of age_low and age_high\n")
            f.write("- **Missing data handling:** Both skip country-years with missing data\n")
            f.write("- **Age sorting:** Both sort by age before interpolation\n\n")
            
            f.write("### Model Parameters\n")
            f.write("- **Python:** Uses Pydantic config with centralized parameters\n")
            f.write("- **Julia:** Uses BiologicalParameters and EconomicParameters structs\n")
            f.write("- **Comparison needed:** Verify all parameters match exactly\n\n")
            
            f.write("### Numerical Methods\n")
            f.write("- **Optimization:** Both use similar lifecycle optimization\n")
            f.write("- **VSL Calibration:** Python uses brentq + adaptive grid search with fallback tolerance\n")
            f.write("- **Julia calibration:** Verify calibration approach matches\n\n")
            
            f.write("### Output Format\n")
            f.write("- **Python:** Outputs to `analysis.csv`\n")
            f.write("- **Julia:** Outputs to `international_comp.csv`\n\n")
            
            # Recommendations
            f.write("\n## Recommendations\n\n")
            
            # Check for large differences
            large_diffs = []
            for var_name, stats in comparisons.items():
                if abs(stats['rel_diff_mean']) > 5.0:  # More than 5% difference
                    large_diffs.append((var_name, stats['rel_diff_mean']))
            
            if large_diffs:
                f.write("### ⚠️ Large Differences Detected\n\n")
                f.write("The following variables show mean differences >5%:\n\n")
                for var_name, diff in large_diffs:
                    f.write(f"- **{var_name}:** {diff:.2f}% difference\n")
                f.write("\n**Action items:**\n")
                f.write("1. Verify model parameters match exactly between implementations\n")
                f.write("2. Check calibration procedures (especially VSL calibration)\n")
                f.write("3. Review numerical precision and tolerance settings\n")
                f.write("4. Examine data preprocessing differences\n\n")
            else:
                f.write("### ✓ Results are Highly Consistent\n\n")
                f.write("All variables show mean differences <5%, indicating strong agreement ")
                f.write("between implementations.\n\n")
            
            # Coverage differences
            if coverage['only_python'] > 0 or coverage['only_julia'] > 0:
                f.write("### Data Coverage Differences\n\n")
                f.write("Some country-year combinations are present in only one implementation. ")
                f.write("This may be due to:\n")
                f.write("- Different data availability checks\n")
                f.write("- Different handling of missing data\n")
                f.write("- Different convergence criteria\n\n")
            
            f.write("---\n\n")
            f.write("*End of Report*\n")
        
        print(f"✓ Report saved to: {output_path}")
    
    def run_comparison(self, skip_runs: bool = False, output_path: Path = None) -> None:
        """Run complete comparison suite.
        
        Args:
            skip_runs: If True, skip running the scripts and only compare existing outputs
            output_path: Path for comparison report
        """
        if output_path is None:
            output_path = self.repo_root / "output" / f"comparison_report_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.md"
        
        performance = {}
        
        # Run analyses
        if not skip_runs:
            # Run Python
            py_time, py_success = self.run_python_analysis()
            performance['python'] = {'time': py_time, 'success': py_success}
            
            # Run Julia
            jl_time, jl_success = self.run_julia_analysis()
            performance['julia'] = {'time': jl_time, 'success': jl_success}
        else:
            print("\nSkipping script execution, comparing existing outputs...")
            performance['python'] = {'time': 0, 'success': True}
            performance['julia'] = {'time': 0, 'success': True}
        
        # Load outputs
        py_df, jl_df = self.load_outputs()
        
        if py_df is None or jl_df is None:
            print("\nERROR: Could not load both output files. Comparison cannot proceed.")
            return
        
        # Compare coverage
        coverage = self.compare_data_coverage(py_df, jl_df)
        
        # Align and compare values
        if coverage['common'] > 0:
            merged_df = self.align_dataframes(py_df, jl_df)
            comparisons = self.compare_numerical_values(merged_df)
        else:
            print("\nWARNING: No common country-year combinations found!")
            comparisons = {}
        
        # Generate report
        self.generate_report(output_path, performance, coverage, comparisons, skip_runs)
        
        print("\n" + "="*70)
        print("COMPARISON COMPLETE")
        print("="*70)
        print(f"\nReport saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare Julia and Python implementations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--skip-runs', action='store_true',
        help='Skip running the scripts, only compare existing outputs'
    )
    parser.add_argument('--output', type=Path, default=None,
        help='Path for comparison report (default: output/comparison_report_YYYY-MM-DD-HH-MM-SS.md)'
    )
    
    args = parser.parse_args()
    
    # Get repository root
    repo_root = Path(__file__).parent.absolute()
    
    print("="*70)
    print("JULIA VS PYTHON IMPLEMENTATION COMPARISON")
    print("="*70)
    print(f"Repository: {repo_root}")
    print(f"Skip runs: {args.skip_runs}")
    
    # Run comparison
    comparison = ImplementationComparison(repo_root)
    comparison.run_comparison(skip_runs=args.skip_runs, output_path=args.output)


if __name__ == "__main__":
    main()


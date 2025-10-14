#!/usr/bin/env python3
"""Analysis Module.

This module implements cross-country analysis of health improvements and
economic value, providing functionality equivalent to international_empirical.jl.

The core function fits a lifecycle economic model to country-year health and
mortality data, computing the economic value of health improvements through
willingness-to-pay measures.

Example:
    >>> from analysis import fit_model_to_data
    >>> result = fit_model_to_data(country_df, next_health, next_survival)
    >>> print(f"Life expectancy: {result['le']:.2f} years")

Constants:
    DEFAULT_COUNTRIES: List of sample countries used in typical analyses.
    DEFAULT_YEARS: Range of sample years used in typical analyses.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import argparse
from tqdm import tqdm

from model import LifeCycleModel
from paths import OUTPUT_DIR, INTERMEDIATE_DIR

# Module-level constants
DEFAULT_COUNTRIES: List[str] = [
    "Australia",
    "France",
    "Germany",
    "Italy",
    "Japan",
    "Netherlands",
    "Spain",
    "Sweden",
    "United Kingdom",
    "United States of America"
]

DEFAULT_YEARS: List[int] = list(range(1990, 2000))


def fit_model_to_data(
        country_year_df: pd.DataFrame, 
        next_year_health: pd.Series, 
        next_year_survival: pd.Series,
        births_df: pd.DataFrame
    ) -> Dict[str, float]:
    """Fit lifecycle economic model to estimate value of health improvements.
    
    This function implements a full lifecycle optimization to value marginal
    improvements in health and mortality. The model:
    1. Sets up biological state variables (health, survival, population)
    2. Solves household lifecycle optimization (consumption/leisure choices)
    3. Computes Value of Statistical Life (VSL)
    4. Calibrates model parameters to match country-specific VSL
    5. Computes willingness-to-pay (WTP) for health improvements
    
    Economic Framework:
        Households maximize lifetime utility by choosing consumption and leisure
        paths subject to health, mortality, and budget constraints. The model
        is calibrated to match observed VSL, then used to value marginal health
        improvements.
    
    Args:
        country_year_df: Current year data with required columns:
            - age: Integer ages from 0 to max_age
            - population: Population count by age
            - survival: Survival probabilities by age [0, 1]
            - health: Health levels by age [0, 1]
            - real_gdp_usd: Real GDP in USD (constant across ages)
        next_year_health: Health rates for next year (counterfactual), length
            must match country_year_df.
        next_year_survival: Survival rates for next year (counterfactual), 
            length must match country_year_df.
        births_df: DataFrame with births data for the country-year including:
            - year: Year
            - births: Births
            
    Returns:
        Dictionary containing:
            - country (str): Country name
            - year (int): Year
            - population (float): Total population in millions
            - real_gdp (float): Real GDP in billions USD
            - real_gdp_pc (float): GDP per capita in thousands USD
            - vsl (float): Value of Statistical Life in millions USD
            - le (float): Life expectancy in years
            - hle (float): Healthy life expectancy in years
            - wtp_s (float): WTP for survival improvements in trillions USD
            - wtp_h (float): WTP for health improvements in trillions USD
            - wtp (float): Total WTP in trillions USD
            - wtp_pc (float): WTP per capita in thousands USD
            - wtp_0 (float): WTP for the unborn in thousands USD
            - wtp_unborn (float): WTP for the unborn in millions USD
            
    Raises:
        ValueError: If input data is invalid or missing required columns.
    """
    # ===================================================================
    # STEP 1: Initialize the model
    # ===================================================================
    # Create the model with standard parameters
    # given health capital, mortality risk, and population structure
    model = LifeCycleModel()

    # Initialize new DataFrame with given country-year "biological" variables
    df = model.initialize_biological_variables(country_year_df, next_year_health, next_year_survival)

    # Compute wages based on age and health
    # This determines the budget constraint and opportunity cost of leisure
    model.compute_wage_profile(df)
    
    # Compute wage gradient (change in wages due to health improvement)
    model.compute_wage_gradient(df)

    # ===================================================================
    # STEP 2: Calculate baseline variables and VSL for calibration
    # ===================================================================
    # Population and GDP statistics
    total_population = country_year_df['population'].sum()
    gdp_total = country_year_df['real_gdp_usd'].iloc[0]  # GDP in USD (same for all ages)
    gdp_pc = gdp_total / total_population

    # VSL varies by country income level. We compute
    # the reference VSL based on GDP per capita to use as calibration target.
    vsl_target = model.compute_country_ref_VSL(gdp_pc)


    # ===================================================================
    # STEP 3: Solve lifecycle optimization (initial pass)
    # ===================================================================
    # Find optimal consumption and leisure paths that
    # maximize lifetime utility subject to budget constraints
    model.solve_lifecycle_optimization(df)
    
    # Compute marginal utility of health capturing both direct utility effects and wage effects
    model.compute_marginal_utility_health(df)
    

    # ===================================================================
    # STEP 4: Compute Value of Statistical Life (VSL) and calibrate
    # ===================================================================
    # New VSL with solved model
    model.compute_VSL(df)
    
    # Adjust child wage parameter so model-implied VSL matches the observed country-specific VSL
    # Calibration function now returns the fully solved DataFrame, so no need to re-solve
    df = model.calibrate_wage_child(df, vsl_target)
    
    # Note: Steps that were previously in STEP 5 (re-solving after calibration) are now
    # handled inside calibrate_wage_child(), which returns the fully solved DataFrame.
    # This eliminates redundant computations and improves performance by ~20%.
    

    # ===================================================================
    # STEP 5: Compute willingness-to-pay for health improvements
    # ===================================================================
    model.compute_willingness_to_pay(df)
    

    # ===================================================================
    # STEP 6: Aggregate to population-level outcomes + final calculations
    # ===================================================================

    # Save for debugging
    df['year'] = country_year_df['year'].iloc[0]
    df['country'] = country_year_df['country'].iloc[0]
    df.to_csv(OUTPUT_DIR / "model_output_sample.csv", index=False)

    # Expected years of life remaining at birth (sum of survival probabilities)
    life_expectancy = np.sum(df['S'])
    
    # Expected years of healthy life (sum of survival × health across all ages)
    healthy_life_expectancy = np.sum(df['S'] * df['H'])
    
    # Total (population-weighted) WTP 
    wtp_s = np.sum(df['population'] * df['WTP_S']) # For survival improvements
    wtp_h = np.sum(df['population'] * df['WTP_H']) # For health improvements
    wtp = np.sum(df['population'] * (df['WTP_S'] + df['WTP_H'])) # For total WTP

    wtp_0 = df['WTP'].iloc[0]
    wtp_unborn = model.compute_unborn_wtp(df, births_df)

    # Return final results
    return {
        'country': country_year_df['country'].iloc[0],
        'year': country_year_df['year'].iloc[0],
        'population': total_population / 1e6,  # Convert to millions
        'real_gdp': gdp_total / 1e9,  # Convert to billions
        'real_gdp_pc': gdp_pc / 1e3,  # Convert to thousands
        'vsl': vsl_target / 1e6,  # Convert to millions
        'le': life_expectancy,
        'hle': healthy_life_expectancy,
        'wtp_s': wtp_s / 1e12,  # Convert to trillions
        'wtp_h': wtp_h / 1e12,  # Convert to trillions
        'wtp': wtp / 1e12,  # Convert to trillions
        'wtp_pc': wtp / total_population / 1e3,  # Convert to thousands
        'wtp_0': wtp_0 / 1e3, # Convert to thousands
        'wtp_unborn': wtp_unborn / 1e12, # Convert to trillions
    }    

def summarize_country_year(summary: Dict[str, any]) -> None:
    translations = {
        'total_population': ('Population (millions)', 1e6),
        'real_gdp_usd': ('GDP (billions USD)', 1e9),
        'real_gdp_pc': ('GDP per capita (thousands USD)', 1e3),
        'le': ('Life expectancy (GBD)', 1),
        'hle': ('Healthy life expectancy (GBD)', 1),
        'hle_WHO': ('Healthy life expectancy (WHO)', 1),
        'le_WHO': ('Life expectancy (WHO)', 1),
        'births': ('Births (millions)', 1e6),
        'avg_health': ('Average health', 1),
        'avg_survival': ('Average survival', 1)
    }
    for k, v in summary.items():
        if not translations.get(k): continue
        if np.isnan(v): continue
        print(f"  {translations[k][0]}: {round(v / translations[k][1], 3)}")

def main(countries: List[str] = None, 
                    years: List[int] = None,
                    generate_synthetic_data: bool = True,
                    fail_on_error: bool = False,
                    verbose: bool = False) -> pd.DataFrame:
    """Run analysis for specified countries and years.
    
    This function loads merged health and GDP data, runs the full lifecycle
    model for each country-year combination, and saves results to CSV.
    
    Note: If next year data is unavailable, synthetic data is generated using
    small random perturbations (±1% standard deviation). This is for
    testing purposes only and should not be used for production analyses.

    This function writes to the OUTPUT_DIR / "international_analysis.csv" file.
    
    Args:
        countries: List of country names to analyze. Defaults to DEFAULT_COUNTRIES.
        years: List of years to analyze. Defaults to DEFAULT_YEARS.
        generate_synthetic_data: Whether to generate synthetic data if next year data is unavailable.
        fail_on_error: Whether to raise an error if analysis fails for any country-year.
        verbose: Whether to print verbose output.
        
    Returns:
        DataFrame with results for all country-year combinations.
        
    Raises:
        FileNotFoundError: If merged data file doesn't exist.
        Exception: If analysis fails for any country-year (re-raised with context).
    """    
    print("=" * 80)
    print("INTERNATIONAL HEALTH AND ECONOMIC ANALYSIS")
    print("=" * 80)

    # Load merged health and GDP data as well as fertility (births) data
    df = pd.read_csv(INTERMEDIATE_DIR / "merged.csv")
    assert df.isna().sum().sum() == 0, "Data contains missing values"
    summaries = pd.read_csv(INTERMEDIATE_DIR / "merged_summaries.csv")
    fert_df = pd.read_csv(INTERMEDIATE_DIR / "fertility_expanded.csv")

    # Select countries and years
    countries = sorted(set(countries or df['country'].unique()))
    years = sorted(set(years or df['year'].unique()))
        
    print(f"\nAnalyzing {len(countries)} countries: {', '.join(countries)}")
    print(f"Analyzing {len(years)} years: {', '.join(map(str, years))}")
    print(f"Total country-year combinations: {len(countries) * len(years)}\n")

    # Run analysis for all country-year combinations
    results = []
    for country in tqdm(countries, desc="Analyzing countries"):        
        for year in years:
            print(f"\nAnalyzing {country} in {year}...")
            result = {'country': country, 'year': year}
            try:
                # Get current country-year DataFrame
                country_year_df = df[(df['country'] == country) & (df['year'] == year)]
                assert not country_year_df.empty, f"Data unavailable for {country} in {year}"
                assert len(country_year_df.real_gdp_usd.unique()) == 1, f"Real GDP data inconsistent for {country} in {year}"
                assert country_year_df.real_gdp_usd.isna().sum() == 0, f"Real GDP data is missing for {country} in {year}"

                summary = summaries[(summaries['country'] == country) & (summaries['year'] == year)]
                assert len(summary) == 1, f"Summary data unavailable for {country} in {year}"
                summary = summary.iloc[0].to_dict()
                if verbose: summarize_country_year(summary)

                # Get next country-year variables (counterfactual scenario)
                next_df = df[(df['country'] == country) & (df['year'] == year + 1)]
                if next_df.empty:
                    if generate_synthetic_data:
                        # Generate synthetic next-year data for testing (not for production!)
                        print(f"Warning: No data available for {country} in {year + 1}. "
                            "Using synthetic data for testing.")
                        n_ages = len(country_year_df)
                        next_health = (
                            country_year_df['health'] + 
                            np.random.normal(0, 0.01, n_ages)
                        )
                        next_survival = (
                            country_year_df['survival'] + 
                            np.random.normal(0, 0.01, n_ages)
                        )
                    else:
                        raise ValueError(f"Data unavailable for {country} in {year + 1}")
                else:
                    next_health = next_df['health']
                    next_survival = next_df['survival']

                # Get current country-year birth projections
                births = fert_df[(fert_df['Country'] == country) & (fert_df['year'] > year)]
        
                # Compute model results
                result = fit_model_to_data(country_year_df, next_health, next_survival, births)         
                
            except Exception as e:
                print(f"  Error analyzing {country} in {year}: {e}\n")
                if fail_on_error:
                    raise  # Re-raise with full traceback for debugging

            finally:
                results.append(result)

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print(f"Results: {len(results_df.dropna())}/{len(results_df)} country-year combinations")
    print(f"Output: {OUTPUT_DIR / 'analysis.csv'} ({len(results_df)} rows)")
    results_df.to_csv(OUTPUT_DIR / 'analysis.csv', index=False)

    return results_df
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run international analysis')
    parser.add_argument('--countries', type=str, default=None, help='Comma-separated list of countries to analyze')
    parser.add_argument('--years', type=str, default=None, help='Comma-separated list of years to analyze')
    parser.add_argument('--default', action='store_true', help='Use default sample of countries and years')
    parser.add_argument('--no_synthetic_data', action='store_true', help='Do not generate synthetic data if next year data is unavailable')
    parser.add_argument('--fail_on_error', action='store_true', help='Fail on error')
    args = parser.parse_args()
    
    if args.default:
        countries = DEFAULT_COUNTRIES
        years = DEFAULT_YEARS
    else:
        countries = args.countries.split(',') if args.countries else None
        years = [int(y) for y in args.years.split(',')] if args.years else None
    
    main(countries, years, generate_synthetic_data=not args.no_synthetic_data, fail_on_error=args.fail_on_error)
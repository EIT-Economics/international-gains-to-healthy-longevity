#!/usr/bin/env python3
"""
Main Execution Script for Health and Longevity Analysis

This script provides a unified interface for running all the analyses
that were previously done in separate R and Julia scripts.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Callable

from analysis.data_processing import DataProcessor
from models.health_longevity_model import HealthLongevityModel, ModelOptions
from models.biological_model import BiologicalParameters
from models.economic_model import EconomicParameters
from analysis.international_analysis import InternationalAnalyzer
from analysis.scenario_analysis import ScenarioAnalyzer
from analysis.social_welfare import SocialWelfareAnalyzer
from visualization.plot_3d_surfaces import SurfacePlotter
from visualization.plot_comparisons import ComparisonPlotter
from visualization.plotting import create_historical_plot, create_oneyear_plot, create_exploratory_plots

DEFAULT_COUNTRIES = [
    "Australia",
    "Canada",
    "France",
    "Germany",
    "Israel",
    "Italy",
    "Japan",
    "Netherlands",
    "New Zealand",
    "Spain",
    "Sweden",
    "United Kingdom",
    "United States of America"
]

DEFAULT_YEARS = list(np.arange(1990, 2021))

class HealthLongevityAnalysis:
    """Main analysis class that coordinates all components"""
    
    def __init__(self, data_dir: str = "intermediate", output_dir: str = "output", figures_dir: str = "figures"):
        """
        Initialize the analysis
        
        Args:
            data_dir: Directory containing data files
            output_dir: Directory for output files
            figures_dir: Directory for figures
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(exist_ok=True)        
    
    @staticmethod
    def _run_international_comparison(fn: Callable,
                                   countries: List[str] = DEFAULT_COUNTRIES, 
                                   years: List[int] = DEFAULT_YEARS,
                                   fail_silently: bool = False) -> pd.DataFrame:
        """
        Run international comparison (sequentially) across countries and years.
        
        Args:
            fn: Function to run for each country-year combination
            countries: List of country names
            years: List of years to analyze
            
        Returns:
            DataFrame with results for all country-year combinations
        """
        results = []
        for country in countries:
            for year in years:
                try:
                    result = fn(country, year)
                    if 'error' not in result:
                        results.append(result)
                    else:
                        print(f"Warning: {result['error']}")
                except Exception as e:
                    print(f"Error analyzing {country} in {year}: {e}")
                    if not fail_silently:
                        raise
        
        return pd.DataFrame(results)

    def _export_results(self, results: pd.DataFrame, filename: str):
        """Export results to CSV file"""
        results.to_csv(self.output_dir / filename, index=False)
        print(f"Results exported to {self.output_dir / filename}")

    def run_international_analysis(self, countries: List[str] = DEFAULT_COUNTRIES, 
                                  years: List[int] = DEFAULT_YEARS) -> pd.DataFrame:
        """
        Run international analysis.

        Saves 'international_analysis.csv' to the output directory.
        
        Args:
            countries: List of countries to analyze
            years: List of years to analyze
            
        Returns:
            DataFrame with international analysis results
        """                
        print("Running international analysis...")
        analyzer = InternationalAnalyzer(str(self.data_dir))
        results = self._run_international_comparison(analyzer.run_country_analysis, countries, years)
        self._export_results(results, "international_analysis.csv")
        return results
    
    def run_3d_surface_analysis(self, param1: str = "M1", param2: str = "D1",
                               le_range: np.ndarray = np.arange(60, 101, 1), 
                               hle_range: np.ndarray = np.arange(55, 101, 1),
                               plot_3d_surface: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run 3D surface analysis (equivalent to 3d_plots.jl)
        
        Args:
            param1: First parameter name
            param2: Second parameter name
            le_range: Range of life expectancy values
            hle_range: Range of healthy life expectancy values
            plot_3d_surface: Whether to plot 3D surface
            
        Returns:
            Dictionary with surface analysis results
        """
        print("Running 3D surface analysis...")
        
        # Set up model
        bio_params = BiologicalParameters()
        econ_params = EconomicParameters()
        options = ModelOptions(param=param1)
        model = HealthLongevityModel(bio_params, econ_params, options)
        
        # Compute surfaces
        struldbrugg_data = model.compute_3d_surface(param1, param2, le_range, hle_range)
        dorian_data = model.compute_3d_surface(param2, param1, le_range, hle_range)
        
        # Create plots   
        if plot_3d_surface:   
            plotter = SurfacePlotter()
            if not struldbrugg_data.empty:
                struldbrugg_figures = plotter.plot_struldbrugg_elongation(
                    struldbrugg_data, str(self.figures_dir)
                )
            if not dorian_data.empty:
                dorian_figures = plotter.plot_dorian_gray_elongation(
                    dorian_data, str(self.figures_dir)
                )
        
        # Save data
        self._export_results(struldbrugg_data, "struldbrugg_surface.csv")
        self._export_results(dorian_data, "dorian_surface.csv")
        return {'struldbrugg': struldbrugg_data, 'dorian': dorian_data}
    
    def run_scenario_analysis(self, 
                scenarios: List[str] = ['wolverine', 'peter_pan', 'rectangularization', 'elongation'], 
                plot_wtp: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run scenario analysis (equivalent to scenarios_wolverine.jl)
        
        Args:
            scenarios: List of scenarios to run
            plot_wtp: Whether to plot WTP
            
        Returns:
            Dictionary with scenario analysis results
        """
        print("Running scenario analysis...")
        results = {}
        scenario_analyzer = ScenarioAnalyzer()
        
        # Wolverine scenario
        if 'wolverine' in scenarios:
            reset_values = np.linspace(0, 50, 11)
            wolverine_results = scenario_analyzer.run_wolverine_scenario(reset_values)
            results['wolverine'] = wolverine_results
            self._export_results(wolverine_results, "wolverine_scenario.csv")
        
        # Peter Pan scenario
        if 'peter_pan' in scenarios:
            delta2_values = np.linspace(0.5, 1.0, 11)
            peter_pan_results = scenario_analyzer.run_peter_pan_scenario(delta2_values)
            results['peter_pan'] = peter_pan_results
            self._export_results(peter_pan_results, "peter_pan_scenario.csv")
        
        # Rectangularization scenario
        if 'rectangularization' in scenarios:
            gamma_values = np.linspace(0.05, 0.15, 5)
            psi_values = np.linspace(0.03, 0.08, 5)
            rect_results = scenario_analyzer.run_rectangularization_scenario(gamma_values, psi_values)
            results['rectangularization'] = rect_results
            self._export_results(rect_results, "rectangularization_scenario.csv")
        
        # Elongation scenario
        if 'elongation' in scenarios:
            m1_values = np.linspace(0.2, 0.5, 5)
            d1_values = np.linspace(0.4, 0.8, 5)
            elongation_results = scenario_analyzer.run_elongation_scenario(m1_values, d1_values)
            results['elongation'] = elongation_results
            self._export_results(elongation_results, "elongation_scenario.csv")
        
        # Create comparison plots
        if plot_wtp:
            plotter = ComparisonPlotter()
            for scenario_name, scenario_data in results.items():
                if not scenario_data.empty:
                    # Plot WTP by age
                    wtp_plot = plotter.plot_wtp_by_age(
                        scenario_data, save_path=str(self.figures_dir / f"{scenario_name}_wtp.pdf")
                    )
        
        return results
    
    def run_social_welfare_analysis(self, countries: List[str] = DEFAULT_COUNTRIES,
                                   years: List[int] = [2020]) -> pd.DataFrame:
        """
        Run social welfare analysis (equivalent to social_WTP.jl)
        
        Args:
            countries: List of countries to analyze
            years: List of years to analyze
            
        Returns:
            DataFrame with social welfare analysis results
        """        
        print("Running social welfare analysis...")
        analyzer = SocialWelfareAnalyzer(str(self.data_dir))
        results = self._run_international_comparison(analyzer.run_social_wtp_analysis, countries, years)
        self._export_results(results, "social_welfare_analysis.csv")
        return results
    
    def run_us_empirical_analysis(self, diseases: List[str] = None) -> pd.DataFrame:
        """
        Run US empirical analysis (equivalent to US_empirical_WTP.jl)
        
        Args:
            diseases: List of diseases to analyze
            
        Returns:
            DataFrame with US empirical analysis results
        """
        if diseases is None:
            diseases = ["Cardiovascular diseases", "Neoplasms", "Chronic respiratory diseases",
                       "Neurological disorders", "Diabetes and kidney diseases"]
        
        print("Running US empirical analysis...")
        analyzer = SocialWelfareAnalyzer(str(self.data_dir))
        results = analyzer.compute_disease_specific_analysis(
            "United States of America", 2019, diseases
        )
        self._export_results(results, "us_empirical_analysis.csv")
        return results
    
    def run_all_analyses(self) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        """Run all analyses helper function."""
        print("Running all analyses...")
        results = {}
        results['international'] = self.run_international_analysis()
        results['3d_surfaces'] = self.run_3d_surface_analysis()
        results['scenarios'] = self.run_scenario_analysis()
        results['social_welfare'] = self.run_social_welfare_analysis()
        results['us_empirical'] = self.run_us_empirical_analysis()
        print("All analyses completed")
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Health and Longevity Analysis")
    parser.add_argument("--process_data", action="store_true", help="Process raw input data")
    parser.add_argument("--analysis", choices=['all', 'international', '3d_surfaces', 
                                              'scenarios', 'social_welfare', 'us_empirical'],
                       default='all', help="Type of analysis to run")
    parser.add_argument("--create_infographic", action="store_true", help="Create infographic plots")
    parser.add_argument("--create_exploratory", action="store_true", help="Create exploratory plots")
    
    args = parser.parse_args()
    
    # Process raw input data (creating necessary intermediate data files)
    if args.process_data:   
        DataProcessor().run_all_processing() 

    # Create exploratory plots
    if args.create_exploratory:
        create_exploratory_plots()

    # Initialize analysis
    analysis = HealthLongevityAnalysis()
    
    # Run specified analysis
    if args.analysis == 'all':
        results = analysis.run_all_analyses()
    elif args.analysis == 'international':
        results = analysis.run_international_analysis()
    elif args.analysis == '3d_surfaces':
        results = analysis.run_3d_surface_analysis()
    elif args.analysis == 'scenarios':
        results = analysis.run_scenario_analysis()
    elif args.analysis == 'social_welfare':
        results = analysis.run_social_welfare_analysis(["United States of America"], [2020])
    elif args.analysis == 'us_empirical':
        results = analysis.run_us_empirical_analysis()

    # Create infographic plots
    if args.create_infographic:
        create_historical_plot()
        create_oneyear_plot()
    
    print("Analysis completed successfully!")


if __name__ == "__main__":
    main()

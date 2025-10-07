#!/usr/bin/env python3
"""
Social Welfare Analysis Module

This module implements the functionality from social_WTP.jl,
providing social welfare analysis with population and fertility data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union
from pathlib import Path

from models.health_longevity_model import HealthLongevityModel, ModelOptions
from models.biological_model import BiologicalParameters
from models.economic_model import EconomicParameters

class SocialWelfareAnalyzer:
    """Analyzer for social welfare and population-level analysis"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the social welfare analyzer
        
        Args:
            data_dir: Directory containing processed data files
        """
        self.data_dir = Path(data_dir)
        self.population_df = self.load_population_data(self.data_dir)
        self.fertility_df = self.load_fertility_data(self.data_dir)
        self.le_df = self.load_life_expectancy_data(self.data_dir)
        self.gdp_df = self.load_gdp_data(self.data_dir)
        self.results = {}
    
    @staticmethod
    def load_population_data(data_dir: Path, source: str = "wpp") -> pd.DataFrame:
        """Load population data."""
        if source == "gbd":
            pop_file = data_dir / "GBD" / "mortality_rates.csv"
        else:
            # Default to WPP estimates
            pop_file = data_dir / "WPP" / "population.csv"
        assert pop_file.exists(), f"Population data file {pop_file} not found"
        return pd.read_csv(pop_file).query('Variant == "Estimates"')
        
    @staticmethod
    def load_fertility_data(data_dir: Path) -> pd.DataFrame:
        """Load fertility projections from WPP."""
        fert_file = data_dir / "WPP" / "fertility.csv"
        assert fert_file.exists(), f"Fertility file {fert_file} not found"
        return pd.read_csv(fert_file)
    
    @staticmethod
    def load_life_expectancy_data(data_dir: Path) -> pd.DataFrame:
        """Load life expectancy data."""
        le_file = data_dir / "WPP" / "life_expectancy.csv"
        assert le_file.exists(), f"Life expectancy file {le_file} not found"
        return pd.read_csv(le_file).query('Variant == "Estimates"')

    @staticmethod
    def load_gdp_data(data_dir: Path) -> pd.DataFrame:
        """Load GDP data."""
        gdp_file = data_dir / "WB" / "real_gdp_data.csv"
        assert gdp_file.exists(), f"GDP file {gdp_file} not found"
        return pd.read_csv(gdp_file)
    
    @staticmethod
    def compute_phased_intervention(model: HealthLongevityModel, 
                                  delta2_final: float, phasing: int = 5) -> pd.DataFrame:
        """
        Compute phased intervention scenario
        
        Args:
            model: Health longevity model
            delta2_final: Final delta2 value
            phasing: Number of phases
            
        Returns:
            DataFrame with phased intervention results
        """
        results = []
        
        for phase in range(phasing + 1):
            # Compute delta2 for this phase
            if phase == 0:
                delta2_phase = 1.0  # Baseline
            else:
                delta2_phase = 1.0 - (phase / phasing) * (1.0 - delta2_final)
            
            # Update model parameters
            model.bio_params.delta2 = delta2_phase
            
            # Compute life cycle variables
            df = model.compute_life_cycle_variables()
            # print("HERE WE HAVE LIFE CYCLE MODEL")
            # print(df.info())
            
            # Compute WTP for this phase
            wtp_results = {}
            for age in model.options.age_range:
                if age < len(df):
                    wtp_results[f'wtp_{age}'] = df['wtp_total'].iloc[age]
                else:
                    wtp_results[f'wtp_{age}'] = 0
            
            results.append({
                'phase': phase,
                'delta2': delta2_phase,
                'life_expectancy': model.compute_life_expectancy(),
                'healthy_life_expectancy': model.compute_healthy_life_expectancy(),
                'total_wtp': np.sum(df['wtp_total'] * df['survival']),
                **wtp_results
            })
        
        return pd.DataFrame(results)
    
    def run_social_wtp_analysis(self, country: str, year: int, 
                               intervention_type: str = 'phased',
                               delta2_final: float = 0.8) -> Dict[str, Union[float, str]]:
        """
        Run social WTP analysis for a country and year
        
        Args:
            country: Country name
            year: Year
            intervention_type: Type of intervention ('phased', 'immediate')
            delta2_final: Final delta2 value for intervention
            
        Returns:
            Dictionary with social WTP results
        """
        # Load data
        pop_data = self.population_df[
            (self.population_df['Country'] == country) & (self.population_df['Year'] == year)
        ]
        
        le_data = self.le_df[(self.le_df['Country'] == country) & (self.le_df['Year_high'] >= year) & (self.le_df['Year_low'] <= year)]
        assert len(le_data) <= 1, f"Multiple life expectancy data found for {country} in {year}"
        baseline_le = 78.9 if le_data.empty else le_data['Life_Expectancy'].iloc[0] # Using default fallback

        fert_data = self.fertility_df[(self.fertility_df['Country'] == country) & (self.fertility_df['Variant'] != "Estimates")]

        gdp_data = self.gdp_df[(self.gdp_df['location_name'] == country) & (self.gdp_df['year'] == year)]
        assert len(gdp_data) <= 1, f"Multiple GDP data found for {country} in {year}"
        GDP = gdp_data['real_gdp_usd'].iloc[0] if not gdp_data.empty else None
        
        if pop_data.empty:
            return {'error': f'No population data available for {country} in {year}'}
        
        # Set up model
        bio_params = BiologicalParameters()
        econ_params = EconomicParameters()
        
        # Fit model parameters to match observed LE/HLE (simple root-finding for delta2)
        # This is a simplified approach; for more accuracy, use a proper optimizer
        target_le = baseline_le
        delta2_guess = 1.0
        for d2 in np.linspace(0.7, 1.0, 31):
            bio_params.delta2 = d2
            model = HealthLongevityModel(bio_params, econ_params, ModelOptions())
            le = model.compute_life_expectancy()
            if abs(le - target_le) < 0.2:
                delta2_guess = d2
                break
        bio_params.delta2 = delta2_guess
        
        options = ModelOptions(
            param='delta2',
            age_start=65,  # Intervention starts at age 65
            phasing=5 if intervention_type == 'phased' else 1
        )
        
        model = HealthLongevityModel(bio_params, econ_params, options)
        
        # Compute baseline scenario
        baseline_df = model.compute_life_cycle_variables()
        baseline_hle = model.compute_healthy_life_expectancy()
        
        # Compute intervention scenario
        if intervention_type == 'phased':
            intervention_results = self.compute_phased_intervention(model, delta2_final, options.phasing)
            # Use final phase for output
            model.bio_params.delta2 = delta2_final
            intervention_df = model.compute_life_cycle_variables()
        else:
            # Immediate intervention
            model.bio_params.delta2 = delta2_final
            intervention_df = model.compute_life_cycle_variables()
        
        intervention_le = model.compute_life_expectancy()
        intervention_hle = model.compute_healthy_life_expectancy()
        
        # Merge population by age with model output
        social_wtp = self.aggregate_wtp_by_band(pop_data[['Age', 'Population']], intervention_df)
        pop_sum = (pop_data['Population'] * 1000).sum()
 

        # Individual WTP at age 0
        individual_wtp = intervention_df['wtp_total'].iloc[0] if len(intervention_df) > 0 else 0
        
        # Unborn WTP (future generations, discounted per birth year)
        unborn_wtp = 0
        if not fert_data.empty and len(fert_data) > 0:
            for _, row in fert_data.iterrows():
                birth_year = row['Year_mid'] # @TODO: Look into this
                births = row['Fertility_Rate'] * 1000  # WPP births in thousands
                years_ahead = birth_year - year
                assert years_ahead >= 0, "Birth year must be >= current year"
                discount = model.econ_model.discount_factor(years_ahead)
                unborn_wtp += individual_wtp * births * discount
        
        # Output columns to match Julia
        return {
            'Country': country,
            'Year': year,
            'LE': baseline_le,
            'HLE': baseline_hle,
            'Pop': pop_data['Population'].sum() / 1000,  # in millions
            'GDP_pc': GDP / pop_data['Population'].sum() / 1000 if GDP else np.nan, # in USD
            'VSL': baseline_df['vsl'].sum() / 1e6,  # millions
            'WTP_1y': individual_wtp,  
            'WTP_avg': social_wtp / pop_sum / 1e3 if pop_sum > 0 else 0,  # thousands
            'WTP_0': individual_wtp / 1e3,  # thousands
            'WTP_unborn': unborn_wtp / 1e12,  # trillions
            'Social_WTP': social_wtp / 1e12,  # trillions
        }
    
    @staticmethod
    def compute_disease_specific_analysis(country: str, year: int,
                                        diseases: List[str], 
                                        reduction_factor: float = 0.5) -> pd.DataFrame:
        """
        Compute disease-specific health improvements
        
        Args:
            country: Country name
            year: Year
            diseases: List of diseases to analyze
            reduction_factor: Factor by which to reduce disease burden
            
        Returns:
            DataFrame with disease-specific results
        """
        results = []
        
        for disease in diseases:
            # Set up model with disease-specific parameters (@TODO: add disease-specific parameters)
            bio_params = BiologicalParameters()
            econ_params = EconomicParameters()
            
            # Adjust parameters based on disease reduction
            # This is simplified - in practice, you'd have disease-specific models
            
            options = ModelOptions(param='delta2')
            model = HealthLongevityModel(bio_params, econ_params, options)
            
            # Compute baseline
            baseline_df = model.compute_life_cycle_variables()
            baseline_le = model.compute_life_expectancy()
            baseline_hle = model.compute_healthy_life_expectancy()
            
            # Compute with disease reduction
            model.bio_params.delta2 = 1.0 - (1.0 - reduction_factor) * 0.1  # Simplified disease effect
            intervention_df = model.compute_life_cycle_variables()
            intervention_le = model.compute_life_expectancy()
            intervention_hle = model.compute_healthy_life_expectancy()
            
            # Compute WTP
            wtp = intervention_df['wtp_total'].iloc[0] if len(intervention_df) > 0 else 0
            
            results.append({
                'country': country,
                'year': year,
                'disease': disease,
                'reduction_factor': reduction_factor,
                'le_improvement': intervention_le - baseline_le,
                'hle_improvement': intervention_hle - baseline_hle,
                'wtp': wtp / 1e3  # Convert to thousands
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def aggregate_wtp_by_band(pop_band_df, intervention_df):
        """
        Aggregate WTP by age band, assuming uniform age distribution within each band.
        pop_band_df: DataFrame with columns ['Age', 'Population']
        intervention_df: DataFrame with columns ['age', 'wtp_total']
        Returns total social WTP (float)
        """
        total_wtp = 0
        for _, row in pop_band_df.iterrows():
            band = row['Age']   # e.g., '0-4', '5-9', etc.
            pop = row['Population'] * 1000  # WPP is in thousands
            start, end = map(int, band.split('-'))
            ages_in_band = list(range(start, end + 1))
            # Get WTP for ages in this band
            wtp_vals = intervention_df.loc[intervention_df['age'].isin(ages_in_band), 'wtp_total']
            if len(wtp_vals) == 0:
                avg_wtp = 0
            else:
                avg_wtp = wtp_vals.mean()  # Uniform distribution assumption
            total_wtp += pop * avg_wtp
        return total_wtp



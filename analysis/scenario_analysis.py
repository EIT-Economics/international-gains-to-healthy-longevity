#!/usr/bin/env python3
"""
Scenario Analysis Module

This module implements the functionality from scenarios_wolverine.jl,
providing analysis of different aging intervention scenarios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

from models.health_longevity_model import HealthLongevityModel, ModelOptions
from models.biological_model import BiologicalParameters
from models.economic_model import EconomicParameters

class ScenarioAnalyzer:
    """Analyzer for different aging intervention scenarios"""
    
    @staticmethod
    def run_wolverine_scenario(reset_values: List[float], 
                             age_start: float = 0.0) -> pd.DataFrame:
        """
        Run Wolverine scenario (frailty reset)
        
        Args:
            reset_values: List of reset values to test
            age_start: Age at which intervention starts
            
        Returns:
            DataFrame with scenario results
        """
        results = []
        
        for reset_val in reset_values:
            # Set up model with Wolverine parameters
            bio_params = BiologicalParameters()
            bio_params.Reset = reset_val
            
            econ_params = EconomicParameters()
            options = ModelOptions(
                param='Reset',
                age_start=age_start,
                wolverine=50.0  # Wolverine starts at age 50
            )
            
            model = HealthLongevityModel(bio_params, econ_params, options)
            
            # Compute life cycle variables
            df = model.compute_life_cycle_variables()
            
            # Compute summary statistics
            le = model.compute_life_expectancy()
            hle = model.compute_healthy_life_expectancy()
            
            # Compute WTP at different ages
            wtp_results = {}
            for age in options.age_range:
                if age < len(df):
                    wtp_results[f'wtp_{age}'] = df['wtp_total'].iloc[age]
                else:
                    wtp_results[f'wtp_{age}'] = 0
            
            results.append({
                'scenario': 'Wolverine',
                'reset_value': reset_val,
                'life_expectancy': le,
                'healthy_life_expectancy': hle,
                'total_wtp': np.sum(df['wtp_total'] * df['survival']),
                **wtp_results
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def run_peter_pan_scenario(delta2_values: List[float], 
                              age_start: float = 30.0) -> pd.DataFrame:
        """
        Run Peter Pan scenario (slowed aging)
        
        Args:
            delta2_values: List of delta2 values to test
            age_start: Age at which intervention starts
            
        Returns:
            DataFrame with scenario results
        """
        results = []
        
        for delta2_val in delta2_values:
            # Set up model with Peter Pan parameters
            bio_params = BiologicalParameters()
            bio_params.delta2 = delta2_val
            
            econ_params = EconomicParameters()
            options = ModelOptions(
                param='delta2',
                age_start=age_start,
                wolverine=0.0
            )
            
            model = HealthLongevityModel(bio_params, econ_params, options)
            
            # Compute life cycle variables
            df = model.compute_life_cycle_variables()
            
            # Compute summary statistics
            le = model.compute_life_expectancy()
            hle = model.compute_healthy_life_expectancy()
            
            # Compute WTP at different ages
            wtp_results = {}
            for age in options.age_range:
                if age < len(df):
                    wtp_results[f'wtp_{age}'] = df['wtp_total'].iloc[age]
                else:
                    wtp_results[f'wtp_{age}'] = 0
            
            results.append({
                'scenario': 'Peter Pan',
                'delta2_value': delta2_val,
                'life_expectancy': le,
                'healthy_life_expectancy': hle,
                'total_wtp': np.sum(df['wtp_total'] * df['survival']),
                **wtp_results
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def run_rectangularization_scenario(gamma_values: List[float], 
                                       psi_values: List[float]) -> pd.DataFrame:
        """
        Run rectangularization scenario
        
        Args:
            gamma_values: List of gamma values to test
            psi_values: List of psi values to test
            
        Returns:
            DataFrame with scenario results
        """
        results = []
        
        for gamma_val in gamma_values:
            for psi_val in psi_values:
                # Set up model with rectangularization parameters
                bio_params = BiologicalParameters()
                bio_params.gamma = gamma_val
                bio_params.psi = psi_val
                
                econ_params = EconomicParameters()
                options = ModelOptions(
                    param='gamma',
                    no_compress=True
                )
                
                model = HealthLongevityModel(bio_params, econ_params, options)
                
                # Compute life cycle variables
                df = model.compute_life_cycle_variables()
                
                # Compute summary statistics
                le = model.compute_life_expectancy()
                hle = model.compute_healthy_life_expectancy()
                
                results.append({
                    'scenario': 'Rectangularization',
                    'gamma': gamma_val,
                    'psi': psi_val,
                    'life_expectancy': le,
                    'healthy_life_expectancy': hle,
                    'total_wtp': np.sum(df['wtp_total'] * df['survival'])
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def run_elongation_scenario(m1_values: List[float], 
                               d1_values: List[float]) -> pd.DataFrame:
        """
        Run elongation scenario (Struldbrugg and Dorian Gray)
        
        Args:
            m1_values: List of M1 values to test
            d1_values: List of D1 values to test
            
        Returns:
            DataFrame with scenario results
        """
        results = []
        
        for m1_val in m1_values:
            for d1_val in d1_values:
                # Set up model with elongation parameters
                bio_params = BiologicalParameters()
                bio_params.M1 = m1_val
                bio_params.D1 = d1_val
                
                econ_params = EconomicParameters()
                options = ModelOptions(param='M1')
                
                model = HealthLongevityModel(bio_params, econ_params, options)
                
                # Compute life cycle variables
                df = model.compute_life_cycle_variables()
                
                # Compute summary statistics
                le = model.compute_life_expectancy()
                hle = model.compute_healthy_life_expectancy()
                
                results.append({
                    'scenario': 'Elongation',
                    'M1': m1_val,
                    'D1': d1_val,
                    'life_expectancy': le,
                    'healthy_life_expectancy': hle,
                    'total_wtp': np.sum(df['wtp_total'] * df['survival'])
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def run_repeated_reversals(starting_ages: List[float], 
                             delta2_values: List[float]) -> pd.DataFrame:
        """
        Run repeated reversals scenario
        
        Args:
            starting_ages: List of starting ages
            delta2_values: List of delta2 values to test
            
        Returns:
            DataFrame with scenario results
        """
        results = []
        
        for start_age in starting_ages:
            for delta2_val in delta2_values:
                # Set up model with repeated reversal parameters
                bio_params = BiologicalParameters()
                bio_params.delta2 = delta2_val
                bio_params.MaxAge = 500  # Extended for repeated reversals
                
                econ_params = EconomicParameters()
                options = ModelOptions(
                    param='delta2',
                    age_start=start_age,
                    wolverine=0.0
                )
                
                model = HealthLongevityModel(bio_params, econ_params, options)
                
                # Compute life cycle variables
                df = model.compute_life_cycle_variables()
                
                # Compute summary statistics
                le = model.compute_life_expectancy()
                hle = model.compute_healthy_life_expectancy()
                
                # Compute WTP at different ages
                wtp_results = {}
                for age in [0, 20, 40, 60, 80]:
                    if age < len(df):
                        wtp_results[f'wtp_{age}'] = df['wtp_total'].iloc[age]
                    else:
                        wtp_results[f'wtp_{age}'] = 0
                
                results.append({
                    'scenario': 'Repeated Reversals',
                    'starting_age': start_age,
                    'delta2': delta2_val,
                    'life_expectancy': le,
                    'healthy_life_expectancy': hle,
                    'total_wtp': np.sum(df['wtp_total'] * df['survival']),
                    **wtp_results
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def compare_scenarios(self, scenario_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compare different scenarios
        
        Args:
            scenario_results: Dictionary with scenario names and results
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for scenario_name, results_df in scenario_results.items():
            if not results_df.empty:
                # Compute summary statistics for each scenario
                summary = {
                    'scenario': scenario_name,
                    'avg_life_expectancy': results_df['life_expectancy'].mean(),
                    'avg_healthy_life_expectancy': results_df['healthy_life_expectancy'].mean(),
                    'avg_total_wtp': results_df['total_wtp'].mean(),
                    'max_life_expectancy': results_df['life_expectancy'].max(),
                    'max_healthy_life_expectancy': results_df['healthy_life_expectancy'].max(),
                    'max_total_wtp': results_df['total_wtp'].max()
                }
                comparison_data.append(summary)
        
        return pd.DataFrame(comparison_data)
    
    @staticmethod
    def export_results(results_df: pd.DataFrame, filename: str):
        """Export results to CSV file"""
        results_df.to_csv(filename, index=False)
        print(f"Scenario analysis results exported to {filename}")
    
    @staticmethod
    def create_wtp_table(results_df: pd.DataFrame, 
                        age_columns: List[str] = None) -> pd.DataFrame:
        """
        Create WTP table for different ages
        
        Args:
            results_df: Results DataFrame
            age_columns: List of age columns to include
            
        Returns:
            DataFrame with WTP table
        """
        if age_columns is None:
            age_columns = [f'wtp_{age}' for age in [0, 20, 40, 60, 80]]
        
        # Select relevant columns
        wtp_columns = ['scenario'] + [col for col in age_columns if col in results_df.columns]
        wtp_table = results_df[wtp_columns].copy()
        
        return wtp_table

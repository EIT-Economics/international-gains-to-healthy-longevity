#!/usr/bin/env python3
"""
Comparison Plotting Module

This module provides comparison plots and tables for different scenarios,
equivalent to the plotting functionality in the Julia scripts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

class ComparisonPlotter:
    """Plotter for comparison plots and tables"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the comparison plotter
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_life_cycle_variables(self, data: pd.DataFrame, 
                                variables: List[str] = None,
                                title: str = "Life Cycle Variables",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot life cycle variables
        
        Args:
            data: DataFrame with life cycle data
            variables: List of variables to plot
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if variables is None:
            variables = ['health', 'survival', 'wage', 'consumption', 'leisure', 'wtp_total']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, var in enumerate(variables):
            if i < len(axes) and var in data.columns:
                ax = axes[i]
                ax.plot(data['age'], data[var], 'b-', linewidth=2)
                ax.set_xlabel('Age')
                ax.set_ylabel(var.replace('_', ' ').title())
                ax.set_title(f'{var.replace("_", " ").title()} by Age')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(variables), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Life cycle variables plot saved to {save_path}")
        
        return fig
    
    def plot_scenario_comparison(self, scenario_data: Dict[str, pd.DataFrame],
                               variable: str = 'wtp_total',
                               title: str = "Scenario Comparison",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of different scenarios
        
        Args:
            scenario_data: Dictionary with scenario names and DataFrames
            variable: Variable to compare
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(scenario_data)))
        
        for i, (scenario_name, data) in enumerate(scenario_data.items()):
            if not data.empty and 'age' in data.columns and variable in data.columns:
                ax.plot(data['age'], data[variable], 'o-', 
                       label=scenario_name, color=colors[i], linewidth=2, markersize=4)
        
        ax.set_xlabel('Age')
        ax.set_ylabel(variable.replace('_', ' ').title())
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scenario comparison plot saved to {save_path}")
        
        return fig
    
    def plot_wtp_by_age(self, data: pd.DataFrame, 
                       ages: List[int] = None,
                       title: str = "WTP by Age",
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot WTP by age for different scenarios
        
        Args:
            data: DataFrame with WTP data
            ages: List of ages to plot
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if ages is None:
            ages = [0, 20, 40, 60, 80]
        
        # Support both long and wide input formats
        # Long format: columns include 'age', 'wtp_total', optional 'wtp_s', 'wtp_h'
        # Wide format: columns like 'wtp_0', 'wtp_20', ..., optionally 'wtp_s_0', 'wtp_h_0', etc.

        wide_total_cols = {a: f"wtp_{a}" for a in ages}
        has_wide = all(col in data.columns for col in wide_total_cols.values())

        if 'age' in data.columns and not has_wide:
            # Long format
            age_data = data[data['age'].isin(ages)].copy()
        else:
            # Wide format: aggregate across rows if multiple (e.g., scenarios); use mean
            totals = []
            comps_s = []
            comps_h = []
            for a in ages:
                col_total = f"wtp_{a}"
                val_total = data[col_total].mean() if col_total in data.columns else np.nan
                totals.append(val_total)

                col_s = f"wtp_s_{a}"
                col_h = f"wtp_h_{a}"
                comps_s.append(data[col_s].mean() if col_s in data.columns else np.nan)
                comps_h.append(data[col_h].mean() if col_h in data.columns else np.nan)

            age_data = pd.DataFrame({
                'age': ages,
                'wtp_total': totals,
                'wtp_s': comps_s,
                'wtp_h': comps_h
            })
        
        if age_data.empty:
            print("Warning: No data found for specified ages")
            return plt.figure()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot WTP by age
        ax1.bar(age_data['age'], age_data['wtp_total'], alpha=0.7, color='skyblue')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('WTP')
        ax1.set_title('WTP by Age')
        ax1.grid(True, alpha=0.3)
        
        # Plot WTP components (if available)
        if 'wtp_s' in age_data.columns and 'wtp_h' in age_data.columns and not age_data[['wtp_s','wtp_h']].isna().all().all():
            x = np.arange(len(age_data))
            width = 0.35
            
            ax2.bar(x - width/2, age_data['wtp_s'], width, label='WTP_S (Survival)', alpha=0.7)
            ax2.bar(x + width/2, age_data['wtp_h'], width, label='WTP_H (Health)', alpha=0.7)
            
            ax2.set_xlabel('Age')
            ax2.set_ylabel('WTP')
            ax2.set_title('WTP Components by Age')
            ax2.set_xticks(x)
            ax2.set_xticklabels(age_data['age'])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"WTP by age plot saved to {save_path}")
        
        return fig
    
    def plot_parameter_sensitivity(self, sensitivity_data: pd.DataFrame,
                                 param_col: str, value_col: str,
                                 title: str = "Parameter Sensitivity",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot parameter sensitivity analysis
        
        Args:
            sensitivity_data: DataFrame with sensitivity data
            param_col: Parameter column name
            value_col: Value column name
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Life expectancy sensitivity
        if 'life_expectancy' in sensitivity_data.columns:
            axes[0, 0].plot(sensitivity_data[param_col], sensitivity_data['life_expectancy'], 'bo-')
            axes[0, 0].set_xlabel(param_col)
            axes[0, 0].set_ylabel('Life Expectancy')
            axes[0, 0].set_title('Life Expectancy Sensitivity')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Healthy life expectancy sensitivity
        if 'healthy_life_expectancy' in sensitivity_data.columns:
            axes[0, 1].plot(sensitivity_data[param_col], sensitivity_data['healthy_life_expectancy'], 'ro-')
            axes[0, 1].set_xlabel(param_col)
            axes[0, 1].set_ylabel('Healthy Life Expectancy')
            axes[0, 1].set_title('Healthy Life Expectancy Sensitivity')
            axes[0, 1].grid(True, alpha=0.3)
        
        # WTP sensitivity
        if 'wtp_at_birth' in sensitivity_data.columns:
            axes[1, 0].plot(sensitivity_data[param_col], sensitivity_data['wtp_at_birth'], 'go-')
            axes[1, 0].set_xlabel(param_col)
            axes[1, 0].set_ylabel('WTP at Birth')
            axes[1, 0].set_title('WTP at Birth Sensitivity')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Total WTP sensitivity
        if 'total_wtp' in sensitivity_data.columns:
            axes[1, 1].plot(sensitivity_data[param_col], sensitivity_data['total_wtp'], 'mo-')
            axes[1, 1].set_xlabel(param_col)
            axes[1, 1].set_ylabel('Total WTP')
            axes[1, 1].set_title('Total WTP Sensitivity')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter sensitivity plot saved to {save_path}")
        
        return fig
    
    def create_summary_table(self, data: pd.DataFrame, 
                           group_col: str = 'scenario',
                           value_cols: List[str] = None) -> pd.DataFrame:
        """
        Create summary table
        
        Args:
            data: DataFrame with data
            group_col: Column to group by
            value_cols: Columns to summarize
            
        Returns:
            Summary DataFrame
        """
        if value_cols is None:
            value_cols = ['life_expectancy', 'healthy_life_expectancy', 'total_wtp']
        
        # Filter to available columns
        available_cols = [col for col in value_cols if col in data.columns]
        
        if not available_cols:
            return pd.DataFrame()
        
        summary = data.groupby(group_col)[available_cols].agg(['mean', 'std', 'min', 'max']).round(2)
        
        return summary
    
    def plot_historical_comparison(self, data: pd.DataFrame,
                                 title: str = "Historical Comparison",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot historical comparison (equivalent to historical.pdf from Julia)
        
        Args:
            data: DataFrame with historical data
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if 'period' in data.columns and 'wtp_pc' in data.columns:
            # Plot WTP per capita by period
            for country in data['country'].unique():
                country_data = data[data['country'] == country]
                ax.plot(country_data['period'], country_data['wtp_pc'], 'o-', 
                       label=country, alpha=0.7)
            
            ax.set_xlabel('Period')
            ax.set_ylabel('WTP per Capita (thousands)')
            ax.set_title(title)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Historical comparison plot saved to {save_path}")
        
        return fig
    
    def plot_country_summary(self, data: pd.DataFrame,
                           title: str = "Country Summary",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot country summary table (equivalent to oneyear.pdf from Julia)
        
        Args:
            data: DataFrame with country data
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table_data = data.values
        col_labels = data.columns
        
        table = ax.table(cellText=table_data,
                        colLabels=col_labels,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Color alternating rows
        for i in range(len(data)):
            for j in range(len(data.columns)):
                cell = table[(i+1, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')
                else:
                    cell.set_facecolor('white')
                cell.set_edgecolor('gray')
                cell.set_linewidth(0.5)
        
        ax.set_title(title, fontsize=14, pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Country summary plot saved to {save_path}")
        
        return fig
    
    def close_all_figures(self):
        """Close all matplotlib figures"""
        plt.close('all')

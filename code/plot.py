#!/usr/bin/env python3
"""Plotting Utilities for Health and Longevity Analysis.

This module provides functions to create visualizations of health trends,
mortality patterns, and economic value of longevity gains. Plots are saved
to the figures directory in high-resolution PDF format.

Functions:
    create_exploratory_plots: Generate mortality and health trend plots (US).
    create_historical_plot: Create GDP vs longevity gains heatmap by decade.
    create_oneyear_plot: Generate country summary table for one-year gains.
    main: Run all plotting functions sequentially.

Usage:
    $ python plot.py  # Generate all plots
    
    Or import specific functions:
    >>> from plot import create_exploratory_plots
    >>> create_exploratory_plots()
"""

import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

from paths import OUTPUT_DIR, FIGURES_DIR, INTERMEDIATE_GBD_DIR

# Plotting parameters
DPI = 300  # High resolution for publications
FIGSIZE_LARGE = (12, 8)  # For detailed plots
FIGSIZE_MEDIUM = (10, 3.5)  # For compact comparisons


def create_exploratory_plots(country: str = 'United States of America',
                            n_years: int = 5) -> None:
    """Create exploratory plots of mortality and health trends by age.
    
    Generates two plots showing how mortality rates and disability rates vary
    by age across different years. Useful for visualizing trends in population
    health over time.
    
    Args:
        country: Country name to plot. Default is 'United States of America'.
        n_years: Number of years to plot (starting from earliest). Default is 5.
        
    Saves:
        mortality_trends_{country}.pdf: Mortality rates by age and year.
        health_trends_{country}.pdf: Disability (YLD) rates by age and year.
        
    Raises:
        FileNotFoundError: If GBD data files don't exist.
    """
    print(f"Creating exploratory plots for {country}...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Load and process mortality data
    mort_df = pd.read_csv(INTERMEDIATE_GBD_DIR / "mortality_rates.csv")
    mort_df = mort_df.loc[mort_df['location_name'] == country]
    
    # Compute age midpoints for plotting
    mort_df['age'] = (mort_df['age_high'] + mort_df['age_low']) / 2.0
        
    # Plot mortality by age for different years
    plt.figure(figsize=FIGSIZE_LARGE)
    for year in mort_df['year'].unique()[:5]:  # Plot first 5 years
        year_data = mort_df.loc[mort_df['year'] == year]
        plt.plot(year_data['age'], year_data['Total'], 
                label=f'Year {year}', alpha=0.7)
    
    plt.xlabel('Age')
    plt.ylabel('Mortality Rate')
    plt.title('Mortality Rates by Age and Year')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mortality_trends_US.pdf", dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # Plot (US) health (i.e. disability) trends
    health_df = pd.read_csv(INTERMEDIATE_GBD_DIR / "morbidity_rates.csv")
    health_df = health_df.loc[health_df['location_name'] == 'United States of America']
    health_df['age'] = (health_df['age_high'] + health_df['age_low']) / 2.0
        
    # Plot health by age for different years
    plt.figure(figsize=FIGSIZE_LARGE)
    for year in health_df['year'].unique()[:n_years]:  # Plot first n years
        year_data = health_df.loc[health_df['year'] == year]
        plt.plot(year_data['age'], year_data['Total'], 
                label=f'Year {year}', alpha=0.7)
    
    plt.xlabel('Age')
    plt.ylabel('Disability Rate (YLD)')
    plt.title('Disability Rates by Age and Year')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "health_trends_US.pdf", dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Exploratory plots saved to {FIGURES_DIR}")

def create_historical_plot():
    """
    Create historical GDP vs longevity gains heatmap using `output/international_analysis.csv`.
    
    Bins years by decade and averages real_gdp_pc and wtp_pc across countries.
    Shows GDP growth, longevity gains, and combined gains by decade.
    
    Saves the following outputs to the figures directory:
        "historical.pdf"
    """
    print("Creating historical plot...")
    
    # Load data from international analysis
    analysis_data = pd.read_csv(OUTPUT_DIR / "international_analysis.csv")
    
    # Create decade bins
    analysis_data['decade'] = (analysis_data['year'] // 10) * 10
    analysis_data['decade_label'] = analysis_data['decade'].astype(str) + 's'
    
    # Calculate GDP growth as difference between end and beginning of decade
    analysis_data = analysis_data.sort_values(['country', 'year'])
    
    # Get first and last year of each decade for each country
    decade_start_end = analysis_data.groupby(['country', 'decade']).agg({
        'real_gdp_pc': ['first', 'last'],
        'wtp_pc': 'mean'  # Average WTP over the decade
    }).reset_index()
    
    # Flatten column names
    decade_start_end.columns = ['country', 'decade', 'gdp_start', 'gdp_end', 'wtp_avg']
    
    # Calculate GDP growth as difference (end - start)
    decade_start_end['gdp_growth_pc'] = decade_start_end['gdp_end'] - decade_start_end['gdp_start']
    
    # Calculate longevity gains as GDP growth + average WTP
    decade_start_end['longevity_gains'] = decade_start_end['gdp_growth_pc'] + decade_start_end['wtp_avg']
    
    # Create period labels for decades
    decade_start_end['period'] = decade_start_end['decade'].astype(int).astype(str) + 's'
    

    # Recreate plot 
    plot_data = []
    for _, row in decade_start_end.iterrows():
        plot_data.extend([
            {'country': row['country'], 'period': row['period'], 'value': row['gdp_growth_pc'], 'metric': 'GDP growth per capita'},
            {'country': row['country'], 'period': row['period'], 'value': row['longevity_gains'], 'metric': 'GDP + longevity gains per capita'}
        ])
    
    plot_data = pd.DataFrame(plot_data)
    
    # Create plot
    countries = sorted(plot_data['country'].unique())
    periods = sorted(plot_data['period'].unique())
    
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)
    
    # Custom colormap
    cmap = LinearSegmentedColormap.from_list('custom', ['red', 'white', 'lime'], N=100)
    
    for idx, metric in enumerate(['GDP growth per capita', 'GDP + longevity gains per capita']):
        data = plot_data[plot_data['metric'] == metric]
        
        # Create matrix
        matrix = np.full((len(countries), len(periods)), np.nan)
        for _, row in data.iterrows():
            i = countries.index(row['country'])
            j = periods.index(row['period'])
            matrix[i, j] = row['value']
        
        # Plot
        im = axes[idx].imshow(matrix, cmap=cmap, aspect='auto', vmin=-100, vmax=100)
        
        # Add text
        for i in range(len(countries)):
            for j in range(len(periods)):
                if not np.isnan(matrix[i, j]):
                    axes[idx].text(j, i, f"{matrix[i, j]:.1f}", ha='center', va='center', fontsize=8)
        
        # Labels
        axes[idx].set_xticks(range(len(periods)))
        axes[idx].set_xticklabels(periods, rotation=45, ha='right')
        axes[idx].set_yticks(range(len(countries)))
        axes[idx].set_yticklabels(countries if idx == 0 else [''] * len(countries))
        axes[idx].set_title(metric, fontsize=12, fontweight='bold')
    
    # Overall labels
    fig.text(0.5, 0.02, 'Period', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, left=0.1)
    
    # Save
    plt.savefig(FIGURES_DIR / "historical.pdf", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Historical plot saved to: {FIGURES_DIR / 'historical.pdf'}")

def create_oneyear_plot():
    """
    Create country summary table using `output/social_welfare_analysis.csv`.
    
    Saves the following outputs to the figures directory:
        "oneyear.pdf"
    """
    print("Creating one year summary table...")
    
    # Load and process data from social_wtp_table.csv
    df = pd.read_csv(OUTPUT_DIR / "social_wtp_table.csv") # Change to social_welfare_analysis.csv
    
    # Select and rename columns to match the expected format
    df = df[['Country', 'Pop', 'WTP_1y']].copy()
    df = df.rename(columns={
        'Pop': 'Population (millions)',
        'WTP_1y': 'Value of Additional Life Year (trillion USD)'
    })
    df = df.sort_values('Country').reset_index(drop=True)
    
    # Create table
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style table
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Data cell styling
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            cell = table[(i, j)]
            cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
            cell.set_edgecolor('gray')
            cell.set_linewidth(0.5)
    
    # Title
    plt.title('Country Summary: Value of Additional Life Year', 
              fontsize=14, pad=20)
    
    # Save
    plt.savefig(FIGURES_DIR / "oneyear.pdf", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"One year plot saved to: {FIGURES_DIR / 'oneyear.pdf'}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting utilities')
    parser.add_argument('--exploratory', action='store_true', help='Create exploratory plots')
    parser.add_argument('--historical', action='store_true', help='Create historical plot')
    parser.add_argument('--oneyear', action='store_true', help='Create one year plot')
    args = parser.parse_args()
    if args.exploratory:
        create_exploratory_plots()
    if args.historical:
        create_historical_plot()
    if args.oneyear:
        create_oneyear_plot()
    if not args.exploratory and not args.historical and not args.oneyear:
        # If no arguments are provided, create all plots
        create_exploratory_plots()
        create_historical_plot()
        create_oneyear_plot()
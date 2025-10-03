#!/usr/bin/env python3
"""
Simple Infographic Plotting

Creates two simple plots: historical GDP vs longevity gains heatmap and country summary table.

As per the original R code, this requires the following data files to be present:
- Table3.csv
- Andrew_international.xlsx
- Table2.csv

The script will save the plots to the figures directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def create_historical_plot():
    """Create historical GDP vs longevity gains heatmap, saves to historical.pdf in figures directory."""
    print("Creating historical plot...")
    
    # Load data
    table_data = pd.read_csv(INPUT_DIR / "Table3.csv")
    raw_data = pd.read_excel(INPUT_DIR / "Andrew_international.xlsx", sheet_name="All Data")
    
    # Process data
    raw_data['country'] = raw_data['country'].str.replace("United Kingdom", "UK").str.replace("United States of America", "USA")
    raw_data = raw_data[['country', 'year', 'real_gdp_pc']]
    raw_data = raw_data[(raw_data['country'].isin(table_data['country'])) & 
                       (raw_data['year'].astype(str).isin(['1990', '1999', '2009', '2018']))]
    
    # Calculate GDP growth
    raw_data = raw_data.sort_values(['country', 'year'])
    raw_data['gdp_growth_pc'] = raw_data.groupby('country')['real_gdp_pc'].diff()
    raw_data['year_lag'] = raw_data.groupby('country')['year'].shift(1)
    raw_data = raw_data.dropna(subset=['year_lag'])
    raw_data['period'] = raw_data['year_lag'].astype(int).astype(str) + '-' + raw_data['year'].astype(int).astype(str)
    
    # Merge and create long format
    historical_data = table_data.copy()
    historical_data['period'] = historical_data['period'].str.replace('2000', '1999').str.replace('2010', '2009')
    historical_data = historical_data.merge(raw_data[['country', 'period', 'gdp_growth_pc']], on=['country', 'period'], how='left')
    historical_data['both'] = historical_data['gdp_growth_pc'] + historical_data['wtp_pc']
    
    # Create plot data
    plot_data = []
    for _, row in historical_data.iterrows():
        plot_data.extend([
            {'country': row['country'], 'period': row['period'], 'value': row['gdp_growth_pc'], 'metric': 'GDP growth per capita'},
            {'country': row['country'], 'period': row['period'], 'value': row['both'], 'metric': 'GDP + longevity gains per capita'}
        ])
    
    df = pd.DataFrame(plot_data)
    
    # Create plot
    countries = ['Australia'] + sorted([c for c in df['country'].unique() if c not in ['Australia', 'USA']]) + ['USA']
    periods = sorted(df['period'].unique())
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    
    # Custom colormap
    cmap = LinearSegmentedColormap.from_list('custom', ['red', 'white', 'lime'], N=100)
    
    for idx, metric in enumerate(['GDP growth per capita', 'GDP + longevity gains per capita']):
        data = df[df['metric'] == metric]
        
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
    plt.savefig(FIGURES_DIR / "historical.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Historical plot saved to: {FIGURES_DIR / 'historical.pdf'}")

def create_oneyear_plot():
    """Create country summary table, saves to oneyear.pdf in figures directory."""
    print("Creating one year summary table...")
    
    # Load and process data
    df = pd.read_csv(INPUT_DIR / "Table2.csv")
    df = df.drop(columns=['Life Expectancy', 'Healthy Life Expectancy'])
    df = df.rename(columns={'country': 'Country'}).sort_values('Country').reset_index(drop=True)
    
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
    plt.title('Country Summary: Population and Economic Value of Additional Life Year', 
              fontsize=14, pad=20)
    
    # Save
    plt.savefig(FIGURES_DIR / "oneyear.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"One year plot saved to: {FIGURES_DIR / 'oneyear.pdf'}")

def main():
    """Create both plots."""
    create_historical_plot()
    create_oneyear_plot()

if __name__ == "__main__":
    main()
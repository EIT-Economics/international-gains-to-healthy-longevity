#!/usr/bin/env python3
"""
3D Surface Plotting Module

This module implements the functionality from 3d_plots.jl,
providing 3D surface plots and heatmaps for WTP analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, Tuple, Optional

class SurfacePlotter:
    """Plotter for 3D surfaces and heatmaps"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the surface plotter
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_3d_surface(self, data: pd.DataFrame, x_col: str, y_col: str, z_col: str,
                       title: str = "3D Surface Plot", 
                       xlabel: str = "X", ylabel: str = "Y", zlabel: str = "Z",
                       colormap: str = 'viridis', 
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create 3D surface plot
        
        Args:
            data: DataFrame with surface data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            z_col: Column name for z-axis
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            zlabel: Z-axis label
            colormap: Colormap for the surface
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        print("Data columns: ", data.columns)
        assert x_col in data.columns, f"Column {x_col} not found in data"
        assert y_col in data.columns, f"Column {y_col} not found in data"
        assert z_col in data.columns, f"Column {z_col} not found in data"
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        x_unique = sorted(data[x_col].unique())
        y_unique = sorted(data[y_col].unique())
        X, Y = np.meshgrid(x_unique, y_unique)
        
        # Create Z matrix
        Z = np.full_like(X, np.nan)
        for i, x_val in enumerate(x_unique):
            for j, y_val in enumerate(y_unique):
                mask = (data[x_col] == x_val) & (data[y_col] == y_val)
                if mask.any():
                    Z[j, i] = data.loc[mask, z_col].iloc[0]
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap=colormap, alpha=0.8, linewidth=0, antialiased=True)
        
        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D surface plot saved to {save_path}")
        
        return fig
    
    def plot_heatmap(self, data: pd.DataFrame, x_col: str, y_col: str, z_col: str,
                    title: str = "Heatmap", 
                    xlabel: str = "X", ylabel: str = "Y",
                    colormap: str = 'coolwarm', 
                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create heatmap plot
        
        Args:
            data: DataFrame with heatmap data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            z_col: Column name for z-axis
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            colormap: Colormap for the heatmap
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create pivot table
        pivot_data = data.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc='mean')
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap=colormap, 
                   cbar_kws={'label': z_col}, ax=ax)
        
        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        
        return fig
    
    def plot_contour(self, data: pd.DataFrame, x_col: str, y_col: str, z_col: str,
                    title: str = "Contour Plot", 
                    xlabel: str = "X", ylabel: str = "Y",
                    levels: int = 20, colormap: str = 'viridis',
                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create contour plot
        
        Args:
            data: DataFrame with contour data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            z_col: Column name for z-axis
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            levels: Number of contour levels
            colormap: Colormap for the contours
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create pivot table
        pivot_data = data.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc='mean')
        
        # Create contour plot
        contour = ax.contourf(pivot_data.columns, pivot_data.index, pivot_data.values, 
                            levels=levels, cmap=colormap)
        
        # Add contour lines
        ax.contour(pivot_data.columns, pivot_data.index, pivot_data.values, 
                  levels=levels, colors='black', alpha=0.5, linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label(z_col)
        
        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Contour plot saved to {save_path}")
        
        return fig
    
    def plot_struldbrugg_elongation(self, data: pd.DataFrame, 
                                   save_dir: str = "figures") -> Dict[str, plt.Figure]:
        """
        Plot Struldbrugg elongation scenario (equivalent to Julia 3d_plots.jl)
        
        Args:
            data: DataFrame with Struldbrugg data
            save_dir: Directory to save plots
            
        Returns:
            Dictionary with figure objects
        """
        figures = {}
        
        # 3D surface plot
        fig1 = self.plot_3d_surface(
            data, 'LE', 'HLE', 'WTP_1y',
            title="WTP for M1 (Struldbrugg Elongation)",
            xlabel="Life Expectancy", ylabel="Healthy Life Expectancy", zlabel="WTP",
            colormap='viridis',
            save_path=f"{save_dir}/WTP_M1_surface.pdf"
        )
        figures['surface'] = fig1
        
        # Heatmap
        fig2 = self.plot_heatmap(
            data, 'LE', 'HLE', 'WTP_1y',
            title="WTP for M1 (Heatmap)",
            xlabel="Life Expectancy", ylabel="Healthy Life Expectancy",
            colormap='coolwarm',
            save_path=f"{save_dir}/WTP_M1_heatmap.pdf"
        )
        figures['heatmap'] = fig2
        
        # Contour plot
        fig3 = self.plot_contour(
            data, 'LE', 'HLE', 'WTP_1y',
            title="WTP for M1 (Contour)",
            xlabel="Life Expectancy", ylabel="Healthy Life Expectancy",
            colormap='hot',
            save_path=f"{save_dir}/WTP_M1_contour.pdf"
        )
        figures['contour'] = fig3
        
        return figures
    
    def plot_dorian_gray_elongation(self, data: pd.DataFrame, 
                                   save_dir: str = "figures") -> Dict[str, plt.Figure]:
        """
        Plot Dorian Gray elongation scenario
        
        Args:
            data: DataFrame with Dorian Gray data
            save_dir: Directory to save plots
            
        Returns:
            Dictionary with figure objects
        """
        figures = {}
        
        # 3D surface plot
        fig1 = self.plot_3d_surface(
            data, 'HLE', 'LE', 'WTP_1y',
            title="WTP for D1 (Dorian Gray Elongation)",
            xlabel="Healthy Life Expectancy", ylabel="Life Expectancy", zlabel="WTP",
            colormap='viridis',
            save_path=f"{save_dir}/WTP_D1_surface.pdf"
        )
        figures['surface'] = fig1
        
        # Heatmap
        fig2 = self.plot_heatmap(
            data, 'LE', 'HLE', 'WTP_1y',
            title="WTP for D1 (Heatmap)",
            xlabel="Life Expectancy", ylabel="Healthy Life Expectancy",
            colormap='coolwarm',
            save_path=f"{save_dir}/WTP_D1_heatmap.pdf"
        )
        figures['heatmap'] = fig2
        
        # Contour plot
        fig3 = self.plot_contour(
            data, 'LE', 'HLE', 'WTP_1y',
            title="WTP for D1 (Contour)",
            xlabel="Life Expectancy", ylabel="Healthy Life Expectancy",
            colormap='hot',
            save_path=f"{save_dir}/WTP_D1_contour.pdf"
        )
        figures['contour'] = fig3
        
        return figures
    
    def plot_rectangularization(self, data: pd.DataFrame, param_name: str,
                               save_dir: str = "figures") -> Dict[str, plt.Figure]:
        """
        Plot rectangularization scenario
        
        Args:
            data: DataFrame with rectangularization data
            param_name: Parameter name (gamma or psi)
            save_dir: Directory to save plots
            
        Returns:
            Dictionary with figure objects
        """
        figures = {}
        
        # 3D surface plot
        fig1 = self.plot_3d_surface(
            data, 'LE', 'HLE', 'WTP_1y',
            title=f"WTP for {param_name} (Rectangularization)",
            xlabel="Life Expectancy", ylabel="Healthy Life Expectancy", zlabel="WTP",
            colormap='viridis',
            save_path=f"{save_dir}/WTP_{param_name}_surface.pdf"
        )
        figures['surface'] = fig1
        
        # Heatmap
        fig2 = self.plot_heatmap(
            data, 'LE', 'HLE', 'WTP_1y',
            title=f"WTP for {param_name} (Heatmap)",
            xlabel="Life Expectancy", ylabel="Healthy Life Expectancy",
            colormap='coolwarm',
            save_path=f"{save_dir}/WTP_{param_name}_heatmap.pdf"
        )
        figures['heatmap'] = fig2
        
        return figures
    
    def create_comparison_plot(self, data_dict: Dict[str, pd.DataFrame],
                            x_col: str, y_col: str, 
                            title: str = "Comparison Plot",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comparison plot for multiple datasets
        
        Args:
            data_dict: Dictionary with dataset names and DataFrames
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(data_dict)))
        
        for i, (name, data) in enumerate(data_dict.items()):
            if not data.empty and x_col in data.columns and y_col in data.columns:
                ax.plot(data[x_col], data[y_col], 'o-', label=name, color=colors[i], alpha=0.7)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        return fig
    
    def close_all_figures(self):
        """Close all matplotlib figures"""
        plt.close('all')

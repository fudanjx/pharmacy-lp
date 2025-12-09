"""
Visualizer module for pharmacy roster scheduling.
This module provides visualization capabilities for the roster schedule.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os


class RosterVisualizer:
    """Class for visualizing the roster schedule."""

    def __init__(self, 
                 staff_names: List[str],
                 schedule_by_staff: Dict[str, List[Tuple]],
                 weekend_count: int = 28,
                 shift_distribution_stats: Dict = None):
        """
        Initialize the RosterVisualizer.
        
        Args:
            staff_names (List[str]): List of all staff names
            schedule_by_staff (Dict): Schedule organized by staff member
            weekend_count (int): Total number of weekends in the period
            shift_distribution_stats (Dict): Statistics about shift balance (optional)
        """
        self.staff_names = staff_names
        self.schedule_by_staff = schedule_by_staff
        self.weekend_count = weekend_count
        self.shift_distribution_stats = shift_distribution_stats
        self.days = ["Saturday", "Sunday"]
        self.shifts = ["early", "mid", "late"]
        
    def create_shift_distribution_chart(self, 
                                        output_path: str = "shift_distribution.png") -> None:
        """
        Create a bar chart showing the distribution of shifts among staff members.
        
        Args:
            output_path (str): Path to save the chart image
        """
        # Count shifts by staff
        shift_counts = {}
        for staff in self.staff_names:
            if staff in self.schedule_by_staff:
                shift_counts[staff] = len(self.schedule_by_staff[staff])
            else:
                shift_counts[staff] = 0
        
        # Create DataFrame for plotting
        df = pd.DataFrame(list(shift_counts.items()), columns=['Staff', 'Shifts'])
        df = df.sort_values('Shifts', ascending=False)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(df['Staff'], df['Shifts'])
        
        # Color bars based on shift count
        for i, bar in enumerate(bars):
            if df['Shifts'].iloc[i] >= 5:
                bar.set_color('steelblue')
            else:
                bar.set_color('lightblue')
        
        plt.axhline(y=4, color='r', linestyle='--', alpha=0.7, label='4 shifts')
        plt.axhline(y=5, color='g', linestyle='--', alpha=0.7, label='5 shifts')
        
        # Add labels and title
        plt.xlabel('Staff')
        plt.ylabel('Number of Shifts')
        plt.title('Distribution of Shifts Among Staff')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save the chart
        plt.savefig(output_path)
        plt.close()
    
    def create_weekend_heatmap(self, output_path: str = "weekend_heatmap.png") -> None:
        """
        Create a heatmap showing which weekends each staff member works.
        
        Args:
            output_path (str): Path to save the chart image
        """
        # Create a matrix to represent weekend assignments
        # 1 if staff works that weekend, 0 otherwise
        data = np.zeros((len(self.staff_names), self.weekend_count))
        
        # Fill the matrix
        for i, staff in enumerate(self.staff_names):
            if staff in self.schedule_by_staff:
                weekends_worked = set()
                for (weekend, _), _ in self.schedule_by_staff[staff]:
                    weekends_worked.add(weekend)
                
                for weekend in weekends_worked:
                    data[i, weekend] = 1
        
        # Create plot
        plt.figure(figsize=(15, 8))
        plt.imshow(data, cmap='Blues', aspect='auto')
        
        # Add labels and title
        plt.xlabel('Weekend Number')
        plt.ylabel('Staff')
        plt.title('Weekend Work Schedule')
        plt.colorbar(ticks=[0, 1], label='Working')
        
        # Add x and y tick labels
        plt.yticks(range(len(self.staff_names)), self.staff_names)
        plt.xticks(range(self.weekend_count), [f'{i+1}' for i in range(self.weekend_count)])
        
        # Add grid
        plt.grid(False)
        
        # Save the chart
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def create_shift_type_distribution(self, output_path: str = "shift_types.png") -> None:
        """
        Create a stacked bar chart showing the distribution of shift types for each staff
        with balance indicators.
        
        Args:
            output_path (str): Path to save the chart image
        """
        # Count shift types by staff
        early_counts = {staff: 0 for staff in self.staff_names}
        mid_counts = {staff: 0 for staff in self.staff_names}
        late_counts = {staff: 0 for staff in self.staff_names}
        
        for staff in self.staff_names:
            if staff in self.schedule_by_staff:
                for _, shift in self.schedule_by_staff[staff]:
                    if shift == "early":
                        early_counts[staff] += 1
                    elif shift == "mid":
                        mid_counts[staff] += 1
                    elif shift == "late":
                        late_counts[staff] += 1
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Staff': self.staff_names,
            'Early': [early_counts[staff] for staff in self.staff_names],
            'Mid': [mid_counts[staff] for staff in self.staff_names],
            'Late': [late_counts[staff] for staff in self.staff_names]
        })
        
        # Add balance information if available
        balance_info = {}
        if self.shift_distribution_stats:
            staff_distribution = self.shift_distribution_stats.get("staff_distribution", {})
            for staff in self.staff_names:
                if staff in staff_distribution:
                    balance_info[staff] = staff_distribution[staff]["is_balanced"]
                else:
                    balance_info[staff] = True  # Assume balanced if no stats
        
        # Sort by total shifts
        df['Total'] = df['Early'] + df['Mid'] + df['Late']
        df = df.sort_values('Total', ascending=False)
        
        # Create plot with enhanced figure size to accommodate balance indicators
        plt.figure(figsize=(14, 8))
        
        # Plot stacked bars with colors indicating balance status
        bar_width = 0.65
        bottom_vals = np.zeros(len(df))
        
        # Determine bar edge colors based on balance status
        edge_colors = []
        for staff in df['Staff']:
            if balance_info.get(staff, True):
                edge_colors.append('green')  # Balanced
            else:
                edge_colors.append('red')    # Imbalanced
        
        # Plot each shift type with balance-indicating borders
        p1 = plt.bar(df['Staff'], df['Early'], bar_width, label='Early', 
                    color='lightblue', edgecolor=edge_colors, linewidth=2)
        bottom_vals += df['Early']
        
        p2 = plt.bar(df['Staff'], df['Mid'], bar_width, bottom=bottom_vals, label='Mid', 
                    color='steelblue', edgecolor=edge_colors, linewidth=2)
        bottom_vals += df['Mid']
        
        p3 = plt.bar(df['Staff'], df['Late'], bar_width, bottom=bottom_vals, label='Late', 
                    color='navy', edgecolor=edge_colors, linewidth=2)
        
        # Add title with balance summary if stats available
        title = 'Distribution of Shift Types Among Staff'
        if self.shift_distribution_stats:
            balance_pct = self.shift_distribution_stats.get("balance_percentage", 0)
            title += f' (Balance: {balance_pct:.1f}% of staff)'
        
        plt.xlabel('Staff')
        plt.ylabel('Number of Shifts')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        
        # Create custom legend including balance indicators
        handles = [p1, p2, p3]
        labels = ['Early', 'Mid', 'Late']
        
        # Add balance indicator legend items
        from matplotlib.patches import Patch
        handles.extend([
            Patch(color='white', edgecolor='green', linewidth=2),
            Patch(color='white', edgecolor='red', linewidth=2)
        ])
        labels.extend(['Balanced', 'Imbalanced'])
        
        plt.legend(handles, labels, loc='upper right')
        
        # Add shift count labels on bars and balance indicators
        for i, staff in enumerate(df['Staff']):
            early = df['Early'].iloc[i]
            mid = df['Mid'].iloc[i]
            late = df['Late'].iloc[i]
            
            # Only add labels for non-zero values
            y_early = early / 2
            if early > 0:
                plt.text(i, y_early, str(early), ha='center', va='center', color='black', fontweight='bold')
            
            y_mid = early + mid / 2
            if mid > 0:
                plt.text(i, y_mid, str(mid), ha='center', va='center', color='white', fontweight='bold')
            
            y_late = early + mid + late / 2
            if late > 0:
                plt.text(i, y_late, str(late), ha='center', va='center', color='white', fontweight='bold')
            
            # Add balance indicator symbol at the top of each bar
            total_height = early + mid + late
            if balance_info.get(staff, True):
                plt.text(i, total_height + 0.1, '✓', ha='center', va='bottom', 
                        color='green', fontsize=14, fontweight='bold')
            else:
                plt.text(i, total_height + 0.1, '⚠', ha='center', va='bottom', 
                        color='red', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def create_shift_balance_report(self, output_path: str = "shift_balance_report.png") -> None:
        """
        Create a detailed shift balance report visualization.
        
        Args:
            output_path (str): Path to save the chart image
        """
        if not self.shift_distribution_stats:
            print("No shift distribution statistics available for balance report")
            return
            
        staff_distribution = self.shift_distribution_stats.get("staff_distribution", {})
        if not staff_distribution:
            print("No staff distribution data available")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Subplot 1: Balance Status Overview (Pie Chart)
        balanced_count = self.shift_distribution_stats.get("balanced_staff_count", 0)
        imbalanced_count = len(staff_distribution) - balanced_count
        
        if balanced_count > 0 or imbalanced_count > 0:
            sizes = [balanced_count, imbalanced_count]
            labels = [f'Balanced ({balanced_count})', f'Imbalanced ({imbalanced_count})']
            colors = ['lightgreen', 'lightcoral']
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Shift Balance Status Overview')
        
        # Subplot 2: Shift Spread Distribution (Histogram)
        spreads = [info["shift_spread"] for info in staff_distribution.values()]
        ax2.hist(spreads, bins=range(0, max(spreads) + 2), alpha=0.7, color='steelblue', edgecolor='black')
        ax2.set_xlabel('Shift Spread (Max - Min shifts per type)')
        ax2.set_ylabel('Number of Staff')
        ax2.set_title('Distribution of Shift Type Spreads')
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Individual Staff Balance Scores
        staff_names = list(staff_distribution.keys())[:15]  # Limit to first 15 for readability
        balance_scores = [staff_distribution[staff]["shift_spread"] for staff in staff_names]
        colors_bar = ['red' if not staff_distribution[staff]["is_balanced"] else 'green' for staff in staff_names]
        
        ax3.bar(range(len(staff_names)), balance_scores, color=colors_bar, alpha=0.7)
        ax3.set_xlabel('Staff (First 15)')
        ax3.set_ylabel('Shift Spread')
        ax3.set_title('Individual Staff Balance Scores')
        ax3.set_xticks(range(len(staff_names)))
        ax3.set_xticklabels([name[:8] + '...' if len(name) > 8 else name for name in staff_names], 
                           rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Shift Type Distribution Heatmap
        shift_matrix = []
        staff_labels = []
        for staff, info in list(staff_distribution.items())[:15]:  # Limit for readability
            counts = info["shift_counts"]
            shift_matrix.append([counts["early"], counts["mid"], counts["late"]])
            staff_labels.append(staff[:10] + '...' if len(staff) > 10 else staff)
        
        if shift_matrix:
            im = ax4.imshow(shift_matrix, cmap='Blues', aspect='auto')
            ax4.set_xticks([0, 1, 2])
            ax4.set_xticklabels(['Early', 'Mid', 'Late'])
            ax4.set_yticks(range(len(staff_labels)))
            ax4.set_yticklabels(staff_labels)
            ax4.set_title('Shift Counts by Staff (First 15)')
            
            # Add text annotations
            for i in range(len(staff_labels)):
                for j in range(3):
                    text = ax4.text(j, i, shift_matrix[i][j], ha="center", va="center", color="black")
            
            # Add colorbar
            plt.colorbar(im, ax=ax4, shrink=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Shift balance report saved to {output_path}")
    
    def generate_all_visualizations(self, output_dir: str = "./visualizations") -> None:
        """
        Generate all visualizations and save to the specified directory.
        
        Args:
            output_dir (str): Directory to save the visualization files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        self.create_shift_distribution_chart(
            os.path.join(output_dir, "shift_distribution.png")
        )
        self.create_weekend_heatmap(
            os.path.join(output_dir, "weekend_heatmap.png")
        )
        self.create_shift_type_distribution(
            os.path.join(output_dir, "shift_types.png")
        )
        
        # Generate shift balance report if statistics are available
        if self.shift_distribution_stats:
            self.create_shift_balance_report(
                os.path.join(output_dir, "shift_balance_report.png")
            )


if __name__ == "__main__":
    # This is just for testing
    from data_loader import DataLoader
    from model import RosterModel
    
    # Load data
    loader = DataLoader("Availability.xls")
    loader.load_data()
    
    # Run model
    model = RosterModel(
        loader.get_staff_names(),
        loader.get_weekend_days(),
        loader.get_availability(),
        loader.get_acc_trained_staff(),
        loader.get_staff_shifts()
    )
    
    # Get results
    model.create_model()
    solution = model.solve()
    
    # Create visualizer
    visualizer = RosterVisualizer(
        loader.get_staff_names(),
        solution,
        len(loader.weekends) if hasattr(loader, 'weekends') else 28
    )
    
    # Generate visualizations
    visualizer.generate_all_visualizations()
    print("Visualizations generated in ./visualizations directory")
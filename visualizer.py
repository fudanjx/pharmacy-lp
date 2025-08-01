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
                 weekend_count: int = 28):
        """
        Initialize the RosterVisualizer.
        
        Args:
            staff_names (List[str]): List of all staff names
            schedule_by_staff (Dict): Schedule organized by staff member
            weekend_count (int): Total number of weekends in the period
        """
        self.staff_names = staff_names
        self.schedule_by_staff = schedule_by_staff
        self.weekend_count = weekend_count
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
        Create a stacked bar chart showing the distribution of shift types for each staff.
        
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
        
        # Sort by total shifts
        df['Total'] = df['Early'] + df['Mid'] + df['Late']
        df = df.sort_values('Total', ascending=False)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot stacked bars
        bar_width = 0.65
        bottom_vals = np.zeros(len(df))
        
        # Plot each shift type
        p1 = plt.bar(df['Staff'], df['Early'], bar_width, label='Early', color='skyblue')
        bottom_vals += df['Early']
        
        p2 = plt.bar(df['Staff'], df['Mid'], bar_width, bottom=bottom_vals, label='Mid', color='steelblue')
        bottom_vals += df['Mid']
        
        p3 = plt.bar(df['Staff'], df['Late'], bar_width, bottom=bottom_vals, label='Late', color='navy')
        
        # Add labels and title
        plt.xlabel('Staff')
        plt.ylabel('Number of Shifts')
        plt.title('Distribution of Shift Types Among Staff')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        
        # Add shift count labels on bars
        for i, staff in enumerate(df['Staff']):
            early = df['Early'].iloc[i]
            mid = df['Mid'].iloc[i]
            late = df['Late'].iloc[i]
            
            # Only add labels for non-zero values
            y_early = early / 2
            if early > 0:
                plt.text(i, y_early, str(early), ha='center', va='center', color='black')
            
            y_mid = early + mid / 2
            if mid > 0:
                plt.text(i, y_mid, str(mid), ha='center', va='center', color='black')
            
            y_late = early + mid + late / 2
            if late > 0:
                plt.text(i, y_late, str(late), ha='center', va='center', color='black')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
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
"""
Roster Generator module for pharmacy scheduling.
This module handles processing optimization results and generating formatted output.
"""

import pandas as pd
from typing import Dict, List, Tuple
import os


class RosterGenerator:
    """Class for generating and exporting roster schedules from model results."""
    
    def __init__(self, 
                 staff_names: List[str],
                 schedule_by_weekend: Dict[Tuple[int, str], Dict[str, str]],
                 schedule_by_staff: Dict[str, List[Tuple[Tuple[int, str], str]]],
                 weekend_count: int = 28):
        """
        Initialize the RosterGenerator with scheduling data.
        
        Args:
            staff_names (List[str]): List of all staff names
            schedule_by_weekend (Dict): Schedule organized by weekend/day/shift
            schedule_by_staff (Dict): Schedule organized by staff member
            weekend_count (int): Total number of weekends in the period
        """
        self.staff_names = staff_names
        self.schedule_by_weekend = schedule_by_weekend
        self.schedule_by_staff = schedule_by_staff
        self.weekend_count = weekend_count
        self.days = ["Saturday", "Sunday"]
        self.shifts = ["early", "mid", "late"]
    
    def generate_weekend_schedule_df(self) -> pd.DataFrame:
        """
        Generate a DataFrame showing weekend schedule organized by date.
        
        Returns:
            pd.DataFrame: Schedule with weekends as rows and shifts as columns
        """
        # Create empty DataFrame for the schedule
        columns = ["Weekend", "Day"] + self.shifts
        rows = []
        
        # Fill in the schedule data
        for weekend in range(self.weekend_count):
            for day in self.days:
                row = [f"Weekend {weekend+1}", day]
                
                # Add staff assigned to each shift
                for shift in self.shifts:
                    if (weekend, day) in self.schedule_by_weekend and shift in self.schedule_by_weekend[(weekend, day)]:
                        staff = self.schedule_by_weekend[(weekend, day)][shift]
                        row.append(staff)
                    else:
                        row.append("UNASSIGNED")
                
                rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=columns)
        return df
    
    def generate_staff_schedule_df(self) -> pd.DataFrame:
        """
        Generate a DataFrame showing staff assignments.
        
        Returns:
            pd.DataFrame: Schedule with staff as rows and total assignments
        """
        # Create empty DataFrame
        columns = ["Staff", "Total Shifts"] + [f"Assignment {i+1}" for i in range(5)]  # Max 5 shifts per staff
        rows = []
        
        # Fill in staff assignments
        for staff in self.staff_names:
            assignments = []
            if staff in self.schedule_by_staff:
                for (weekend, day), shift in self.schedule_by_staff[staff]:
                    assignments.append(f"W{weekend+1} {day[:3]} {shift}")
            
            # Add row for this staff
            row = [staff, len(assignments)]
            # Add up to 5 assignments (padding with empty strings if fewer)
            for i in range(5):
                if i < len(assignments):
                    row.append(assignments[i])
                else:
                    row.append("")
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=columns)
        return df
    
    def export_to_excel(self, output_path: str = "Roster_Schedule.xlsx") -> None:
        """
        Export the schedule to Excel format with multiple sheets.
        
        Args:
            output_path (str): Path to save the Excel file
        """
        # Create a pandas Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write each DataFrame to a different worksheet
            self.generate_weekend_schedule_df().to_excel(writer, sheet_name='Weekend Schedule', index=False)
            self.generate_staff_schedule_df().to_excel(writer, sheet_name='Staff Assignments', index=False)
            
            # Add summary statistics
            self._add_summary_sheet(writer)
    
    def _add_summary_sheet(self, writer) -> None:
        """
        Add a summary statistics sheet to the Excel workbook.
        
        Args:
            writer: ExcelWriter object
        """
        # Count shifts per staff member
        staff_counts = {staff: len(assignments) for staff, assignments in self.schedule_by_staff.items()}
        
        # Calculate weekend distribution
        weekend_counts = {}
        for staff, assignments in self.schedule_by_staff.items():
            weekends = set()
            for (weekend, _), _ in assignments:
                weekends.add(weekend)
            weekend_counts[staff] = len(weekends)
        
        # Create summary DataFrame
        summary_data = []
        for staff in self.staff_names:
            summary_data.append({
                'Staff': staff,
                'Total Shifts': staff_counts.get(staff, 0),
                'Weekends Worked': weekend_counts.get(staff, 0),
                'Avg Shifts/Weekend': round(staff_counts.get(staff, 0) / max(weekend_counts.get(staff, 1), 1), 2)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    def export_to_csv(self, output_dir: str = "./") -> None:
        """
        Export the schedule to CSV files.
        
        Args:
            output_dir (str): Directory to save the CSV files
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Export schedules to CSV
        self.generate_weekend_schedule_df().to_csv(
            os.path.join(output_dir, "weekend_schedule.csv"), 
            index=False
        )
        self.generate_staff_schedule_df().to_csv(
            os.path.join(output_dir, "staff_assignments.csv"), 
            index=False
        )
    
    @classmethod
    def from_model_results(cls, 
                           staff_names: List[str],
                           roster_results: Dict[str, List[Tuple]],
                           weekend_count: int = 28) -> 'RosterGenerator':
        """
        Create RosterGenerator from model results.
        
        Args:
            staff_names (List[str]): List of all staff names
            roster_results (Dict): Results from the optimizer model
            weekend_count (int): Total number of weekends
            
        Returns:
            RosterGenerator: Instance initialized with processed results
        """
        # Process the results into two formats
        schedule_by_weekend = {}
        schedule_by_staff = roster_results
        
        for staff, assignments in roster_results.items():
            for day_tuple, shift in assignments:
                if day_tuple not in schedule_by_weekend:
                    schedule_by_weekend[day_tuple] = {}
                schedule_by_weekend[day_tuple][shift] = staff
        
        return cls(staff_names, schedule_by_weekend, schedule_by_staff, weekend_count)


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
    schedule_by_weekend = model.get_schedule_by_weekend()
    
    # Generate roster
    generator = RosterGenerator.from_model_results(
        loader.get_staff_names(),
        solution,
        len(loader.weekends) if hasattr(loader, 'weekends') else 28
    )
    
    # Export
    generator.export_to_excel("test_roster.xlsx")
    print("Test roster exported to test_roster.xlsx")
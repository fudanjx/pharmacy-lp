"""
Data Loader module for pharmacy roster scheduling.
This module handles loading and processing availability data from Excel files.
"""

import pandas as pd
import os
from typing import Dict, List, Tuple, Set


class DataLoader:
    """Class for loading and processing pharmacy staff availability data."""

    def __init__(self, file_path: str = "Availability.xls"):
        """
        Initialize DataLoader with the path to the availability file.

        Args:
            file_path (str): Path to the Excel file containing availability data.
        """
        self.file_path = file_path
        self.staff_data = None
        self.weekends = None
        self.staff_names = None
        self.acc_trained_staff = set()
        
        # Mapping for shift assignment distribution
        self.staff_shifts = {}
        
        # Special staff with 4 shifts instead of 5
        self.staff_with_four_shifts = {
            "Peggy", "Golda", "Shu Juan", "Chee Cing", "Shi Mun", 
            "Daphne", "Jia Yi", "Diana", "Min Hui", "Caitlin", "Sylvian"
        }

    def load_data(self) -> None:
        """
        Load data from the Excel file and prepare it for the model.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Availability file not found: {self.file_path}")
        
        # Load Excel file
        self.staff_data = pd.read_excel(self.file_path)
        
        # Debug: Print Excel file structure
        print("\n--- Excel File Structure ---")
        print(f"Columns: {self.staff_data.columns.tolist()}")
        print(f"Shape: {self.staff_data.shape}")
        print("First few rows:")
        print(self.staff_data.head(3).to_string())
        print("-------------------------\n")
        
        # Extract staff names (first column) and convert to strings
        self.staff_names = [
            str(name).strip() if pd.notna(name) else f"Staff_{i}" 
            for i, name in enumerate(self.staff_data.iloc[:, 0].tolist())
        ]
        
        # Process ACC-trained staff information from the "ACC" column (column B/index 1)
        try:
            # Look for ACC column names
            acc_column_names = ["ACC", "Staff with ACC"]
            acc_column = None
            
            # Try to find an ACC column by name
            for col_name in acc_column_names:
                if col_name in self.staff_data.columns:
                    acc_column = col_name
                    break
                    
            # If not found by name, use the second column (index 1)
            if acc_column is None and len(self.staff_data.columns) > 1:
                acc_column = self.staff_data.columns[1]
                print(f"Using column '{acc_column}' for ACC information")
            else:
                print("Warning: Could not find ACC column")
                
            if acc_column:
                acc_data = self.staff_data[acc_column]
                print(f"ACC column values: {acc_data.tolist()[:5]}...")
                
                self.acc_trained_staff = {
                    self.staff_names[i] for i, trained in enumerate(acc_data) 
                    if pd.notna(trained) and (
                        trained == 1 or 
                        (isinstance(trained, str) and trained.strip() == 'Yes') or 
                        trained is True
                    )
                }
                print(f"ACC-trained staff: {self.acc_trained_staff}")
        except Exception as e:
            # If ACC column not found, leave as empty set
            print(f"Error processing ACC staff: {e}")
            import traceback
            traceback.print_exc()
            pass
        
        # Convert staff_with_four_shifts from list/set to set of normalized names
        self.staff_with_four_shifts = {self._normalize_name(name) for name in self.staff_with_four_shifts}
        
        # Apply default shift assignments
        self._apply_shift_assignments()
        
        # Verify the total number of shifts is exactly as expected
        total_shifts = sum(self.staff_shifts.values())
        expected_shifts = 8*5 + 11*4  # 8 staff with 5 shifts, 11 staff with 4 shifts
        print(f"Total shift assignments: {total_shifts} (expected {expected_shifts})")
        
        # Ensure we have exactly the right number of shifts
        if total_shifts != expected_shifts:
            print("WARNING: Shift allocation doesn't match expectations!")
            # Count how many staff have 4 vs 5 shifts
            staff_with_4 = sum(1 for shifts in self.staff_shifts.values() if shifts == 4)
            staff_with_5 = sum(1 for shifts in self.staff_shifts.values() if shifts == 5)
            print(f"Staff with 4 shifts: {staff_with_4} (expected 11)")
            print(f"Staff with 5 shifts: {staff_with_5} (expected 8)")
    
    def get_availability(self) -> Dict[str, Dict[Tuple[int, str], List[str]]]:
        """
        Get staff availability for each weekend and shift type.
        
        Returns:
            Dict: Availability data in the format:
                {staff_name: {(weekend_num, day): [available_shifts]}}
        """
        availability = {}
        
        # Skip the first two columns (names and ACC info)
        start_col = 2
        
        # Debug information
        print(f"Processing availability data from columns {start_col} onwards")
        if start_col < len(self.staff_data.columns):
            print(f"First availability column: {self.staff_data.columns[start_col]}")
            
        # Print all column names for debugging
        print(f"All columns: {self.staff_data.columns.tolist()}")
        
        # Iterate through staff members
        for idx, staff in enumerate(self.staff_names):
            staff_avail = {}
            weekend = 0
            
            # Get availability columns (skip first two columns - name and ACC info)
            avail_columns = self.staff_data.columns[start_col:]
            
            # Determine if columns represent weekends by pairs or individual days
            # If we have exactly 20 days, assume they're all individual weekend days
            if len(avail_columns) == 20:
                print("Detected 20 weekend days - treating each column as individual day")
                for day_idx in range(len(avail_columns)):
                    # Determine if it's Saturday or Sunday based on index
                    day_type = "Saturday" if day_idx % 2 == 0 else "Sunday"
                    weekend_idx = day_idx // 2
                    
                    # Get the column value
                    avail_val = self.staff_data.iloc[idx, day_idx + start_col]
                    if pd.notna(avail_val) and (avail_val == 1 or 
                                               (isinstance(avail_val, str) and avail_val.strip() == '1') or
                                               avail_val is True):
                        staff_avail[(weekend_idx, day_type)] = ["early", "mid", "late"]
            else:
                # Process columns in pairs (each pair represents a weekend: Saturday + Sunday)
                for weekend_idx in range(len(avail_columns) // 2):
                    # Calculate column indices for this weekend
                    sat_col_idx = weekend_idx * 2
                    sun_col_idx = sat_col_idx + 1
                    
                    # Skip if we're out of columns
                    if sat_col_idx >= len(avail_columns) or sun_col_idx >= len(avail_columns):
                        continue
                    
                    # Get actual column positions
                    actual_sat_col = sat_col_idx + start_col
                    actual_sun_col = sun_col_idx + start_col
                
                    # Skip if these are not availability columns
                    sat_col_name = self.staff_data.columns[actual_sat_col]
                    sun_col_name = self.staff_data.columns[actual_sun_col]
                    if ("ACC" in str(sat_col_name) or "Notes" in str(sat_col_name) or
                        "ACC" in str(sun_col_name) or "Notes" in str(sun_col_name)):
                        continue
                    
                    # Process Saturday availability
                    sat_val = self.staff_data.iloc[idx, actual_sat_col]
                    if pd.notna(sat_val) and (sat_val == 1 or (isinstance(sat_val, str) and sat_val.strip() == '1')):
                        staff_avail[(weekend_idx, "Saturday")] = ["early", "mid", "late"]
                    
                    # Process Sunday availability
                    sun_val = self.staff_data.iloc[idx, actual_sun_col]
                    if pd.notna(sun_val) and (sun_val == 1 or (isinstance(sun_val, str) and sun_val.strip() == '1')):
                        staff_avail[(weekend_idx, "Sunday")] = ["early", "mid", "late"]
            
            # Debug information for first staff member
            if staff == self.staff_names[0]:
                weekend_count = len(avail_columns) // 2
                print(f"Processed {weekend_count} weekends for availability")
                weekend_days = [(w, d) for (w, d) in staff_avail.keys()]
                print(f"Sample availability days for {staff}: {weekend_days[:6]}...")
                
                # Print detailed availability for the first staff member
                print(f"\nDetailed availability for {staff}:")
                for weekend in range(weekend_count):
                    sat_avail = (weekend, "Saturday") in staff_avail
                    sun_avail = (weekend, "Sunday") in staff_avail
                    print(f"  Weekend {weekend}: Saturday: {sat_avail}, Sunday: {sun_avail}")
                
            availability[staff] = staff_avail
            
        return availability
    
    def get_weekend_days(self) -> List[Tuple[int, str]]:
        """
        Get a list of all weekend days in the scheduling period.
        
        Returns:
            List[Tuple[int, str]]: List of (weekend_num, day) tuples
        """
        weekend_days = []
        
        # Count the actual number of weekends from the data
        if self.staff_data is not None:
            # Skip the first two columns (name and ACC)
            # Each pair of columns represents one weekend (Sat+Sun)
            avail_columns = self.staff_data.columns[2:]
            weekend_count = len(avail_columns) // 2
            
            # Ensure this matches the PRD expectation of 14 weekends (28 days)
            # If it doesn't match, print a warning but respect the actual data
            expected_weekends = 14
            if weekend_count != expected_weekends:
                print(f"WARNING: Detected {weekend_count} weekends in Excel data, but PRD specifies {expected_weekends} weekends")
            else:
                print(f"Detected {weekend_count} weekends from Excel data (matches PRD)")
                
            print(f"Weekend column headers: {[str(col) for col in avail_columns[:4]]}...")
        else:
            weekend_count = 14  # Default based on the PRD
            print(f"Using default weekend count from PRD: {weekend_count}")
        
        for weekend in range(weekend_count):
            weekend_days.append((weekend, "Saturday"))
            weekend_days.append((weekend, "Sunday"))
            
        return weekend_days
    
    def get_staff_names(self) -> List[str]:
        """
        Get the list of all staff names.
        
        Returns:
            List[str]: List of staff names
        """
        return self.staff_names
    
    def get_acc_trained_staff(self) -> Set[str]:
        """
        Get the set of ACC-trained staff members.
        
        Returns:
            Set[str]: Set of ACC-trained staff names
        """
        return self.acc_trained_staff
    
    def set_staff_with_four_shifts(self, staff_list):
        """
        Override the default list of staff who should receive 4 shifts.
        
        Args:
            staff_list (list): List of staff names who should get 4 shifts
        """
        # Convert to set of normalized names for comparison
        self.staff_with_four_shifts = {self._normalize_name(name) for name in staff_list}
        
        # Re-apply shift assignments based on the new list
        self._apply_shift_assignments()
        
        # Debug info
        print(f"Updated staff_with_four_shifts: {self.staff_with_four_shifts}")
    
    def _normalize_name(self, name):
        """
        Normalize a staff name for consistent comparison.
        
        Args:
            name: The staff name to normalize
            
        Returns:
            str: Normalized staff name
        """
        return str(name).strip() if pd.notna(name) else ""
    
    def _apply_shift_assignments(self):
        """
        Apply shift assignments based on staff_with_four_shifts.
        """
        # Reset shift assignments
        self.staff_shifts = {}
        
        # Apply shift assignments based on the current staff_with_four_shifts set
        for staff in self.staff_names:
            normalized_name = self._normalize_name(staff)
            if normalized_name in self.staff_with_four_shifts:
                self.staff_shifts[staff] = 4
            else:
                self.staff_shifts[staff] = 5
        
        # Debug output
        staff_with_4 = [staff for staff, shifts in self.staff_shifts.items() if shifts == 4]
        staff_with_5 = [staff for staff, shifts in self.staff_shifts.items() if shifts == 5]
        print(f"Staff with 4 shifts: {staff_with_4}")
        print(f"Staff with 5 shifts: {staff_with_5}")
            
    def get_staff_shifts(self) -> Dict[str, int]:
        """
        Get the number of shifts assigned to each staff member.
        
        Returns:
            Dict[str, int]: Dictionary mapping staff names to their shift count
        """
        return self.staff_shifts

    def get_shift_balance_targets(self) -> Dict[str, Dict[str, int]]:
        """
        Calculate optimal shift type distribution targets for each staff member.
        
        Returns:
            Dict[str, Dict[str, int]]: Dictionary mapping staff names to their target shift counts
                                     Format: {staff: {"min_per_shift": int, "max_per_shift": int, 
                                                      "ideal_distribution": [int, int, int]}}
        """
        shift_targets = {}
        
        for staff in self.staff_names:
            total_shifts = self.staff_shifts[staff]
            
            # Calculate base distribution
            ideal_per_shift = total_shifts // 3  # Base amount per shift type
            remainder = total_shifts % 3         # Extra shifts to distribute
            
            # Calculate min/max bounds
            min_per_shift = max(0, ideal_per_shift)
            max_per_shift = ideal_per_shift + (1 if remainder > 0 else 0)
            
            # Create ideal distribution array [early, mid, late]
            ideal_distribution = [ideal_per_shift] * 3
            # Distribute remainder shifts (prioritize early, then mid, then late)
            for i in range(remainder):
                ideal_distribution[i] += 1
            
            shift_targets[staff] = {
                "total_shifts": total_shifts,
                "min_per_shift": min_per_shift,
                "max_per_shift": max_per_shift,
                "ideal_distribution": ideal_distribution,
                "remainder": remainder
            }
        
        return shift_targets


if __name__ == "__main__":
    # Test the DataLoader
    loader = DataLoader("Availability.xls")
    loader.load_data()
    
    print(f"Staff members: {len(loader.get_staff_names())}")
    print(f"ACC trained: {loader.get_acc_trained_staff()}")
    print(f"Weekend days: {len(loader.get_weekend_days())}")
    
    availability = loader.get_availability()
    for staff, avail in list(availability.items())[:3]:  # Show first 3 staff members
        print(f"{staff}: {len(avail)} weekends available")
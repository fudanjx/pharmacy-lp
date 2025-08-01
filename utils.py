"""
Utility functions for pharmacy roster scheduling.
This module provides helper functions used across the application.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any, Optional


def validate_file_path(file_path: str) -> bool:
    """
    Check if a file exists and is readable.
    
    Args:
        file_path (str): Path to the file to validate
        
    Returns:
        bool: True if file exists and is readable, False otherwise
    """
    return os.path.exists(file_path) and os.path.isfile(file_path) and os.access(file_path, os.R_OK)


def validate_excel_file(file_path: str) -> bool:
    """
    Check if a file is a valid Excel file.
    
    Args:
        file_path (str): Path to the Excel file to validate
        
    Returns:
        bool: True if file is a valid Excel file, False otherwise
    """
    if not validate_file_path(file_path):
        return False
        
    # Check file extension
    if not file_path.endswith(('.xls', '.xlsx', '.xlsm')):
        return False
    
    # Try to read the file with pandas
    try:
        pd.read_excel(file_path, nrows=1)
        return True
    except Exception:
        return False


def create_output_directory(dir_path: str) -> str:
    """
    Create output directory if it doesn't exist.
    
    Args:
        dir_path (str): Directory path
        
    Returns:
        str: Path to the created directory
    """
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def generate_quarter_dates(year: int, quarter: int) -> List[Tuple[datetime, datetime]]:
    """
    Generate a list of weekend dates for a specific quarter.
    
    Args:
        year (int): Year
        quarter (int): Quarter (1-4)
        
    Returns:
        List[Tuple[datetime, datetime]]: List of (Saturday, Sunday) date tuples for each weekend
    """
    if quarter < 1 or quarter > 4:
        raise ValueError("Quarter must be between 1 and 4")
    
    # Determine start month based on quarter
    start_month = 1 + (quarter - 1) * 3
    
    # Create start date (first day of the quarter)
    start_date = datetime(year, start_month, 1)
    
    # Find the first Saturday after the start date
    days_until_saturday = (5 - start_date.weekday()) % 7
    if days_until_saturday == 0:
        days_until_saturday = 7
    
    first_saturday = start_date + timedelta(days=days_until_saturday)
    
    # Create end date (last day of the quarter)
    if quarter < 4:
        end_month = start_month + 3
        end_year = year
    else:
        end_month = 1
        end_year = year + 1
    
    end_date = datetime(end_year, end_month, 1) - timedelta(days=1)
    
    # Generate all weekend dates
    weekends = []
    current_saturday = first_saturday
    
    while current_saturday <= end_date:
        current_sunday = current_saturday + timedelta(days=1)
        weekends.append((current_saturday, current_sunday))
        current_saturday += timedelta(days=7)
    
    return weekends


def format_date(date: datetime, format_str: str = "%Y-%m-%d") -> str:
    """
    Format a date as string.
    
    Args:
        date (datetime): Date to format
        format_str (str): Format string
        
    Returns:
        str: Formatted date string
    """
    return date.strftime(format_str)


def calculate_staff_statistics(staff_schedules: Dict[str, List[Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate statistics for each staff member.
    
    Args:
        staff_schedules (Dict): Staff schedules
        
    Returns:
        Dict: Statistics for each staff member
    """
    stats = {}
    
    for staff, assignments in staff_schedules.items():
        # Count total shifts
        total_shifts = len(assignments)
        
        # Count shift types
        shift_counts = {"early": 0, "mid": 0, "late": 0}
        for _, shift in assignments:
            shift_counts[shift] += 1
        
        # Count unique weekends
        unique_weekends = set([day[0] for day, _ in assignments])
        
        # Store stats
        stats[staff] = {
            "total_shifts": total_shifts,
            "weekends_worked": len(unique_weekends),
            "shift_distribution": shift_counts
        }
    
    return stats


def evaluate_schedule_fairness(stats: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate the fairness of the schedule.
    
    Args:
        stats (Dict): Staff statistics
        
    Returns:
        Dict: Fairness metrics
    """
    # Extract metrics
    total_shifts = [s["total_shifts"] for s in stats.values()]
    weekends_worked = [s["weekends_worked"] for s in stats.values()]
    
    early_shifts = [s["shift_distribution"]["early"] for s in stats.values()]
    mid_shifts = [s["shift_distribution"]["mid"] for s in stats.values()]
    late_shifts = [s["shift_distribution"]["late"] for s in stats.values()]
    
    # Calculate standard deviations
    import numpy as np
    
    metrics = {
        "total_shifts_std": float(np.std(total_shifts)),
        "weekends_worked_std": float(np.std(weekends_worked)),
        "early_shifts_std": float(np.std(early_shifts)),
        "mid_shifts_std": float(np.std(mid_shifts)), 
        "late_shifts_std": float(np.std(late_shifts)),
    }
    
    # Calculate fairness score (lower is better)
    metrics["fairness_score"] = sum([
        metrics["total_shifts_std"] * 2,  # Weigh total shifts more
        metrics["weekends_worked_std"] * 1.5,  # Weigh weekend distribution
        metrics["early_shifts_std"],
        metrics["mid_shifts_std"],
        metrics["late_shifts_std"]
    ])
    
    return metrics


def print_schedule_summary(
    schedule_by_weekend: Dict[Tuple[int, str], Dict[str, str]],
    weekend_count: int,
    file: Optional[str] = None
) -> None:
    """
    Print a summary of the schedule.
    
    Args:
        schedule_by_weekend (Dict): Schedule by weekend
        weekend_count (int): Number of weekends
        file (Optional[str]): File to write output to, or None for stdout
    """
    days = ["Saturday", "Sunday"]
    shifts = ["early", "mid", "late"]
    
    output = []
    output.append("=== SCHEDULE SUMMARY ===")
    
    for weekend in range(weekend_count):
        output.append(f"\nWeekend {weekend+1}:")
        for day in days:
            output.append(f"  {day}:")
            for shift in shifts:
                staff = schedule_by_weekend.get((weekend, day), {}).get(shift, "UNASSIGNED")
                output.append(f"    {shift}: {staff}")
    
    # Write to file or print to stdout
    if file:
        with open(file, 'w') as f:
            f.write('\n'.join(output))
    else:
        print('\n'.join(output))


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test quarter dates
    weekends = generate_quarter_dates(2023, 3)
    print(f"Q3 2023 has {len(weekends)} weekends")
    print(f"First weekend: {format_date(weekends[0][0])} - {format_date(weekends[0][1])}")
    print(f"Last weekend: {format_date(weekends[-1][0])} - {format_date(weekends[-1][1])}")
    
    # Test directory creation
    test_dir = "test_output"
    create_output_directory(test_dir)
    print(f"Created directory: {test_dir}")
    
    # Clean up test directory if empty
    try:
        os.rmdir(test_dir)
    except:
        pass
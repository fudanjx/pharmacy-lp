"""
Main entry point for the pharmacy roster scheduling application.
"""

import os
import argparse
import time
from datetime import datetime

from data_loader import DataLoader
from model import RosterModel
from roster_generator import RosterGenerator
from visualizer import RosterVisualizer
from utils import validate_excel_file, create_output_directory, print_schedule_summary


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pharmacy Roster Scheduling using Linear Programming"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="Availability.xls",
        help="Path to the Excel file with staff availability data"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="Roster_Schedule.xlsx",
        help="Path to save the output Excel file with the roster"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["excel", "csv", "both"],
        default="excel",
        help="Output format for the roster schedule"
    )
    
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Generate visualizations of the roster schedule"
    )
    
    parser.add_argument(
        "--visualize-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualization files"
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a summary of the schedule to the console"
    )
    
    parser.add_argument(
        "--relax-constraints",
        action="store_true",
        help="Relax hard constraints if no solution can be found (last resort)"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the scheduling process."""
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"Starting pharmacy roster scheduling at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Validate input file
    if not validate_excel_file(args.input):
        print(f"Error: Invalid Excel file: {args.input}")
        return 1
    
    # Create output directories if needed
    if args.format in ["csv", "both"]:
        csv_dir = os.path.splitext(args.output)[0] + "_csv"
        create_output_directory(csv_dir)
    
    if args.visualize:
        create_output_directory(args.visualize_dir)
    
    # Start timing
    start_time = time.time()
    
    # Load data
    print(f"Loading availability data from {args.input}...")
    loader = DataLoader(args.input)
    loader.load_data()
    
    print(f"Loaded {len(loader.get_staff_names())} staff members")
    print(f"Found {len(loader.get_acc_trained_staff())} ACC-trained staff members")
    
    # Create and solve model
    print("Creating and solving the linear programming model...")
    model = RosterModel(
        loader.get_staff_names(),
        loader.get_weekend_days(),
        loader.get_availability(),
        loader.get_acc_trained_staff(),
        loader.get_staff_shifts(),
        relax_constraints=args.relax_constraints
    )
    
    model.create_model()
    solution = model.solve()
    
    if not solution:
        print("Failed to find a solution. Check your constraints and availability data.")
        return 1
    
    schedule_by_weekend = model.get_schedule_by_weekend()
    
    # Get the actual number of weekends from the data loader
    weekend_count = len(loader.get_weekend_days()) // 2
    print(f"Using {weekend_count} weekends for roster generation")
    
    # Generate roster
    print(f"Generating roster schedule...")
    generator = RosterGenerator.from_model_results(
        loader.get_staff_names(),
        solution,
        weekend_count  # Actual number of weekends in the data
    )
    
    # Export roster
    if args.format in ["excel", "both"]:
        print(f"Exporting roster to Excel: {args.output}")
        generator.export_to_excel(args.output)
    
    if args.format in ["csv", "both"]:
        print(f"Exporting roster to CSV files in: {csv_dir}")
        generator.export_to_csv(csv_dir)
    
    # Generate visualizations if requested
    if args.visualize:
        print(f"Generating visualizations in: {args.visualize_dir}")
        visualizer = RosterVisualizer(
            loader.get_staff_names(),
            solution,
            weekend_count  # Using the actual number of weekends
        )
        visualizer.generate_all_visualizations(args.visualize_dir)
    
    # Print schedule summary if requested
    if args.summary:
        print("\n")
        print_schedule_summary(schedule_by_weekend, weekend_count)
    
    # Print timing information
    elapsed_time = time.time() - start_time
    print(f"\nProcess completed in {elapsed_time:.2f} seconds")
    
    # Print completion message
    print(f"\nRoster scheduling completed successfully!")
    print(f"Results saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
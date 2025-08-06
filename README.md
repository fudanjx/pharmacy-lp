# Pharmacy Roster Scheduler

A Python application that uses linear programming to optimize weekend shift assignments for pharmacists based on availability and constraints. Includes both a command-line interface and a user-friendly web interface built with Streamlit.

App demo is availale at https://ah-pharmacy-roster.streamlit.app/

## Overview

This application creates optimal quarterly staff schedules for pharmacy weekend shifts, using the PuLP linear programming library. It balances multiple constraints including staff availability, workload distribution, and specialized skills requirements.

## Key Features

- **Automated Scheduling:** Generates optimized rosters that satisfy multiple constraints simultaneously
- **Equal Distribution:** Balances workload by ensuring fair shift distribution among staff
- **Constraint Handling:** Implements both hard constraints (must be satisfied) and soft constraints (optimized but can be relaxed)
- **Customizable Outputs:** Generates schedules in multiple formats (Excel, CSV) with visualization options
- **Staff Availability:** Respects staff availability preferences submitted for the planning period

## Requirements

The application requires Python 3.6+ and the following dependencies:
- PuLP (linear programming solver)
- Pandas (data manipulation)
- Openpyxl (Excel output generation)
- XLRD (Excel input processing)
- Matplotlib (visualizations)
- Streamlit (web interface)

Development requirements include:
- pytest
- flake8
- black

## Installation

1. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

For detailed instructions on setting up and using the application, please see the [Getting Started Guide](docs/getting_started.md).

## Part 1: Command Line Interface (using the first commit)

For batch processing or automation, you can use the command line interface:

```bash
python main.py
```

### Command Line Options

```
usage: main.py [-h] [--input INPUT] [--output OUTPUT] [--format {excel,csv,both}] [--visualize] [--visualize-dir VISUALIZE_DIR] [--summary]

Pharmacy Roster Scheduling using Linear Programming

options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Path to the Excel file with staff availability data
  --output OUTPUT, -o OUTPUT
                        Path to save the output Excel file with the roster
  --format {excel,csv,both}, -f {excel,csv,both}
                        Output format for the roster schedule
  --visualize, -v       Generate visualizations of the roster schedule
  --visualize-dir VISUALIZE_DIR
                        Directory to save visualization files
  --summary             Print a summary of the schedule to the console
```

### CLI Examples

Generate a roster with specific input and output files:
```bash
python main.py --input Availability.xls --output MyRoster.xlsx
```

Generate both Excel and CSV outputs with visualizations:
```bash
python main.py --format both --visualize
```

Print a summary of the generated schedule:
```bash
python main.py --summary
```

## Part 2: Streamlit Web Interface

The application features a user-friendly web interface built with Streamlit, providing an intuitive way to use the roster scheduler without command-line knowledge.

### Live Demo

**A live demo is available at: [https://ah-pharmacy-roster.streamlit.app/](https://ah-pharmacy-roster.streamlit.app/)**

### Running the Web Interface Locally

1. Make sure you've installed the required dependencies:
```bash
pip install -r requirements.txt
```

2. Launch the Streamlit app:
```bash
streamlit run app.py
```

3. Your default web browser should automatically open to the application (usually at http://localhost:8501)

### Using the Web Interface (using the latest commit)

The web interface guides you through the roster generation process with these steps:

1. **Introduction**: Overview of the application and its features
2. **Upload Data**: Upload the staff availability Excel file or download a template
3. **Configure Settings**: Set which staff should be assigned 4 shifts vs 5 shifts
4. **Generate Roster**: Process the data to create an optimized schedule
5. **View Results**: View, visualize, and download the generated roster

The interface provides:
- Interactive data tables with filtering options
- Visualizations of the roster distribution
- Download options for Excel and CSV formats
- Ability to highlight specific staff assignments

## Constraints and Requirements

The schedule optimizes for the following constraints:

### Hard Constraints (Must be satisfied)
- Each shift must be assigned to exactly one staff member
- Staff can only be assigned to shifts when they are available
- Each staff member must receive their target number of shifts (4 or 5)
- No staff member can work more than one shift per day

### Soft Constraints (Optimized but can be relaxed)
- Staff should not work on consecutive weekends
- Staff should not work on both Saturday and Sunday of the same weekend
- At least one ACC-trained staff member should be scheduled for Saturdays

### Tiered Constraint Relaxation

The solver implements an intelligent tiered relaxation approach when no feasible solution can be found with all constraints enforced:

1. **Tier 1**: First attempts to solve with all constraints enforced
2. **Tier 2**: If no solution found, relaxes the ACC-trained staff requirement (if allowed)
3. **Tier 3**: If still no solution, also relaxes consecutive weekend constraints and/or same weekend constraints (if allowed)

In the web interface, users can control which constraints can be relaxed through the Advanced Options:
- **Allow Consecutive Weekends if Needed**: Allows staff to work consecutive weekends when necessary
- **Allow Both Days on Same Weekend if Needed**: Allows staff to work both days of the same weekend when necessary
- **ACC staff are not mandatory for every Saturday**: Allows Saturdays without ACC-trained staff when necessary

The results page will indicate which constraints were relaxed to find a solution.

### Best Effort ACC Coverage

The system uses a "best effort" approach to maintain ACC-trained staff coverage on Saturdays even when full coverage is not possible:

- When ACC constraints need to be relaxed, the system doesn't simply abandon the requirement
- Instead, it converts them to soft constraints with penalties to maximize coverage where possible
- The solver prioritizes assigning ACC-trained staff to as many Saturdays as possible
- Results show clear statistics about ACC coverage, including which Saturdays (if any) don't have coverage

This approach ensures optimal distribution of specialized staff even when constraints must be relaxed.

## Input Data Format

The application expects an Excel file (`Availability.xls`) with:
- Staff names in the first column
- ACC-trained designation in the second column
- Availability for each weekend day (1 = available, 0 = unavailable) in subsequent columns

## Output

### Excel Output
The generated Excel file includes multiple sheets:
- Weekend Schedule - Shows who is assigned to each shift on each weekend day
- Staff Assignments - Lists all shifts assigned to each staff member
- Summary - Provides statistics on shift distribution

### Visualizations
When the `--visualize` flag is used, the program generates:
- Shift distribution chart - Bar chart showing shifts per staff member
- Weekend heatmap - Visual representation of which weekends each staff works
- Shift type distribution - Stacked bar chart showing early/mid/late shift distribution

The web interface includes additional visualizations:
- Staff availability heatmap - Visual representation of submitted staff availability
- Interactive filtering options for all visualization types

## Project Structure

- `main.py` - Entry point and CLI interface
- `app.py` - Streamlit web interface
- `data_loader.py` - Handles reading availability data from Excel
- `model.py` - Defines the linear programming model and constraints
- `roster_generator.py` - Processes model results and generates output files
- `visualizer.py` - Creates visual representations of the roster
- `utils.py` - Helper functions for data processing

## License & Usage
This project is released under a Custom License for Non-Commercial Use. See the [LICENSE](LICENSE) file for complete details.

✅ **Educational & Research Use**: You are welcome to use, copy, modify, and distribute this code for educational and research purposes.

⚠️ **Commercial Use Prohibited**: Commercial usage (including but not limited to selling, integrating into paid products or services, or using it for profit-driven purposes) is strictly prohibited without prior written permission from the copyright holder.

🔒 **Attribution Required**: All usage must include appropriate attribution to the original authors by retaining the License, copyright notices, and disclaimers.

📧 **Commercial Licensing**: For commercial use permissions, please contact the copyright holder by opening an issue in this repository.

By using this software, you agree to comply with these terms. Unauthorized commercial exploitation may result in legal action.

## Acknowledgments

This application uses the PuLP library for linear programming optimization and follows best practices for roster scheduling in healthcare environments.

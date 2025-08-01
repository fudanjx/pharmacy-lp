# Getting Started with Pharmacy Roster Scheduler

This guide will walk you through setting up and using the Pharmacy Roster Scheduler application with the Streamlit web interface.

## Installation

### Step 1: Set Up a Python Environment

It's recommended to use a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate
```

### Step 2: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

### Step 3: Launch the Application

Start the Streamlit web interface:

```bash
streamlit run app.py
```

Your web browser should automatically open to the application (typically at http://localhost:8501).

## Using the Application

### 1. Prepare Your Data

The application expects an Excel file with staff availability information. You'll need:

- **Staff names** in the first column
- **ACC training designation** in the second column (Yes/No)
- **Weekend day availability** in subsequent columns (1 = available, 0 = not available)

You can download a template from the application's "Upload Data" page.

### 2. Navigate the Interface

The application has five main sections:

1. **Introduction**: Overview of the application
2. **Upload Data**: Where you'll upload your staff availability Excel file
3. **Configure Settings**: Set staff shift allocations (4 or 5 shifts per staff)
4. **Generate Roster**: Process the data and create the optimized schedule
5. **View Results**: View, analyze, and download the generated roster

### 3. Upload Your Data

In the "Upload Data" section:

1. Click the "Download Template" button to get a sample Excel file
2. Fill in your staff availability data
3. Upload your completed Excel file
4. Verify that the data preview looks correct

### 4. Configure Staff Shift Allocation

In the "Configure Settings" section:

1. Choose which staff members should be assigned 4 shifts (the rest will get 5 shifts)
2. You can use the default allocation or make manual selections
3. Review the feasibility check to ensure your allocation works with the available slots

### 5. Generate the Roster

In the "Generate Roster" section:

1. Review the settings summary
2. Set any advanced options if needed
3. Click "Generate Roster Schedule" to start the optimization process
4. Wait for the process to complete (this may take a few moments)

### 6. View and Download Results

In the "View Results" section:

1. Explore the generated roster across different tabs:
   - **Summary**: Overall statistics about the roster
   - **Weekend Schedule**: Shows all weekend shifts and assignments
   - **Staff Assignments**: Individual staff schedules
   - **Visualizations**: Graphical representations of the schedule
2. Download the roster as an Excel file or export to CSV
3. Use filters to focus on specific staff members or weekends

### 7. Start Over or Make Adjustments

If you need to make changes:

- Use the "Generate New Roster" button to try different constraints with the same data
- Use the "Start Over" button to begin a completely new schedule
- Use the sidebar navigation to move between different sections of the app

## Troubleshooting

### Common Issues

1. **"No feasible solution found"**: This means the constraints cannot all be satisfied together. Try:
   - Checking if staff availability is sufficient
   - Enabling advanced options like "Allow Consecutive Weekends if Needed"
   - Verifying the ACC-trained staff requirements

2. **Excel file format issues**: Ensure your file follows the expected format:
   - First column: Staff names
   - Second column: ACC training (Yes/No)
   - Subsequent columns: Availability for each weekend day (1/0)

3. **Performance issues**: For large staff rosters, the optimization may take longer. Be patient while the solver works.

## Getting Help

If you encounter issues or have questions, please refer to:

- The project README.md file for general information
- The project documentation in the docs/ directory
- Submit issues or questions through your organization's standard support channels
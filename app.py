"""
Streamlit Web Interface for the Pharmacy Roster Scheduler.

This application provides a user-friendly interface for generating optimal
pharmacy staff schedules using linear programming techniques.
"""

import os
import io
import base64
import tempfile
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
from annotated_text import annotated_text

# Import the core functionality from the existing application
from data_loader import DataLoader
from model import RosterModel
from roster_generator import RosterGenerator
from visualizer import RosterVisualizer
from utils import validate_excel_file

# Set page configuration
st.set_page_config(
    page_title="Pharmacy Roster Scheduler",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define custom CSS for styling
def local_css():
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stAlert {
        background-color: rgba(255, 255, 255, 0.95);
        border: 1px solid #ddd;
    }
    .info-box {
        background-color: #EFF6FF;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1E40AF;
    }
    .success-box {
        background-color: #ECFDF5;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #047857;
    }
    .warning-box {
        background-color: #FFFBEB;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #D97706;
    }
    .footnote {
        color: #6B7280;
        font-size: 0.8rem;
    }
    .section {
        margin-top: 2rem;
        margin-bottom: 2rem;
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    </style>
    """, unsafe_allow_html=True)

# Apply custom CSS
local_css()

# Note: Session state initialization has been moved after the title section
# for better organization and to ensure it's done before any UI elements are rendered

def reset_app():
    """Reset the application to its initial state."""
    # Store current page
    current_page = "Introduction"  # Default to Introduction
    if 'page' in st.session_state:
        current_page = st.session_state.page
        
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Re-initialize essential state
    st.session_state.page = "Introduction"  # Always go back to Introduction on reset
    st.session_state.schedule_generated = False
    st.session_state.processing_complete = False
    st.session_state.staff_names = []
    st.session_state.acc_trained_staff = []
    st.session_state.staff_with_four_shifts = []
    st.session_state.availability_df = None
    st.session_state.uploaded_file = None
    
    # Refresh the page
    st.rerun()

def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Generate a download link for binary files."""
    with open(bin_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href

def download_template():
    """Allow users to download the template availability file."""
    template_path = "Availability.xls"
    if os.path.exists(template_path):
        with open(template_path, "rb") as file:
            btn = st.download_button(
                label="📋 Download Template",
                data=file,
                file_name="Availability_Template.xls",
                mime="application/vnd.ms-excel",
                help="Download a template file to fill with staff availability data"
            )
    else:
        st.warning("Template file not found. Please contact support.")

# Main application title
st.title("Pharmacy Roster Scheduler")

# Setup pages for navigation
app_pages = ["Introduction", "Upload Data", "Configure Settings", "Generate Roster", "View Results"]

# Initialize all required session state variables
if 'page' not in st.session_state:
    st.session_state.page = "Introduction"
if 'staff_names' not in st.session_state:
    st.session_state.staff_names = []
if 'acc_trained_staff' not in st.session_state:
    st.session_state.acc_trained_staff = []
if 'staff_with_four_shifts' not in st.session_state:
    st.session_state.staff_with_four_shifts = []
if 'schedule_generated' not in st.session_state:
    st.session_state.schedule_generated = False
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'availability_df' not in st.session_state:
    st.session_state.availability_df = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'solution' not in st.session_state:
    st.session_state.solution = None
if 'output_file' not in st.session_state:
    st.session_state.output_file = None
if 'schedule_summary' not in st.session_state:
    st.session_state.schedule_summary = None

# Main application layout with sidebar and tabs
with st.sidebar:
    st.header("Navigation")
    selected_page = st.radio("Go to", app_pages, index=app_pages.index(st.session_state.page))
    # Update the page in session state if changed through the radio button
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()
    
    st.sidebar.divider()
    st.sidebar.subheader("Actions")
    if st.sidebar.button("🔄 Reset Application", help="Clear all data and start over"):
        reset_app()

# Helper functions for navigation
def get_next_page():
    """Get the next page in the workflow."""
    current_idx = app_pages.index(st.session_state.page)
    if current_idx < len(app_pages) - 1:
        return app_pages[current_idx + 1]
    return None

def get_prev_page():
    """Get the previous page in the workflow."""
    current_idx = app_pages.index(st.session_state.page)
    if current_idx > 0:
        return app_pages[current_idx - 1]
    return None

def nav_to(target_page):
    """Navigate to the specified page."""
    st.session_state.page = target_page
    st.rerun()

# Main content
if st.session_state.page == "Introduction":
    st.header("Welcome to the Pharmacy Roster Scheduler")
    
    st.markdown("""
    <div class="info-box">
    This application uses linear programming to create optimal weekend shift assignments for pharmacists 
    based on staff availability and various constraints.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Key Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Automated Scheduling:** Generates optimized rosters for pharmacy weekend shifts
        - **Equal Distribution:** Balances workload among staff members
        - **Constraint Handling:** Respects both hard and soft constraints
        """)
    
    with col2:
        st.markdown("""
        - **Staff Availability:** Respects staff availability preferences
        - **Visualization:** Provides clear visual reports of the schedule
        - **Customizable:** Adjustable parameters for your specific needs
        """)
    
    st.subheader("How to Use This Application")
    st.markdown("""
    1. **Upload Data**: Provide a staff availability Excel file (download template if needed)
    2. **Configure Settings**: Set up shift allocation rules for staff members
    3. **Generate Roster**: Process the data to create an optimized schedule
    4. **View Results**: Review, visualize, and download the generated roster
    """)
    
    st.markdown("""
    <div class="footnote">
    Use the navigation panel on the left to move between different sections of the application.
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.subheader("About")
    st.markdown("""
    This application implements a linear programming model to solve the complex scheduling problem
    for pharmacy weekend shifts. It balances multiple constraints including staff availability,
    workload distribution, consecutive weekend restrictions, and specialized staff requirements.
    """)

# Placeholder for other pages - will be implemented in subsequent steps
elif st.session_state.page == "Upload Data":
    st.header("Upload Staff Availability Data")
    st.info("Please upload an Excel file containing staff availability information.")
    
    # Template download option
    st.markdown("### Need a template?")
    st.markdown("Download the template file to see the expected format:")
    download_template()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xls', 'xlsx'], help="Upload your staff availability data in Excel format")
    
    if uploaded_file is not None:
        try:
            # Store uploaded file in session state
            st.session_state.uploaded_file = uploaded_file
            
            # Read the data with proper handling of date columns
            df = pd.read_excel(uploaded_file)
            
            # Convert datetime columns to string to avoid mixed types warning
            for col in df.columns:
                if isinstance(col, pd.Timestamp) or isinstance(col, datetime):
                    # Get the date part of the timestamp as a string
                    date_str = col.strftime('%Y-%m-%d')
                    # Create a new dataframe with renamed columns
                    df = df.rename(columns={col: date_str})
            
            st.session_state.availability_df = df
            
            # Display preview of the data
            st.markdown("### Data Preview")
            st.dataframe(df.head(5), use_container_width=True)
            
            # Extract staff names and ACC information
            # Extract staff names and convert to strings to ensure consistent typing
            staff_names = [str(name) for name in df.iloc[:, 0].tolist() if pd.notna(name)]
            st.session_state.staff_names = staff_names
            
            # Process ACC-trained staff information
            acc_column = None
            acc_column_names = ["ACC", "Staff with ACC"]
            
            # Try to find ACC column by name
            for col_name in acc_column_names:
                if col_name in df.columns:
                    acc_column = col_name
                    break
                    
            # If not found by name, use the second column (index 1)
            if acc_column is None and len(df.columns) > 1:
                acc_column = df.columns[1]
                
            if acc_column:
                acc_data = df[acc_column]
                acc_trained_staff = [
                    staff_names[i] for i, trained in enumerate(acc_data) 
                    if pd.notna(trained) and (
                        trained == 1 or 
                        (isinstance(trained, str) and trained.strip() == 'Yes') or 
                        trained is True
                    )
                ]
                st.session_state.acc_trained_staff = acc_trained_staff
                
                # Display ACC-trained staff
                st.markdown("### Detected ACC-Trained Staff")
                if len(acc_trained_staff) > 0:
                    st.success(f"Found {len(acc_trained_staff)} ACC-trained staff members: {', '.join(acc_trained_staff)}")
                else:
                    st.warning("No ACC-trained staff detected. Please check your data.")
            
            # Analyze weekend data
            avail_columns = df.columns[2:]  # Skip name and ACC columns
            weekend_count = len(avail_columns) // 2
            if weekend_count == 0:
                st.error("No weekend data found. Please check your Excel file format.")
            else:
                st.markdown("### Weekend Information")
                st.success(f"Detected {weekend_count} weekends ({weekend_count*2} days) in the data")
                
                # Sample of weekend dates
                weekend_cols = [str(col).split()[0] if isinstance(col, datetime) else str(col) for col in avail_columns[:6]]
                st.markdown(f"Weekend dates: {', '.join(weekend_cols)}...")
                
                # Suggest to proceed to next step
                st.markdown("### Next Steps")
                st.markdown("✅ Data uploaded successfully. Proceed to **Configure Settings** to set up staff shift allocations.")
                if st.button("Continue to Configure Settings", key="to_configure_btn"):
                    nav_to("Configure Settings")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your file follows the template format. You can download the template above.")
            st.session_state.availability_df = None

elif st.session_state.page == "Configure Settings":
    st.header("Configure Roster Settings")
    if st.session_state.availability_df is None:
        st.warning("Please upload staff availability data first.")
        if st.button("Go to Upload Data", key="to_upload_btn"):
            nav_to("Upload Data")
    else:
        st.markdown("""
        <div class="info-box">
        <p>According to the schedule requirements, staff members need to be assigned shifts to cover all weekend slots.</p>
        <p><strong>Terminology:</strong></p>
        <ul>
            <li><strong>Available Shifts</strong>: Total number of shifts that can be filled by staff based on current allocation</li>
            <li><strong>Required Shifts</strong>: Total number of weekend slots that need to be filled (weekends × days × shifts per day)</li>
        </ul>
        <p>When Available < Required: Not enough staff availability to cover all slots (infeasible)</p>
        <p>When Available > Required: More staff availability than needed (some staff availability will go unused)</p>
        <p>When Available = Required: Perfect allocation (optimal)</p>
        </div>
        """, unsafe_allow_html=True)
        
        if len(st.session_state.staff_names) == 0:
            st.error("No staff names found in the data. Please check your Excel file.")
        else:
            st.markdown("### Staff Shift Allocation")
            st.markdown(f"Total staff members: **{len(st.session_state.staff_names)}**")
            
            # Calculate the optimal distribution of staff shifts
            # Get the actual weekend count from data
            weekend_columns = st.session_state.availability_df.columns[2:]  # Skip name and ACC columns
            weekend_count = len(weekend_columns) // 2
            
            # Calculate actual shifts needed based on weekends
            total_shifts_needed = weekend_count * 2 * 3  # weekends * days * shifts per day
            total_staff = len(st.session_state.staff_names)
            
            # Calculate average shifts per staff member
            avg_shifts = total_shifts_needed / total_staff
            
            # Determine the base shifts (everyone gets at least this many)
            base_shifts = int(avg_shifts)  # Floor value
            
            # Calculate how many staff need an extra shift
            staff_with_extra = total_shifts_needed - (base_shifts * total_staff)
            
            # For display purposes, calculate staff with each shift count
            staff_with_higher_count = int(staff_with_extra)
            staff_with_lower_count = total_staff - staff_with_higher_count
            higher_shift_count = base_shifts + 1
            lower_shift_count = base_shifts
            
            # Check feasibility with the current staff count
            st.markdown("### Shift Allocation Recommendation")
            
            # Calculate total shifts that would be assigned with current allocation (staff availability)
            available_shifts = (higher_shift_count * staff_with_higher_count) + (lower_shift_count * staff_with_lower_count)
            
            # Calculate weekend data for display
            weekend_columns = st.session_state.availability_df.columns[2:]  # Skip name and ACC columns
            weekend_count = len(weekend_columns) // 2
            required_shifts = weekend_count * 2 * 3  # weekends * days * shifts per day (slots that need to be filled)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Total Staff", 
                    value=total_staff
                )
                st.metric(
                    label="Total Weekends", 
                    value=weekend_count
                )
                
            with col2:
                if staff_with_higher_count > 0:
                    st.metric(
                        label="Optimal Distribution", 
                        value=f"{staff_with_higher_count} staff with {higher_shift_count} shifts, {staff_with_lower_count} staff with {lower_shift_count} shifts"
                    )
                else:
                    # Everyone gets the same number of shifts
                    st.metric(
                        label="Optimal Distribution", 
                        value=f"All staff with {lower_shift_count} shifts"
                    )
                    
                # Calculate total staff availability based on the new distribution
                available_shifts = (staff_with_higher_count * higher_shift_count) + (staff_with_lower_count * lower_shift_count)
                st.metric(
                    label="Total Shifts", 
                    value=f"Available: {available_shifts}, Required: {required_shifts}",
                    help="Available = total staff availability, Required = total weekend slots to fill"
                )
                
            # Check if allocation is feasible
            # if available_shifts < required_shifts:
            #     st.error(f"Insufficient staff availability! Available: {available_shifts}, Required: {required_shifts}")
            #     st.markdown("Please adjust staff allocations to increase the total staff availability.")
            #     is_feasible = False
            # elif available_shifts > required_shifts:
            #     st.warning(f"More staff availability than required! Available: {available_shifts}, Required: {required_shifts}")
            #     st.markdown(f"Note: {available_shifts - required_shifts} shifts of staff availability will go unused.")
            #     is_feasible = True
            # else:
            #     st.success(f"Perfect allocation! Available: {available_shifts}, Required: {required_shifts}")
            #     is_feasible = True
            
            # Manual selection of staff with higher shift count
            if staff_with_higher_count > 0:
                st.markdown(f"### Select Staff with {higher_shift_count} Shifts")
                st.markdown(f"Select {staff_with_higher_count} staff members who should receive {higher_shift_count} shifts (all others will receive {lower_shift_count} shifts):")
                
                # Convert all staff names to strings to avoid type comparison errors
                staff_name_strings = [str(name) for name in st.session_state.staff_names if name is not None]
                staff_with_higher = st.multiselect(
                    f"Staff with {higher_shift_count} shifts", 
                    options=sorted(staff_name_strings),
                    default=[],
                    help=f"Select {staff_with_higher_count} staff members who should be assigned {higher_shift_count} shifts"
                )
                # Update variable name for consistency with the rest of the code
                staff_with_five = staff_with_higher
            else:
                # No need for selection if everyone gets the same number of shifts
                st.markdown(f"### Staff Shift Allocation")
                st.info(f"All staff members will be assigned {lower_shift_count} shifts each.")
                st.markdown("No manual selection needed since all staff receive the same number of shifts.")
                staff_with_five = []  # Empty list for consistency with rest of code
            
            # Store the selected staff in session state with proper normalization
            if staff_with_higher_count > 0:
                # Function to normalize staff names consistently
                def normalize_name(name):
                    return str(name).strip() if pd.notna(name) else ""
                
                # Normalize all names for consistent comparison
                normalized_staff_names = [normalize_name(name) for name in st.session_state.staff_names]
                normalized_staff_five = [normalize_name(name) for name in staff_with_five]
                
                # Determine staff with lower shifts (all those not selected for higher shifts)
                staff_with_four = []
                for i, staff in enumerate(st.session_state.staff_names):
                    if normalize_name(staff) not in normalized_staff_five:
                        staff_with_four.append(staff)
                
                # Store the normalized names in session state
                st.session_state.staff_with_four_shifts = staff_with_four
                st.session_state.higher_shift_count = higher_shift_count
                st.session_state.lower_shift_count = lower_shift_count
                
                # Debug info
                print(f"\nStaff with {higher_shift_count} shifts (UI): {[str(s) for s in staff_with_five]}")
                print(f"Staff with {lower_shift_count} shifts (UI): {[str(s) for s in staff_with_four]}")
            else:
                # Everyone gets the same shifts
                st.session_state.staff_with_four_shifts = list(st.session_state.staff_names)
                st.session_state.higher_shift_count = lower_shift_count  # Both are the same
                st.session_state.lower_shift_count = lower_shift_count
            
            # Show the breakdown if there's a distinction between shift counts
            if staff_with_higher_count > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### Staff with {higher_shift_count} Shifts")
                    if staff_with_five:
                        for name in sorted(staff_with_five):
                            st.markdown(f"- {name}")
                    else:
                        st.info(f"No staff selected for {higher_shift_count} shifts")
                
                with col2:
                    st.markdown(f"### Staff with {lower_shift_count} Shifts")
                    if staff_with_four:
                        for name in sorted(staff_with_four):
                            st.markdown(f"- {name}")
                    else:
                        st.info(f"No staff selected for {lower_shift_count} shifts")
            else:
                # No breakdown needed if everyone gets the same shifts
                st.markdown(f"### Staff with {lower_shift_count} Shifts")
                st.text("All staff members will receive the same number of shifts.")
                st.markdown("\n".join([f"- {name}" for name in sorted(st.session_state.staff_names)]))
            
            # Display allocation summary
            st.markdown("### Current Allocation Summary")
            
            # Calculate staff count and total shifts based on current selection
            if staff_with_higher_count > 0:
                # When we have different shift counts
                staff_with_higher_selected = len(staff_with_five)
                staff_with_lower_selected = len(st.session_state.staff_names) - staff_with_higher_selected
                total_shifts = (staff_with_higher_selected * higher_shift_count) + (staff_with_lower_selected * lower_shift_count)
                
                # Create a summary dataframe
                summary_data = {
                    "Category": [f"Staff with {lower_shift_count} shifts", f"Staff with {higher_shift_count} shifts", "Total staff", "Total shifts"],
                    "Count": [staff_with_lower_selected, staff_with_higher_selected, len(st.session_state.staff_names), total_shifts]
                }
            else:
                # When everyone gets the same shifts
                total_shifts = len(st.session_state.staff_names) * lower_shift_count
                
                # Create a summary dataframe
                summary_data = {
                    "Category": [f"Staff with {lower_shift_count} shifts", "Total staff", "Total shifts"],
                    "Count": [len(st.session_state.staff_names), len(st.session_state.staff_names), total_shifts]
                }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, hide_index=True, use_container_width=True)
            
            # Display weekend data
            weekend_columns = st.session_state.availability_df.columns[2:]  # Skip name and ACC columns
            weekend_count = len(weekend_columns) // 2
            required_shifts = weekend_count * 2 * 3  # weekends * days * shifts per day
            available_shifts = total_shifts  # Staff availability aggregate
            
            # Check if allocation is feasible
            st.markdown("### Feasibility Check")
            
            if available_shifts < required_shifts:
                st.error(f"Insufficient staff availability! Available: {available_shifts}, Required: {required_shifts}")
                st.markdown("Please adjust staff allocations to increase the number of available shifts.")
            elif available_shifts > required_shifts:
                st.warning(f"More staff availability than required! Available: {available_shifts}, Required: {required_shifts}")
                st.markdown(f"Note: {available_shifts - required_shifts} shifts of staff availability will go unused.")
            else:
                st.success(f"Perfect allocation! Available: {available_shifts}, Required: {required_shifts}")
                
                # Check if we have the correct number of staff selected for higher shift count
                if staff_with_higher_count > 0 and len(staff_with_five) != staff_with_higher_count:
                    st.warning(f"⚠️ You've selected {len(staff_with_five)} staff for {higher_shift_count} shifts, but the optimal allocation requires exactly {staff_with_higher_count} staff.")
                    st.markdown(f"Please adjust your selection to match the required {staff_with_higher_count} staff members for {higher_shift_count} shifts.")
                else:
                    # Next step button
                    st.markdown("### Next Steps")
                    st.markdown("✅ Staff shift configuration complete. Proceed to **Generate Roster** to create the schedule.")
                    if st.button("Continue to Generate Roster", key="to_generate_btn"):
                        nav_to("Generate Roster")

elif st.session_state.page == "Generate Roster":
    st.header("Generate Roster Schedule")
    if st.session_state.staff_with_four_shifts is None or len(st.session_state.staff_with_four_shifts) == 0:
        st.warning("Please configure staff shift settings first.")
        if st.button("Go to Configure Settings"):
            st.session_state.page = "Configure Settings"
            st.experimental_rerun()
    else:
        st.markdown("""
        <div class="info-box">
        The roster generator will create an optimized schedule based on the staff availability data and the 
        constraints you have configured. This process may take a few moments to complete.
        </div>
        """, unsafe_allow_html=True)
        
        # Display roster settings summary
        st.markdown("### Roster Settings Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Summary**")
            st.markdown(f"- Staff Members: **{len(st.session_state.staff_names)}**")
            weekend_columns = st.session_state.availability_df.columns[2:]  # Skip name and ACC columns
            weekend_count = len(weekend_columns) // 2
            st.markdown(f"- Weekends: **{weekend_count}** ({weekend_count*2} days)")
            st.markdown(f"- ACC-Trained Staff: **{len(st.session_state.acc_trained_staff)}**")
            
        with col2:
            st.markdown("**Shift Allocation**")
            staff_with_four_count = len(st.session_state.staff_with_four_shifts)
            staff_with_higher_count = len(st.session_state.staff_names) - staff_with_four_count
            
            # Get shift counts from session state
            higher_shift_count = st.session_state.get('higher_shift_count', 5)
            lower_shift_count = st.session_state.get('lower_shift_count', 4)
            
            if higher_shift_count != lower_shift_count:
                st.markdown(f"- Staff with {lower_shift_count} shifts: **{staff_with_four_count}**")
                st.markdown(f"- Staff with {higher_shift_count} shifts: **{staff_with_higher_count}**")
            else:
                st.markdown(f"- All staff with {lower_shift_count} shifts: **{len(st.session_state.staff_names)}**")
                
            total_shifts = (staff_with_four_count * lower_shift_count) + (staff_with_higher_count * higher_shift_count)
            st.markdown(f"- Total shifts: **{total_shifts}**")
        
        # Advanced options (if needed)
        with st.expander("Advanced Options"):
            st.markdown("These options affect how the roster is generated and constraints are applied.")
            
            allow_consecutive_weekends = st.checkbox(
                "Allow Consecutive Weekends if Needed", 
                value=True, 
                help="If checked, the model can assign consecutive weekends if no other solution is possible"
            )
            
            allow_same_weekend = st.checkbox(
                "Allow Both Days on Same Weekend if Needed", 
                value=True,
                help="If checked, the model can assign both Saturday and Sunday of the same weekend to a staff member if necessary"
            )
            
            require_acc = st.checkbox(
                "Require ACC Staff on Saturdays", 
                value=True, 
                help="If checked, the model will try to ensure at least one ACC-trained staff member is scheduled for each Saturday"
            )
        
        # Button to generate roster
        generate_button = st.button("🔄 Generate Roster Schedule", type="primary", use_container_width=True)
        
        if generate_button or st.session_state.processing_complete:
            if not st.session_state.processing_complete:
                # Show progress information
                progress_container = st.container()
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Update status
                    status_text.text("Step 1/4: Preparing data...")
                    progress_bar.progress(10)
                    
                    # Create a temporary file to hold the uploaded data
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.xls') as tmp_file:
                        tmp_file.write(st.session_state.uploaded_file.getvalue())
                        temp_file_path = tmp_file.name
                    
                    # Step 2: Create data loader and load data
                    status_text.text("Step 2/4: Loading and processing data...")
                    progress_bar.progress(25)
                    
                    loader = DataLoader(temp_file_path)
                    loader.load_data()
                    
                    # Calculate needed shifts based on available weekend days
                    weekend_count = len(loader.get_weekend_days()) // 2
                    total_shifts_needed = weekend_count * 2 * 3  # weekends * days * shifts per day
                    
                    # Get shift counts from session state
                    higher_shift_count = st.session_state.get('higher_shift_count', 5)
                    lower_shift_count = st.session_state.get('lower_shift_count', 4)
                    
                    # Override DataLoader's staff_with_four_shifts with our UI selections
                    # This ensures the roster uses our UI selections, not the default hard-coded list
                    print(f"Setting staff_with_four_shifts from UI: {sorted(st.session_state.staff_with_four_shifts)}")
                    loader.set_staff_with_four_shifts(st.session_state.staff_with_four_shifts)
                    
                    # If we're using different shift counts than 4/5, override those values too
                    if higher_shift_count != 5 or lower_shift_count != 4:
                        print(f"Using custom shift counts: {higher_shift_count}/{lower_shift_count} instead of default 5/4")
                        for staff in loader.staff_names:
                            if staff in st.session_state.staff_with_four_shifts:
                                loader.staff_shifts[staff] = lower_shift_count
                            else:
                                loader.staff_shifts[staff] = higher_shift_count
                    
                    # Print some debug info
                    higher_count = len(loader.staff_names) - len(st.session_state.staff_with_four_shifts)
                    lower_count = len(st.session_state.staff_with_four_shifts)
                    print(f"Assigned {higher_count} staff with {higher_shift_count} shifts")
                    print(f"Assigned {lower_count} staff with {lower_shift_count} shifts")
                    print(f"Total shifts allocated: {(higher_count * higher_shift_count) + (lower_count * lower_shift_count)}")
                    print(f"Available shifts: {weekend_count * 2 * 3}")
                    
                    # Print actual assignment to help debug
                    print("\nStaff shift assignments:")
                    
                    # Validate shift assignments match UI selections
                    mismatch_found = False
                    for staff, shifts in sorted(loader.staff_shifts.items()):
                        staff_str = str(staff).strip()
                        expected_shifts = lower_shift_count if staff_str in [str(s).strip() for s in st.session_state.staff_with_four_shifts] else higher_shift_count
                        if shifts != expected_shifts:
                            print(f"MISMATCH! {staff}: expected {expected_shifts}, got {shifts} shifts")
                            mismatch_found = True
                        else:
                            print(f"{staff}: {shifts} shifts")
                    
                    if mismatch_found:
                        print("\nWARNING: Some staff shift assignments don't match UI selections!")
                        print("This may affect the roster generation. Check your selections and the data loader.")
                    
                    # Step 3: Create and solve model
                    status_text.text("Step 3/4: Creating and solving the model...")
                    progress_bar.progress(50)
                    
                    model = RosterModel(
                        loader.get_staff_names(),
                        loader.get_weekend_days(),
                        loader.get_availability(),
                        loader.get_acc_trained_staff(),
                        loader.staff_shifts
                    )
                    
                    model.create_model()
                    solution = model.solve()
                    
                    if not solution:
                        st.session_state.processing_complete = True
                        st.error("Failed to find a feasible solution. The model is infeasible with the current constraints and availability data.")
                        st.markdown("""
                        <div class="warning-box">
                        The solver could not find a feasible solution. This might be due to:
                        1. Insufficient staff availability for the required shifts
                        2. Conflicting constraints that cannot be satisfied simultaneously
                        3. Not enough ACC-trained staff to cover all Saturdays
                        
                        Try adjusting the advanced options or reviewing staff availability data.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        schedule_by_weekend = model.get_schedule_by_weekend()
                        
                        # Step 4: Generate roster
                        status_text.text("Step 4/4: Generating roster files...")
                        progress_bar.progress(75)
                        
                        weekend_count = len(loader.get_weekend_days()) // 2
                        generator = RosterGenerator.from_model_results(
                            loader.get_staff_names(),
                            solution,
                            weekend_count
                        )
                        
                        # Create a temporary output file
                        output_file = os.path.join(tempfile.gettempdir(), "Roster_Schedule.xlsx")
                        generator.export_to_excel(output_file)
                        
                        # Save results in session state
                        st.session_state.solution = solution
                        st.session_state.output_file = output_file
                        st.session_state.schedule_summary = schedule_by_weekend
                        st.session_state.schedule_generated = True
                        st.session_state.processing_complete = True
                        
                        # Complete the progress bar
                        progress_bar.progress(100)
                        status_text.text("✅ Roster generation complete!")
                        
                        # Navigate to results page
                        st.markdown("### Generation Complete!")
                        st.markdown("Your roster has been successfully generated. You can now view and download the results.")
                        if st.button("View Results", key="view_results_btn"):
                            nav_to("View Results")
                
                except Exception as e:
                    st.error(f"An error occurred during roster generation: {str(e)}")
                    st.session_state.processing_complete = True
                    
                finally:
                    # Clean up temporary file if it was created
                    if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass
            
            else:
                # If processing is already complete, just show the completion message
                if st.session_state.schedule_generated:
                    st.success("✅ Roster generation complete!")
                    st.markdown("Your roster has been successfully generated. You can now view and download the results.")
                    if st.button("View Results", key="view_results_btn2"):
                        nav_to("View Results")
                else:
                    st.error("The roster generation process failed. Please try again with different settings.")
                    
        # Button to go back
        if st.button("Back to Configure Settings", key="back_to_config_btn"):
            st.session_state.processing_complete = False
            nav_to("Configure Settings")

elif st.session_state.page == "View Results":
    st.header("View Roster Results")
    if not st.session_state.schedule_generated:
        st.warning("No roster has been generated yet. Please complete the previous steps first.")
        if st.button("Go to Generate Roster", key="goto_generate_btn"):
            nav_to("Generate Roster")
    else:
        st.success("Roster generated successfully! You can view and download the results below.")
        
        # Create tabs for different views
        tabs = st.tabs(["Summary", "Weekend Schedule", "Staff Assignments", "Visualizations"])
        
        # Process solution data for display
        solution = st.session_state.solution
        schedule_by_weekend = st.session_state.schedule_summary
        # Ensure staff names are strings
        staff_names = [str(name) for name in st.session_state.staff_names if pd.notna(name)]
        
        # Tab 1: Summary
        with tabs[0]:
            st.markdown("### Roster Summary")
            
            # Staff shift counts
            shift_counts = {}
            for staff, assignments in solution.items():
                shift_counts[staff] = len(assignments)
            
            shift_data = pd.DataFrame({
                "Staff": list(shift_counts.keys()),
                "Assigned Shifts": list(shift_counts.values())
            })
            
            # Sort by number of shifts descending, then by name
            shift_data = shift_data.sort_values(by=["Assigned Shifts", "Staff"], ascending=[False, True])
            
            # Display summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Total Staff", 
                    value=len(staff_names)
                )
            
            with col2:
                st.metric(
                    label="Total Shifts Assigned", 
                    value=sum(shift_counts.values())
                )
            
            with col3:
                st.metric(
                    label="Total Weekends", 
                    value=len(set([day[0] for staff_assignments in solution.values() for day, _ in staff_assignments]))
                )
            
            # Display staff shift statistics
            st.markdown("#### Staff Shift Allocation")
            st.dataframe(shift_data, use_container_width=True)
        
        # Tab 2: Weekend Schedule
        with tabs[1]:
            st.markdown("### Weekend Schedule")
            
            # Create a cleaner weekend schedule dataframe
            weekends = sorted(set([day[0] for day in schedule_by_weekend.keys()]))
            
            all_schedule_data = []
            
            for weekend in weekends:
                for day_type in ["Saturday", "Sunday"]:
                    day_data = (weekend, day_type)
                    if day_data in schedule_by_weekend:
                        shifts = schedule_by_weekend[day_data]
                        early = shifts.get("early", "-")
                        mid = shifts.get("mid", "-")
                        late = shifts.get("late", "-")
                        
                        all_schedule_data.append({
                            "Weekend": int(weekend + 1),  # 1-indexed for display, ensure it's an integer
                            "Day": day_type,
                            "Early Shift": early,
                            "Mid Shift": mid,
                            "Late Shift": late
                        })
            
            weekend_df = pd.DataFrame(all_schedule_data)
            
            # Ensure Weekend column is integer type
            weekend_df['Weekend'] = weekend_df['Weekend'].astype(int)
            
            # Sort by Weekend and Day (Saturday first, then Sunday)
            weekend_df = weekend_df.sort_values(by=['Weekend', 'Day'], 
                                              key=lambda x: x.map({'Saturday': 0, 'Sunday': 1}) if x.name == 'Day' else x)
            
            # Display the weekend schedule
            st.dataframe(weekend_df, use_container_width=True)
            
            # Option to highlight specific staff member
            selected_staff = st.selectbox(
                "Highlight shifts for staff member:", 
                ["(None)"] + sorted(staff_names)
            )
            
            if selected_staff != "(None)":
                # Filter for the selected staff member
                staff_schedule = weekend_df[
                    (weekend_df["Early Shift"] == selected_staff) | 
                    (weekend_df["Mid Shift"] == selected_staff) | 
                    (weekend_df["Late Shift"] == selected_staff)
                ]
                
                if not staff_schedule.empty:
                    st.markdown(f"#### Shifts for {selected_staff}")
                    st.dataframe(staff_schedule, use_container_width=True)
                else:
                    st.info(f"No shifts assigned to {selected_staff}")
        
        # Tab 3: Staff Assignments
        with tabs[2]:
            st.markdown("### Staff Assignments")
            
            # Create a dataframe of all staff assignments
            staff_assignments = []
            
            for staff, assignments in solution.items():
                for (weekend, day), shift in assignments:
                    staff_assignments.append({
                        "Staff": staff,
                        "Weekend": weekend + 1,  # 1-indexed for display
                        "Day": day,
                        "Shift": shift
                    })
            
            staff_df = pd.DataFrame(staff_assignments)
            
            # Sort by staff name, then by weekend
            staff_df = staff_df.sort_values(by=["Staff", "Weekend", "Day"])
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                filter_staff = st.multiselect(
                    "Filter by staff:",
                    options=sorted(staff_names),
                    default=[]
                )
            
            with col2:
                filter_weekend = st.multiselect(
                    "Filter by weekend:",
                    options=[w+1 for w in range(len(weekends))],
                    default=[]
                )
            
            # Apply filters
            filtered_df = staff_df
            if filter_staff:
                filtered_df = filtered_df[filtered_df["Staff"].isin(filter_staff)]
            if filter_weekend:
                filtered_df = filtered_df[filtered_df["Weekend"].isin(filter_weekend)]
            
            # Display filtered assignments
            st.dataframe(filtered_df, use_container_width=True)
        
        # Tab 4: Visualizations
        with tabs[3]:
            st.markdown("### Roster Visualizations")
            
            # Create some simple visualizations
            
            # 1. Shift counts per staff
            st.subheader("Shift Distribution")
            fig1 = px.bar(
                shift_data,
                x="Staff",
                y="Assigned Shifts",
                color="Assigned Shifts",
                title="Number of Shifts Assigned per Staff Member"
            )
            fig1.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig1, use_container_width=True)
            
            # 2. Weekend heatmap
            st.subheader("Weekend Assignment Heatmap")
            
            # Create a matrix of staff vs weekends
            heatmap_data = {}
            for staff in staff_names:
                # Ensure staff name is a string for dictionary key
                staff_key = str(staff).strip()
                heatmap_data[staff_key] = [0] * len(weekends)
            
            for staff, assignments in solution.items():
                # Ensure staff name is consistently handled as a string
                staff_key = str(staff).strip()
                for (weekend, _), _ in assignments:
                    # Handle cases where staff might not be in heatmap_data
                    if staff_key in heatmap_data:
                        heatmap_data[staff_key][weekend] += 1
                    else:
                        print(f"Warning: Staff '{staff_key}' not found in heatmap_data keys. Available keys: {list(heatmap_data.keys())[:5]}...")
            
            heatmap_df = pd.DataFrame(heatmap_data).T
            heatmap_df.columns = [f"W{i+1}" for i in range(len(weekends))]
            
            fig2 = px.imshow(
                heatmap_df,
                labels=dict(x="Weekend", y="Staff", color="Shifts"),
                title="Staff Weekend Assignments",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Download options
        st.divider()
        st.markdown("### Download Options")
        
        # Download Excel roster
        if st.session_state.output_file and os.path.exists(st.session_state.output_file):
            with open(st.session_state.output_file, "rb") as f:
                excel_data = f.read()
                
            st.download_button(
                label="📥 Download Excel Roster",
                data=excel_data,
                file_name="Pharmacy_Roster.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.error("Excel file not found. There may have been an error during roster generation.")
        
        # # Option to export to CSV
        # if st.button("Export to CSV"):
        #     # Create CSV export of the main schedules
        #     weekend_csv = weekend_df.to_csv(index=False)
        #     staff_csv = staff_df.to_csv(index=False)
            
        #     col1, col2 = st.columns(2)
        #     with col1:
        #         st.download_button(
        #             label="Download Weekend Schedule CSV",
        #             data=weekend_csv,
        #             file_name="Weekend_Schedule.csv",
        #             mime="text/csv"
        #         )
            
        #     with col2:
        #         st.download_button(
        #             label="Download Staff Assignments CSV",
        #             data=staff_csv,
        #             file_name="Staff_Assignments.csv",
        #             mime="text/csv"
        #         )
        
        # Navigation buttons
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate New Roster", key="new_roster_btn", use_container_width=True):
                st.session_state.processing_complete = False
                st.session_state.schedule_generated = False
                nav_to("Generate Roster")
        
        with col2:
            if st.button("Start Over", use_container_width=True):
                reset_app()

# Navigation buttons
st.divider()

# Add custom CSS for navigation container
st.markdown("""
<style>
.navigation-container {
    margin-top: 2rem;
    padding: 1rem;
    background-color: #f9f9f9;
    border-radius: 0.5rem;
    border: 1px solid #eee;
}
.stButton button {
    width: 100%;
    padding: 0.75rem 0;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# Navigation section
st.markdown('<div class="navigation-container">', unsafe_allow_html=True)
st.markdown("### Navigation")

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    prev_page = get_prev_page()
    if prev_page:
        if st.button(f"← Previous: {prev_page}", key="prev_page_btn", use_container_width=True):
            nav_to(prev_page)

with col3:
    next_page = get_next_page()
    # Only show next button if appropriate conditions are met
    can_proceed = True
    
    # Define conditions for proceeding to next page
    if st.session_state.page == "Upload Data" and st.session_state.availability_df is None:
        can_proceed = False
    elif st.session_state.page == "Configure Settings" and (st.session_state.staff_with_four_shifts is None or len(st.session_state.staff_with_four_shifts) == 0):
        can_proceed = False
    elif st.session_state.page == "Generate Roster" and not st.session_state.processing_complete:
        can_proceed = False
    
    if next_page and can_proceed:
        if st.button(f"Next: {next_page} →", key="next_page_btn", type="primary", use_container_width=True):
            nav_to(next_page)
    elif next_page:
        st.button(f"Next: {next_page} →", key="next_page_disabled_btn", disabled=True, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# App footer
st.divider()
st.markdown("""
<div class="footnote" style="text-align: center">
Pharmacy Roster Scheduler • Developed with Streamlit • © 2025
</div>
""", unsafe_allow_html=True)
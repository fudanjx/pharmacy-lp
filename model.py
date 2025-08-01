"""
Model module for pharmacy roster scheduling.
This module defines and solves the linear programming model for staff scheduling.
"""

import pulp
from typing import Dict, List, Tuple, Set
from pulp import LpVariable, LpProblem, LpMinimize, LpStatus, LpBinary


class RosterModel:
    """Class for defining and solving the linear programming model for staff scheduling."""

    def __init__(self, 
                 staff_names: List[str],
                 weekend_days: List[Tuple[int, str]],
                 availability: Dict[str, Dict[Tuple[int, str], List[str]]],
                 acc_trained_staff: Set[str],
                 staff_shifts: Dict[str, int]):
        """
        Initialize the RosterModel with necessary data.

        Args:
            staff_names (List[str]): List of all staff names
            weekend_days (List[Tuple[int, str]]): List of weekend days as (weekend_num, day) tuples
            availability (Dict): Staff availability for each day and shift
            acc_trained_staff (Set[str]): Set of ACC-trained staff
            staff_shifts (Dict[str, int]): Number of shifts assigned to each staff
        """
        self.staff_names = staff_names
        self.weekend_days = weekend_days
        self.availability = availability
        self.acc_trained_staff = acc_trained_staff
        self.staff_shifts = staff_shifts
        self.shifts = ["early", "mid", "late"]
        self.weekends = set([w for w, _ in weekend_days])
        
        # Will store the problem and variables
        self.problem = None
        self.assignments = {}
        
        # Will store the results
        self.results = {}

    def create_model(self) -> None:
        """Create the linear programming model with all constraints."""
        # Create the LP problem
        self.problem = LpProblem("Pharmacy_Roster_Scheduling", LpMinimize)
        
        # Count variables for debugging
        total_vars = 0
        available_shifts_per_staff = {staff: 0 for staff in self.staff_names}
        
        # Create binary decision variables
        # x[staff, day, shift] = 1 if staff is assigned to shift on day, 0 otherwise
        self.assignments = {}
        for staff in self.staff_names:
            for day in self.weekend_days:
                for shift in self.shifts:
                    # Only create variables for available shifts
                    if day in self.availability[staff] and shift in self.availability[staff][day]:
                        self.assignments[(staff, day, shift)] = LpVariable(
                            f"x_{staff}_{day[0]}_{day[1]}_{shift}", 
                            cat=LpBinary
                        )
                        total_vars += 1
                        available_shifts_per_staff[staff] += 1
        
        # Print availability statistics for debugging
        print(f"Created {total_vars} assignment variables")
        print(f"Staff with fewest available shifts: {min(available_shifts_per_staff.items(), key=lambda x: x[1])}")
        print(f"Staff with most available shifts: {max(available_shifts_per_staff.items(), key=lambda x: x[1])}")
        
        # Check for staff with fewer available shifts than their target
        for staff, available_count in available_shifts_per_staff.items():
            target_shifts = self.staff_shifts[staff]
            if available_count < target_shifts:
                print(f"WARNING: {staff} has only {available_count} available shifts but needs {target_shifts}")
        
        # Create weekend assignment tracking variables
        # y[staff, weekend] = 1 if staff is assigned any shift on the weekend, 0 otherwise
        weekend_assignments = {}
        for staff in self.staff_names:
            for weekend in self.weekends:
                weekend_assignments[(staff, weekend)] = LpVariable(
                    f"y_{staff}_{weekend}", 
                    cat=LpBinary
                )
        
        # Objective: Minimize a dummy variable (we'll use constraints to enforce balance)
        # This is a feasibility problem rather than an optimization problem
        dummy = LpVariable("dummy")
        self.problem += dummy
        
        # Constraint 1: Each shift must be assigned to exactly one staff member
        for day in self.weekend_days:
            for shift in self.shifts:
                self.problem += (
                    pulp.lpSum(
                        self.assignments.get((staff, day, shift), 0) 
                        for staff in self.staff_names
                    ) == 1,
                    f"One_staff_for_{day[0]}_{day[1]}_{shift}"
                )
        
        # Check if we have enough weekends to satisfy all staff shift requirements
        total_shifts_required = sum(self.staff_shifts.values())
        total_shifts_available = len(self.weekend_days) * len(self.shifts)
        print(f"Total shifts required: {total_shifts_required}, Total shifts available: {total_shifts_available}")
        
        if total_shifts_required > total_shifts_available:
            print(f"WARNING: Not enough weekend days to satisfy shift requirements. Adjusting constraints.")
            # We need to make the constraint a <= instead of == to allow fewer shifts
            
            # Constraint 2: Each staff member should be assigned at most their target number of shifts
            for staff in self.staff_names:
                self.problem += (
                    pulp.lpSum(
                        self.assignments.get((staff, day, shift), 0)
                        for day in self.weekend_days
                        for shift in self.shifts
                    ) <= self.staff_shifts[staff],
                    f"Max_shift_count_for_{staff}"
                )
                
            # Add constraint to ensure we use all available shifts
            self.problem += (
                pulp.lpSum(
                    self.assignments.get((staff, day, shift), 0)
                    for staff in self.staff_names
                    for day in self.weekend_days
                    for shift in self.shifts
                    if (staff, day, shift) in self.assignments
                ) == total_shifts_available,
                "Use_all_available_shifts"
            )
        else:
            # Constraint 2: Each staff member should be assigned exactly their target number of shifts
            for staff in self.staff_names:
                self.problem += (
                    pulp.lpSum(
                        self.assignments.get((staff, day, shift), 0)
                        for day in self.weekend_days
                        for shift in self.shifts
                    ) == self.staff_shifts[staff],
                    f"Shift_count_for_{staff}"
                )
        
        # Constraint 3: No staff should work more than one shift per day
        for staff in self.staff_names:
            for day in self.weekend_days:
                self.problem += (
                    pulp.lpSum(
                        self.assignments.get((staff, day, shift), 0)
                        for shift in self.shifts
                    ) <= 1,
                    f"One_shift_per_day_{staff}_{day[0]}_{day[1]}"
                )
        
        # Constraint 4: No staff should work consecutive days within the same weekend
        # Use soft constraint with penalties for this requirement
        print("Adding constraints for consecutive days within same weekend")
        same_weekend_penalties = {}
        
        for staff in self.staff_names:
            for weekend in self.weekends:
                sat_day = (weekend, "Saturday")
                sun_day = (weekend, "Sunday")
                
                # Create penalty variable for this staff and weekend
                penalty_var = LpVariable(f"same_weekend_penalty_{staff}_{weekend}", 0, 1, LpBinary)
                same_weekend_penalties[(staff, weekend)] = penalty_var
                
                # Sum of shifts on Saturday and Sunday for this weekend
                sat_shifts = pulp.lpSum(self.assignments.get((staff, sat_day, shift), 0) 
                                      for shift in self.shifts)
                sun_shifts = pulp.lpSum(self.assignments.get((staff, sun_day, shift), 0) 
                                      for shift in self.shifts)
                
                # Staff can only work on one day of the weekend unless penalty is applied
                self.problem += (
                    sat_shifts + sun_shifts <= 1 + penalty_var,
                    f"No_consecutive_days_in_weekend_{staff}_{weekend}"
                )
        
        # Add penalties to objective function (to minimize violations)
        same_weekend_penalty_sum = pulp.lpSum(same_weekend_penalties.values())
        self.problem += same_weekend_penalty_sum * 2  # Add to objective with higher weight
        print(f"Added {len(same_weekend_penalties)} soft constraints for consecutive days within same weekend")
        
        # Constraint 5: Link weekend assignment variables
        for staff in self.staff_names:
            for weekend in self.weekends:
                # Get all shifts for this staff on this weekend (both Saturday and Sunday)
                weekend_shifts = [
                    self.assignments.get((staff, (weekend, day), shift), 0)
                    for day in ["Saturday", "Sunday"]
                    for shift in self.shifts
                ]
                
                # y[staff, weekend] = 1 if any shift is assigned, 0 otherwise
                self.problem += (
                    weekend_assignments[(staff, weekend)] >= 
                    (1/len(weekend_shifts)) * pulp.lpSum(weekend_shifts),
                    f"Weekend_link_lower_{staff}_{weekend}"
                )
                
                self.problem += (
                    weekend_assignments[(staff, weekend)] <= pulp.lpSum(weekend_shifts),
                    f"Weekend_link_upper_{staff}_{weekend}"
                )
        
        # Constraint 6: No consecutive weekend assignments
        # First check if this constraint might be too restrictive
        avg_shifts_per_staff = sum(self.staff_shifts.values()) / len(self.staff_names)
        weekends_per_staff = avg_shifts_per_staff / 3  # Assuming 3 shifts per weekend
        weekends_ratio = weekends_per_staff / len(self.weekends)
        
        print(f"Weekends needed per staff: {weekends_per_staff:.2f} out of {len(self.weekends)} ({weekends_ratio:.2%})")
        
        # If the ratio is high, relax the constraint to allow some consecutive weekends
        if weekends_ratio > 0.4:  # More than 40% of weekends needed, might be hard to avoid consecutive weekends
            print("WARNING: Staff might need to work many weekends. Relaxing consecutive weekend constraint.")
            # Use a soft constraint with penalties instead
            consecutive_penalties = {}
            for staff in self.staff_names:
                for i, weekend in enumerate(sorted(self.weekends)[:-1]):
                    next_weekend = sorted(self.weekends)[i+1]
                    penalty_var = LpVariable(f"consec_penalty_{staff}_{weekend}_{next_weekend}", 0, 1, LpBinary)
                    consecutive_penalties[(staff, weekend, next_weekend)] = penalty_var
                    
                    # Link penalty variable to consecutive weekends
                    self.problem += (
                        weekend_assignments[(staff, weekend)] + 
                        weekend_assignments[(staff, next_weekend)] <= 1 + penalty_var,
                        f"Consec_weekend_link_{staff}_{weekend}_{next_weekend}"
                    )
            
            # Add penalties to objective function
            penalty_sum = pulp.lpSum(consecutive_penalties.values())
            self.problem += penalty_sum  # Add to the objective (which was just a dummy variable)
            print(f"Added {len(consecutive_penalties)} soft constraints for consecutive weekends")
        else:
            # Use hard constraints as before
            for staff in self.staff_names:
                for i, weekend in enumerate(sorted(self.weekends)[:-1]):
                    next_weekend = sorted(self.weekends)[i+1]
                    self.problem += (
                        weekend_assignments[(staff, weekend)] + 
                        weekend_assignments[(staff, next_weekend)] <= 1,
                        f"No_consecutive_weekends_{staff}_{weekend}_{next_weekend}"
                    )
        
        # Constraint 7: ACC-trained staff on Saturday (soft constraint)
        # We'll use a binary variable to track if an ACC staff is assigned on each Saturday
        acc_coverage = {}
        
        # Only apply ACC coverage constraint if we have ACC-trained staff
        if len(self.acc_trained_staff) > 0:
            print(f"Adding ACC coverage constraints with {len(self.acc_trained_staff)} ACC-trained staff")
            for weekend in self.weekends:
                acc_coverage[weekend] = LpVariable(f"acc_coverage_{weekend}", cat=LpBinary)
                
                # Link acc_coverage to ACC staff assignments
                self.problem += (
                    acc_coverage[weekend] <= pulp.lpSum(
                        self.assignments.get((staff, (weekend, "Saturday"), shift), 0)
                        for staff in self.acc_trained_staff
                        for shift in self.shifts
                    ),
                    f"ACC_coverage_link_{weekend}"
                )
                
                        # Make ACC coverage a soft constraint with high penalty for not having coverage
                # This allows the model to find a solution even when ACC coverage is impossible
                acc_penalty = LpVariable(f"acc_penalty_{weekend}", 0, 1, LpBinary)
                self.problem += (
                    acc_coverage[weekend] + acc_penalty == 1,
                    f"ACC_coverage_Saturday_{weekend}"
                )
                
                # Add penalty to objective
                self.problem += acc_penalty * 5  # High penalty weight to prioritize ACC coverage
        else:
            print("No ACC-trained staff found, skipping ACC coverage constraint")

    def _relax_constraints(self) -> None:
        """Relax hard constraints to make model easier to solve"""
        # Identify and relax problematic constraints
        # We'll focus on the consecutive weekends constraints as they're often problematic
        
        # Look for constraints with 'No_consecutive_weekends' in the name and remove them
        consec_constraints = [c for c in self.problem.constraints if "No_consecutive_weekends" in c]
        for c in consec_constraints:
            del self.problem.constraints[c]
            
        print(f"Removed {len(consec_constraints)} hard consecutive weekend constraints")
        
        # Replace with soft constraints if they don't exist yet
        if not any("consec_penalty" in str(v) for v in self.problem.variables()):
            print("Adding soft consecutive weekend constraints instead")
            consecutive_penalties = {}
            
            # Get the weekend assignments variables
            weekend_assignments = {}
            for var in self.problem.variables():
                if var.name.startswith("y_"):
                    # Parse variable name to get staff and weekend
                    parts = var.name.split("_")
                    if len(parts) >= 3:
                        staff = parts[1]
                        weekend = int(parts[2])
                        weekend_assignments[(staff, weekend)] = var
            
            # Add soft constraints
            for staff in self.staff_names:
                for i, weekend in enumerate(sorted(self.weekends)[:-1]):
                    next_weekend = sorted(self.weekends)[i+1]
                    if (staff, weekend) in weekend_assignments and (staff, next_weekend) in weekend_assignments:
                        penalty_var = LpVariable(f"consec_penalty_{staff}_{weekend}_{next_weekend}", 0, 1, LpBinary)
                        consecutive_penalties[(staff, weekend, next_weekend)] = penalty_var
                        
                        # Link penalty variable to consecutive weekends
                        self.problem += (
                            weekend_assignments[(staff, weekend)] + 
                            weekend_assignments[(staff, next_weekend)] <= 1 + penalty_var,
                            f"Consec_weekend_soft_{staff}_{weekend}_{next_weekend}"
                        )
            
            # Add penalties to objective function
            if consecutive_penalties:
                penalty_sum = pulp.lpSum(consecutive_penalties.values())
                self.problem += penalty_sum * 2  # Add to the objective with weight

    def solve(self) -> Dict:
        """
        Solve the linear programming model.
        
        Returns:
            Dict: The staff roster solution
        """
        if not self.problem:
            self.create_model()
        
        # Solve the problem with increased timeLimit to allow more time to find a solution
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=120)
        self.problem.solve(solver)
        
        # Check if a solution was found
        if LpStatus[self.problem.status] != "Optimal":
            print(f"No optimal solution found. Status: {LpStatus[self.problem.status]}")
            print("Attempting to relax constraints and solve again...")
            
            # Relax the hard constraints for consecutive weekends to make them soft
            self._relax_constraints()
            
            # Try solving again
            self.problem.solve(solver)
            
            if LpStatus[self.problem.status] != "Optimal":
                print(f"Still no optimal solution after relaxing constraints. Status: {LpStatus[self.problem.status]}")
                return {}
            else:
                print("Found solution after relaxing constraints")
        
        # Extract results
        roster = {}
        for staff in self.staff_names:
            roster[staff] = []
            for day in self.weekend_days:
                for shift in self.shifts:
                    var = self.assignments.get((staff, day, shift))
                    if var and var.value() == 1:
                        roster[staff].append((day, shift))
        
        self.results = roster
        return roster
    
    def get_schedule_by_weekend(self) -> Dict[Tuple[int, str], Dict[str, str]]:
        """
        Organize the schedule by weekend day and shift.
        
        Returns:
            Dict: Schedule in the format {(weekend, day): {shift: staff}}
        """
        if not self.results:
            return {}
            
        schedule = {}
        for staff, assignments in self.results.items():
            for (weekend, day), shift in assignments:
                if (weekend, day) not in schedule:
                    schedule[(weekend, day)] = {}
                schedule[(weekend, day)][shift] = staff
        
        return schedule


if __name__ == "__main__":
    # This is just for testing
    from data_loader import DataLoader
    
    loader = DataLoader("Availability.xls")
    loader.load_data()
    
    model = RosterModel(
        loader.get_staff_names(),
        loader.get_weekend_days(),
        loader.get_availability(),
        loader.get_acc_trained_staff(),
        loader.get_staff_shifts()
    )
    
    model.create_model()
    solution = model.solve()
    
    # Print a sample of the solution
    for staff, assignments in list(solution.items())[:3]:
        print(f"{staff}: {len(assignments)} shifts")
        for day, shift in assignments:
            print(f"  Weekend {day[0]} {day[1]}, {shift} shift")
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
                 staff_shifts: Dict[str, int],
                 allow_consecutive_weekends: bool = True,
                 allow_same_weekend: bool = True,
                 relax_acc_requirement: bool = True,
                 enforce_shift_balance: bool = True,
                 shift_balance_tolerance: int = 1):
        """
        Initialize the RosterModel with necessary data.

        Args:
            staff_names (List[str]): List of all staff names
            weekend_days (List[Tuple[int, str]]): List of weekend days as (weekend_num, day) tuples
            availability (Dict): Staff availability for each day and shift
            acc_trained_staff (Set[str]): Set of ACC-trained staff
            staff_shifts (Dict[str, int]): Number of shifts assigned to each staff
            enforce_shift_balance (bool): Whether to enforce balanced distribution of shift types
            shift_balance_tolerance (int): Allowed deviation from ideal shift type distribution
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
        
        # Flags to control which constraints can be relaxed (based on user options)
        self.allow_consecutive_weekends = allow_consecutive_weekends
        self.allow_same_weekend = allow_same_weekend
        self.relax_acc_requirement = relax_acc_requirement
        self.enforce_shift_balance = enforce_shift_balance
        self.shift_balance_tolerance = shift_balance_tolerance
        
        # Track which constraints were relaxed in the final solution
        self.relaxed_consecutive_weekends = False
        self.relaxed_same_weekend = False
        self.relaxed_acc_requirement = False
        self.relaxed_shift_balance = False
        
        # Track ACC coverage statistics
        self.acc_coverage_stats = None
        
        # Track shift distribution statistics
        self.shift_distribution_stats = None

    def create_model(self, skip_acc_constraints=False, 
                 skip_consecutive_weekend_constraints=False,
                 skip_same_weekend_constraints=False,
                 skip_shift_balance_constraints=False) -> None:
        """Create the linear programming model with specified constraints.
        
        Args:
            skip_acc_constraints: If True, don't add ACC-trained staff constraints
            skip_consecutive_weekend_constraints: If True, don't add consecutive weekend constraints
            skip_same_weekend_constraints: If True, don't add same-weekend day constraints
            skip_shift_balance_constraints: If True, don't add shift type balance constraints
        """
        print("Creating linear programming model for roster optimization...")
        if skip_acc_constraints:
            print("- Skipping ACC-trained staff constraints")
        if skip_consecutive_weekend_constraints:
            print("- Skipping consecutive weekend constraints")
        if skip_same_weekend_constraints:
            print("- Skipping same weekend day constraints")
        if skip_shift_balance_constraints:
            print("- Skipping shift type balance constraints")
            
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
        if not skip_same_weekend_constraints:
            if self.allow_same_weekend:
                # SOFT CONSTRAINT - discouraged but allowed if needed for feasibility
                print("Adding SOFT constraints for same weekend days (penalty-based)")
                same_weekend_penalties = {}

                for staff in self.staff_names:
                    for weekend in self.weekends:
                        sat_day = (weekend, "Saturday")
                        sun_day = (weekend, "Sunday")

                        # Create penalty variable
                        penalty_var = LpVariable(f"same_weekend_penalty_{staff}_{weekend}", 0, 1, LpBinary)
                        same_weekend_penalties[(staff, weekend)] = penalty_var

                        sat_shifts = pulp.lpSum(self.assignments.get((staff, sat_day, shift), 0)
                                              for shift in self.shifts)
                        sun_shifts = pulp.lpSum(self.assignments.get((staff, sun_day, shift), 0)
                                              for shift in self.shifts)

                        # Soft constraint with penalty
                        self.problem += (
                            sat_shifts + sun_shifts <= 1 + penalty_var,
                            f"No_same_weekend_SOFT_{staff}_{weekend}"
                        )

                # Add penalties to objective function
                same_weekend_penalty_sum = pulp.lpSum(same_weekend_penalties.values())
                self.problem += same_weekend_penalty_sum * 2
                print(f"Added {len(same_weekend_penalties)} soft constraints for same weekend days")
            else:
                # HARD CONSTRAINT - absolutely prohibited, no exceptions
                print("Adding HARD constraints for same weekend days (strict enforcement)")
                constraint_count = 0

                for staff in self.staff_names:
                    for weekend in self.weekends:
                        sat_day = (weekend, "Saturday")
                        sun_day = (weekend, "Sunday")

                        sat_shifts = pulp.lpSum(self.assignments.get((staff, sat_day, shift), 0)
                                              for shift in self.shifts)
                        sun_shifts = pulp.lpSum(self.assignments.get((staff, sun_day, shift), 0)
                                              for shift in self.shifts)

                        # Hard constraint - no penalty variable
                        self.problem += (
                            sat_shifts + sun_shifts <= 1,
                            f"No_same_weekend_HARD_{staff}_{weekend}"
                        )
                        constraint_count += 1

                print(f"Added {constraint_count} hard constraints for same weekend days")
        else:
            print("Skipping same weekend day constraints as requested")
        
        # Note: The penalties are now added inside the if not skip_same_weekend_constraints block
        
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
        if not skip_consecutive_weekend_constraints:
            # Calculate ratio of required weekends per staff
            weekends_per_staff = {}
            for staff in self.staff_names:
                target_shifts = self.staff_shifts[staff]
                weekends_per_staff[staff] = target_shifts / 3

            avg_weekends_ratio = sum(weekends_per_staff.values()) / len(weekends_per_staff) / len(self.weekends)

            if self.allow_consecutive_weekends:
                # SOFT CONSTRAINT - use penalties
                print(f"Adding SOFT constraints for consecutive weekends (avg ratio: {avg_weekends_ratio:.2f})")
                consecutive_penalties = {}

                for staff in self.staff_names:
                    sorted_weekends = sorted(self.weekends)
                    for i in range(len(sorted_weekends) - 1):
                        weekend = sorted_weekends[i]
                        next_weekend = sorted_weekends[i + 1]

                        penalty_var = LpVariable(f"consecutive_penalty_{staff}_{weekend}_{next_weekend}", 0, 1, LpBinary)
                        consecutive_penalties[(staff, weekend, next_weekend)] = penalty_var

                        self.problem += (
                            weekend_assignments[(staff, weekend)] +
                            weekend_assignments[(staff, next_weekend)] <= 1 + penalty_var,
                            f"No_consecutive_weekends_SOFT_{staff}_{weekend}_{next_weekend}"
                        )

                consecutive_penalty_sum = pulp.lpSum(consecutive_penalties.values())
                # Use higher weight for lower ratios where it's more achievable
                weight = 5 if avg_weekends_ratio < 0.4 else 3
                self.problem += consecutive_penalty_sum * weight
                print(f"Added {len(consecutive_penalties)} soft constraints for consecutive weekends")
            else:
                # HARD CONSTRAINT - strictly prohibited regardless of ratio
                print(f"Adding HARD constraints for consecutive weekends (strict enforcement, ratio: {avg_weekends_ratio:.2f})")
                constraint_count = 0

                for staff in self.staff_names:
                    sorted_weekends = sorted(self.weekends)
                    for i in range(len(sorted_weekends) - 1):
                        weekend = sorted_weekends[i]
                        next_weekend = sorted_weekends[i + 1]

                        # Hard constraint - no penalty variable
                        self.problem += (
                            weekend_assignments[(staff, weekend)] +
                            weekend_assignments[(staff, next_weekend)] <= 1,
                            f"No_consecutive_weekends_HARD_{staff}_{weekend}_{next_weekend}"
                        )
                        constraint_count += 1

                print(f"Added {constraint_count} hard constraints for consecutive weekends")
        else:
            print("Skipping consecutive weekend constraints as requested")
        
        # Constraint 7: ACC-trained staff on Saturday (soft constraint)
        # We'll use a binary variable to track if an ACC staff is assigned on each Saturday
        acc_coverage = {}
        
        # Only apply ACC coverage constraint if we have ACC-trained staff and the constraint is not skipped
        if not skip_acc_constraints and len(self.acc_trained_staff) > 0:
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
                
                # Try to enforce ACC coverage on Saturdays
                self.problem += (
                    acc_coverage[weekend] == 1,
                    f"ACC_coverage_Saturday_{weekend}"
                )
        elif skip_acc_constraints:
            print("Skipping ACC-trained staff constraints as requested")
        else:
            print("No ACC-trained staff found, skipping ACC coverage constraint")

        # Constraint 8: Shift Type Balance (soft constraints to encourage even distribution)
        if not skip_shift_balance_constraints and self.enforce_shift_balance:
            print(f"Adding shift type balance constraints with tolerance {self.shift_balance_tolerance}")
            shift_balance_penalties = {}
            
            for staff in self.staff_names:
                total_shifts = self.staff_shifts[staff]
                
                # Calculate ideal distribution targets for this staff member
                # For 4 shifts: aim for [1,1,2] or [1,2,1] or [2,1,1] 
                # For 5 shifts: aim for [2,2,1] or [2,1,2] or [1,2,2]
                ideal_per_shift = total_shifts // 3  # Base amount per shift type
                remainder = total_shifts % 3         # Extra shifts to distribute
                
                # Each staff should get at least 'ideal_per_shift' of each type
                # and at most 'ideal_per_shift + 1' of any type
                min_per_shift = max(0, ideal_per_shift)
                max_per_shift = ideal_per_shift + (1 if remainder > 0 else 0) + self.shift_balance_tolerance
                
                # For each shift type, add soft constraints
                for shift_type in self.shifts:
                    # Count assignments for this staff and shift type
                    shift_assignments = pulp.lpSum([
                        self.assignments.get((staff, day, shift_type), 0)
                        for day in self.weekend_days
                    ])
                    
                    # Add penalty variables for under and over assignment
                    under_penalty = LpVariable(f"under_penalty_{staff}_{shift_type}", 0, None, LpBinary)
                    over_penalty = LpVariable(f"over_penalty_{staff}_{shift_type}", 0, None, LpBinary)
                    
                    shift_balance_penalties[(staff, shift_type, 'under')] = under_penalty
                    shift_balance_penalties[(staff, shift_type, 'over')] = over_penalty
                    
                    # Soft constraint for minimum assignments
                    # If shift_assignments < min_per_shift, then under_penalty = 1
                    if min_per_shift > 0:
                        self.problem += (
                            shift_assignments + under_penalty >= min_per_shift,
                            f"Min_shifts_{staff}_{shift_type}"
                        )
                    
                    # Soft constraint for maximum assignments  
                    # If shift_assignments > max_per_shift, then over_penalty = 1
                    self.problem += (
                        shift_assignments - over_penalty <= max_per_shift,
                        f"Max_shifts_{staff}_{shift_type}"
                    )
            
            # Add penalties to objective function
            if shift_balance_penalties:
                penalty_sum = pulp.lpSum(shift_balance_penalties.values())
                self.problem += penalty_sum * 3  # Weight to prioritize shift balance
                print(f"Added {len(shift_balance_penalties)} shift balance penalty variables")
        elif skip_shift_balance_constraints:
            print("Skipping shift type balance constraints as requested")
        else:
            print("Shift balance enforcement disabled")

    def _extract_results(self) -> Dict:
        """Extract the results from the solved model."""
        roster = {}
        for staff in self.staff_names:
            roster[staff] = []
            for day in self.weekend_days:
                for shift in self.shifts:
                    var = self.assignments.get((staff, day, shift))
                    if var and var.value() == 1:
                        roster[staff].append((day, shift))
        
        # Count Saturdays with ACC coverage
        if len(self.acc_trained_staff) > 0:
            saturdays_with_acc = 0
            saturdays_without_acc = []
            total_saturdays = len(self.weekends)
            
            for weekend in self.weekends:
                has_acc = False
                for staff in self.acc_trained_staff:
                    for shift in self.shifts:
                        var = self.assignments.get((staff, (weekend, "Saturday"), shift))
                        if var and var.value() == 1:
                            has_acc = True
                            break
                    if has_acc:
                        break
                        
                if has_acc:
                    saturdays_with_acc += 1
                else:
                    saturdays_without_acc.append(weekend)
            
            print(f"ACC coverage achieved on {saturdays_with_acc}/{total_saturdays} Saturdays")
            if saturdays_without_acc:
                print(f"Saturdays without ACC coverage: {saturdays_without_acc}")
                
            self.acc_coverage_stats = {
                "saturdays_with_acc": saturdays_with_acc,
                "total_saturdays": total_saturdays,
                "saturdays_without_acc": saturdays_without_acc
            }
        
        # Track shift distribution statistics
        shift_distribution = {}
        total_imbalance = 0
        staff_with_imbalance = []
        
        for staff in self.staff_names:
            # Count each shift type for this staff
            shift_counts = {"early": 0, "mid": 0, "late": 0}
            if staff in roster:
                for (day, shift) in roster[staff]:
                    shift_counts[shift] += 1
            
            # Calculate ideal distribution
            total_shifts = self.staff_shifts[staff]
            ideal_per_shift = total_shifts // 3
            remainder = total_shifts % 3
            
            # Calculate balance score (deviation from ideal)
            imbalance_score = 0
            min_shifts = min(shift_counts.values())
            max_shifts = max(shift_counts.values())
            shift_spread = max_shifts - min_shifts
            
            # Check if within tolerance
            max_allowed_spread = 1 + self.shift_balance_tolerance
            is_balanced = shift_spread <= max_allowed_spread
            
            if not is_balanced:
                staff_with_imbalance.append(staff)
                imbalance_score = shift_spread - max_allowed_spread
                total_imbalance += imbalance_score
            
            shift_distribution[staff] = {
                "shift_counts": shift_counts.copy(),
                "total_shifts": total_shifts,
                "ideal_per_shift": ideal_per_shift,
                "remainder": remainder,
                "shift_spread": shift_spread,
                "is_balanced": is_balanced,
                "imbalance_score": imbalance_score
            }
        
        # Calculate overall balance metrics
        total_staff = len(self.staff_names)
        balanced_staff_count = total_staff - len(staff_with_imbalance)
        balance_percentage = (balanced_staff_count / total_staff) * 100 if total_staff > 0 else 0
        
        self.shift_distribution_stats = {
            "staff_distribution": shift_distribution,
            "total_staff": total_staff,
            "balanced_staff_count": balanced_staff_count,
            "staff_with_imbalance": staff_with_imbalance,
            "balance_percentage": balance_percentage,
            "total_imbalance_score": total_imbalance,
            "relaxed": self.relaxed_shift_balance
        }
        
        print(f"Shift balance: {balanced_staff_count}/{total_staff} staff ({balance_percentage:.1f}%) have balanced shifts")
        if staff_with_imbalance:
            print(f"Staff with imbalanced shifts: {staff_with_imbalance}")
        
        self.results = roster
        return roster

    def _relax_acc_constraints(self) -> None:
        """Relax ACC-trained staff constraints to a best-effort approach."""
        # Find and remove the hard ACC coverage constraints
        acc_constraints = [c for c in self.problem.constraints if "ACC_coverage_Saturday" in c]
        for c in acc_constraints:
            del self.problem.constraints[c]
        
        print(f"Converting {len(acc_constraints)} hard ACC constraints to soft constraints")
        
        # Add soft constraints with penalties
        if len(self.acc_trained_staff) > 0:
            acc_penalties = {}
            
            # For each weekend
            for weekend in self.weekends:
                # Create a penalty variable for this weekend
                penalty_var = LpVariable(f"acc_penalty_{weekend}", 0, 1, LpBinary)
                acc_penalties[weekend] = penalty_var
                
                # Find the existing acc_coverage variable for this weekend
                acc_var = None
                for var in self.problem.variables():
                    if var.name == f"acc_coverage_{weekend}":
                        acc_var = var
                        break
                
                if acc_var:
                    # Add soft constraint: acc_coverage + penalty = 1
                    # This means: either we have ACC coverage (acc_coverage=1, penalty=0)
                    # or we don't (acc_coverage=0, penalty=1)
                    self.problem += (
                        acc_var + penalty_var == 1,
                        f"ACC_coverage_soft_{weekend}"
                    )
            
            # Add penalties to objective function with high weight to prioritize ACC coverage
            if acc_penalties:
                penalty_sum = pulp.lpSum(acc_penalties.values())
                self.problem += penalty_sum * 5  # Higher weight than other penalties
                print(f"Added {len(acc_penalties)} soft constraints for ACC coverage")
    
    def _relax_consecutive_weekend_constraints(self) -> None:
        """Relax consecutive weekend constraints."""
        # Find and remove hard consecutive weekend constraints
        consec_constraints = [c for c in self.problem.constraints if "No_consecutive_weekends" in c]
        for c in consec_constraints:
            del self.problem.constraints[c]
            
        print(f"Removed {len(consec_constraints)} consecutive weekend constraints")
        
        # Replace with soft constraints (with penalties)
        if not any("consec_penalty" in str(v) for v in self.problem.variables()):
            print("Adding soft consecutive weekend constraints instead")
            consecutive_penalties = {}
            
            # Get the weekend assignments variables
            weekend_assignments = {}
            for var in self.problem.variables():
                if var.name.startswith("y_"):
                    try:
                        # Parse variable name to get staff and weekend
                        parts = var.name.split("_")
                        if len(parts) >= 3:
                            # The weekend number is always the last element after splitting
                            weekend = int(parts[-1])  # Use the last part as weekend
                            # Reconstruct staff name by joining all middle parts
                            staff = "_".join(parts[1:-1])  # Join all parts between prefix and weekend
                            weekend_assignments[(staff, weekend)] = var
                    except ValueError as e:
                        print(f"Warning: Could not parse variable name '{var.name}': {e}")
                        continue
            
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
                
    def _relax_same_weekend_constraints(self) -> None:
        """Relax constraints that prevent staff from working both days of the same weekend."""
        # Find and remove constraints that prevent staff from working both days of the same weekend
        same_weekend_constraints = [c for c in self.problem.constraints if "No_consecutive_days_in_weekend" in c]
        for c in same_weekend_constraints:
            del self.problem.constraints[c]
            
        print(f"Removed {len(same_weekend_constraints)} same weekend day constraints")

    def solve(self) -> Dict:
        """
        Solve the linear programming model with tiered relaxation approach.
        Uses separate model instances for each relaxation tier to avoid solver errors.
        
        Returns:
            Dict: The staff roster solution or empty dict if no solution found
        """
        # Reset relaxation tracking flags
        self.relaxed_consecutive_weekends = False
        self.relaxed_same_weekend = False
        self.relaxed_acc_requirement = False
        self.relaxed_shift_balance = False
        
        # Tier 1: Solve with strict constraints
        print("Tier 1: Attempting to solve with all constraints...")
        self.problem = None  # Reset problem to ensure fresh start
        self.create_model()  # Create model with all constraints
        
        # Configure solver parameters
        solver = pulp.PULP_CBC_CMD(
            msg=True,             # Show messages for debugging
            timeLimit=120,        # Allow 2 minutes to solve
            gapRel=0.05,         # Accept solutions within 5% of optimal
            options=['presolve on', 'strong branching on', 'cuts on']
        )
        
        try:
            self.problem.solve(solver)
            
            # If optimal solution found, return it
            if LpStatus[self.problem.status] == "Optimal":
                print("Found optimal solution with all constraints satisfied!")
                return self._extract_results()
        except Exception as e:
            print(f"Error in Tier 1 solve: {e}")
        
        # Tier 2: Relax ACC-trained staff requirement if allowed
        if self.relax_acc_requirement:
            print("Tier 2: Creating new model with relaxed ACC staff requirement...")
            try:
                # Create a new model without ACC constraints
                self.problem = None  # Reset problem
                self.create_model(skip_acc_constraints=True)
                
                self.problem.solve(solver)
                
                if LpStatus[self.problem.status] == "Optimal":
                    print("Found solution after relaxing ACC-trained staff requirement")
                    self.relaxed_acc_requirement = True
                    return self._extract_results()
            except Exception as e:
                print(f"Error in Tier 2 solve: {e}")
        else:
            print("ACC constraint relaxation disabled by user - skipping tier 2")
        
        # Tier 3: Relax weekend constraints if allowed
        if self.allow_consecutive_weekends or self.allow_same_weekend:
            print("Tier 3: Creating new model with multiple relaxed constraints...")
            
            # Track which constraints to skip
            skip_acc = self.relax_acc_requirement
            skip_consecutive = self.allow_consecutive_weekends
            skip_same_weekend = self.allow_same_weekend
            
            try:
                # Create a new model with appropriate constraints skipped
                self.problem = None  # Reset problem
                self.create_model(
                    skip_acc_constraints=skip_acc,
                    skip_consecutive_weekend_constraints=skip_consecutive,
                    skip_same_weekend_constraints=skip_same_weekend
                )
                
                self.problem.solve(solver)
                
                if LpStatus[self.problem.status] == "Optimal":
                    print("Found solution with relaxed constraints")
                    if skip_acc:
                        self.relaxed_acc_requirement = True
                    if skip_consecutive:
                        self.relaxed_consecutive_weekends = True
                    if skip_same_weekend:
                        self.relaxed_same_weekend = True
                    return self._extract_results()
            except Exception as e:
                print(f"Error in Tier 3 solve: {e}")
        else:
            print("Weekend constraint relaxation disabled by user - skipping tier 3")
        
        # Tier 4: Relax shift balance constraints if enabled and no solution found yet
        if self.enforce_shift_balance:
            print("Tier 4: Creating new model with relaxed shift balance constraints...")
            
            # Track which constraints to skip (all previously attempted plus shift balance)
            skip_acc = self.relax_acc_requirement
            skip_consecutive = self.allow_consecutive_weekends
            skip_same_weekend = self.allow_same_weekend
            skip_shift_balance = True  # Always try relaxing shift balance in this tier
            
            try:
                # Create a new model with shift balance constraints skipped
                self.problem = None  # Reset problem
                self.create_model(
                    skip_acc_constraints=skip_acc,
                    skip_consecutive_weekend_constraints=skip_consecutive,
                    skip_same_weekend_constraints=skip_same_weekend,
                    skip_shift_balance_constraints=skip_shift_balance
                )
                
                self.problem.solve(solver)
                
                if LpStatus[self.problem.status] == "Optimal":
                    print("Found solution after relaxing shift balance constraints")
                    if skip_acc:
                        self.relaxed_acc_requirement = True
                    if skip_consecutive:
                        self.relaxed_consecutive_weekends = True
                    if skip_same_weekend:
                        self.relaxed_same_weekend = True
                    if skip_shift_balance:
                        self.relaxed_shift_balance = True
                    return self._extract_results()
            except Exception as e:
                print(f"Error in Tier 4 solve: {e}")
        else:
            print("Shift balance enforcement disabled - skipping tier 4")
        
        # If we get here, no solution was found with the allowed relaxations
        attempted_relaxations = []
        if self.relax_acc_requirement:
            attempted_relaxations.append("ACC staff requirements")
        if self.allow_consecutive_weekends:
            attempted_relaxations.append("consecutive weekend constraints")
        if self.allow_same_weekend:
            attempted_relaxations.append("same weekend day constraints")
        if self.enforce_shift_balance:
            attempted_relaxations.append("shift balance constraints")
            
        if attempted_relaxations:
            print(f"No solution found even after relaxing: {', '.join(attempted_relaxations)}")
        else:
            print("No solution found with strict constraints (no relaxations were allowed)")
            
        return {}
        
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
    
    def get_acc_coverage_stats(self) -> Dict:
        """Get statistics on ACC-trained staff coverage on Saturdays.
        
        Returns:
            Dict: Statistics about ACC coverage or None if not applicable
        """
        return self.acc_coverage_stats
    
    def get_shift_distribution_stats(self) -> Dict:
        """Get statistics on shift type distribution balance.
        
        Returns:
            Dict: Statistics about shift distribution balance or None if not applicable
        """
        return self.shift_distribution_stats
    
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
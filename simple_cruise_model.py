import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any

# Only import these when needed in main()
# import matplotlib.pyplot as plt
# import argparse

class Year:
    """
    Simplified class for tracking economic parameters for each simulation year.
    """
    def __init__(self, initial_inflation_rate: float, initial_unemployment_rate: float, 
                 initial_isa_cap: float, initial_isa_threshold: float, num_years: int):
        self.year_count = 1
        self.inflation_rate = initial_inflation_rate
        self.stable_inflation_rate = initial_inflation_rate
        self.unemployment_rate = initial_unemployment_rate
        self.stable_unemployment_rate = initial_unemployment_rate
        self.isa_cap = initial_isa_cap
        self.isa_threshold = initial_isa_threshold
        self.deflator = 1.0
        # Store random seed for reproducibility if needed
        self.random_seed = None

    def next_year(self, random_seed: Optional[int] = None) -> None:
        """
        Advance to the next year and update economic conditions.
        
        Args:
            random_seed: Optional seed for random number generation for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            self.random_seed = random_seed
            
        self.year_count += 1
        
        # More realistic inflation model with bounds
        inflation_shock = np.random.normal(0, 0.01)
        self.inflation_rate = (
            self.stable_inflation_rate * 0.45 + 
            self.inflation_rate * 0.5 + 
            inflation_shock
        )
        # Ensure inflation stays within reasonable bounds
        self.inflation_rate = max(-0.02, min(0.15, self.inflation_rate))
        
        # More realistic unemployment model with bounds
        unemployment_shock = np.random.lognormal(0, 1) / 100
        self.unemployment_rate = (
            self.stable_unemployment_rate * 0.33 + 
            self.unemployment_rate * 0.25 + 
            unemployment_shock
        )
        # Ensure unemployment stays within reasonable bounds
        self.unemployment_rate = max(0.02, min(0.30, self.unemployment_rate))
        
        # Update ISA parameters with inflation
        self.isa_cap *= (1 + self.inflation_rate)
        self.isa_threshold *= (1 + self.inflation_rate)
        self.deflator *= (1 + self.inflation_rate)


class Student:
    """
    Simplified class for tracking student payments without debt seniority.
    """
    def __init__(self, degree, num_years: int):
        self.degree = degree
        self.num_years = num_years
        self.earnings_power = 0.0
        self.earnings = [0.0] * num_years
        self.payments = [0.0] * num_years
        self.real_payments = [0.0] * num_years
        self.is_graduated = False
        self.is_employed = False
        self.is_home = False
        self.years_paid = 0
        self.hit_cap = False
        self.cap_value_when_hit = 0.0  # Store the actual cap value when the student hits it
        self.years_experience = 0
        self.graduation_year = degree.years_to_complete


class Degree:
    """
    Class representing different degree options with associated parameters.
    
    Attributes:
        name: Identifier for the degree type (e.g., 'BA', 'MA', 'ASST', 'NURSE', 'TRADE', 'NA')
        mean_earnings: Average annual earnings for graduates with this degree
        stdev: Standard deviation of earnings for this degree type
        experience_growth: Annual percentage growth in earnings due to experience
        years_to_complete: Number of years required to complete the degree
        leave_labor_force_probability: Probability that a graduate leaves the labor force after graduation
    """
    def __init__(self, name: str, mean_earnings: float, stdev: float, 
                 experience_growth: float, years_to_complete: int, leave_labor_force_probability: float):
        self.name = name
        self.mean_earnings = mean_earnings
        self.stdev = stdev
        self.experience_growth = experience_growth
        self.years_to_complete = years_to_complete
        self.leave_labor_force_probability = leave_labor_force_probability
    
    def __repr__(self) -> str:
        """String representation of the Degree object for debugging."""
        return (f"Degree(name='{self.name}', mean_earnings={self.mean_earnings}, "
                f"stdev={self.stdev}, growth={self.experience_growth:.2%}, "
                f"years={self.years_to_complete}, leave_labor_force_probability={self.leave_labor_force_probability:.1%})")


def _calculate_graduation_delay(base_years_to_complete: int, degree_name: str = '') -> int:
    """
    Calculate a realistic graduation delay based on degree-specific distributions.
    
    For BA and ASST degrees:
    - 50% graduate on time (no delay)
    - 25% graduate 1 year late (50% of remaining)
    - 12.5% graduate 2 years late (50% of remaining)
    - 6.25% graduate 3 years late (50% of remaining)
    - The rest (6.25%) graduate 4 years late
    
    For MA, NURSE, and TRADE degrees:
    - 75% graduate on time (no delay)
    - 20% graduate 1 year late
    - 2.5% graduate 2 years late
    - 2.5% graduate 3 years late
    
    Args:
        base_years_to_complete: The nominal years to complete the degree
        degree_name: The type of degree (BA, MA, ASST, NURSE, TRADE, etc.)
        
    Returns:
        Total years to complete including delay
    """
    rand = np.random.random()
    
    # Apply special distribution for Masters, Nurse, and Trade degrees
    if degree_name in ['MA', 'NURSE', 'TRADE']:
        if rand < 0.75:
            return base_years_to_complete  # Graduate on time
        elif rand < 0.95:
            return base_years_to_complete + 1  # 1 year late
        elif rand < 0.975:
            return base_years_to_complete + 2  # 2 years late
        else:
            return base_years_to_complete + 3  # 3 years late
    else:
        # Default distribution for other degrees (BA, ASST, NA, etc.)
        if rand < 0.5:
            return base_years_to_complete  # Graduate on time
        elif rand < 0.75:
            return base_years_to_complete + 1  # 1 year late
        elif rand < 0.875:
            return base_years_to_complete + 2  # 2 years late
        elif rand < 0.9375:
            return base_years_to_complete + 3  # 3 years late
        else:
            return base_years_to_complete + 4  # 4 years late


def simulate_simple(
    students: List[Student], 
    year: Year, 
    num_years: int, 
    isa_percentage: float, 
    limit_years: int, 
    performance_fee_pct: float = 0.15, 
    gamma: bool = False, 
    price_per_student: float = 30000, 
    new_malengo_fee: bool = False,
    apply_graduation_delay: bool = False
) -> Dict[str, Any]:
    """
    Run a single simulation for the given students over the specified number of years
    with a simple repayment structure.
    
    Parameters:
        students: List of Student objects
        year: Year object representing economic conditions
        num_years: Total number of years to simulate
        isa_percentage: Percentage of income to pay in ISA (as decimal, e.g., 0.14 for 14%)
        limit_years: Maximum number of years to pay the ISA
        performance_fee_pct: Percentage of payments that goes to Malengo (if using old structure)
        gamma: Whether to use gamma distribution instead of normal for earnings
        price_per_student: Cost per student (for new Malengo fee structure)
        new_malengo_fee: Whether to use the new Malengo fee structure (2% of inflation-adjusted investment)
        apply_graduation_delay: Whether to apply realistic graduation delays
    
    Returns:
        Dictionary of simulation results including student data and payment information
    """
    # Initialize arrays to track payments
    total_payments = np.zeros(num_years)
    total_real_payments = np.zeros(num_years)
    malengo_payments = np.zeros(num_years)
    malengo_real_payments = np.zeros(num_years)
    investor_payments = np.zeros(num_years)
    investor_real_payments = np.zeros(num_years)
    
    # Track student status for fee calculations
    student_graduated = np.zeros(len(students), dtype=bool)
    student_hit_cap = np.zeros(len(students), dtype=bool)
    student_is_na = np.zeros(len(students), dtype=bool)
    
    # Apply graduation delay if requested
    if apply_graduation_delay:
        for student in students:
            # Store the original years to complete
            base_years = student.degree.years_to_complete
            # Calculate and apply the delay
            student.graduation_year = _calculate_graduation_delay(base_years, student.degree.name)
    else:
        # Without delay, graduation year is just the degree completion time
        for student in students:
            student.graduation_year = student.degree.years_to_complete
    
    # Simulation loop
    for i in range(num_years):
        # Process each student
        for student_idx, student in enumerate(students):
            # Skip if student hasn't completed degree yet
            if i < student.graduation_year:
                continue
                
            # Handle graduation year
            if i == student.graduation_year:
                _process_graduation(student, student_idx, student_graduated, student_is_na, gamma)
                
            # Determine employment status
            _update_employment_status(student, year)
            
            # Process employed students
            if student.is_employed:
                # Update earnings based on experience
                student.earnings[i] = student.earnings_power * year.deflator * (1 + student.degree.experience_growth) ** student.years_experience
                student.years_experience += 1
                
                # Process payments if earnings exceed threshold
                if student.earnings[i] > year.isa_threshold:
                    student.years_paid += 1
                    
                    # Check if student has reached payment year limit
                    if student.years_paid > limit_years:
                        student_hit_cap[student_idx] = True
                        continue
                        
                    # Skip if student already hit payment cap
                    if student.hit_cap:
                        continue
                    
                    # Calculate payment
                    potential_payment = isa_percentage * student.earnings[i]
                    
                    # Check if payment would exceed cap
                    if (np.sum(student.payments) + potential_payment) > year.isa_cap:
                        student.payments[i] = year.isa_cap - np.sum(student.payments)
                        student.real_payments[i] = student.payments[i] / year.deflator
                        student.hit_cap = True
                        student_hit_cap[student_idx] = True
                        student.cap_value_when_hit = year.isa_cap  # Save the cap value at the time it's hit
                    else:
                        student.payments[i] = potential_payment
                        student.real_payments[i] = potential_payment / year.deflator
                    
                    # Add to total payments
                    total_payments[i] += student.payments[i]
                    total_real_payments[i] += student.real_payments[i]
            else:
                # Reduce experience for unemployed students
                student.years_experience = max(0, student.years_experience - 3)
                
                # Stop Malengo fees if student is unemployed after graduation (except NA degrees)
                if student_graduated[student_idx] and not student_is_na[student_idx]:
                    student_hit_cap[student_idx] = True
        
        # Calculate Malengo's fee for the new structure
        if new_malengo_fee:
            _calculate_malengo_fees(
                students, student_idx, student_graduated, student_hit_cap, 
                student_is_na, price_per_student, year, i, 
                malengo_payments, malengo_real_payments
            )
            
            # Calculate investor payments
            investor_payments[i] = total_payments[i] - malengo_payments[i]
            investor_real_payments[i] = total_real_payments[i] - malengo_real_payments[i]
        else:
            # Old fee structure: Malengo gets a percentage of all payments
            malengo_payments[i] = total_payments[i] * performance_fee_pct
            malengo_real_payments[i] = total_real_payments[i] * performance_fee_pct
            investor_payments[i] = total_payments[i] - malengo_payments[i]
            investor_real_payments[i] = total_real_payments[i] - malengo_real_payments[i]

        # Advance to next year
        year.next_year()

    # Prepare and return results
    data = {
        'Student': students,
        'Degree': [student.degree for student in students],
        'Earnings': [student.earnings for student in students],
        'Payments': [student.payments for student in students],
        'Real_Payments': [student.real_payments for student in students],
        'Total_Payments': total_payments,
        'Total_Real_Payments': total_real_payments,
        'Malengo_Payments': malengo_payments,
        'Malengo_Real_Payments': malengo_real_payments,
        'Investor_Payments': investor_payments,
        'Investor_Real_Payments': investor_real_payments
    }

    return data


def _process_graduation(student: Student, student_idx: int, 
                       student_graduated: np.ndarray, student_is_na: np.ndarray,
                       gamma: bool) -> None:
    """Helper function to process a student's graduation."""
    student.is_graduated = True
    student_graduated[student_idx] = True
    student_is_na[student_idx] = (student.degree.name == 'NA')
    
    # Determine if student returns home
    student.is_home = np.random.binomial(1, student.degree.leave_labor_force_probability) == 1
    
    # Set initial earnings power based on degree
    if gamma:
        student.earnings_power = max(0, np.random.gamma(student.degree.mean_earnings, student.degree.stdev))
    else:
        student.earnings_power = max(0, np.random.normal(student.degree.mean_earnings, student.degree.stdev))
        
    # Adjust earnings for students who return home
    if student.is_home:
        if gamma:
            student.earnings_power = max(0, np.random.gamma(67600/4761, 4761/26))
        else:
            student.earnings_power = max(0, np.random.normal(2600, 690))


def _update_employment_status(student: Student, year: Year) -> None:
    """Helper function to update a student's employment status."""
    # NA degree holders are always unemployed
    if student.degree.name == 'NA':
        student.is_employed = False
    elif year.unemployment_rate < 1:
        student.is_employed = np.random.binomial(1, 1 - year.unemployment_rate) == 1
    else:
        student.is_employed = False


def _calculate_malengo_fees(
    students: List[Student], 
    student_idx: int,
    student_graduated: np.ndarray, 
    student_hit_cap: np.ndarray,
    student_is_na: np.ndarray, 
    price_per_student: float, 
    year: Year, 
    current_year: int,
    malengo_payments: np.ndarray, 
    malengo_real_payments: np.ndarray
) -> None:
    """Helper function to calculate Malengo's fees under the new structure."""
    for student_idx, student in enumerate(students):
        # Malengo gets 2% of inflation-adjusted initial investment only if:
        # 1. Student has graduated
        # 2. Student has not hit any cap (payment cap, years cap, or stopped payments)
        # 3. Student does not have an NA degree
        if (student_graduated[student_idx] and 
            not student_hit_cap[student_idx] and
            not student_is_na[student_idx] and
            student.is_employed):  # Only apply fee if student is employed
            annual_fee = price_per_student * 0.02 * year.deflator
            malengo_payments[current_year] += annual_fee
            malengo_real_payments[current_year] += annual_fee / year.deflator


def run_simple_simulation(
    program_type: str,
    num_students: int,
    num_sims: int,
    scenario: str = 'baseline',
    salary_adjustment_pct: float = 0,
    salary_std_adjustment_pct: float = 0,
    initial_unemployment_rate: float = 0.08,
    initial_inflation_rate: float = 0.02,
    performance_fee_pct: float = 0.15,
    leave_labor_force_probability: float = 0.05,
    ba_pct: float = 0,
    ma_pct: float = 0, 
    asst_pct: float = 0,
    nurse_pct: float = 0,
    na_pct: float = 0,
    trade_pct: float = 0,  
    asst_shift_pct: float = 0,
    # Degree parameters
    ba_salary: float = 41300,
    ba_std: float = 6000,
    ba_growth: float = 0.03,  # Growth rate in decimal form (e.g., 0.03 = 3% growth)
    ma_salary: float = 46709,
    ma_std: float = 6600,
    ma_growth: float = 0.04,  # Growth rate in decimal form (e.g., 0.04 = 4% growth)
    asst_salary: float = 31500,
    asst_std: float = 2800,
    asst_growth: float = 0.005,  # Growth rate in decimal form (e.g., 0.005 = 0.5% growth)
    nurse_salary: float = 40000,
    nurse_std: float = 4000,
    nurse_growth: float = 0.02,  # Growth rate in decimal form (e.g., 0.02 = 2% growth)
    na_salary: float = 2200,
    na_std: float = 640,
    na_growth: float = 0.01,  # Growth rate in decimal form (e.g., 0.01 = 1% growth)
    trade_salary: float = 35000,  
    trade_std: float = 3000,      
    trade_growth: float = 0.02,   # Growth rate in decimal form (e.g., 0.02 = 2% growth)
    asst_shift_salary: float = None,
    asst_shift_std: float = None,
    asst_shift_growth: float = None,
    # ISA parameters
    isa_percentage: Optional[float] = None,
    isa_threshold: float = 27000,
    isa_cap: Optional[float] = None,
    new_malengo_fee: bool = True,
    # Additional parameters
    random_seed: Optional[int] = None,
    num_years: int = 25,
    limit_years: int = 10,
    apply_graduation_delay: bool = False
) -> Dict[str, Any]:
    """
    Run multiple simulations for Uganda, Kenya, or Rwanda program with simplified parameters.
    
    Parameters:
        program_type: 'Uganda', 'Kenya', or 'Rwanda'
        num_students: Number of students to simulate
        num_sims: Number of simulations to run
        scenario: Predefined scenario to use ('baseline', 'conservative', 'optimistic', or 'custom')
        salary_adjustment_pct: Percentage adjustment to average salaries (legacy parameter)
        salary_std_adjustment_pct: Percentage adjustment to salary standard deviations (legacy parameter)
        initial_unemployment_rate: Starting unemployment rate (decimal form, e.g., 0.08 = 8%)
        initial_inflation_rate: Starting inflation rate (decimal form, e.g., 0.02 = 2%)
        performance_fee_pct: Percentage of payments that goes to Malengo (old fee structure)
        leave_labor_force_probability: Probability of a student leaving the labor force after graduation
        ba_pct, ma_pct, asst_pct, nurse_pct, na_pct, trade_pct, asst_shift_pct: Custom degree distribution (if scenario='custom')
        ba_salary, ba_std, ba_growth: Custom parameters for BA degree (growth in decimal form)
        ma_salary, ma_std, ma_growth: Custom parameters for MA degree (growth in decimal form)
        asst_salary, asst_std, asst_growth: Custom parameters for Assistant degree (growth in decimal form)
        nurse_salary, nurse_std, nurse_growth: Custom parameters for Nurse degree (growth in decimal form)
        na_salary, na_std, na_growth: Custom parameters for NA degree (growth in decimal form)
        trade_salary, trade_std, trade_growth: Custom parameters for Trade degree (growth in decimal form)
        asst_shift_salary, asst_shift_std, asst_shift_growth: Custom parameters for ASST_SHIFT degree (defaults to ASST values)
        isa_percentage: Custom ISA percentage (defaults based on program type)
        isa_threshold: Custom ISA threshold
        isa_cap: Custom ISA cap (defaults based on program type)
        new_malengo_fee: Whether to use the new Malengo fee structure
        random_seed: Optional seed for random number generation
        num_years: Total number of years to simulate
        limit_years: Maximum number of years to pay the ISA
        apply_graduation_delay: Whether to apply realistic graduation delays
    
    Returns:
        Dictionary of aggregated results from multiple simulations
    
    Note:
        All growth rates should be provided in decimal form (e.g., 0.03 for 3% growth)
        rather than as percentages.
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Set default ISA parameters based on program type if not provided
    if isa_percentage is None:
        if program_type == 'Uganda':
            isa_percentage = 0.14
        elif program_type == 'Kenya':
            isa_percentage = 0.12
        elif program_type == 'Rwanda':
            isa_percentage = 0.12  
        else:
            isa_percentage = 0.12  # Default
    
    if isa_cap is None:
        if program_type == 'Uganda':
            isa_cap = 72500
        elif program_type == 'Kenya':
            isa_cap = 49950
        elif program_type == 'Rwanda':
            isa_cap = 45000  
        else:
            isa_cap = 50000  # Default
    
    # Program-specific parameters
    if program_type == 'Uganda':
        price_per_student = 29000
    elif program_type == 'Kenya':
        price_per_student = 16650
    elif program_type == 'Rwanda':
        price_per_student = 16650  
    else:
        raise ValueError("Program type must be 'Uganda', 'Kenya', or 'Rwanda'")
    
    # Set default values for asst_shift if not provided
    if asst_shift_salary is None:
        asst_shift_salary = asst_salary
    if asst_shift_std is None:
        asst_shift_std = asst_std
    if asst_shift_growth is None:
        asst_shift_growth = asst_growth
    
    # Define all possible degree types with custom parameters
    base_degrees = _create_degree_definitions(
        ba_salary, ba_std, ba_growth,
        ma_salary, ma_std, ma_growth,
        asst_salary, asst_std, asst_growth,
        nurse_salary, nurse_std, nurse_growth,
        na_salary, na_std, na_growth,
        trade_salary, trade_std, trade_growth,
        asst_shift_salary, asst_shift_std, asst_shift_growth
    )
    
    # Set up degree distribution based on scenario
    degrees, probs = _setup_degree_distribution(
        scenario, program_type, base_degrees, leave_labor_force_probability,
        ba_pct, ma_pct, asst_pct, nurse_pct, na_pct, trade_pct, asst_shift_pct
    )
    
    # Prepare containers for results
    total_payment = {}
    investor_payment = {}
    malengo_payment = {}
    
    # Add containers for nominal (non-inflation-adjusted) payments
    nominal_total_payment = {}
    nominal_investor_payment = {}
    nominal_malengo_payment = {}
    
    df_list = []
    
    # Track statistics across simulations
    employment_stats = []
    ever_employed_stats = []  # Add this line to track ever employed rates
    repayment_stats = []
    cap_stats = []
    
    # Calculate total investment
    total_investment = num_students * price_per_student
    
    # Run multiple simulations
    for trial in range(num_sims):
        # Initialize year class with a unique seed for each trial if random_seed is provided
        trial_seed = random_seed + trial if random_seed is not None else None
        year = Year(
            initial_inflation_rate=initial_inflation_rate,
            initial_unemployment_rate=initial_unemployment_rate,
            initial_isa_cap=isa_cap,
            initial_isa_threshold=isa_threshold,
            num_years=num_years
        )
        
        # Assign degrees to each student
        students = _create_students(num_students, degrees, probs, num_years)
        
        # Run the simulation and store results
        sim_results = simulate_simple(
            students=students,
            year=year,
            num_years=num_years,
            limit_years=limit_years,
            isa_percentage=isa_percentage,
            performance_fee_pct=performance_fee_pct,
            gamma=False,
            price_per_student=price_per_student,
            new_malengo_fee=new_malengo_fee,
            apply_graduation_delay=apply_graduation_delay
        )
        df_list.append(sim_results)
        
        # Calculate and store statistics for this simulation
        stats = _calculate_simulation_statistics(
            students, num_students, num_years, limit_years
        )
        
        employment_stats.append(stats['employment_rate'])
        ever_employed_stats.append(stats['ever_employed_rate'])  # Add this line to collect ever employed rates
        repayment_stats.append(stats['repayment_rate'])
        cap_stats.append(stats['cap_stats'])
        
        # Extract and store real payments (inflation-adjusted)
        total_payment[trial] = np.sum(pd.DataFrame([student.real_payments for student in students]), axis=0)
        investor_payment[trial] = df_list[trial]['Investor_Real_Payments']
        malengo_payment[trial] = df_list[trial]['Malengo_Real_Payments']
        
        # Extract and store nominal payments (not adjusted for inflation)
        nominal_total_payment[trial] = np.sum(pd.DataFrame([student.payments for student in students]), axis=0)
        nominal_investor_payment[trial] = df_list[trial]['Investor_Payments']
        nominal_malengo_payment[trial] = df_list[trial]['Malengo_Payments']
    
    # Calculate summary statistics
    summary_stats = _calculate_summary_statistics(
        total_payment, investor_payment, malengo_payment,
        nominal_total_payment, nominal_investor_payment, nominal_malengo_payment,
        total_investment, degrees, probs, num_students,
        employment_stats, ever_employed_stats, repayment_stats, cap_stats
    )
    
    # Add simulation parameters to results
    summary_stats.update({
        'program_type': program_type,
        'total_investment': total_investment,
        'price_per_student': price_per_student,
        'isa_percentage': isa_percentage,
        'isa_threshold': isa_threshold,
        'isa_cap': isa_cap,
        'performance_fee_pct': performance_fee_pct,
        'leave_labor_force_probability': leave_labor_force_probability,
        'custom_degrees': True
    })
    
    return summary_stats


def _create_degree_definitions(
    ba_salary: float, ba_std: float, ba_growth: float,
    ma_salary: float, ma_std: float, ma_growth: float,
    asst_salary: float, asst_std: float, asst_growth: float,
    nurse_salary: float, nurse_std: float, nurse_growth: float,
    na_salary: float, na_std: float, na_growth: float,
    trade_salary: float, trade_std: float, trade_growth: float,
    asst_shift_salary: float, asst_shift_std: float, asst_shift_growth: float
) -> Dict[str, Dict[str, Any]]:
    """
    Helper function to create degree definitions.
    
    Note:
        All growth rates should be provided in decimal form (e.g., 0.03 for 3% growth)
        and will be used directly without further conversion.
    """
    return {
        'BA': {
            'name': 'BA',
            'mean_earnings': ba_salary,
            'stdev': ba_std,
            'experience_growth': ba_growth,  # Growth rate already in decimal form
            'years_to_complete': 4
        },
        'MA': {
            'name': 'MA',
            'mean_earnings': ma_salary,
            'stdev': ma_std,
            'experience_growth': ma_growth,  # Growth rate already in decimal form
            'years_to_complete': 6
        },
        'ASST': {
            'name': 'ASST',
            'mean_earnings': asst_salary,
            'stdev': asst_std,
            'experience_growth': asst_growth,  # Growth rate already in decimal form
            'years_to_complete': 3
        },
        'ASST_SHIFT': {
            'name': 'ASST_SHIFT',
            'mean_earnings': asst_shift_salary,
            'stdev': asst_shift_std,
            'experience_growth': asst_shift_growth,  # Growth rate already in decimal form
            'years_to_complete': 6
        },
        'NURSE': {
            'name': 'NURSE',
            'mean_earnings': nurse_salary,
            'stdev': nurse_std,
            'experience_growth': nurse_growth,  # Growth rate already in decimal form
            'years_to_complete': 4
        },
        'NA': {
            'name': 'NA',
            'mean_earnings': na_salary,
            'stdev': na_std,
            'experience_growth': na_growth,  # Growth rate already in decimal form
            'years_to_complete': 4
        },
        'TRADE': {
            'name': 'TRADE',
            'mean_earnings': trade_salary,
            'stdev': trade_std,
            'experience_growth': trade_growth,  # Growth rate already in decimal form
            'years_to_complete': 3
        }
    }


def _setup_degree_distribution(
    scenario: str, 
    program_type: str, 
    base_degrees: Dict[str, Dict[str, Any]], 
    leave_labor_force_probability: float,
    ba_pct: float, 
    ma_pct: float, 
    asst_pct: float, 
    nurse_pct: float, 
    na_pct: float,
    trade_pct: float,
    asst_shift_pct: float = 0
) -> Tuple[List[Degree], List[float]]:
    """Helper function to set up degree distribution based on scenario."""
    degrees = []
    probs = []
    
    # Create a copy of base_degrees to modify
    modified_degrees = {k: v.copy() for k, v in base_degrees.items()}
    
    # For Kenya and Rwanda programs, add 1 year to all degrees for language training
    if program_type == 'Kenya' or program_type == 'Rwanda':
        for degree_type in modified_degrees:
            modified_degrees[degree_type]['years_to_complete'] += 1
    
    if scenario == 'baseline':
        if program_type == 'Uganda':
            degrees = [
                Degree(
                    name=modified_degrees['BA']['name'],
                    mean_earnings=modified_degrees['BA']['mean_earnings'],
                    stdev=modified_degrees['BA']['stdev'],
                    experience_growth=modified_degrees['BA']['experience_growth'],
                    years_to_complete=modified_degrees['BA']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['MA']['name'],
                    mean_earnings=modified_degrees['MA']['mean_earnings'],
                    stdev=modified_degrees['MA']['stdev'],
                    experience_growth=modified_degrees['MA']['experience_growth'],
                    years_to_complete=modified_degrees['MA']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['ASST_SHIFT']['name'],
                    mean_earnings=modified_degrees['ASST_SHIFT']['mean_earnings'],
                    stdev=modified_degrees['ASST_SHIFT']['stdev'],
                    experience_growth=modified_degrees['ASST_SHIFT']['experience_growth'],
                    years_to_complete=modified_degrees['ASST_SHIFT']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['NA']['name'],
                    mean_earnings=modified_degrees['NA']['mean_earnings'],
                    stdev=modified_degrees['NA']['stdev'],
                    experience_growth=modified_degrees['NA']['experience_growth'],
                    years_to_complete=modified_degrees['NA']['years_to_complete'],
                    leave_labor_force_probability=1  
                )
            ]
            probs = [0.45, 0.24, 0.27, 0.04]  
        elif program_type == 'Kenya':
            # Split ASST into ASST and ASST_SHIFT (33% of ASST becomes ASST_SHIFT)
            asst_regular_pct = 0.60 * 0.67  # 67% of original ASST percentage
            asst_shift_pct = 0.60 * 0.33    # 33% of original ASST percentage
            
            degrees = [
                Degree(
                    name=modified_degrees['NURSE']['name'],
                    mean_earnings=modified_degrees['NURSE']['mean_earnings'],
                    stdev=modified_degrees['NURSE']['stdev'],
                    experience_growth=modified_degrees['NURSE']['experience_growth'],
                    years_to_complete=modified_degrees['NURSE']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['ASST']['name'],
                    mean_earnings=modified_degrees['ASST']['mean_earnings'],
                    stdev=modified_degrees['ASST']['stdev'],
                    experience_growth=modified_degrees['ASST']['experience_growth'],
                    years_to_complete=modified_degrees['ASST']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['ASST_SHIFT']['name'],
                    mean_earnings=modified_degrees['ASST_SHIFT']['mean_earnings'],
                    stdev=modified_degrees['ASST_SHIFT']['stdev'],
                    experience_growth=modified_degrees['ASST_SHIFT']['experience_growth'],
                    years_to_complete=modified_degrees['ASST_SHIFT']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['NA']['name'],
                    mean_earnings=modified_degrees['NA']['mean_earnings'],
                    stdev=modified_degrees['NA']['stdev'],
                    experience_growth=modified_degrees['NA']['experience_growth'],
                    years_to_complete=modified_degrees['NA']['years_to_complete'],
                    leave_labor_force_probability=1  
                )
            ]
            probs = [0.25, asst_regular_pct, asst_shift_pct, 0.15]
        elif program_type == 'Rwanda':
            # Split ASST into ASST and ASST_SHIFT (33% of ASST becomes ASST_SHIFT)
            asst_regular_pct = 0.40 * 0.67  # 67% of original ASST percentage
            asst_shift_pct = 0.40 * 0.33    # 33% of original ASST percentage
           
            degrees = [
                Degree(
                    name=modified_degrees['TRADE']['name'],
                    mean_earnings=modified_degrees['TRADE']['mean_earnings'],
                    stdev=modified_degrees['TRADE']['stdev'],
                    experience_growth=modified_degrees['TRADE']['experience_growth'],
                    years_to_complete=modified_degrees['TRADE']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['ASST']['name'],
                    mean_earnings=modified_degrees['ASST']['mean_earnings'],
                    stdev=modified_degrees['ASST']['stdev'],
                    experience_growth=modified_degrees['ASST']['experience_growth'],
                    years_to_complete=modified_degrees['ASST']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['ASST_SHIFT']['name'],
                    mean_earnings=modified_degrees['ASST_SHIFT']['mean_earnings'],
                    stdev=modified_degrees['ASST_SHIFT']['stdev'],
                    experience_growth=modified_degrees['ASST_SHIFT']['experience_growth'],
                    years_to_complete=modified_degrees['ASST_SHIFT']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['NA']['name'],
                    mean_earnings=modified_degrees['NA']['mean_earnings'],
                    stdev=modified_degrees['NA']['stdev'],
                    experience_growth=modified_degrees['NA']['experience_growth'],
                    years_to_complete=modified_degrees['NA']['years_to_complete'],
                    leave_labor_force_probability=1  
                )
            ]
            probs = [0.40, asst_regular_pct, asst_shift_pct, 0.20]
    
    elif scenario == 'conservative':
        if program_type == 'Uganda':
            # Uganda conservative: Updated (32% BA, 11% MA, 42% ASST, 15% NA)
            degrees = [
                Degree(
                    name=modified_degrees['BA']['name'],
                    mean_earnings=modified_degrees['BA']['mean_earnings'],
                    stdev=modified_degrees['BA']['stdev'],
                    experience_growth=modified_degrees['BA']['experience_growth'],
                    years_to_complete=modified_degrees['BA']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['MA']['name'],
                    mean_earnings=modified_degrees['MA']['mean_earnings'],
                    stdev=modified_degrees['MA']['stdev'],
                    experience_growth=modified_degrees['MA']['experience_growth'],
                    years_to_complete=modified_degrees['MA']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['ASST_SHIFT']['name'],
                    mean_earnings=modified_degrees['ASST_SHIFT']['mean_earnings'],
                    stdev=modified_degrees['ASST_SHIFT']['stdev'],
                    experience_growth=modified_degrees['ASST_SHIFT']['experience_growth'],
                    years_to_complete=modified_degrees['ASST_SHIFT']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['NA']['name'],
                    mean_earnings=modified_degrees['NA']['mean_earnings'],
                    stdev=modified_degrees['NA']['stdev'],
                    experience_growth=modified_degrees['NA']['experience_growth'],
                    years_to_complete=modified_degrees['NA']['years_to_complete'],
                    leave_labor_force_probability=1  
                )
            ]
            probs = [0.32, 0.11, 0.42, 0.15]
        elif program_type == 'Kenya':
            # Split ASST into ASST and ASST_SHIFT (33% of ASST becomes ASST_SHIFT)
            asst_regular_pct = 0.50 * 0.67  # 67% of original ASST percentage
            asst_shift_pct = 0.50 * 0.33    # 33% of original ASST percentage
            
            degrees = [
                Degree(
                    name=modified_degrees['NURSE']['name'],
                    mean_earnings=modified_degrees['NURSE']['mean_earnings'],
                    stdev=modified_degrees['NURSE']['stdev'],
                    experience_growth=modified_degrees['NURSE']['experience_growth'],
                    years_to_complete=modified_degrees['NURSE']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['ASST']['name'],
                    mean_earnings=modified_degrees['ASST']['mean_earnings'],
                    stdev=modified_degrees['ASST']['stdev'],
                    experience_growth=modified_degrees['ASST']['experience_growth'],
                    years_to_complete=modified_degrees['ASST']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['ASST_SHIFT']['name'],
                    mean_earnings=modified_degrees['ASST_SHIFT']['mean_earnings'],
                    stdev=modified_degrees['ASST_SHIFT']['stdev'],
                    experience_growth=modified_degrees['ASST_SHIFT']['experience_growth'],
                    years_to_complete=modified_degrees['ASST_SHIFT']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['NA']['name'],
                    mean_earnings=modified_degrees['NA']['mean_earnings'],
                    stdev=modified_degrees['NA']['stdev'],
                    experience_growth=modified_degrees['NA']['experience_growth'],
                    years_to_complete=modified_degrees['NA']['years_to_complete'],
                    leave_labor_force_probability=1  
                )
            ]
            probs = [0.20, asst_regular_pct, asst_shift_pct, 0.30]
        elif program_type == 'Rwanda':
            # Split ASST into ASST and ASST_SHIFT (33% of ASST becomes ASST_SHIFT)
            asst_regular_pct = 0.40 * 0.67  # 67% of original ASST percentage
            asst_shift_pct = 0.40 * 0.33    # 33% of original ASST percentage

            degrees = [
                Degree(
                    name=modified_degrees['TRADE']['name'],
                    mean_earnings=modified_degrees['TRADE']['mean_earnings'],
                    stdev=modified_degrees['TRADE']['stdev'],
                    experience_growth=modified_degrees['TRADE']['experience_growth'],
                    years_to_complete=modified_degrees['TRADE']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['ASST']['name'],
                    mean_earnings=modified_degrees['ASST']['mean_earnings'],
                    stdev=modified_degrees['ASST']['stdev'],
                    experience_growth=modified_degrees['ASST']['experience_growth'],
                    years_to_complete=modified_degrees['ASST']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['ASST_SHIFT']['name'],
                    mean_earnings=modified_degrees['ASST_SHIFT']['mean_earnings'],
                    stdev=modified_degrees['ASST_SHIFT']['stdev'],
                    experience_growth=modified_degrees['ASST_SHIFT']['experience_growth'],
                    years_to_complete=modified_degrees['ASST_SHIFT']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['NA']['name'],
                    mean_earnings=modified_degrees['NA']['mean_earnings'],
                    stdev=modified_degrees['NA']['stdev'],
                    experience_growth=modified_degrees['NA']['experience_growth'],
                    years_to_complete=modified_degrees['NA']['years_to_complete'],
                    leave_labor_force_probability=1  
                )
            ]
            probs = [0.2, asst_regular_pct, asst_shift_pct, 0.4]
    
    elif scenario == 'optimistic':
        if program_type == 'Uganda':
            # Uganda optimistic scenario (63% BA, 33% MA, 2.5% ASST, 1.5% NA) - reduced NA by 1%
            degrees = [
                Degree(
                    name=modified_degrees['BA']['name'],
                    mean_earnings=modified_degrees['BA']['mean_earnings'],
                    stdev=modified_degrees['BA']['stdev'],
                    experience_growth=modified_degrees['BA']['experience_growth'],
                    years_to_complete=modified_degrees['BA']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['MA']['name'],
                    mean_earnings=modified_degrees['MA']['mean_earnings'],
                    stdev=modified_degrees['MA']['stdev'],
                    experience_growth=modified_degrees['MA']['experience_growth'],
                    years_to_complete=modified_degrees['MA']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['ASST_SHIFT']['name'],
                    mean_earnings=modified_degrees['ASST_SHIFT']['mean_earnings'],
                    stdev=modified_degrees['ASST_SHIFT']['stdev'],
                    experience_growth=modified_degrees['ASST_SHIFT']['experience_growth'],
                    years_to_complete=modified_degrees['ASST_SHIFT']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['NA']['name'],
                    mean_earnings=modified_degrees['NA']['mean_earnings'],
                    stdev=modified_degrees['NA']['stdev'],
                    experience_growth=modified_degrees['NA']['experience_growth'],
                    years_to_complete=modified_degrees['NA']['years_to_complete'],
                    leave_labor_force_probability=1  
                )
            ]
            probs = [0.63, 0.33, 0.025, 0.015]
        elif program_type == 'Kenya':
            # Split ASST into ASST and ASST_SHIFT (33% of ASST becomes ASST_SHIFT)
            asst_regular_pct = 0.40 * 0.67  # 67% of original ASST percentage
            asst_shift_pct = 0.40 * 0.33    # 33% of original ASST percentage
            
            degrees = [
                Degree(
                    name=modified_degrees['NURSE']['name'],
                    mean_earnings=modified_degrees['NURSE']['mean_earnings'],
                    stdev=modified_degrees['NURSE']['stdev'],
                    experience_growth=modified_degrees['NURSE']['experience_growth'],
                    years_to_complete=modified_degrees['NURSE']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['ASST']['name'],
                    mean_earnings=modified_degrees['ASST']['mean_earnings'],
                    stdev=modified_degrees['ASST']['stdev'],
                    experience_growth=modified_degrees['ASST']['experience_growth'],
                    years_to_complete=modified_degrees['ASST']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['ASST_SHIFT']['name'],
                    mean_earnings=modified_degrees['ASST_SHIFT']['mean_earnings'],
                    stdev=modified_degrees['ASST_SHIFT']['stdev'],
                    experience_growth=modified_degrees['ASST_SHIFT']['experience_growth'],
                    years_to_complete=modified_degrees['ASST_SHIFT']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                )
            ]
            probs = [0.60, asst_regular_pct, asst_shift_pct]
        elif program_type == 'Rwanda':
            # Trade optimistic scenario - Split ASST into ASST and ASST_SHIFT (33% of ASST becomes ASST_SHIFT)
            asst_regular_pct = 0.35 * 0.67  # 67% of original ASST percentage
            asst_shift_pct = 0.35 * 0.33    # 33% of original ASST percentage
            
            degrees = [
                Degree(
                    name=modified_degrees['TRADE']['name'],
                    mean_earnings=modified_degrees['TRADE']['mean_earnings'],
                    stdev=modified_degrees['TRADE']['stdev'],
                    experience_growth=modified_degrees['TRADE']['experience_growth'],
                    years_to_complete=modified_degrees['TRADE']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['ASST']['name'],
                    mean_earnings=modified_degrees['ASST']['mean_earnings'],
                    stdev=modified_degrees['ASST']['stdev'],
                    experience_growth=modified_degrees['ASST']['experience_growth'],
                    years_to_complete=modified_degrees['ASST']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['ASST_SHIFT']['name'],
                    mean_earnings=modified_degrees['ASST_SHIFT']['mean_earnings'],
                    stdev=modified_degrees['ASST_SHIFT']['stdev'],
                    experience_growth=modified_degrees['ASST_SHIFT']['experience_growth'],
                    years_to_complete=modified_degrees['ASST_SHIFT']['years_to_complete'],
                    leave_labor_force_probability=leave_labor_force_probability
                ),
                Degree(
                    name=modified_degrees['NA']['name'],
                    mean_earnings=modified_degrees['NA']['mean_earnings'],
                    stdev=modified_degrees['NA']['stdev'],
                    experience_growth=modified_degrees['NA']['experience_growth'],
                    years_to_complete=modified_degrees['NA']['years_to_complete'],
                    leave_labor_force_probability=1  
                )
            ]
            probs = [0.60, asst_regular_pct, asst_shift_pct, 0.05]
    
    elif scenario == 'custom':
        # Use user-provided degree distribution
        if ba_pct > 0:
            degrees.append(Degree(
                name=modified_degrees['BA']['name'],
                mean_earnings=modified_degrees['BA']['mean_earnings'],
                stdev=modified_degrees['BA']['stdev'],
                experience_growth=modified_degrees['BA']['experience_growth'],
                years_to_complete=modified_degrees['BA']['years_to_complete'],
                leave_labor_force_probability=leave_labor_force_probability
            ))
            probs.append(ba_pct)
        
        if ma_pct > 0:
            degrees.append(Degree(
                name=modified_degrees['MA']['name'],
                mean_earnings=modified_degrees['MA']['mean_earnings'],
                stdev=modified_degrees['MA']['stdev'],
                experience_growth=modified_degrees['MA']['experience_growth'],
                years_to_complete=modified_degrees['MA']['years_to_complete'],
                leave_labor_force_probability=leave_labor_force_probability
            ))
            probs.append(ma_pct)
        
        if asst_pct > 0:
            degrees.append(Degree(
                name=modified_degrees['ASST']['name'],
                mean_earnings=modified_degrees['ASST']['mean_earnings'],
                stdev=modified_degrees['ASST']['stdev'],
                experience_growth=modified_degrees['ASST']['experience_growth'],
                years_to_complete=modified_degrees['ASST']['years_to_complete'],
                leave_labor_force_probability=leave_labor_force_probability
            ))
            probs.append(asst_pct)
            
        if asst_shift_pct > 0:
            degrees.append(Degree(
                name=modified_degrees['ASST_SHIFT']['name'],
                mean_earnings=modified_degrees['ASST_SHIFT']['mean_earnings'],
                stdev=modified_degrees['ASST_SHIFT']['stdev'],
                experience_growth=modified_degrees['ASST_SHIFT']['experience_growth'],
                years_to_complete=modified_degrees['ASST_SHIFT']['years_to_complete'],
                leave_labor_force_probability=leave_labor_force_probability
            ))
            probs.append(asst_shift_pct)

        if nurse_pct > 0:
            degrees.append(Degree(
                name=modified_degrees['NURSE']['name'],
                mean_earnings=modified_degrees['NURSE']['mean_earnings'],
                stdev=modified_degrees['NURSE']['stdev'],
                experience_growth=modified_degrees['NURSE']['experience_growth'],
                years_to_complete=modified_degrees['NURSE']['years_to_complete'],
                leave_labor_force_probability=leave_labor_force_probability
            ))
            probs.append(nurse_pct)
        
        if na_pct > 0:
            degrees.append(Degree(
                name=modified_degrees['NA']['name'],
                mean_earnings=modified_degrees['NA']['mean_earnings'],
                stdev=modified_degrees['NA']['stdev'],
                experience_growth=modified_degrees['NA']['experience_growth'],
                years_to_complete=modified_degrees['NA']['years_to_complete'],
                leave_labor_force_probability=1  # NA degree has fixed high leave labor force probability
            ))
            probs.append(na_pct)
        
        if trade_pct > 0:
            degrees.append(Degree(
                name=modified_degrees['TRADE']['name'],
                mean_earnings=modified_degrees['TRADE']['mean_earnings'],
                stdev=modified_degrees['TRADE']['stdev'],
                experience_growth=modified_degrees['TRADE']['experience_growth'],
                years_to_complete=modified_degrees['TRADE']['years_to_complete'],
                leave_labor_force_probability=leave_labor_force_probability  # Use the user-provided leave_labor_force_probability, not a fixed value
            ))
            probs.append(trade_pct)
        
        # Normalize probabilities to ensure they sum to 1
        if sum(probs) > 0:
            probs = [p/sum(probs) for p in probs]
        else:
            raise ValueError("At least one degree type must have a non-zero percentage")
    else:
        raise ValueError("Invalid scenario. Must be 'baseline', 'conservative', 'optimistic', or 'custom'")
    
    return degrees, probs


def _create_students(
    num_students: int, 
    degrees: List[Degree], 
    probs: List[float], 
    num_years: int
) -> List[Student]:
    """Helper function to create and assign degrees to students."""
    # Assign degrees to each student
    test_array = np.array([np.random.multinomial(1, probs) for _ in range(num_students)])
    degree_labels = np.array(degrees)[test_array.argmax(axis=1)]
    
    # Create student objects
    students = []
    for i in range(num_students):
        students.append(Student(degree_labels[i], num_years))
    
    return students


def _calculate_simulation_statistics(
    students: List[Student], 
    num_students: int, 
    num_years: int, 
    limit_years: int
) -> Dict[str, Any]:
    """Helper function to calculate statistics for a single simulation."""
    # Track statistics for this simulation
    students_employed = 0
    students_made_payments = 0
    students_hit_payment_cap = 0
    students_hit_years_cap = 0
    students_hit_no_cap = 0
    
    # Track repayments by category
    total_repayment_cap_hit = 0
    total_years_cap_hit = 0
    total_no_cap_hit = 0
    
    # Track annual employment rates
    annual_employment_rates = []
    post_graduation_years = 0
    
    # For tracking the actual cap values 
    cap_values = []
    
    # Count students in different categories
    for student in students:
        # Check if student was ever employed
        was_employed = False
        employment_periods = 0
        post_grad_periods = 0
        
        for i in range(num_years):
            if i >= student.degree.years_to_complete:
                post_grad_periods += 1
                if student.earnings[i] > 0:
                    was_employed = True
                    employment_periods += 1
        
        # Calculate this student's employment rate
        if post_grad_periods > 0:
            student_employment_rate = employment_periods / post_grad_periods
        else:
            student_employment_rate = 0
            
        # Add to total employment rate calculation
        if post_grad_periods > 0:
            annual_employment_rates.append(student_employment_rate)
            post_graduation_years += post_grad_periods
        
        if was_employed:
            students_employed += 1
            
        # Check if student made any payments
        made_payment = sum(student.payments) > 0
        if made_payment:
            students_made_payments += 1
            
        # Check which cap (if any) the student hit
        if student.hit_cap:
            students_hit_payment_cap += 1
            total_repayment_cap_hit += sum(student.real_payments)
            cap_values.append(student.cap_value_when_hit)  # Store the cap value when hit
        elif student.years_paid >= limit_years:
            students_hit_years_cap += 1
            total_years_cap_hit += sum(student.real_payments)
        else:
            if made_payment:  # Only count students who made payments
                students_hit_no_cap += 1
                total_no_cap_hit += sum(student.real_payments)
    
    # Calculate averages (with safe division)
    avg_repayment_cap_hit = total_repayment_cap_hit / max(1, students_hit_payment_cap)
    avg_repayment_years_hit = total_years_cap_hit / max(1, students_hit_years_cap)
    avg_repayment_no_cap = total_no_cap_hit / max(1, students_hit_no_cap)
    
    # Calculate average annual employment rate
    avg_annual_employment_rate = sum(annual_employment_rates) / max(1, len(annual_employment_rates))
    
    # Calculate average cap value (if any students hit it)
    avg_cap_value = sum(cap_values) / max(1, len(cap_values)) if cap_values else 0
    
    # Return statistics
    return {
        'employment_rate': avg_annual_employment_rate,  # Changed to average annual employment rate
        'ever_employed_rate': students_employed / num_students,  # Keep track of ever employed rate
        'repayment_rate': students_made_payments / num_students,
        'cap_stats': {
            'payment_cap_count': students_hit_payment_cap,
            'years_cap_count': students_hit_years_cap,
            'no_cap_count': students_hit_no_cap,
            'payment_cap_pct': students_hit_payment_cap / num_students,
            'years_cap_pct': students_hit_years_cap / num_students,
            'no_cap_pct': students_hit_no_cap / num_students,
            'avg_repayment_cap_hit': avg_repayment_cap_hit,
            'avg_repayment_years_hit': avg_repayment_years_hit,
            'avg_repayment_no_cap': avg_repayment_no_cap,
            'avg_cap_value': avg_cap_value
        }
    }


def _calculate_summary_statistics(
    total_payment: Dict[int, np.ndarray],
    investor_payment: Dict[int, np.ndarray],
    malengo_payment: Dict[int, np.ndarray],
    nominal_total_payment: Dict[int, np.ndarray],
    nominal_investor_payment: Dict[int, np.ndarray],
    nominal_malengo_payment: Dict[int, np.ndarray],
    total_investment: float,
    degrees: List[Degree],
    probs: List[float],
    num_students: int,
    employment_stats: List[float],
    ever_employed_stats: List[float],
    repayment_stats: List[float],
    cap_stats: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Helper function to calculate summary statistics across all simulations."""
    # Calculate summary statistics for real (inflation-adjusted) payments
    payments_df = pd.DataFrame(total_payment)
    average_total_payment = np.sum(payments_df, axis=0).mean()
    
    # Calculate weighted average duration (avoiding division by zero)
    payment_sums = np.sum(payments_df, axis=0)
    if np.any(payment_sums > 0):
        # Convert to numpy arrays to avoid pandas indexing issues
        payments_np = payments_df.to_numpy()
        payment_sums_np = payment_sums.to_numpy()
        
        # Create weights matrix
        weights = np.zeros_like(payments_np)
        for i in range(len(payment_sums_np)):
            if payment_sums_np[i] > 0:
                weights[:, i] = payments_np[:, i] / payment_sums_np[i]
        
        # Calculate weighted average
        years = np.arange(1, len(payments_df) + 1)
        weighted_durations = np.sum(years[:, np.newaxis] * weights, axis=0)
        average_duration = np.mean(weighted_durations)
    else:
        average_duration = 0
    
    # Calculate real IRR (safely handle negative values)
    if average_total_payment > 0 and average_duration > 0:
        IRR = np.log(max(1, average_total_payment) / total_investment) / average_duration
    else:
        IRR = -0.1  # Default negative return
    
    # Calculate real investor payments
    investor_payments_df = pd.DataFrame(investor_payment)
    average_investor_payment = np.sum(investor_payments_df, axis=0).mean()
    
    # Calculate real Malengo payments
    malengo_payments_df = pd.DataFrame(malengo_payment)
    average_malengo_payment = np.sum(malengo_payments_df, axis=0).mean()
    
    # Calculate real investor IRR using total investment as base
    if average_investor_payment > 0 and average_duration > 0:
        investor_IRR = np.log(max(1, average_investor_payment) / total_investment) / average_duration
    else:
        investor_IRR = -0.1
    
    # Calculate real payment quantiles
    payment_quantiles = {}
    for quantile in [0, 0.25, 0.5, 0.75, 1.0]:
        quantile_payment = np.sum(payments_df, axis=0).quantile(quantile)
        if quantile_payment > 0 and average_duration > 0:
            payment_quantiles[quantile] = np.log(max(1, quantile_payment) / total_investment) / average_duration
        else:
            payment_quantiles[quantile] = -0.1 - (0.1 * (1-quantile))  # Lower default for lower quantiles
    
    # Calculate real investor payment quantiles
    investor_payment_quantiles = {}
    for quantile in [0, 0.25, 0.5, 0.75, 1.0]:
        investor_quantile_payment = np.sum(investor_payments_df, axis=0).quantile(quantile)
        if investor_quantile_payment > 0 and average_duration > 0:
            investor_payment_quantiles[quantile] = np.log(max(1, investor_quantile_payment) / total_investment) / average_duration
        else:
            investor_payment_quantiles[quantile] = -0.1 - (0.1 * (1-quantile))  # Lower default for lower quantiles
    
    # Prepare real payment data for plotting
    payment_by_year = payments_df.mean(axis=1)
    investor_payment_by_year = investor_payments_df.mean(axis=1)
    malengo_payment_by_year = malengo_payments_df.mean(axis=1)
    
    # Calculate average employment and repayment statistics
    avg_employment_rate = np.mean(employment_stats)
    avg_ever_employed_rate = np.mean(ever_employed_stats)
    avg_repayment_rate = np.mean(repayment_stats)
    
    # Calculate average cap statistics
    avg_cap_stats = {
        'payment_cap_pct': np.mean([s['payment_cap_pct'] for s in cap_stats]),
        'years_cap_pct': np.mean([s['years_cap_pct'] for s in cap_stats]),
        'no_cap_pct': np.mean([s['no_cap_pct'] for s in cap_stats]),
        'avg_cap_value': np.mean([s.get('avg_cap_value', 0) for s in cap_stats])
    }
    
    # Calculate degree counts and percentages
    degree_counts = {degree.name: probs[i] * num_students for i, degree in enumerate(degrees)}
    degree_pcts = {degree.name: probs[i] for i, degree in enumerate(degrees)}
    
    # Calculate summary statistics for nominal (non-inflation-adjusted) payments
    nominal_payments_df = pd.DataFrame(nominal_total_payment)
    avg_nominal_total_payment = np.sum(nominal_payments_df, axis=0).mean()
    
    # Calculate nominal investor payments
    nominal_investor_payments_df = pd.DataFrame(nominal_investor_payment)
    avg_nominal_investor_payment = np.sum(nominal_investor_payments_df, axis=0).mean()
    
    # Calculate nominal Malengo payments
    nominal_malengo_payments_df = pd.DataFrame(nominal_malengo_payment)
    avg_nominal_malengo_payment = np.sum(nominal_malengo_payments_df, axis=0).mean()
    
    # Calculate nominal IRR values using the same duration as real IRR
    if avg_nominal_total_payment > 0 and average_duration > 0:
        nominal_IRR = np.log(max(1, avg_nominal_total_payment) / total_investment) / average_duration
    else:
        nominal_IRR = -0.1
        
    if avg_nominal_investor_payment > 0 and average_duration > 0:
        nominal_investor_IRR = np.log(max(1, avg_nominal_investor_payment) / total_investment) / average_duration
    else:
        nominal_investor_IRR = -0.1
    
    # Calculate nominal payment quantiles
    nominal_payment_quantiles = {}
    for quantile in [0, 0.25, 0.5, 0.75, 1.0]:
        nominal_quantile_payment = np.sum(nominal_payments_df, axis=0).quantile(quantile)
        if nominal_quantile_payment > 0 and average_duration > 0:
            nominal_payment_quantiles[quantile] = np.log(max(1, nominal_quantile_payment) / total_investment) / average_duration
        else:
            nominal_payment_quantiles[quantile] = -0.1 - (0.1 * (1-quantile))
    
    # Calculate nominal investor payment quantiles
    nominal_investor_payment_quantiles = {}
    for quantile in [0, 0.25, 0.5, 0.75, 1.0]:
        nominal_investor_quantile_payment = np.sum(nominal_investor_payments_df, axis=0).quantile(quantile)
        if nominal_investor_quantile_payment > 0 and average_duration > 0:
            nominal_investor_payment_quantiles[quantile] = np.log(max(1, nominal_investor_quantile_payment) / total_investment) / average_duration
        else:
            nominal_investor_payment_quantiles[quantile] = -0.1 - (0.1 * (1-quantile))
    
    # Prepare nominal payment data for plotting
    nominal_payment_by_year = nominal_payments_df.mean(axis=1)
    nominal_investor_payment_by_year = nominal_investor_payments_df.mean(axis=1)
    nominal_malengo_payment_by_year = nominal_malengo_payments_df.mean(axis=1)
    
    return {
        # Real (inflation-adjusted) IRR values
        'IRR': IRR,
        'investor_IRR': investor_IRR,
        'average_total_payment': average_total_payment,
        'average_investor_payment': average_investor_payment,
        'average_malengo_payment': average_malengo_payment,
        'average_duration': average_duration,
        'payment_by_year': payment_by_year,
        'investor_payment_by_year': investor_payment_by_year,
        'malengo_payment_by_year': malengo_payment_by_year,
        'payments_df': payments_df,
        'investor_payments_df': investor_payments_df,
        'malengo_payments_df': malengo_payments_df,
        'payment_quantiles': payment_quantiles,
        'investor_payment_quantiles': investor_payment_quantiles,
        
        # Nominal (non inflation-adjusted) IRR values
        'nominal_IRR': nominal_IRR,
        'nominal_investor_IRR': nominal_investor_IRR,
        'average_nominal_total_payment': avg_nominal_total_payment,
        'average_nominal_investor_payment': avg_nominal_investor_payment,
        'average_nominal_malengo_payment': avg_nominal_malengo_payment,
        'nominal_payment_by_year': nominal_payment_by_year,
        'nominal_investor_payment_by_year': nominal_investor_payment_by_year,
        'nominal_malengo_payment_by_year': nominal_malengo_payment_by_year,
        'nominal_payments_df': nominal_payments_df,
        'nominal_investor_payments_df': nominal_investor_payments_df,
        'nominal_malengo_payments_df': nominal_malengo_payments_df,
        'nominal_payment_quantiles': nominal_payment_quantiles,
        'nominal_investor_payment_quantiles': nominal_investor_payment_quantiles,
        
        # Other statistics
        'adjusted_mean_salary': degrees[0].mean_earnings if len(degrees) > 0 else 0,
        'adjusted_salary_std': degrees[0].stdev if len(degrees) > 0 else 0,
        'employment_rate': avg_employment_rate,
        'ever_employed_rate': avg_ever_employed_rate,
        'repayment_rate': avg_repayment_rate,
        'cap_stats': avg_cap_stats,
        'degree_counts': degree_counts,
        'degree_pcts': degree_pcts
    }


def main():
    """
    Command-line entrypoint for running simulations.
    
    Example usage:
    python simple_isa_model.py --program Uganda --scenario baseline --students 200 --sims 50
    """
    import matplotlib.pyplot as plt
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ISA Monte Carlo simulations.')
    
    # Basic simulation parameters
    parser.add_argument('--program', type=str, default='Uganda', choices=['Uganda', 'Kenya', 'Rwanda'],
                        help='Program type')
    parser.add_argument('--scenario', type=str, default='baseline', 
                        choices=['baseline', 'conservative', 'optimistic', 'custom'],
                        help='Scenario to run')
    parser.add_argument('--students', type=int, default=200, help='Number of students to simulate')
    parser.add_argument('--sims', type=int, default=20, help='Number of simulations to run')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--graduation-delay', action='store_true', 
                        help='Apply realistic graduation delays based on degree type')
    
    # Growth rate parameters
    parser.add_argument('--ba-growth', type=float, default=0.03, 
                        help='Annual growth rate for BA salaries (in decimal form, e.g., 0.03 for 3%)')
    parser.add_argument('--ma-growth', type=float, default=0.04, 
                        help='Annual growth rate for MA salaries (in decimal form, e.g., 0.04 for 4%)')
    parser.add_argument('--asst-growth', type=float, default=0.005, 
                        help='Annual growth rate for ASST salaries (in decimal form, e.g., 0.005 for 0.5%)')
    parser.add_argument('--asst-shift-growth', type=float, default=0.005, 
                        help='Annual growth rate for ASST_SHIFT salaries (in decimal form, e.g., 0.005 for 0.5%)')
    parser.add_argument('--nurse-growth', type=float, default=0.02, 
                        help='Annual growth rate for NURSE salaries (in decimal form, e.g., 0.02 for 2%)')
    parser.add_argument('--na-growth', type=float, default=0.01, 
                        help='Annual growth rate for NA salaries (in decimal form, e.g., 0.01 for 1%)')
    parser.add_argument('--trade-growth', type=float, default=0.02, 
                        help='Annual growth rate for TRADE salaries (in decimal form, e.g., 0.02 for 2%)')
    
    # Comparison mode for generating comparison charts
    parser.add_argument('--comparison', action='store_true', 
                        help='Enable comparison mode to generate charts comparing scenarios')
    
    args = parser.parse_args()
    
    # Special comparison mode that runs multiple scenarios and plots them
    if args.comparison:
        # Run all main scenarios
        results_baseline = run_simple_simulation(
            program_type=args.program,
            num_students=args.students,
            num_sims=args.sims,
            scenario='baseline',
            ba_growth=args.ba_growth,
            ma_growth=args.ma_growth,
            asst_growth=args.asst_growth,
            asst_shift_growth=args.asst_shift_growth,
            nurse_growth=args.nurse_growth,
            na_growth=args.na_growth,
            trade_growth=args.trade_growth,
            random_seed=args.seed,
            apply_graduation_delay=args.graduation_delay
        )
        
        results_conservative = run_simple_simulation(
            program_type=args.program,
            num_students=args.students,
            num_sims=args.sims,
            scenario='conservative',
            ba_growth=args.ba_growth,
            ma_growth=args.ma_growth,
            asst_growth=args.asst_growth,
            asst_shift_growth=args.asst_shift_growth,
            nurse_growth=args.nurse_growth,
            na_growth=args.na_growth,
            trade_growth=args.trade_growth,
            random_seed=args.seed,
            apply_graduation_delay=args.graduation_delay
        )
        
        results_optimistic = run_simple_simulation(
            program_type=args.program,
            num_students=args.students,
            num_sims=args.sims,
            scenario='optimistic',
            ba_growth=args.ba_growth,
            ma_growth=args.ma_growth,
            asst_growth=args.asst_growth,
            asst_shift_growth=args.asst_shift_growth,
            nurse_growth=args.nurse_growth,
            na_growth=args.na_growth,
            trade_growth=args.trade_growth,
            random_seed=args.seed,
            apply_graduation_delay=args.graduation_delay
        )
        
        # Create comparison plots
        # 1. IRR comparison
        plt.figure(figsize=(10, 6))
        scenarios = ['Baseline', 'Conservative', 'Optimistic']
        real_irrs = [
            results_baseline['investor_IRR'] * 100,
            results_conservative['investor_IRR'] * 100,
            results_optimistic['investor_IRR'] * 100
        ]
        nominal_irrs = [
            results_baseline['nominal_investor_IRR'] * 100,
            results_conservative['nominal_investor_IRR'] * 100,
            results_optimistic['nominal_investor_IRR'] * 100
        ]
        
        bar_width = 0.35
        index = np.arange(len(scenarios))
        
        plt.bar(index, real_irrs, bar_width, label='Real IRR')
        plt.bar(index + bar_width, nominal_irrs, bar_width, label='Nominal IRR')
        
        plt.xlabel('Scenario')
        plt.ylabel('IRR (%)')
        plt.title(f'{args.program} - Investor IRR Comparison')
        plt.xticks(index + bar_width / 2, scenarios)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{args.program}_irr_comparison.png')
        
        # 2. Cumulative payment comparison
        plt.figure(figsize=(12, 7))
        years = range(1, len(results_baseline['investor_payment_by_year']) + 1)
        
        plt.plot(years, results_baseline['investor_payment_by_year'].cumsum(), label='Baseline (Real)')
        plt.plot(years, results_conservative['investor_payment_by_year'].cumsum(), label='Conservative (Real)')
        plt.plot(years, results_optimistic['investor_payment_by_year'].cumsum(), label='Optimistic (Real)')
        
        plt.plot(years, results_baseline['nominal_investor_payment_by_year'].cumsum(), label='Baseline (Nominal)', linestyle='--')
        plt.plot(years, results_conservative['nominal_investor_payment_by_year'].cumsum(), label='Conservative (Nominal)', linestyle='--')
        plt.plot(years, results_optimistic['nominal_investor_payment_by_year'].cumsum(), label='Optimistic (Nominal)', linestyle='--')
        
        plt.axhline(y=results_baseline['total_investment'], color='black', linestyle='-', label='Initial Investment')
        
        plt.xlabel('Year')
        plt.ylabel('Cumulative Payments')
        plt.title(f'{args.program} - Cumulative Investor Returns')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{args.program}_cumulative_returns.png')
        
        print(f"\nComparison results saved to {args.program}_irr_comparison.png and {args.program}_cumulative_returns.png")
        
    else:
        # Run with selected scenario and growth rates
        results = run_simple_simulation(
            program_type=args.program,
            num_students=args.students,
            num_sims=args.sims,
            scenario=args.scenario,
            ba_growth=args.ba_growth,
            ma_growth=args.ma_growth,
            asst_growth=args.asst_growth,
            asst_shift_growth=args.asst_shift_growth,
            nurse_growth=args.nurse_growth,
            na_growth=args.na_growth,
            trade_growth=args.trade_growth,
            random_seed=args.seed,
            apply_graduation_delay=args.graduation_delay
        )
    
        # Print key results
        print("\nSimulation Results:")
        print(f"Total Investment: ${results['total_investment']:.2f}")
        print(f"Average Total Payment (Real): ${results['average_total_payment']:.2f}")
        print(f"Average Total Payment (Nominal): ${results['average_nominal_total_payment']:.2f}")
        print(f"Real IRR: {results['IRR']*100:.2f}%")
        print(f"Nominal IRR: {results['nominal_IRR']*100:.2f}%")
        print(f"Real Investor IRR: {results['investor_IRR']*100:.2f}%")
        print(f"Nominal Investor IRR: {results['nominal_investor_IRR']*100:.2f}%")
        print(f"Average Duration: {results['average_duration']:.2f} years")
        print(f"Annual Employment Rate: {results['employment_rate']*100:.2f}%")
        print(f"Ever Employed Rate: {results.get('ever_employed_rate', 0)*100:.2f}%")
        print(f"Repayment Rate: {results['repayment_rate']*100:.2f}%")
        
        # Print degree distribution
        print("\nDegree Distribution:")
        for degree, count in results['degree_counts'].items():
            print(f"{degree}: {count:.1f} students ({results['degree_pcts'][degree]*100:.1f}%)")
        
        # Print cap statistics
        print("\nCap Statistics:")
        print(f"Payment Cap: {results['cap_stats']['payment_cap_pct']*100:.2f}% of students")
        print(f"Years Cap: {results['cap_stats']['years_cap_pct']*100:.2f}% of students")
        print(f"No Cap: {results['cap_stats']['no_cap_pct']*100:.2f}% of students")
        print(f"Average Cap Value When Hit: ${results['cap_stats'].get('avg_cap_value', 0):.2f}")
        
        # Create and save plots
        # 1. Annual payments plot
        plt.figure(figsize=(10, 6))
        years = range(1, len(results['payment_by_year']) + 1)
        
        plt.plot(years, results['investor_payment_by_year'], label='Investor (Real)')
        plt.plot(years, results['malengo_payment_by_year'], label='Malengo (Real)')
        plt.plot(years, results['nominal_investor_payment_by_year'], label='Investor (Nominal)', linestyle='--')
        plt.plot(years, results['nominal_malengo_payment_by_year'], label='Malengo (Nominal)', linestyle='--')
        
        plt.xlabel('Year')
        plt.ylabel('Average Payment')
        plt.title(f'{results["program_type"]} - Annual Payments')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{args.program}_{args.scenario}_annual_payments.png')
        
        # 2. Cumulative returns plot
        plt.figure(figsize=(10, 6))
        
        plt.plot(years, results['investor_payment_by_year'].cumsum(), label='Investor Returns (Real)')
        plt.plot(years, results['nominal_investor_payment_by_year'].cumsum(), label='Investor Returns (Nominal)', linestyle='--')
        plt.axhline(y=results['total_investment'], color='black', linestyle='-', label='Initial Investment')
        
        # Calculate breakeven point (real returns)
        cumulative_returns = results['investor_payment_by_year'].cumsum()
        breakeven_year = None
        for i, value in enumerate(cumulative_returns):
            if value >= results['total_investment']:
                breakeven_year = i + 1
                break
        
        # Calculate breakeven point (nominal returns)
        nominal_cumulative_returns = results['nominal_investor_payment_by_year'].cumsum()
        nominal_breakeven_year = None
        for i, value in enumerate(nominal_cumulative_returns):
            if value >= results['total_investment']:
                nominal_breakeven_year = i + 1
                break
        
        # Add breakeven point to plot if it exists
        if breakeven_year:
            plt.plot(breakeven_year, results['total_investment'], 'ro', 
                    label=f'Breakeven (Real): Year {breakeven_year}')
        
        if nominal_breakeven_year:
            plt.plot(nominal_breakeven_year, results['total_investment'], 'go', 
                    label=f'Breakeven (Nominal): Year {nominal_breakeven_year}')
        
        plt.xlabel('Year')
        plt.ylabel('Cumulative Returns')
        plt.title(f'{results["program_type"]} - Cumulative Investor Returns')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{args.program}_{args.scenario}_cumulative_returns.png')
        
        print(f"\nPlots saved to {args.program}_{args.scenario}_annual_payments.png and {args.program}_{args.scenario}_cumulative_returns.png")


if __name__ == "__main__":
    main() 
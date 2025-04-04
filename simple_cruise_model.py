import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any
from dataclasses import dataclass
from simulation_config import StateConfig, SimulationConfig, DEFAULT_CONFIG, BASELINE_CONFIG, OPTIMISTIC_CONFIG, PESSIMISTIC_CONFIG

class CruiseCareerSequence:
    """Represents a person going through a sequence of training and work states"""
    
    def __init__(
        self,
        state_configs: List[StateConfig],
        random_seed: Optional[int] = None
    ):
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.state_configs = state_configs
        self.num_states = len(state_configs)
        
        # Current state tracking
        self.current_state_index = 0
        self.months_in_current_state = 0
        
        # Financial tracking
        self.total_training_costs = 0.0
        self.total_payments = 0.0
        self.current_salary = 0.0
        self.last_cruise_salary = 0.0  # Track last cruise salary for increases
        
        # History tracking
        self.monthly_salaries = []
        self.monthly_payments = []
        self.training_costs_by_state = []
        self.completed_states = []
        
        # Status flags
        self.dropout = False
        self.completed = False
        
        # For per-state dropout
        self.dropout_checked = False
        
        # Initialize first state
        self._enter_new_state()

    def _enter_new_state(self) -> None:
        """Handle entry into a new state"""
        if self.current_state_index >= self.num_states:
            self.completed = True
            return
            
        config = self.state_configs[self.current_state_index]
        
        # Add training cost for this state
        self.total_training_costs += config.training_cost
        self.training_costs_by_state.append(config.training_cost)
        
        # Reset state duration counter
        self.months_in_current_state = 0
        
        # Reset dropout check for new state
        self.dropout_checked = False
        
        # Check for dropout at the start of the state
        if self._check_dropout():
            self.dropout = True

    def _calculate_monthly_salary(self) -> float:
        """Calculate monthly salary based on current state and progression"""
        if self.current_state_index >= self.num_states:
            return 0.0
            
        config = self.state_configs[self.current_state_index]
        
        # No salary during training states
        if config.base_salary == 0:
            return 0.0
            
        # First cruise - use base salary with variation
        if self.last_cruise_salary == 0:
            variation_amount = config.base_salary * (config.salary_variation_pct / 100)
            new_salary = np.random.normal(config.base_salary, variation_amount)
            self.last_cruise_salary = new_salary
            return max(0, new_salary)
            
        # Subsequent cruises - increase from last salary
        increase = self.last_cruise_salary * (config.salary_increase_pct / 100)
        variation_amount = increase * (config.salary_variation_pct / 100)
        new_salary = self.last_cruise_salary + np.random.normal(increase, variation_amount)
        self.last_cruise_salary = new_salary
        return max(0, new_salary)

    def _check_dropout(self) -> bool:
        """Check if person drops out in current state"""
        if self.current_state_index >= self.num_states or self.dropout_checked:
            return False
            
        config = self.state_configs[self.current_state_index]
        self.dropout_checked = True  # Mark that we've already checked dropout for this state
        return np.random.random() < config.dropout_rate

    def advance_month(self) -> Dict[str, Any]:
        """Advance one month and return monthly results"""
        if self.dropout or self.completed:
            return self._get_monthly_summary()

        # Calculate salary and payments
        self.current_salary = self._calculate_monthly_salary()
        
        if self.current_state_index < self.num_states:
            config = self.state_configs[self.current_state_index]
            monthly_payment = self.current_salary * config.payment_fraction
        else:
            monthly_payment = 0
            
        self.total_payments += monthly_payment
        
        # Track history
        self.monthly_salaries.append(self.current_salary)
        self.monthly_payments.append(monthly_payment)
        
        # Update state duration
        self.months_in_current_state += 1
        
        # Check for state completion
        if self.current_state_index < self.num_states:
            config = self.state_configs[self.current_state_index]
            if self.months_in_current_state >= config.duration_months:
                self.completed_states.append(self.current_state_index)
                self.current_state_index += 1
                self._enter_new_state()
            
        return self._get_monthly_summary()

    def _get_monthly_summary(self) -> Dict[str, Any]:
        """Return summary of current status"""
        current_payment_fraction = (
            self.state_configs[self.current_state_index].payment_fraction 
            if self.current_state_index < self.num_states and not self.dropout
            else 0
        )
        
        current_state_name = (
            self.state_configs[self.current_state_index].name
            if self.current_state_index < self.num_states and hasattr(self.state_configs[self.current_state_index], 'name')
            else f"State {self.current_state_index}"
        )
        
        return {
            'state_index': self.current_state_index,
            'state_name': current_state_name,
            'months_in_state': self.months_in_current_state,
            'current_salary': self.current_salary,
            'monthly_payment': self.current_salary * current_payment_fraction,
            'total_training_costs': self.total_training_costs,
            'total_payments': self.total_payments,
            'net_cash_flow': self.total_payments - self.total_training_costs,
            'dropout': self.dropout,
            'completed': self.completed,
            'completed_states': self.completed_states.copy()
        }


def create_default_state_configs(num_cruises: int = 3) -> List[StateConfig]:
    """Create a default sequence of states with reasonable parameters
    
    Args:
        num_cruises: Number of cruises to simulate (default: 3)
    """
    # Start with the training states
    states = [
        # Basic Training
        StateConfig(
            training_cost=2000,    # Initial training cost
            dropout_rate=0.15,     # 15% dropout rate
            base_salary=0,         # No salary during training
            salary_increase_pct=0, # No increase during training
            salary_variation_pct=0,
            duration_months=3,     # 3 months
            payment_fraction=0,    # No payments during training
            name="Basic Training"
        ),
        # Advanced Training
        StateConfig(
            training_cost=2000,    # Advanced training cost
            dropout_rate=0.10,     # 10% dropout rate
            base_salary=0,         # No salary during training
            salary_increase_pct=0, # No increase during training
            salary_variation_pct=0,
            duration_months=3,     # 3 months
            payment_fraction=0,    # No payments during training
            name="Advanced Training"
        ),
    ]
    
    # Add first cruise
    states.append(
        StateConfig(
            training_cost=0,       # No additional training cost
            dropout_rate=0.15,     # 15% dropout rate
            base_salary=5000,      # First cruise base salary
            salary_increase_pct=0, # Initial salary, no increase
            salary_variation_pct=6,# ±6% variation (±$300 on $5000)
            duration_months=8,     # 8 month cruise
            payment_fraction=0.14, # 14% of salary as payment
            name="First Cruise"
        )
    )
    
    # Add subsequent cruises
    for i in range(1, num_cruises):
        states.append(
            StateConfig(
                training_cost=0,       # No additional training cost
                dropout_rate=0.02,     # 2% dropout rate
                base_salary=0,         # Base salary not used after first cruise
                salary_increase_pct=10,# 10% increase from previous
                salary_variation_pct=5,# ±5% variation in increase
                duration_months=8,
                payment_fraction=0.14,
                name=f"Cruise {i+1}"
            )
        )
    
    return states


def run_simulation(
    num_cruises: int = 3,
    state_configs: Optional[List[StateConfig]] = None,
    max_months: int = 48,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """Run a simulation with given state configurations
    
    Args:
        num_cruises: Number of cruises to simulate (default: 3)
        state_configs: Custom state configurations (if None, uses default configs)
        max_months: Maximum number of months to simulate
        random_seed: Optional random seed for reproducibility
    """
    
    if state_configs is None:
        state_configs = create_default_state_configs(num_cruises)
    
    sequence = CruiseCareerSequence(
        state_configs=state_configs,
        random_seed=random_seed
    )
    
    monthly_results = []
    for _ in range(max_months):
        result = sequence.advance_month()
        monthly_results.append(result)
        
        if sequence.dropout or sequence.completed:
            break
    
    # Calculate ROI
    roi = ((sequence.total_payments - sequence.total_training_costs) / 
           sequence.total_training_costs if sequence.total_training_costs > 0 else 0)
    
    return {
        'final_state_index': sequence.current_state_index,
        'total_training_costs': sequence.total_training_costs,
        'total_payments': sequence.total_payments,
        'net_cash_flow': sequence.total_payments - sequence.total_training_costs,
        'duration_months': len(monthly_results),
        'roi': roi,
        'dropout': sequence.dropout,
        'completed': sequence.completed,
        'completed_states': sequence.completed_states,
        'training_costs_by_state': sequence.training_costs_by_state,
        'monthly_results': monthly_results,
        'monthly_salaries': sequence.monthly_salaries,
        'monthly_payments': sequence.monthly_payments
    }


def calculate_summary_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate comprehensive summary metrics from simulation results"""
    monthly_salaries = results['monthly_salaries']
    monthly_payments = results['monthly_payments']
    total_training_costs = results['total_training_costs']
    total_payments = results['total_payments']
    duration_months = results['duration_months']
    
    # Calculate metrics
    metrics = {
        'total_training_costs': total_training_costs,
        'total_payments': total_payments,
        'net_returns': total_payments - total_training_costs,
        'roi_percentage': ((total_payments - total_training_costs) / total_training_costs * 100 
                         if total_training_costs > 0 else 0),
        'duration_months': duration_months,
        'average_monthly_salary': np.mean([s for s in monthly_salaries if s > 0]) if monthly_salaries else 0,
        'average_monthly_payment': np.mean([p for p in monthly_payments if p > 0]) if monthly_payments else 0,
    }
    
    # Calculate breakeven month (if reached)
    cumulative_payments = np.cumsum(monthly_payments)
    breakeven_months = np.where(cumulative_payments >= total_training_costs)[0]
    metrics['breakeven_month'] = breakeven_months[0] + 1 if len(breakeven_months) > 0 else None
    
    # Calculate simple IRR (annualized return)
    if duration_months > 0 and total_training_costs > 0:
        years = duration_months / 12
        irr = (np.power(total_payments / total_training_costs, 1/years) - 1) * 100 if total_payments > 0 else -100
        metrics['annual_irr'] = irr
    else:
        metrics['annual_irr'] = None
    
    # Calculate repayment rate (percentage of training costs recovered)
    metrics['repayment_rate'] = (total_payments / total_training_costs * 100 
                                if total_training_costs > 0 else 0)
    
    return metrics


def print_simulation_summary(results: Dict[str, Any]) -> None:
    """Print comprehensive summary of simulation results"""
    metrics = calculate_summary_metrics(results)
    
    print("\nSimulation Summary:")
    print("==================")
    
    # Training Progress
    print("\nTraining Progress:")
    if 'state_name' in results['monthly_results'][-1]:
        print(f"Final State: {results['monthly_results'][-1]['state_name']} (index {results['final_state_index']})")
    else:
        print(f"Final State: {results['final_state_index']}")
    print(f"States Completed: {results['completed_states']}")
    print(f"Duration: {metrics['duration_months']} months")
    if results['dropout']:
        if 'state_name' in results['monthly_results'][-1]:
            print(f"Status: Dropped out during {results['monthly_results'][-1]['state_name']}")
        else:
            print(f"Status: Dropped out during state {results['final_state_index']}")
    elif results['completed']:
        print("Status: Completed all states")
    
    # Financial Summary
    print("\nFinancial Summary:")
    print(f"Total Training Costs: ${metrics['total_training_costs']:,.2f}")
    print(f"Total Payments Made: ${metrics['total_payments']:,.2f}")
    print(f"Net Returns: ${metrics['net_returns']:,.2f}")
    
    # Performance Metrics
    print("\nPerformance Metrics:")
    if metrics['breakeven_month']:
        print(f"Breakeven Month: {metrics['breakeven_month']}")
    else:
        print("Breakeven: Not reached")
    print(f"Repayment Rate: {metrics['repayment_rate']:.1f}%")
    if metrics['annual_irr'] is not None:
        print(f"Annual IRR: {metrics['annual_irr']:.1f}%")
    
    # Averages
    print("\nMonthly Averages:")
    print(f"Average Monthly Salary: ${metrics['average_monthly_salary']:,.2f}")
    print(f"Average Monthly Payment: ${metrics['average_monthly_payment']:,.2f}")
    
    # Return Profile
    if metrics['breakeven_month']:
        months_after_breakeven = metrics['duration_months'] - metrics['breakeven_month']
        if months_after_breakeven > 0:
            returns_after_breakeven = sum(results['monthly_payments'][metrics['breakeven_month']:])
            print(f"\nReturns after Breakeven: ${returns_after_breakeven:,.2f}")
            print(f"Months after Breakeven: {months_after_breakeven}")


def compare_cruise_configurations(max_cruises: int = 10, num_simulations: int = 100) -> pd.DataFrame:
    """Compare metrics for different numbers of cruises
    
    Args:
        max_cruises: Maximum number of cruises to simulate
        num_simulations: Number of simulations to run for each configuration
        
    Returns:
        DataFrame with comparison metrics
    """
    results = []
    
    for num_cruises in range(1, max_cruises + 1):
        print(f"Running {num_simulations} simulations for {num_cruises} cruises...")
        simulation_results = []
        
        for i in range(num_simulations):
            # Run simulation with different random seed for each iteration
            sim_result = run_simulation(num_cruises=num_cruises, random_seed=i)
            metrics = calculate_summary_metrics(sim_result)
            
            # Track if simulation completed all states
            completed_all = sim_result['completed']
            
            # Add metrics to results
            simulation_results.append({
                'num_cruises': num_cruises,
                'completed_all_states': completed_all,
                'dropout': sim_result['dropout'],
                'duration_months': metrics['duration_months'],
                'total_training_costs': metrics['total_training_costs'],
                'total_payments': metrics['total_payments'],
                'net_returns': metrics['net_returns'],
                'roi_percentage': metrics['roi_percentage'],
                'breakeven_month': metrics['breakeven_month'],
                'annual_irr': metrics['annual_irr']
            })
        
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(simulation_results)
        
        # Calculate aggregate metrics
        avg_metrics = {
            'num_cruises': num_cruises,
            'completion_rate': df['completed_all_states'].mean() * 100,
            'dropout_rate': df['dropout'].mean() * 100,
            'avg_duration': df['duration_months'].mean(),
            'avg_net_returns': df['net_returns'].mean(),
            'avg_roi': df['roi_percentage'].mean(),
            'breakeven_rate': df['breakeven_month'].notna().mean() * 100,
            'avg_annual_irr': df['annual_irr'].mean()
        }
        
        results.append(avg_metrics)
    
    return pd.DataFrame(results)


def print_cruise_comparison(max_cruises: int = 5, num_simulations: int = 100) -> None:
    """Print a comparison of different cruise configurations
    
    Args:
        max_cruises: Maximum number of cruises to simulate
        num_simulations: Number of simulations to run for each configuration
    """
    comparison_df = compare_cruise_configurations(max_cruises, num_simulations)
    
    print("\nCruise Configuration Comparison:")
    print("===============================")
    print("\nBased on", num_simulations, "simulations per configuration:")
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    # Find optimal configuration based on ROI
    best_roi_idx = comparison_df['avg_roi'].idxmax()
    best_roi_cruises = comparison_df.iloc[best_roi_idx]['num_cruises']
    
    # Find optimal configuration based on net returns
    best_returns_idx = comparison_df['avg_net_returns'].idxmax()
    best_returns_cruises = comparison_df.iloc[best_returns_idx]['num_cruises']
    
    print(f"\nBest ROI: {best_roi_cruises} cruises ({comparison_df.iloc[best_roi_idx]['avg_roi']:.2f}%)")
    print(f"Best Net Returns: {best_returns_cruises} cruises (${comparison_df.iloc[best_returns_idx]['avg_net_returns']:.2f})")


def run_simulation_batch(config: SimulationConfig) -> Dict:
    """Run a batch of simulations with the given configuration
    
    Args:
        config: SimulationConfig object containing simulation parameters
        
    Returns:
        Dictionary containing analysis results
    """
    all_results = []
    state_configs = config.create_state_configs()
    num_states = len(state_configs)
    
    # Initialize state-specific tracking
    state_salary_sums = {i: 0.0 for i in range(num_states)}
    state_payment_sums = {i: 0.0 for i in range(num_states)}
    state_active_months = {i: 0 for i in range(num_states)}
    
    for i in range(config.num_students):
        # Use student index as seed if no random seed provided
        seed = config.random_seed + i if config.random_seed is not None else i
        
        result = run_simulation(
            state_configs=state_configs,
            max_months=config.max_months,
            random_seed=seed
        )
        all_results.append(result)
        
        # Track state-specific metrics
        for month_result in result['monthly_results']:
            state_idx = month_result['state_index']
            if state_idx < num_states:  # Ensure valid state
                state_salary_sums[state_idx] += month_result['current_salary']
                state_payment_sums[state_idx] += month_result['monthly_payment']
                state_active_months[state_idx] += 1
    
    # Calculate average salaries and payments by state
    state_metrics = {}
    for state_idx in range(num_states):
        total_months = config.num_students * state_configs[state_idx].duration_months
        state_metrics[state_idx] = {
            'name': state_configs[state_idx].name,
            'avg_salary': state_salary_sums[state_idx] / total_months,  # Average across all students
            'avg_payment': state_payment_sums[state_idx] / total_months, # Average across all students
            'active_months': state_active_months[state_idx]
        }
    
    # Convert results to DataFrame for analysis
    df = pd.DataFrame([
        {
            'dropout': r['dropout'],
            'completed': r['completed'],
            'duration_months': r['duration_months'],
            'total_training_costs': r['total_training_costs'],
            'total_payments': r['total_payments'],
            'net_cash_flow': r['net_cash_flow'],
            'final_state_index': r['final_state_index']
        }
        for r in all_results
    ])
    
    # Calculate key metrics
    roi = df['net_cash_flow'] / df['total_training_costs']
    
    return {
        'completion_rate': df['completed'].mean() * 100,
        'dropout_rate': df['dropout'].mean() * 100,
        'avg_duration': df['duration_months'].mean(),
        'avg_training_cost': df['total_training_costs'].mean(),
        'avg_total_payments': df['total_payments'].mean(),
        'avg_net_cash_flow': df['net_cash_flow'].mean(),
        'avg_roi': roi.mean() * 100,
        'roi_std': roi.std() * 100,
        'roi_10th': roi.quantile(0.1) * 100,
        'roi_90th': roi.quantile(0.9) * 100,
        'state_distribution': df['final_state_index'].value_counts().sort_index().to_dict(),
        'state_metrics': state_metrics
    }

def print_simulation_results(results: Dict, scenario_name: str = "Default") -> None:
    """Print formatted simulation results
    
    Args:
        results: Dictionary of analysis results
        scenario_name: Name of the scenario being analyzed
    """
    print(f"\n{scenario_name} Scenario Analysis")
    print("=" * (len(scenario_name) + 19))
    
    print("\nCompletion Statistics:")
    print(f"Completion Rate: {results['completion_rate']:.1f}%")
    print(f"Dropout Rate: {results['dropout_rate']:.1f}%")
    print(f"Average Duration: {results['avg_duration']:.1f} months")
    
    print("\nFinancial Statistics:")
    print(f"Average Training Cost: ${results['avg_training_cost']:,.2f}")
    print(f"Average Total Payments: ${results['avg_total_payments']:,.2f}")
    print(f"Average Net Cash Flow: ${results['avg_net_cash_flow']:,.2f}")
    
    print("\nROI Statistics:")
    print(f"Average ROI: {results['avg_roi']:.1f}%")
    print(f"ROI Standard Deviation: {results['roi_std']:.1f}%")
    print(f"ROI 10th Percentile: {results['roi_10th']:.1f}%")
    print(f"ROI 90th Percentile: {results['roi_90th']:.1f}%")
    
    print("\nState-by-State Analysis:")
    print("------------------------")
    print(f"{'State':<20} {'Avg Monthly Salary':>20} {'Avg Monthly Payment':>20} {'Active Months':>15}")
    print("-" * 75)
    
    for state_idx, metrics in results['state_metrics'].items():
        print(f"{metrics['name']:<20} ${metrics['avg_salary']:>19,.2f} ${metrics['avg_payment']:>19,.2f} {metrics['active_months']:>15,d}")
    
    print("\nFinal State Distribution:")
    total_students = sum(results['state_distribution'].values())
    for state_idx, count in results['state_distribution'].items():
        # Check if state_idx exists in state_metrics before accessing it
        if state_idx in results['state_metrics']:
            state_name = results['state_metrics'][state_idx]['name']
        else:
            state_name = f"Unknown State {state_idx}"
        print(f"{state_name}: {count/total_students*100:.1f}%")

def analyze_state_transitions(config: SimulationConfig, num_simulations: int = 500) -> None:
    """Analyze state transitions and dropout rates
    
    Args:
        config: SimulationConfig to use for simulation
        num_simulations: Number of simulations to run
    """
    state_configs = config.create_state_configs()
    state_names = [s.name for s in state_configs]
    
    # Track state transitions
    entered_state = {i: 0 for i in range(len(state_configs))}
    completed_state = {i: 0 for i in range(len(state_configs))}
    dropouts_in_state = {i: 0 for i in range(len(state_configs))}
    
    # Run simulations
    for i in range(num_simulations):
        seed = i if config.random_seed is None else config.random_seed + i
        result = run_simulation(
            state_configs=state_configs,
            max_months=config.max_months,
            random_seed=seed
        )
        
        # Record final state and completion
        final_state = result['final_state_index']
        
        # Record state entries and completions
        for state in range(len(state_configs)):
            if state in result['completed_states'] or state == final_state:
                entered_state[state] += 1
            if state in result['completed_states']:
                completed_state[state] += 1
            if result['dropout'] and state == final_state:
                dropouts_in_state[state] += 1
    
    # Calculate statistics
    print("\nState Transition Analysis:")
    print("=========================")
    print(f"Based on {num_simulations} simulations")
    print("\n{:<20} {:<15} {:<15} {:<15} {:<15}".format(
        "State", "Entered", "Completed", "Dropouts", "Dropout Rate"
    ))
    print("-" * 80)
    
    for state in range(len(state_configs)):
        if entered_state[state] > 0:
            dropout_rate = dropouts_in_state[state] / entered_state[state] * 100
            completion_rate = completed_state[state] / entered_state[state] * 100
            
            configured_dropout = state_configs[state].dropout_rate * 100
            
            print("{:<20} {:<15} {:<15} {:<15} {:<15.1f}%".format(
                state_names[state],
                entered_state[state],
                completed_state[state],
                dropouts_in_state[state],
                dropout_rate
            ))
            
            # Add warning if dropout rate is significantly different from configured rate
            if abs(dropout_rate - configured_dropout) > 5:
                print(f"  NOTE: Actual dropout rate differs from configured rate of {configured_dropout:.1f}%")
    
    # Add explanation for dropout rates
    print("\nDropout Rate Analysis:")
    print("=====================")
    print("Each state now has a single dropout chance that applies to the entire state period,")
    print("rather than a monthly dropout chance that compounds over time.")
    print("This makes dropout rates more intuitive and easier to configure.")
    
    # Explain dropout vs completion rate
    print("\nDropout vs Completion Rate:")
    print("For each state, students either:")
    print("1. Drop out during the state (counted in 'Dropouts')")
    print("2. Complete the state and move to the next one (counted in 'Completed')")
    print("3. Reach the end of the simulation while in that state (not completed but not dropped out)")
    
    # Provide recommendations
    print("\nRecommendations for Setting Dropout Rates:")
    print("1. Basic Training: 10-20% depending on selectivity")
    print("2. Advanced Training: 10-15% for most programs")
    print("3. First Cruise: 10-20% (higher due to first real-world experience)")
    print("4. Subsequent Cruises: 2-5% (lower as students gain experience)")
    print("\nThese rates now directly represent the percentage of students who")
    print("will not complete each state, making the simulation more intuitive.")

if __name__ == "__main__":
    print("Career Cruise Simulator")
    print("======================")
    print("Running scenarios from simulation_config.py")
    
    # Run all scenarios
    configs = {
        "Baseline": BASELINE_CONFIG,
        "Optimistic": OPTIMISTIC_CONFIG,
        "Pessimistic": PESSIMISTIC_CONFIG
    }
    
    for name, config in configs.items():
        print(f"\nRunning {name} scenario with {config.num_students} students...")
        results = run_simulation_batch(config)
        print_simulation_results(results, name)
    
    # Run detailed state transition analysis for baseline scenario
    print("\nAnalyzing state transitions and dropout patterns for baseline scenario...")
    analyze_state_transitions(BASELINE_CONFIG, num_simulations=500)
    
    # Example of running a single simulation with baseline config
    print("\nRunning a single simulation example with baseline configuration:")
    baseline_state_configs = BASELINE_CONFIG.create_state_configs()
    single_result = run_simulation(state_configs=baseline_state_configs, random_seed=42)
    print_simulation_summary(single_result)
    
    # Uncomment to run cruise configuration comparison
    # print("\nRunning cruise configuration comparison...")
    # print_cruise_comparison(max_cruises=5, num_simulations=50) 
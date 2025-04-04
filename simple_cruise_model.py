import numpy as np
import pandas as pd
import numpy_financial as npf  # Add import for numpy-financial
from typing import List, Dict, Union, Optional, Tuple, Any
from dataclasses import dataclass
from simulation_config import StateConfig, SimulationConfig, DEFAULT_CONFIG

class CruiseCareerSequence:
    """Represents a person going through a sequence of training and work states"""
    
    def __init__(
        self,
        state_configs: List[StateConfig],
        random_seed: Optional[int] = None,
        disney_allocation_pct: float = 30.0,
        costa_allocation_pct: float = 70.0
    ):
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.state_configs = state_configs
        self.num_states = len(state_configs)
        
        # Provider allocation
        self.disney_allocation_pct = disney_allocation_pct
        self.costa_allocation_pct = costa_allocation_pct
        
        # Selected provider path
        self.selected_provider = None
        
        # Current state tracking
        self.current_state_index = 0
        
        # Financial tracking
        self.total_training_costs = 0.0
        self.total_payments = 0.0
        self.current_state_salary = 0.0  # Salary for the entire state
        
        # History tracking
        self.state_salaries = []
        self.state_payments = []
        self.training_costs_by_state = []
        self.completed_states = []
        
        # Status flags
        self.dropout = False
        self.completed = False
        
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
        
        # Check for dropout at the start of the state
        if self._check_dropout():
            self.dropout = True
            return  # Exit early if dropout occurs
            
        # If we completed the advanced training, assign to provider
        current_state_name = config.name
        if current_state_name == "Transportation and placement" and not self.selected_provider:
            self._select_provider()
            
        # Calculate salary for this state immediately after entering (if not dropped out)
        self.current_state_salary = self._calculate_state_salary()

    def _select_provider(self) -> None:
        """Select which provider (Disney or Costa) the student will be assigned to"""
        if np.random.random() * 100 < self.disney_allocation_pct:
            self.selected_provider = "Disney"
        else:
            self.selected_provider = "Costa"
        
        # Skip to the first state for this provider
        # We need to find the index of the first cruise state for this provider
        for i, state in enumerate(self.state_configs):
            if state.provider == self.selected_provider and "Cruise 1" in state.name:
                self.current_state_index = i
                break

    def _calculate_state_salary(self) -> float:
        """Calculate salary for the entire state based on current state"""
        if self.current_state_index >= self.num_states:
            return 0.0
            
        config = self.state_configs[self.current_state_index]
        
        # No salary during training states or breaks
        if "Training" in config.name or "Transportation and placement" in config.name or "Break" in config.name:
            return 0.0
            
        # For cruise states, use the configured base salary with variation
        if "Cruise" in config.name:
            variation_amount = config.base_salary * (config.salary_variation_pct / 100)
            salary = np.random.normal(config.base_salary, variation_amount)
            return max(0, salary)
            
        # Default case
        return 0.0

    def _check_dropout(self) -> bool:
        """Check if person drops out in current state"""
        if self.current_state_index >= self.num_states:
            return False
            
        config = self.state_configs[self.current_state_index]
        return np.random.random() < config.dropout_rate

    def advance_state(self) -> Dict[str, Any]:
        """Advance to the next state and return state results"""
        if self.dropout or self.completed:
            return self._get_state_summary()

        # Calculate payment for the current state
        state_payment = 0.0
        if self.current_state_index < self.num_states:
            config = self.state_configs[self.current_state_index]
            # Only calculate payment for cruise states
            if "Cruise" in config.name and not "Break" in config.name:
                state_payment = self.current_state_salary * config.payment_fraction
        
        # Add payment to total
        self.total_payments += state_payment
        
        # Track history
        self.state_salaries.append(self.current_state_salary)
        self.state_payments.append(state_payment)
        
        # Store the current state index before completing it
        current_index = self.current_state_index
        
        # Complete the current state and move to the next
        self.completed_states.append(self.current_state_index)
        
        # Advance to next state, but only if it belongs to the same provider or it's early training
        next_index = self.current_state_index + 1
        if next_index < self.num_states:
            next_config = self.state_configs[next_index]
            current_config = self.state_configs[self.current_state_index]
            
            # If we're still in training, move to the next state
            if not self.selected_provider:
                self.current_state_index = next_index
            # If we have a provider, only move to states for that provider
            elif next_config.provider == self.selected_provider or next_config.provider == "":
                self.current_state_index = next_index
            # Otherwise, find the next state for this provider
            else:
                found_next = False
                for i in range(next_index, self.num_states):
                    if self.state_configs[i].provider == self.selected_provider:
                        self.current_state_index = i
                        found_next = True
                        break
                # If no next state found, we're done
                if not found_next:
                    self.completed = True
        else:
            # No more states
            self.completed = True
            
        self._enter_new_state()
        
        # Create a summary based on the state we just completed
        return {
            'state_index': current_index,
            'state_name': self.state_configs[current_index].name if current_index < self.num_states else f"State {current_index}",
            'state_duration': self.state_configs[current_index].duration_months if current_index < self.num_states else 0,
            'current_state_salary': self.current_state_salary if not "Break" in self.state_configs[current_index].name else 0.0,
            'state_payment': state_payment,
            'payment_fraction': self.state_configs[current_index].payment_fraction if current_index < self.num_states else 0,
            'total_training_costs': self.total_training_costs,
            'total_payments': self.total_payments,
            'net_cash_flow': self.total_payments - self.total_training_costs,
            'dropout': self.dropout,
            'completed': self.completed,
            'completed_states': self.completed_states.copy(),
            'provider': self.selected_provider
        }

    def _get_state_summary(self) -> Dict[str, Any]:
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
        
        # Calculate state duration in months (for backward compatibility)
        state_duration = (
            self.state_configs[self.current_state_index].duration_months
            if self.current_state_index < self.num_states
            else 0
        )
        
        # For payment, use the last recorded payment to ensure consistency
        state_payment = self.state_payments[-1] if self.state_payments else (self.current_state_salary * current_payment_fraction)
        
        return {
            'state_index': self.current_state_index,
            'state_name': current_state_name,
            'state_duration': state_duration,
            'current_state_salary': self.current_state_salary,
            'state_payment': state_payment,
            'payment_fraction': current_payment_fraction,
            'total_training_costs': self.total_training_costs,
            'total_payments': self.total_payments,
            'net_cash_flow': self.total_payments - self.total_training_costs,
            'dropout': self.dropout,
            'completed': self.completed,
            'completed_states': self.completed_states.copy(),
            'provider': self.selected_provider
        }


def create_default_state_configs(num_cruises: int = 3) -> List[StateConfig]:
    """Create a default sequence of states with reasonable parameters
    
    Args:
        num_cruises: Number of cruises to simulate (default: 3)
    """
    # Start with the training states
    states = [
        # Training
        StateConfig(
            training_cost=2000,    # Initial training cost
            dropout_rate=0.10,     # 10% dropout rate
            base_salary=0,         # No salary during training
            salary_increase_pct=0, # No increase during training
            salary_variation_pct=0,
            duration_months=3,     # 3 months
            payment_fraction=0,    # No payments during training
            name="Training"
        ),
        # Transportation and placement
        StateConfig(
            training_cost=2000,    # Advanced training cost
            dropout_rate=0.15,     # 15% dropout rate
            base_salary=0,         # No salary during training
            salary_increase_pct=0, # No increase during training
            salary_variation_pct=0,
            duration_months=3,     # 3 months
            payment_fraction=0,    # No payments during training
            name="Transportation and placement"
        ),
    ]
    
    # Add first cruise
    states.append(
        StateConfig(
            training_cost=0,       # No additional training cost
            dropout_rate=0.03,     # 3% dropout rate
            base_salary=5000,      # First cruise period salary
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
                dropout_rate=0.03,     # 3% dropout rate
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
    random_seed: Optional[int] = None,
    simulation_config: Optional[SimulationConfig] = None
) -> Dict[str, Any]:
    """Run a simulation with given state configurations
    
    Args:
        num_cruises: Number of cruises to simulate (default: 3)
        state_configs: Custom state configurations (if None, uses default configs)
        random_seed: Optional random seed for reproducibility
        simulation_config: Optional simulation configuration to use
    """
    if simulation_config:
        if state_configs is None:
            state_configs = simulation_config.create_state_configs()
        disney_allocation = simulation_config.disney_allocation_pct
        costa_allocation = simulation_config.costa_allocation_pct
    else:
        if state_configs is None:
            state_configs = create_default_state_configs(num_cruises)
        disney_allocation = 30.0  # Default allocations
        costa_allocation = 70.0
    
    sequence = CruiseCareerSequence(
        state_configs=state_configs,
        random_seed=random_seed,
        disney_allocation_pct=disney_allocation,
        costa_allocation_pct=costa_allocation
    )
    
    state_results = []
    while True:
        result = sequence.advance_state()
        state_results.append(result)
        
        if sequence.dropout or sequence.completed:
            break
    
    # Calculate ROI
    roi = ((sequence.total_payments - sequence.total_training_costs) / 
           sequence.total_training_costs if sequence.total_training_costs > 0 else 0)
    
    # For backwards compatibility, compute duration_months from state durations
    total_months = 0
    for i, state_config in enumerate(state_configs):
        if i in sequence.completed_states:
            total_months += state_config.duration_months
    
    # Add current state if not completed all states and not dropped out
    if not sequence.completed and not sequence.dropout and sequence.current_state_index < len(state_configs):
        # We're still in the last state, add its duration
        total_months += state_configs[sequence.current_state_index].duration_months
    
    return {
        'final_state_index': sequence.current_state_index,
        'total_training_costs': sequence.total_training_costs,
        'total_payments': sequence.total_payments,
        'net_cash_flow': sequence.total_payments - sequence.total_training_costs,
        'duration_months': total_months,
        'roi': roi,
        'dropout': sequence.dropout,
        'completed': sequence.completed,
        'completed_states': sequence.completed_states,
        'training_costs_by_state': sequence.training_costs_by_state,
        'state_results': state_results,
        'state_salaries': sequence.state_salaries,
        'state_payments': sequence.state_payments,
        'selected_provider': sequence.selected_provider
    }


def calculate_monthly_irr(results: Dict[str, Any]) -> Optional[float]:
    """Calculate the IRR (Internal Rate of Return) based on monthly cash flows
    
    Args:
        results: Dictionary containing simulation results
        
    Returns:
        Monthly IRR as a percentage (or None if calculation not possible)
    """
    # Extract relevant data
    state_results = results.get('state_results', [])
    
    if not state_results:
        return None
    
    # Create a list to store monthly cash flows
    monthly_cash_flows = []
    
    # Initial cash flow is negative (training cost)
    for state_idx, state_result in enumerate(state_results):
        state_name = state_result.get('state_name', '')
        state_duration = state_result.get('state_duration', 0)
        state_payment = state_result.get('state_payment', 0)
        
        # Skip states with no duration
        if state_duration <= 0:
            continue
        
        # For training states, add full cost as upfront payment in first month
        if "Training" in state_name or "Transportation and placement" in state_name:
            training_cost = state_result.get('total_training_costs', 0) - (
                state_results[state_idx-1].get('total_training_costs', 0) if state_idx > 0 else 0
            )
            
            if training_cost > 0:
                # Add training cost as negative cash flow at start of state
                monthly_cash_flows.append(-training_cost)
                
                # Add 0 cash flow for remaining months of training
                monthly_cash_flows.extend([0] * (state_duration - 1))
        else:
            # For cruise states, distribute payments evenly across months
            monthly_payment = state_payment / state_duration if state_duration > 0 else 0
            monthly_cash_flows.extend([monthly_payment] * state_duration)
    
    # If there are no cash flows or only positive/negative, IRR cannot be calculated
    if not monthly_cash_flows:
        return None
    
    if all(cf <= 0 for cf in monthly_cash_flows):
        return None
    
    if all(cf >= 0 for cf in monthly_cash_flows):
        return None
    
    try:
        # Calculate IRR using numpy's financial function
        cash_flow_array = np.array(monthly_cash_flows)
        
        # Check if the array has both positive and negative values
        has_positive = any(cf > 0 for cf in cash_flow_array)
        has_negative = any(cf < 0 for cf in cash_flow_array)
        
        if not (has_positive and has_negative):
            return None
        
        # Calculate IRR (returns monthly rate as decimal) using numpy-financial
        monthly_irr = npf.irr(cash_flow_array)
        
        # Convert to annual IRR and to percentage
        annual_irr = ((1 + monthly_irr) ** 12 - 1) * 100
        
        return annual_irr
    except Exception as e:
        print(f"IRR calculation error: {str(e)}")
        # IRR calculation might fail for various reasons
        return None


def calculate_summary_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate comprehensive summary metrics from simulation results"""
    state_salaries = results['state_salaries']
    state_payments = results['state_payments']
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
        'num_states': len(state_salaries),
        'average_state_salary': np.mean([s for s in state_salaries if s > 0]) if state_salaries else 0,
        'average_state_payment': np.mean([p for p in state_payments if p > 0]) if state_payments else 0,
    }
    
    # Calculate breakeven state (if reached)
    cumulative_payments = np.cumsum(state_payments)
    breakeven_states = np.where(cumulative_payments >= total_training_costs)[0]
    metrics['breakeven_state'] = breakeven_states[0] + 1 if len(breakeven_states) > 0 else None
    
    # Calculate simple IRR (annualized return) using original method
    if duration_months > 0 and total_training_costs > 0:
        years = duration_months / 12
        irr = (np.power(total_payments / total_training_costs, 1/years) - 1) * 100 if total_payments > 0 else -100
        metrics['annual_irr'] = irr
    else:
        metrics['annual_irr'] = None
    
    # Calculate the more accurate monthly-based IRR
    metrics['monthly_based_irr'] = calculate_monthly_irr(results)
    
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
    if 'state_name' in results['state_results'][-1]:
        print(f"Final State: {results['state_results'][-1]['state_name']} (index {results['final_state_index']})")
    else:
        print(f"Final State: {results['final_state_index']}")
    print(f"States Completed: {results['completed_states']}")
    print(f"Number of States: {metrics['num_states']}")
    print(f"Duration (calculated in months): {metrics['duration_months']} months")
    if results['dropout']:
        if 'state_name' in results['state_results'][-1]:
            print(f"Status: Dropped out during {results['state_results'][-1]['state_name']}")
        else:
            print(f"Status: Dropped out during state {results['final_state_index']}")
    elif results['completed']:
        print("Status: Completed all states")
    
    # Calculate total payments from state-by-state data
    total_payments = sum(results['state_payments'])
    
    # Financial Summary
    print("\nFinancial Summary:")
    print(f"Total Training Costs: ${metrics['total_training_costs']:,.2f}")
    print(f"Total Payments Made: ${total_payments:,.2f}")
    print(f"Net Returns: ${total_payments - metrics['total_training_costs']:,.2f}")
    
    # Performance Metrics
    print("\nPerformance Metrics:")
    if metrics['breakeven_state']:
        print(f"Breakeven State: {metrics['breakeven_state']}")
    else:
        print("Breakeven: Not reached")
    print(f"Repayment Rate: {metrics['repayment_rate']:.1f}%")
    if metrics['annual_irr'] is not None:
        print(f"Simple Annual IRR: {metrics['annual_irr']:.1f}%")
    if metrics['monthly_based_irr'] is not None:
        print(f"Monthly-Based Annual IRR: {metrics['monthly_based_irr']:.1f}%")
    
    # Averages
    print("\nState Averages:")
    print(f"Average State Salary: ${metrics['average_state_salary']:,.2f}")
    print(f"Average State Payment: ${metrics['average_state_payment']:,.2f}")
    
    # Return Profile
    if metrics['breakeven_state']:
        states_after_breakeven = metrics['num_states'] - metrics['breakeven_state']
        if states_after_breakeven > 0:
            returns_after_breakeven = sum(results['state_payments'][metrics['breakeven_state']:])
            print(f"\nReturns after Breakeven: ${returns_after_breakeven:,.2f}")
            print(f"States after Breakeven: {states_after_breakeven}")


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
                'breakeven_state': metrics['breakeven_state'],
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
            'breakeven_rate': df['breakeven_state'].notna().mean() * 100,
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
    
    # Print state config debug info
    print("\nDEBUG - State Configuration:")
    for i, state in enumerate(state_configs):
        print(f"State {i} ({state.name}): payment_fraction={state.payment_fraction}, base_salary={state.base_salary}, provider={state.provider}")
    
    # Initialize state-specific tracking
    state_salary_sums = {i: 0.0 for i in range(num_states)}
    state_salary_counts = {i: 0 for i in range(num_states)} # Track states with salary > 0
    state_count = {i: 0 for i in range(num_states)}
    state_total_costs = {i: 0.0 for i in range(num_states)}
    state_total_payments = {i: 0.0 for i in range(num_states)}
    state_entry_counts = {i: 0 for i in range(num_states)} # Track how many simulations entered each state
    
    # Provider tracking
    provider_counts = {"Disney": 0, "Costa": 0, None: 0}
    
    # Add debug counters for payment tracking
    debug_payment_counts = {i: 0 for i in range(num_states)}
    debug_payment_sums = {i: 0.0 for i in range(num_states)}
    
    for i in range(config.num_students):
        # Use student index as seed if no random seed provided
        seed = config.random_seed + i if config.random_seed is not None else i
        
        result = run_simulation(
            state_configs=state_configs,
            random_seed=seed,
            simulation_config=config
        )
        all_results.append(result)
        
        # Track selected provider
        provider = result.get('selected_provider')
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
        
        # Debug output for first few simulations
        if i < 3:  # Print debug info for first 3 students
            print(f"\nDEBUG - Student {i} Payment Details (Provider: {provider}):")
            for j, state in enumerate(result['state_results']):
                state_idx = state['state_index']
                if state_idx < num_states:
                    payment_fraction = state.get('payment_fraction', 0)
                    print(f"State {j}: Index {state_idx} ({state_configs[state_idx].name}), " +
                          f"State Salary: ${state['current_state_salary']:.2f}, " +
                          f"Payment: ${state['state_payment']:.2f}, " +
                          f"Fraction: {payment_fraction:.2f}")
        
        # Track state-specific metrics
        for state_result in result['state_results']:
            state_idx = state_result['state_index']
            if state_idx < num_states:  # Ensure valid state
                # Track salary data
                state_salary = state_result['current_state_salary']
                state_salary_sums[state_idx] += state_salary
                state_count[state_idx] += 1
                if state_salary > 0:
                    state_salary_counts[state_idx] += 1
                
                # Track payment data - this is the key part
                state_payment = state_result['state_payment']
                state_total_payments[state_idx] += state_payment
                
                # Debug counters
                if state_payment > 0:
                    debug_payment_counts[state_idx] += 1
                    debug_payment_sums[state_idx] += state_payment
    
    # Print provider distribution
    total_students = sum(provider_counts.values())
    print("\nProvider Distribution:")
    for provider, count in provider_counts.items():
        if provider:  # Skip None provider
            percent = (count / total_students) * 100
            print(f"{provider}: {count} students ({percent:.1f}%)")
    
    # Print debug payment info
    print("\nDEBUG - Payment Tracking:")
    for i in range(num_states):
        if debug_payment_counts[i] > 0:
            avg = debug_payment_sums[i] / debug_payment_counts[i]
            print(f"State {i} ({state_configs[i].name}): " +
                  f"States with payments: {debug_payment_counts[i]}, " +
                  f"Total payments: ${debug_payment_sums[i]:.2f}, " +
                  f"Avg payment: ${avg:.2f}")
        else:
            print(f"State {i} ({state_configs[i].name}): No payments recorded")

    # Refined entry count logic: count if state appears in state results at all
    state_entry_counts = {i: 0 for i in range(num_states)}
    for r in all_results:
        unique_states_entered = set(sr['state_index'] for sr in r['state_results'] if sr['state_index'] < num_states)
        for state_idx in unique_states_entered:
            state_entry_counts[state_idx] += 1
            
    # Recalculate total costs based *only* on simulations that entered the state
    state_total_costs = {i: 0.0 for i in range(num_states)}
    for r in all_results:
        unique_states_entered = set(sr['state_index'] for sr in r['state_results'] if sr['state_index'] < num_states)
        for state_idx, cost in enumerate(r['training_costs_by_state']):
            if state_idx < num_states and state_idx in unique_states_entered: # Check if the state was actually entered in this sim
                 state_total_costs[state_idx] += cost
                 
    # Calculate average salaries and payment metrics by state
    state_metrics = {}
    for state_idx in range(num_states):
        num_salary_states = state_salary_counts[state_idx]
        avg_state_salary = (state_salary_sums[state_idx] / num_salary_states 
                      if num_salary_states > 0 else 0.0)
        
        # Calculate average state payment based on salary times payment fraction
        if state_idx < num_states:
            config = state_configs[state_idx]
            if "Cruise" in config.name:
                expected_payment = avg_state_salary * config.payment_fraction
            else:
                expected_payment = 0.0
        else:
            expected_payment = 0.0
        
        # Calculate actual average payment
        avg_payment = (state_total_payments[state_idx] / num_salary_states
                       if num_salary_states > 0 else 0.0)
                       
        state_metrics[state_idx] = {
            'name': state_configs[state_idx].name,
            'provider': state_configs[state_idx].provider if hasattr(state_configs[state_idx], 'provider') else "",
            'avg_state_salary': avg_state_salary,
            'avg_payment': avg_payment,
            'expected_payment': expected_payment, # Add this for debugging
            'state_count': state_count[state_idx],
            'salary_count': num_salary_states
        }
    
    # Convert results to DataFrame for analysis
    df = pd.DataFrame([
        {
            'dropout': r['dropout'],
            'completed': r['completed'],
            'duration_states': len(r['state_results']),
            'total_training_costs': r['total_training_costs'],
            'total_payments': r['total_payments'],
            'net_cash_flow': r['net_cash_flow'],
            'final_state_index': r['final_state_index'],
            'monthly_irr': calculate_monthly_irr(r),  # Calculate monthly IRR for each simulation
            'provider': r.get('selected_provider')
        }
        for r in all_results
    ])
    
    # Calculate key metrics
    roi = df['net_cash_flow'] / df['total_training_costs']
    
    # Calculate average monthly-based IRR, handling None values
    monthly_irr_values = [val for val in df['monthly_irr'] if val is not None]
    avg_monthly_irr = sum(monthly_irr_values) / len(monthly_irr_values) if monthly_irr_values else None
    
    # Calculate provider-specific metrics
    provider_metrics = {}
    for provider in ['Disney', 'Costa']:
        provider_df = df[df['provider'] == provider]
        if len(provider_df) > 0:
            provider_roi = provider_df['net_cash_flow'] / provider_df['total_training_costs']
            provider_metrics[provider] = {
                'count': len(provider_df),
                'avg_training_cost': provider_df['total_training_costs'].mean(),
                'avg_total_payments': provider_df['total_payments'].mean(),
                'avg_net_cash_flow': provider_df['net_cash_flow'].mean(),
                'avg_roi': provider_roi.mean() * 100,
                'roi_std': provider_roi.std() * 100
            }
    
    # Print state total payments for debugging
    print("\nDEBUG - Final State Total Payments:")
    for i in range(num_states):
        state_name = state_configs[i].name if i < len(state_configs) else f"State {i}"
        print(f"State {i} ({state_name}): Total payments = ${state_total_payments.get(i, 0):.2f}")
    
    return {
        'completion_rate': df['completed'].mean() * 100,
        'dropout_rate': df['dropout'].mean() * 100,
        'avg_duration_states': df['duration_states'].mean(),
        'avg_training_cost': df['total_training_costs'].mean(),
        'avg_total_payments': df['total_payments'].mean(),
        'avg_net_cash_flow': df['net_cash_flow'].mean(),
        'avg_roi': roi.mean() * 100,
        'roi_std': roi.std() * 100,
        'roi_10th': roi.quantile(0.1) * 100,
        'roi_90th': roi.quantile(0.9) * 100,
        'avg_annual_irr': df['monthly_irr'].mean(),  # Simple mean (may be None)
        'avg_monthly_irr': avg_monthly_irr,  # Properly calculated average of non-None values
        'state_distribution': df['final_state_index'].value_counts().sort_index().to_dict(),
        'state_metrics': state_metrics,
        'state_total_costs': state_total_costs,
        'state_total_payments': state_total_payments,
        'state_entry_counts': state_entry_counts,
        'provider_metrics': provider_metrics,
        'provider_distribution': {provider: count for provider, count in provider_counts.items() if provider}
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
    print(f"Average Duration (states): {results['avg_duration_states']:.1f}")
    
    print("\nFinancial Statistics:")
    print(f"Average Training Cost: ${results['avg_training_cost']:,.2f}")
    print(f"Average Total Payments: ${results['avg_total_payments']:,.2f}")
    print(f"Average Net Cash Flow: ${results['avg_net_cash_flow']:,.2f}")
    
    print("\nROI Statistics:")
    print(f"Average ROI: {results['avg_roi']:.1f}%")
    print(f"ROI Standard Deviation: {results['roi_std']:.1f}%")
    print(f"ROI 10th Percentile: {results['roi_10th']:.1f}%")
    print(f"ROI 90th Percentile: {results['roi_90th']:.1f}%")
    
    # Print provider-specific metrics
    if 'provider_metrics' in results:
        print("\nProvider-Specific Results:")
        print("--------------------------")
        for provider, metrics in results['provider_metrics'].items():
            print(f"\n{provider} ({metrics['count']} students):")
            print(f"  Avg Training Cost: ${metrics['avg_training_cost']:,.2f}")
            print(f"  Avg Total Payments: ${metrics['avg_total_payments']:,.2f}")
            print(f"  Avg Net Cash Flow: ${metrics['avg_net_cash_flow']:,.2f}")
            print(f"  Avg ROI: {metrics['avg_roi']:.1f}%")
            print(f"  ROI Standard Deviation: {metrics['roi_std']:.1f}%")
    
    # Print provider distribution
    if 'provider_distribution' in results:
        print("\nProvider Distribution:")
        total = sum(results['provider_distribution'].values())
        for provider, count in results['provider_distribution'].items():
            print(f"{provider}: {count} students ({count/total*100:.1f}%)")
    
    print("\nState-by-State Analysis:")
    print("------------------------")
    print(f"{'State':<25} {'Provider':<10} {'Entries':>8} {'State Salary':>15} {'Avg Payment':>15} {'Expected':>15} {'Total Payments':>20}")
    print("-" * 110)
    
    for state_idx, metrics in results['state_metrics'].items():
        # Ensure state name exists before printing
        state_name = metrics.get('name', f'State {state_idx}')
        provider = metrics.get('provider', "")
        entries = results['state_entry_counts'].get(state_idx, 0)
        total_payments = results['state_total_payments'].get(state_idx, 0.0)
        expected_payment = metrics.get('expected_payment', 0.0)
        
        print(f"{state_name:<25} {provider:<10} {entries:>8} ${metrics['avg_state_salary']:>14,.2f} ${metrics['avg_payment']:>14,.2f} ${expected_payment:>14,.2f} ${total_payments:>19,.2f}")
    
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
    print("Each state has a single dropout chance that applies at the beginning of the state,")
    print("making dropout rates more intuitive and easier to configure.")
    
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
    print("\nThese rates directly represent the percentage of students who")
    print("will not complete each state, making the simulation more intuitive.")

if __name__ == "__main__":
    print("Career Cruise Simulator")
    print("======================")
    print("Running scenarios from simulation_config.py")
    
    # Run all scenarios
    configs = {
        "Baseline": DEFAULT_CONFIG
    }
    
    for name, config in configs.items():
        print(f"\nRunning {name} scenario with {config.num_students} students...")
        results = run_simulation_batch(config)
        print_simulation_results(results, name)
    
    # Run detailed state transition analysis for baseline scenario
    print("\nAnalyzing state transitions and dropout patterns for baseline scenario...")
    analyze_state_transitions(DEFAULT_CONFIG, num_simulations=500)
    
    # Example of running a single simulation with baseline config
    print("\nRunning a single simulation example with baseline configuration:")
    baseline_state_configs = DEFAULT_CONFIG.create_state_configs()
    single_result = run_simulation(state_configs=baseline_state_configs, random_seed=42)
    print_simulation_summary(single_result)
    
    # Uncomment to run cruise configuration comparison
    # print("\nRunning cruise configuration comparison...")
    # print_cruise_comparison(max_cruises=5, num_simulations=50) 
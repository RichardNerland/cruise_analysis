from dataclasses import dataclass
from typing import List, Optional

@dataclass
class StateConfig:
    """Configuration for a single state in the training/work sequence"""
    training_cost: float       # Cost of training for this state
    dropout_rate: float        # Probability of dropping out for this entire state (not per month)
    base_salary: float        # Base monthly salary (only used for first cruise)
    salary_increase_pct: float # Percentage increase from previous salary
    salary_variation_pct: float # Plus/minus percentage variation in salary
    duration_months: int       # How long this state lasts
    payment_fraction: float    # Fraction of salary paid as ISA/fee
    name: str = ""             # Name of this state for reporting

@dataclass
class SimulationConfig:
    """Configuration for running multiple student simulations"""
    
    # General simulation parameters
    num_students: int = 100
    random_seed: Optional[int] = None
    max_months: int = 48
    
    # Training Config
    basic_training_cost: float = 1500
    basic_training_dropout_rate: float = 0.10  # 10% chance of dropping out during Training
    basic_training_duration: int = 3
    
    # Transportation and placement Config (optional)
    include_advanced_training: bool = True
    advanced_training_cost: float = 500
    advanced_training_dropout_rate: float = 0.15  # 15% chance of dropping out during Transportation and placement
    advanced_training_duration: int = 3
    
    # First Cruise Config
    first_cruise_base_salary: float = 5000
    first_cruise_dropout_rate: float = 0.03  # 3% chance of dropping out during First Cruise
    first_cruise_salary_variation: float = 6.0  # ±6%
    first_cruise_duration: int = 8
    first_cruise_payment_fraction: float = 0.14  # 14%
    
    # Subsequent Cruises Config
    num_additional_cruises: int = 4  # Number of cruises after first cruise
    subsequent_cruise_dropout_rate: float = 0.03  # 3% chance of dropping out during each subsequent cruise
    subsequent_cruise_salary_increase: float = 10.0  # 10% increase per cruise
    subsequent_cruise_salary_variation: float = 5.0  # ±5%
    subsequent_cruise_duration: int = 8
    subsequent_cruise_payment_fraction: float = 0.14

    def create_state_configs(self) -> List[StateConfig]:
        """Create state configurations based on the current settings"""
        states = []
        
        # Add Training
        states.append(
            StateConfig(
                training_cost=self.basic_training_cost,
                dropout_rate=self.basic_training_dropout_rate,
                base_salary=0,
                salary_increase_pct=0,
                salary_variation_pct=0,
                duration_months=self.basic_training_duration,
                payment_fraction=0,
                name="Training"
            )
        )
        
        # Add Transportation and placement if enabled
        if self.include_advanced_training:
            states.append(
                StateConfig(
                    training_cost=self.advanced_training_cost,
                    dropout_rate=self.advanced_training_dropout_rate,
                    base_salary=0,
                    salary_increase_pct=0,
                    salary_variation_pct=0,
                    duration_months=self.advanced_training_duration,
                    payment_fraction=0,
                    name="Transportation and placement"
                )
            )
        
        # Add First Cruise
        states.append(
            StateConfig(
                training_cost=0,
                dropout_rate=self.first_cruise_dropout_rate,
                base_salary=self.first_cruise_base_salary,
                salary_increase_pct=0,
                salary_variation_pct=self.first_cruise_salary_variation,
                duration_months=self.first_cruise_duration,
                payment_fraction=self.first_cruise_payment_fraction,
                name="First Cruise"
            )
        )
        
        # Add Subsequent Cruises
        for i in range(self.num_additional_cruises):
            states.append(
                StateConfig(
                    training_cost=0,
                    dropout_rate=self.subsequent_cruise_dropout_rate,
                    base_salary=0,
                    salary_increase_pct=self.subsequent_cruise_salary_increase,
                    salary_variation_pct=self.subsequent_cruise_salary_variation,
                    duration_months=self.subsequent_cruise_duration,
                    payment_fraction=self.subsequent_cruise_payment_fraction,
                    name=f"Cruise {i+2}"  # +2 because first cruise is already added
                )
            )
        
        return states

# Baseline scenario - moderate assumptions
BASELINE_CONFIG = SimulationConfig(
    # Training parameters
    basic_training_cost=1500,         # $1500 training cost
    basic_training_dropout_rate=0.10,    # 10% dropout in training
    include_advanced_training=True,
    advanced_training_cost=500,       # $500 transportation and placement cost
    advanced_training_dropout_rate=0.15,  # 15% dropout in transportation and placement
    
    # First cruise parameters
    first_cruise_base_salary=5000,       # $5000 starting salary
    first_cruise_dropout_rate=0.03,      # 3% dropout in first cruise
    first_cruise_salary_variation=6.0,    # ±6% salary variation
    
    # Subsequent cruise parameters
    num_additional_cruises=3,            # Total of 4 cruises
    subsequent_cruise_dropout_rate=0.03,  # 3% dropout per subsequent cruise
    subsequent_cruise_salary_increase=10.0, # 10% salary increase per cruise
    subsequent_cruise_salary_variation=5.0  # ±5% variation in increases
)

# Optimistic scenario - favorable conditions
OPTIMISTIC_CONFIG = SimulationConfig(
    # Training parameters
    basic_training_cost=1500,         # $1500 training cost
    basic_training_dropout_rate=0.08,    # Lower dropout rate
    include_advanced_training=True,
    advanced_training_cost=500,       # $500 transportation and placement cost
    advanced_training_dropout_rate=0.12,  # Lower dropout rate
    
    # First cruise parameters
    first_cruise_base_salary=5500,       # Higher starting salary
    first_cruise_dropout_rate=0.02,      # Lower dropout rate
    first_cruise_salary_variation=5.0,    # Less salary variation
    
    # Subsequent cruise parameters
    num_additional_cruises=4,            # More cruises
    subsequent_cruise_dropout_rate=0.02,  # Lower dropout rate
    subsequent_cruise_salary_increase=15.0, # Higher salary increases
    subsequent_cruise_salary_variation=4.0  # Less variation in increases
)

# Pessimistic scenario - challenging conditions
PESSIMISTIC_CONFIG = SimulationConfig(
    # Training parameters
    basic_training_cost=1500,         # $1500 training cost
    basic_training_dropout_rate=0.15,    # Higher dropout rate
    include_advanced_training=True,
    advanced_training_cost=500,       # $500 transportation and placement cost
    advanced_training_dropout_rate=0.20,  # Higher dropout rate
    
    # First cruise parameters
    first_cruise_base_salary=4500,       # Lower starting salary
    first_cruise_dropout_rate=0.05,      # Higher dropout rate
    first_cruise_salary_variation=8.0,    # More salary variation
    
    # Subsequent cruise parameters
    num_additional_cruises=2,            # Fewer cruises
    subsequent_cruise_dropout_rate=0.05,  # Higher dropout rate
    subsequent_cruise_salary_increase=7.0,  # Lower salary increases
    subsequent_cruise_salary_variation=7.0  # More variation in increases
)

# Default config is the same as baseline for backward compatibility
DEFAULT_CONFIG = BASELINE_CONFIG 
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
    provider: str = ""         # Cruise provider (Disney, Costa, etc.)

@dataclass
class SimulationConfig:
    """Configuration for running multiple student simulations"""
    
    # General simulation parameters
    num_students: int = 100
    random_seed: Optional[int] = None
    
    # Training Config
    basic_training_cost: float = 3000
    basic_training_dropout_rate: float = 0.10  # 10% chance of failing the training
    basic_training_duration: int = 8
    
    # Training offer stage Config (new)
    include_offer_stage: bool = True
    no_offer_rate: float = 0.30  # 30% chance of not receiving an offer
    offer_stage_duration: int = 1  # 1 month duration
    
    # Early contract termination stage Config (new)
    include_early_termination: bool = True
    early_termination_rate: float = 0.10  # 10% chance of early termination
    early_termination_duration: int = 1  # 1 month duration
    
    # Transportation and placement Config (optional)
    include_advanced_training: bool = True
    advanced_training_cost: float = 0
    advanced_training_dropout_rate: float = 0.1  # 5% chance of dropping out during Transportation and placement
    advanced_training_duration: int = 3
    
    # Provider allocation after training
    disney_allocation_pct: float = 30.0  # 30% go to Disney
    costa_allocation_pct: float = 70.0   # 70% go to Costa
    
    # Disney Cruise Config
    disney_first_cruise_salary: float = 5100
    disney_second_cruise_salary: float = 5400
    disney_third_cruise_salary: float = 18000
    disney_cruise_duration: int = 6
    disney_cruise_dropout_rate: float = 0.03
    disney_cruise_salary_variation: float = 5.0
    disney_cruise_payment_fraction: float = 0.14
    
    # Costa Cruise Config
    costa_first_cruise_salary: float = 5100
    costa_second_cruise_salary: float = 5850
    costa_third_cruise_salary: float = 9000
    costa_cruise_duration: int = 7
    costa_cruise_dropout_rate: float = 0.03
    costa_cruise_salary_variation: float = 5.0
    costa_cruise_payment_fraction: float = 0.14
    
    # Break Config
    include_breaks: bool = True  # Whether to include breaks between cruises
    break_duration: int = 2  # Default 2 months for breaks
    break_dropout_rate: float = 0.0  # Default 0% chance of dropping out during breaks
    
    # Cruise Count Config
    num_cruises: int = 3  # Number of cruises per provider

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
        
        # Add Offer Stage if enabled
        if self.include_offer_stage:
            states.append(
                StateConfig(
                    training_cost=0,  # No cost for this stage
                    dropout_rate=self.no_offer_rate,
                    base_salary=0,
                    salary_increase_pct=0,
                    salary_variation_pct=0,
                    duration_months=self.offer_stage_duration,
                    payment_fraction=0,
                    name="Offer Stage"
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
        
        # Add Early Termination Stage if enabled
        if self.include_early_termination:
            states.append(
                StateConfig(
                    training_cost=0,  # No cost for this stage
                    dropout_rate=self.early_termination_rate,
                    base_salary=0,
                    salary_increase_pct=0,
                    salary_variation_pct=0,
                    duration_months=self.early_termination_duration,
                    payment_fraction=0,
                    name="Early Termination Stage"
                )
            )
        
        # Now, we need to handle the provider-specific cruise sequences
        # First generate Disney cruises
        disney_states = self._create_provider_states(
            provider="Disney",
            salaries=[self.disney_first_cruise_salary, self.disney_second_cruise_salary, self.disney_third_cruise_salary],
            duration=self.disney_cruise_duration,
            dropout_rate=self.disney_cruise_dropout_rate,
            salary_variation=self.disney_cruise_salary_variation,
            payment_fraction=self.disney_cruise_payment_fraction
        )
        
        # Then generate Costa cruises
        costa_states = self._create_provider_states(
            provider="Costa",
            salaries=[self.costa_first_cruise_salary, self.costa_second_cruise_salary, self.costa_third_cruise_salary],
            duration=self.costa_cruise_duration,
            dropout_rate=self.costa_cruise_dropout_rate,
            salary_variation=self.costa_cruise_salary_variation,
            payment_fraction=self.costa_cruise_payment_fraction
        )
        
        # Return combined state list - both providers will be included
        # The actual provider selection will happen in the simulation logic
        return states + disney_states + costa_states
    
    def _create_provider_states(self, provider: str, salaries: List[float], duration: int, 
                               dropout_rate: float, salary_variation: float, payment_fraction: float) -> List[StateConfig]:
        """Helper method to create cruise states for a specific provider"""
        states = []
        
        for i, salary in enumerate(salaries[:self.num_cruises]):
            cruise_number = i + 1
            
            # Add the cruise
            states.append(
                StateConfig(
                    training_cost=0,
                    dropout_rate=dropout_rate,
                    base_salary=salary,
                    salary_increase_pct=0,  # Not using percentage increase anymore
                    salary_variation_pct=salary_variation,
                    duration_months=duration,
                    payment_fraction=payment_fraction,
                    name=f"{provider} Cruise {cruise_number}",
                    provider=provider
                )
            )
            
            # Add break after cruise if breaks are enabled and it's not the last cruise
            if self.include_breaks and i < len(salaries) - 1:
                states.append(
                    StateConfig(
                        training_cost=0,
                        dropout_rate=self.break_dropout_rate,
                        base_salary=0,  # No salary during breaks
                        salary_increase_pct=0,
                        salary_variation_pct=0,
                        duration_months=self.break_duration,
                        payment_fraction=0,  # No payments during breaks
                        name=f"{provider} Break {cruise_number}",
                        provider=provider
                    )
                )
        
        return states

# Default simulation configuration with the new cruise provider model
DEFAULT_CONFIG = SimulationConfig()

# Baseline configuration - updated with provider-specific cruise details
BASELINE_CONFIG = SimulationConfig(
    basic_training_cost=1500,
    basic_training_dropout_rate=0.10,
    basic_training_duration=6,
    include_offer_stage=True,
    no_offer_rate=0.30,
    offer_stage_duration=1,
    include_early_termination=True,
    early_termination_rate=0.10,
    early_termination_duration=1,
    advanced_training_cost=500,
    advanced_training_dropout_rate=0.15,
    advanced_training_duration=5,
    disney_allocation_pct=30.0,
    costa_allocation_pct=70.0,
    disney_first_cruise_salary=5100,
    disney_second_cruise_salary=5400,
    disney_third_cruise_salary=18000,
    disney_cruise_duration=6,
    costa_first_cruise_salary=5100,
    costa_second_cruise_salary=5850,
    costa_third_cruise_salary=9000,
    costa_cruise_duration=7,
    include_breaks=True,
    break_duration=2,
    num_cruises=3
)

# Optimistic configuration - lower dropout rates, slightly higher salaries
OPTIMISTIC_CONFIG = SimulationConfig(
    basic_training_cost=1500,
    basic_training_dropout_rate=0.05,  # Lower dropout rate
    basic_training_duration=6,
    include_offer_stage=True,
    no_offer_rate=0.20,  # Lower no-offer rate
    offer_stage_duration=1,
    include_early_termination=True,
    early_termination_rate=0.05,  # Lower early termination rate
    early_termination_duration=1,
    advanced_training_cost=500,
    advanced_training_dropout_rate=0.08,  # Lower dropout rate
    advanced_training_duration=5,
    disney_allocation_pct=30.0,
    costa_allocation_pct=70.0,
    disney_first_cruise_salary=5100,
    disney_second_cruise_salary=5500,  # Slightly higher
    disney_third_cruise_salary=18500,  # Slightly higher
    disney_cruise_duration=6,
    disney_cruise_dropout_rate=0.02,  # Lower dropout rate
    costa_first_cruise_salary=5100,
    costa_second_cruise_salary=5900,  # Slightly higher
    costa_third_cruise_salary=9200,   # Slightly higher
    costa_cruise_duration=7,
    costa_cruise_dropout_rate=0.02,   # Lower dropout rate
    include_breaks=True,
    break_duration=2,
    num_cruises=3
)

# Pessimistic configuration - higher dropout rates, slightly lower salaries
PESSIMISTIC_CONFIG = SimulationConfig(
    basic_training_cost=1500,
    basic_training_dropout_rate=0.15,  # Higher dropout rate
    basic_training_duration=6,
    include_offer_stage=True,
    no_offer_rate=0.40,  # Higher no-offer rate
    offer_stage_duration=1,
    include_early_termination=True,
    early_termination_rate=0.15,  # Higher early termination rate
    early_termination_duration=1,
    advanced_training_cost=500,
    advanced_training_dropout_rate=0.20,  # Higher dropout rate
    advanced_training_duration=5,
    disney_allocation_pct=30.0,
    costa_allocation_pct=70.0,
    disney_first_cruise_salary=5000,  # Slightly lower
    disney_second_cruise_salary=5300,  # Slightly lower
    disney_third_cruise_salary=17500,  # Slightly lower
    disney_cruise_duration=6,
    disney_cruise_dropout_rate=0.04,  # Higher dropout rate
    costa_first_cruise_salary=5000,   # Slightly lower
    costa_second_cruise_salary=5750,  # Slightly lower
    costa_third_cruise_salary=8800,   # Slightly lower
    costa_cruise_duration=7,
    costa_cruise_dropout_rate=0.04,   # Higher dropout rate
    include_breaks=True,
    break_duration=2,
    num_cruises=3
) 
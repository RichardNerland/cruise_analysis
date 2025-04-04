import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import random
import argparse

# Import the cruise model
from simple_cruise_model import (
    run_simulation, 
    run_simulation_batch, 
    create_default_state_configs,
    calculate_summary_metrics, 
    print_simulation_summary
)
from simulation_config import (
    StateConfig, 
    SimulationConfig, 
    BASELINE_CONFIG, 
    OPTIMISTIC_CONFIG, 
    PESSIMISTIC_CONFIG
)

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # Expose the server variable for production

# Enable the app to be embedded in an iframe
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Cruise Career Analysis Tool</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Define the preset scenarios
preset_scenarios = {
    'baseline': {
        'name': 'Baseline',
        'description': 'Moderate assumptions about cruise career progression.',
        'config': BASELINE_CONFIG
    },
    'optimistic': {
        'name': 'Optimistic',
        'description': 'Favorable conditions with higher salaries and lower dropout rates.',
        'config': OPTIMISTIC_CONFIG
    },
    'pessimistic': {
        'name': 'Pessimistic',
        'description': 'Challenging conditions with lower salaries and higher dropout rates.',
        'config': PESSIMISTIC_CONFIG
    }
}

# Dropdown options for number of students
student_options = [
    {'label': '10 simulations', 'value': 10},
    {'label': '50 simulations', 'value': 50},
    {'label': '100 simulations', 'value': 100},
    {'label': '200 simulations', 'value': 200},
    {'label': '500 simulations', 'value': 500}
]

# Dropdown options for number of simulations
sim_options = [
    {'label': '10 Monte Carlo runs (faster)', 'value': 10},
    {'label': '50 Monte Carlo runs', 'value': 50},
    {'label': '100 Monte Carlo runs (recommended)', 'value': 100}
]

# Define the layout of the app
app.layout = html.Div([
    html.H1("Cruise Career Analysis Tool", style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    # Tabs for different sections
    dcc.Tabs([
        dcc.Tab(label='About', children=[
            html.Div([
                html.H1("Cruise Career Analysis Tool", style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#2c3e50'}),
                
                # Background Section
                html.Div([
                    html.H2("Background", style={'color': '#2c3e50', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                    html.P([
                        "This tool simulates career paths for crew members through a sequence of training and cruise assignments, ",
                        "analyzing financial outcomes including training costs, payments, and return on investment."
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                ], style={'marginBottom': '30px'}),
                
                # Objectives Section
                html.Div([
                    html.H2("Objectives", style={'color': '#2c3e50', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                    html.P([
                        "This simulation tool helps stakeholders understand the financial outcomes of cruise careers across various scenarios. The tool specifically aims to:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li("Model expected returns on training investments across different career paths", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Simulate how dropout rates and salary variations affect career trajectories", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Provide program designers with data-driven insights for career structure", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Analyze the impact of different payment structures on return on investment", style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'})
                ], style={'marginBottom': '30px'}),
                
                # Implementation Section
                html.Div([
                    html.H2("Implementation", style={'color': '#2c3e50', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                    html.P([
                        "This interactive dashboard allows users to:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li("Select preset scenarios or customize career parameters", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Adjust training costs, dropout rates, and salary progression", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Customize payment percentages and cruise durations", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Run Monte Carlo simulations to test robustness", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Compare multiple scenarios to identify optimal career structures", style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'})
                ], style={'marginBottom': '30px'})
            ], style={'padding': '20px', 'maxWidth': '1200px', 'margin': '0 auto'})
        ]),
        
        dcc.Tab(label='Simulation', children=[
            html.Div([
                html.H2("Cruise Career Simulation Dashboard", style={'textAlign': 'center', 'marginBottom': '30px'}),
                
                html.Div([
                    # Left panel for inputs
                    html.Div([
                        html.H3("Simulation Parameters", style={'marginBottom': '20px'}),
                        
                        # Preset scenarios selector
                        html.Div([
                            html.Label("Preset Scenarios:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
                            html.P("Select a pre-configured scenario to automatically set career parameters", 
                                 style={'fontSize': '0.85em', 'margin': '2px 0 10px 0'}),
                            dcc.Dropdown(
                                id="preset-scenario",
                                options=[{'label': preset_scenarios[k]['name'], 'value': k} for k in preset_scenarios.keys()],
                                value="baseline",
                                placeholder="Select a preset scenario",
                                style={'fontWeight': 'bold'}
                            ),
                            html.Div(id="preset-description", style={'color': '#666', 'fontSize': '0.9em', 'marginTop': '5px', 'fontStyle': 'italic'})
                        ], style={'marginBottom': '20px', 'backgroundColor': '#e6f7ff', 'padding': '15px', 'borderRadius': '5px'}),
                        
                        # Basic settings
                        html.Div([
                            html.H4("Basic Settings", style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Number of Monte Carlo Runs:"),
                                dcc.Dropdown(
                                    id="num-sims",
                                    options=sim_options,
                                    value=50
                                ),
                                html.P("Each run uses the same parameters but with different random values for salary variation, dropout chance, etc.",
                                      style={'fontSize': '0.8em', 'color': '#666', 'marginTop': '5px'})
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Number of Career Simulations:"),
                                dcc.Dropdown(
                                    id="num-students",
                                    options=student_options,
                                    value=100
                                ),
                                html.P("Each simulation follows one student through their career path. Results are aggregated across all simulations.",
                                      style={'fontSize': '0.8em', 'color': '#666', 'marginTop': '5px'})
                            ], style={'marginBottom': '15px'})
                            
                        ], style={'marginBottom': '20px', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px'}),
                        
                        # Training parameters
                        html.Div([
                            html.H4("Training Parameters", style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Include Advanced Training:"),
                                dcc.RadioItems(
                                    id="include-advanced-training",
                                    options=[
                                        {'label': 'Yes', 'value': True},
                                        {'label': 'No', 'value': False}
                                    ],
                                    value=True,
                                    inline=True
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Basic Training Cost ($):"),
                                dcc.Input(
                                    id="basic-training-cost",
                                    type="number",
                                    value=2000,
                                    min=0,
                                    step=100,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Basic Training Dropout Rate (%):"),
                                dcc.Slider(
                                    id="basic-training-dropout-rate",
                                    min=0,
                                    max=50,
                                    step=1,
                                    value=15,
                                    marks={i: f'{i}%' for i in range(0, 51, 10)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Basic Training Duration (months):"),
                                dcc.Input(
                                    id="basic-training-duration",
                                    type="number",
                                    value=3,
                                    min=1,
                                    max=24,
                                    step=1,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            # Add the advanced training parameters directly in the layout instead of using conditional rendering
                            html.Div([
                                html.Div([
                                    html.Label("Advanced Training Cost ($):"),
                                    dcc.Input(
                                        id="advanced-training-cost",
                                        type="number",
                                        value=2000,
                                        min=0,
                                        step=100,
                                        style={"width": "100%"}
                                    )
                                ], style={'marginBottom': '15px'}),
                                
                                html.Div([
                                    html.Label("Advanced Training Dropout Rate (%):"),
                                    dcc.Slider(
                                        id="advanced-training-dropout-rate",
                                        min=0,
                                        max=50,
                                        step=1,
                                        value=12,
                                        marks={i: f'{i}%' for i in range(0, 51, 10)},
                                    )
                                ], style={'marginBottom': '15px'}),
                                
                                html.Div([
                                    html.Label("Advanced Training Duration (months):"),
                                    dcc.Input(
                                        id="advanced-training-duration",
                                        type="number",
                                        value=3,
                                        min=1,
                                        max=24,
                                        step=1,
                                        style={"width": "100%"}
                                    )
                                ], style={'marginBottom': '15px'})
                            ], id="advanced-training-container", style={'marginBottom': '15px', 'backgroundColor': '#e6f7ff', 'padding': '15px', 'borderRadius': '5px'})
                        ], style={'marginBottom': '20px', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px'}),
                        
                        # Cruise parameters
                        html.Div([
                            html.H4("Cruise Parameters", style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Number of Additional Cruises:"),
                                dcc.Input(
                                    id="num-additional-cruises",
                                    type="number",
                                    value=3,
                                    min=0,
                                    max=10,
                                    step=1,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("First Cruise Base Salary ($):"),
                                dcc.Input(
                                    id="first-cruise-base-salary",
                                    type="number",
                                    value=5000,
                                    min=0,
                                    step=100,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("First Cruise Dropout Rate (%):"),
                                dcc.Slider(
                                    id="first-cruise-dropout-rate",
                                    min=0,
                                    max=50,
                                    step=1,
                                    value=15,
                                    marks={i: f'{i}%' for i in range(0, 51, 10)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("First Cruise Salary Variation (%):"),
                                dcc.Slider(
                                    id="first-cruise-salary-variation",
                                    min=0,
                                    max=20,
                                    step=0.5,
                                    value=6,
                                    marks={i: f'{i}%' for i in range(0, 21, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("First Cruise Duration (months):"),
                                dcc.Input(
                                    id="first-cruise-duration",
                                    type="number",
                                    value=8,
                                    min=1,
                                    max=24,
                                    step=1,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("First Cruise Payment Fraction (%):"),
                                dcc.Slider(
                                    id="first-cruise-payment-fraction",
                                    min=0,
                                    max=30,
                                    step=0.5,
                                    value=14,
                                    marks={i: f'{i}%' for i in range(0, 31, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Hr(),
                            
                            html.Div([
                                html.Label("Subsequent Cruise Dropout Rate (%):"),
                                dcc.Slider(
                                    id="subsequent-cruise-dropout-rate",
                                    min=0,
                                    max=20,
                                    step=0.5,
                                    value=2,
                                    marks={i: f'{i}%' for i in range(0, 21, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Subsequent Cruise Salary Increase (%):"),
                                dcc.Slider(
                                    id="subsequent-cruise-salary-increase",
                                    min=0,
                                    max=25,
                                    step=0.5,
                                    value=10,
                                    marks={i: f'{i}%' for i in range(0, 26, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Subsequent Cruise Salary Variation (%):"),
                                dcc.Slider(
                                    id="subsequent-cruise-salary-variation",
                                    min=0,
                                    max=20,
                                    step=0.5,
                                    value=5,
                                    marks={i: f'{i}%' for i in range(0, 21, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Subsequent Cruise Duration (months):"),
                                dcc.Input(
                                    id="subsequent-cruise-duration",
                                    type="number",
                                    value=8,
                                    min=1,
                                    max=24,
                                    step=1,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Subsequent Cruise Payment Fraction (%):"),
                                dcc.Slider(
                                    id="subsequent-cruise-payment-fraction",
                                    min=0,
                                    max=30,
                                    step=0.5,
                                    value=14,
                                    marks={i: f'{i}%' for i in range(0, 31, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                        ], style={'marginBottom': '20px', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px'}),
                        
                        # Run simulation button
                        html.Div([
                            html.Button(
                                "Run Career Simulations", 
                                id="run-simulation", 
                                n_clicks=0,
                                style={
                                    'backgroundColor': '#4CAF50',
                                    'color': 'white',
                                    'padding': '12px 20px',
                                    'borderRadius': '5px',
                                    'border': 'none',
                                    'fontSize': '16px',
                                    'cursor': 'pointer',
                                    'width': '100%',
                                    'fontWeight': 'bold'
                                }
                            ),
                            html.Div(id="loading-message", style={'marginTop': '10px', 'color': '#888'})
                        ])
                    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'backgroundColor': '#f9f9f9', 'borderRadius': '8px'}),
                    
                    # Right panel for results
                    html.Div([
                        html.H3("Results"),
                        
                        html.Div([
                            html.Div(id="summary-stats", style={'marginBottom': '20px'}),
                            
                            dcc.Tabs([
                                dcc.Tab(label='Overview', children=[
                                    html.Div(id="overview-content")
                                ]),
                                dcc.Tab(label='State Progression', children=[
                                    html.Div(id="state-progression-content")
                                ]),
                                dcc.Tab(label='Monthly Metrics', children=[
                                    html.Div(id="monthly-metrics-content")
                                ]),
                                dcc.Tab(label='Detailed Data', children=[
                                    html.Div(id="detailed-data")
                                ]),
                                dcc.Tab(label='State Tracking', children=[
                                    html.Div(id="state-tracking-content")
                                ]),
                                dcc.Tab(label='Scenario Comparison', children=[
                                    html.Div([
                                        html.H4("Compare Saved Scenarios", style={'marginBottom': '15px'}),
                                        html.P("Save multiple scenarios and compare their results side by side."),
                                        
                                        html.Div([
                                            html.Div([
                                                html.Label("Scenario Name:"),
                                                dcc.Input(
                                                    id="scenario-name-input",
                                                    type="text",
                                                    placeholder="Enter a name for this scenario",
                                                    style={'width': '100%'}
                                                )
                                            ], style={'width': '60%', 'display': 'inline-block'}),
                                            
                                            html.Div([
                                                html.Button(
                                                    "Save Current Scenario", 
                                                    id="save-scenario-button", 
                                                    n_clicks=0,
                                                    style={
                                                        'backgroundColor': '#4CAF50',
                                                        'color': 'white',
                                                        'padding': '10px',
                                                        'borderRadius': '5px',
                                                        'border': 'none',
                                                        'width': '100%',
                                                        'cursor': 'pointer'
                                                    }
                                                )
                                            ], style={'width': '35%', 'display': 'inline-block', 'float': 'right'})
                                        ], style={'marginBottom': '20px'}),
                                        
                                        html.Div([
                                            html.H5("Saved Scenarios", style={'marginBottom': '10px'}),
                                            html.Div(id="saved-scenarios-list"),
                                            html.Div([
                                                html.Button(
                                                    "Compare Selected Scenarios", 
                                                    id="compare-scenarios-button", 
                                                    n_clicks=0,
                                                    style={
                                                        'backgroundColor': '#2196F3',
                                                        'color': 'white',
                                                        'padding': '10px',
                                                        'borderRadius': '5px',
                                                        'border': 'none',
                                                        'marginRight': '10px',
                                                        'cursor': 'pointer'
                                                    }
                                                ),
                                                html.Button(
                                                    "Clear All Scenarios", 
                                                    id="clear-scenarios-button", 
                                                    n_clicks=0,
                                                    style={
                                                        'backgroundColor': '#f44336',
                                                        'color': 'white',
                                                        'padding': '10px',
                                                        'borderRadius': '5px',
                                                        'border': 'none',
                                                        'cursor': 'pointer'
                                                    }
                                                )
                                            ], style={'marginTop': '15px', 'marginBottom': '20px'})
                                        ], style={'marginBottom': '20px'}),
                                        
                                        html.Div(id="scenario-comparison-results")
                                    ])
                                ]),
                                dcc.Tab(label='Active Students', children=[
                                    html.Div(id="active-students-content")
                                ])
                            ], style={'marginTop': '20px'})
                        ])
                    ], style={'width': '65%', 'display': 'inline-block', 'padding': '20px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'backgroundColor': '#f9f9f9', 'borderRadius': '8px'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '20px', 'marginBottom': '30px'}),
            ])
        ])
    ]),
    
    # Data stores
    dcc.Store(id='simulation-results-store'),
    dcc.Store(id='simulation-config-store'),
    dcc.Store(id='saved-scenarios-store', data={})
])

# Callback to update preset description
@app.callback(
    Output("preset-description", "children"),
    [Input("preset-scenario", "value")]
)
def update_preset_description(preset_name):
    if preset_name and preset_name in preset_scenarios:
        return preset_scenarios[preset_name]['description']
    return ""

# Callback to show/hide advanced training container
@app.callback(
    Output("advanced-training-container", "style"),
    [Input("include-advanced-training", "value")]
)
def toggle_advanced_training_visibility(include_advanced):
    if include_advanced:
        return {'display': 'block', 'marginBottom': '15px', 'backgroundColor': '#e6f7ff', 'padding': '15px', 'borderRadius': '5px'}
    else:
        return {'display': 'none'}

# Callback to update all parameters when a preset is selected
@app.callback(
    [
        # Training parameters
        Output("include-advanced-training", "value"),
        Output("basic-training-cost", "value"),
        Output("basic-training-dropout-rate", "value"),
        Output("basic-training-duration", "value"),
        Output("advanced-training-cost", "value"),
        Output("advanced-training-dropout-rate", "value"),
        Output("advanced-training-duration", "value"),
        
        # Cruise parameters
        Output("num-additional-cruises", "value"),
        Output("first-cruise-base-salary", "value"),
        Output("first-cruise-dropout-rate", "value"),
        Output("first-cruise-salary-variation", "value"),
        Output("first-cruise-duration", "value"),
        Output("first-cruise-payment-fraction", "value"),
        Output("subsequent-cruise-dropout-rate", "value"),
        Output("subsequent-cruise-salary-increase", "value"),
        Output("subsequent-cruise-salary-variation", "value"),
        Output("subsequent-cruise-duration", "value"),
        Output("subsequent-cruise-payment-fraction", "value"),
    ],
    [Input("preset-scenario", "value")]
)
def update_from_preset(preset_name):
    if preset_name is None or preset_name not in preset_scenarios:
        preset_name = 'baseline'
    
    # Get the config from the preset
    config = preset_scenarios[preset_name]['config']
    
    # Return all values from the config
    return (
        config.include_advanced_training,
        config.basic_training_cost,
        config.basic_training_dropout_rate * 100,
        config.basic_training_duration,
        config.advanced_training_cost,
        config.advanced_training_dropout_rate * 100,
        config.advanced_training_duration,
        config.num_additional_cruises,
        config.first_cruise_base_salary,
        config.first_cruise_dropout_rate * 100,
        config.first_cruise_salary_variation,
        config.first_cruise_duration,
        config.first_cruise_payment_fraction * 100,
        config.subsequent_cruise_dropout_rate * 100,
        config.subsequent_cruise_salary_increase,
        config.subsequent_cruise_salary_variation,
        config.subsequent_cruise_duration,
        config.subsequent_cruise_payment_fraction * 100
    )

# Callback to create a simulation configuration and store it
@app.callback(
    Output("simulation-config-store", "data"),
    [
        Input("include-advanced-training", "value"),
        Input("basic-training-cost", "value"),
        Input("basic-training-dropout-rate", "value"),
        Input("basic-training-duration", "value"),
        Input("advanced-training-cost", "value"),
        Input("advanced-training-dropout-rate", "value"),
        Input("advanced-training-duration", "value"),
        Input("num-additional-cruises", "value"),
        Input("first-cruise-base-salary", "value"),
        Input("first-cruise-dropout-rate", "value"),
        Input("first-cruise-salary-variation", "value"),
        Input("first-cruise-duration", "value"),
        Input("first-cruise-payment-fraction", "value"),
        Input("subsequent-cruise-dropout-rate", "value"),
        Input("subsequent-cruise-salary-increase", "value"),
        Input("subsequent-cruise-salary-variation", "value"),
        Input("subsequent-cruise-duration", "value"),
        Input("subsequent-cruise-payment-fraction", "value"),
        Input("num-students", "value"),
        Input("num-sims", "value")
    ]
)
def update_simulation_config(
    include_advanced_training, 
    basic_training_cost, 
    basic_training_dropout_rate,
    basic_training_duration,
    advanced_training_cost,
    advanced_training_dropout_rate,
    advanced_training_duration,
    num_additional_cruises,
    first_cruise_base_salary,
    first_cruise_dropout_rate,
    first_cruise_salary_variation,
    first_cruise_duration,
    first_cruise_payment_fraction,
    subsequent_cruise_dropout_rate,
    subsequent_cruise_salary_increase,
    subsequent_cruise_salary_variation,
    subsequent_cruise_duration,
    subsequent_cruise_payment_fraction,
    num_students,
    num_sims
):
    # Convert percentage inputs to decimal values
    basic_training_dropout_rate = basic_training_dropout_rate / 100 if basic_training_dropout_rate else 0.15
    advanced_training_dropout_rate = advanced_training_dropout_rate / 100 if advanced_training_dropout_rate else 0.12
    first_cruise_dropout_rate = first_cruise_dropout_rate / 100 if first_cruise_dropout_rate else 0.15
    first_cruise_payment_fraction = first_cruise_payment_fraction / 100 if first_cruise_payment_fraction else 0.14
    subsequent_cruise_dropout_rate = subsequent_cruise_dropout_rate / 100 if subsequent_cruise_dropout_rate else 0.02
    subsequent_cruise_payment_fraction = subsequent_cruise_payment_fraction / 100 if subsequent_cruise_payment_fraction else 0.14
    
    # Create a config dict that can be serialized to JSON
    config = {
        'include_advanced_training': include_advanced_training,
        'basic_training_cost': basic_training_cost,
        'basic_training_dropout_rate': basic_training_dropout_rate,
        'basic_training_duration': basic_training_duration,
        'advanced_training_cost': advanced_training_cost,
        'advanced_training_dropout_rate': advanced_training_dropout_rate,
        'advanced_training_duration': advanced_training_duration,
        'num_additional_cruises': num_additional_cruises,
        'first_cruise_base_salary': first_cruise_base_salary,
        'first_cruise_dropout_rate': first_cruise_dropout_rate,
        'first_cruise_salary_variation': first_cruise_salary_variation,
        'first_cruise_duration': first_cruise_duration,
        'first_cruise_payment_fraction': first_cruise_payment_fraction,
        'subsequent_cruise_dropout_rate': subsequent_cruise_dropout_rate,
        'subsequent_cruise_salary_increase': subsequent_cruise_salary_increase,
        'subsequent_cruise_salary_variation': subsequent_cruise_salary_variation,
        'subsequent_cruise_duration': subsequent_cruise_duration,
        'subsequent_cruise_payment_fraction': subsequent_cruise_payment_fraction,
        'num_students': num_students,
        'num_sims': num_sims,
    }
    
    return config

# *** START: Helper function for progression calculation ***
def calculate_progression_data(state_metrics, config):
    """
    Calculates the theoretical progression of students through states based on config dropout rates.

    Args:
        state_metrics (dict): The pre-processed state metrics dictionary.
        config (dict): The simulation configuration dictionary.

    Returns:
        list: A list of dictionaries, each containing progression data for a state.
              Keys: 'state', 'entered', 'completed', 'dropouts', 'dropout_rate',
                    'avg_salary', 'avg_payment', 'active_months', 'cash_flow_per_student'
    """
    progression_data = []
    total_entered_initial = config.get('num_students', 100)
    entered_count = total_entered_initial # Initialize for the first state

    # Sort state_metrics by state index to ensure correct order
    sorted_metrics = sorted(state_metrics.items(), key=lambda x: int(x[0]))

    for state_idx_str, metrics in sorted_metrics:
        state_idx = int(state_idx_str)
        if isinstance(metrics, dict):
            state_name = metrics.get('name', f"State {state_idx_str}")

            # Determine 'entered' count
            if state_idx == 0:
                entered = entered_count
            else:
                # Entered count is the 'completed' count from the previous state
                if progression_data: # Check if there's a previous state
                    entered = progression_data[-1].get('completed', 0)
                else:
                    entered = 0 # Should not happen if state_idx > 0

            # Calculate dropouts and completions based on configured dropout rates
            dropout_rate = 0.0 # Default as decimal
            if "Basic Training" in state_name:
                dropout_rate = config.get('basic_training_dropout_rate', 0)
            elif "Advanced Training" in state_name:
                dropout_rate = config.get('advanced_training_dropout_rate', 0)
            elif "First Cruise" in state_name:
                dropout_rate = config.get('first_cruise_dropout_rate', 0)
            else: # Subsequent cruises or completion state
                 # Assume 0 dropout for completion states if not specified as a cruise
                if "Cruise" in state_name:
                    dropout_rate = config.get('subsequent_cruise_dropout_rate', 0)
                else:
                    dropout_rate = 0.0 # e.g., for a final "Completed Program" state

            dropouts = round(entered * dropout_rate)
            completed = entered - dropouts

            # Get financial metrics for the state
            avg_salary = metrics.get('avg_salary', 0)
            avg_payment = metrics.get('avg_payment', 0)
            active_months = metrics.get('active_months', 0)

            # Ensure numeric
            avg_salary = avg_salary if isinstance(avg_salary, (int, float)) else 0
            avg_payment = avg_payment if isinstance(avg_payment, (int, float)) else 0
            active_months = active_months if isinstance(active_months, (int, float)) else 0

            # Calculate cash flow per student for this state based on config duration
            cash_flow = 0
            state_duration = 0 # Default duration
            if "Training" in state_name:
                training_cost = 0
                if "Basic" in state_name:
                    training_cost = config.get('basic_training_cost', 0)
                    state_duration = config.get('basic_training_duration', 0) # Get configured duration
                elif "Advanced" in state_name:
                    # Only apply cost if advanced training is included in the config
                    if config.get('include_advanced_training', False):
                        training_cost = config.get('advanced_training_cost', 0)
                        state_duration = config.get('advanced_training_duration', 0) # Get configured duration
                    else:
                        # If not included, skip this state effectively for cash flow
                        training_cost = 0
                        state_duration = 0
                cash_flow = -training_cost
            elif "Cruise" in state_name: # Calculate payment only for cruise states
                avg_monthly_payment = metrics.get('avg_payment', 0) # Get avg monthly payment from simulation results
                avg_monthly_payment = avg_monthly_payment if isinstance(avg_monthly_payment, (int, float)) else 0

                if "First Cruise" in state_name:
                    state_duration = config.get('first_cruise_duration', 0) # Get configured duration
                else: # Subsequent cruises
                    state_duration = config.get('subsequent_cruise_duration', 0) # Get configured duration

                # Calculate cash flow per student = avg monthly payment * duration of state
                cash_flow = avg_monthly_payment * state_duration
            # Assume 0 cash flow for non-training/non-cruise states (like dropouts/completion)

            progression_data.append({
                'state': state_name,
                'entered': entered,
                'completed': completed,
                'dropouts': dropouts,
                'dropout_rate': dropout_rate * 100, # Convert back to percentage for display
                'avg_salary': avg_salary, # Keep avg monthly salary from simulation
                'avg_payment': metrics.get('avg_payment', 0), # Keep avg monthly payment from simulation
                'active_months': state_duration, # Use configured duration for this metric now
                'cash_flow_per_student': cash_flow # Use the corrected cash flow calculation
            })
        else:
             # Handle unexpected metric format (should not happen with pre-processing)
             progression_data.append({
                'state': f"State {state_idx_str} (error)",
                'entered': entered if state_idx > 0 else entered_count,
                'completed': 0,
                'dropouts': 0,
                'dropout_rate': 0.0,
                'avg_salary': 0,
                'avg_payment': 0,
                'active_months': 0,
                'cash_flow_per_student': 0
            })
        # Update entered_count for the next iteration if needed (handled by logic above)

    return progression_data
# *** END: Helper function for progression calculation ***

# Callback to run the simulation
@app.callback(
    [Output("loading-message", "children"),
     Output("simulation-results-store", "data")],
    [Input("run-simulation", "n_clicks")],
    [State("simulation-config-store", "data")]
)
def run_simulation_callback(n_clicks, config_data):
    if n_clicks is None or n_clicks == 0 or not config_data:
        return "", None
    
    # Create a SimulationConfig object from the stored config
    config = SimulationConfig(
        num_students=config_data.get('num_students', 100),
        include_advanced_training=config_data.get('include_advanced_training', True),
        basic_training_cost=config_data.get('basic_training_cost', 2000),
        basic_training_dropout_rate=config_data.get('basic_training_dropout_rate', 0.15),
        basic_training_duration=config_data.get('basic_training_duration', 3),
        advanced_training_cost=config_data.get('advanced_training_cost', 2000),
        advanced_training_dropout_rate=config_data.get('advanced_training_dropout_rate', 0.12),
        advanced_training_duration=config_data.get('advanced_training_duration', 3),
        num_additional_cruises=config_data.get('num_additional_cruises', 3),
        first_cruise_base_salary=config_data.get('first_cruise_base_salary', 5000),
        first_cruise_dropout_rate=config_data.get('first_cruise_dropout_rate', 0.15),
        first_cruise_salary_variation=config_data.get('first_cruise_salary_variation', 6.0),
        first_cruise_duration=config_data.get('first_cruise_duration', 8),
        first_cruise_payment_fraction=config_data.get('first_cruise_payment_fraction', 0.14),
        subsequent_cruise_dropout_rate=config_data.get('subsequent_cruise_dropout_rate', 0.02),
        subsequent_cruise_salary_increase=config_data.get('subsequent_cruise_salary_increase', 10.0),
        subsequent_cruise_salary_variation=config_data.get('subsequent_cruise_salary_variation', 5.0),
        subsequent_cruise_duration=config_data.get('subsequent_cruise_duration', 8),
        subsequent_cruise_payment_fraction=config_data.get('subsequent_cruise_payment_fraction', 0.14)
    )
    
    random_seed = random.randint(1, 10000)
    config.random_seed = random_seed
    
    try:
        num_careers = config_data.get('num_students', 100)
        results = run_simulation_batch(config)
        
        # *** START: Pre-process state_metrics (Ensure this block is present and correct) ***
        if 'state_metrics' in results and isinstance(results['state_metrics'], dict):
            cleaned_metrics = {}
            for state_idx, metric_info in results['state_metrics'].items():
                cleaned_state_data = {}
                actual_name = f"State {state_idx}" # Default
                
                if isinstance(metric_info, dict):
                    # Copy existing serializable data
                    for key, val in metric_info.items():
                         if isinstance(val, (int, float, str, bool, type(None))):
                             cleaned_state_data[key] = val
                         
                    # Attempt to extract the simple name
                    name_raw = metric_info.get('name')
                    if isinstance(name_raw, dict) and 'name' in name_raw:
                        actual_name = name_raw['name']
                    elif isinstance(name_raw, str):
                        actual_name = name_raw
                        
                    # Ensure the 'name' key holds the simple string name
                    cleaned_state_data['name'] = actual_name
                    
                elif isinstance(metric_info, str): # Handle case where metric is just a name
                     cleaned_state_data['name'] = metric_info
                     cleaned_state_data['avg_salary'] = 0
                     cleaned_state_data['avg_payment'] = 0
                     cleaned_state_data['active_months'] = 0

                cleaned_metrics[str(state_idx)] = cleaned_state_data
                
            results['state_metrics'] = cleaned_metrics # Replace original with cleaned version
        # *** END: Pre-process state_metrics ***

        # Serialization logic (more explicit for state_metrics)
        serializable_results = {}
        for key, value in results.items():
            if key == 'state_metrics':
                # Explicitly build the serializable dict for state_metrics
                temp_metrics_dict = {}
                if isinstance(value, dict): # Value should be the cleaned_metrics dict
                    for state_idx_str, state_data_dict in value.items():
                        # state_data_dict should contain simple types after pre-processing
                        temp_metrics_dict[state_idx_str] = state_data_dict
                serializable_results[key] = temp_metrics_dict # Assign the explicitly built dict
            # *** START Handle New Aggregated State Data ***
            elif key in ['state_total_costs', 'state_total_payments', 'state_entry_counts']:
                 # These should already be dicts with string keys (state index as string) and numeric values
                 serializable_results[key] = {str(k): v for k, v in value.items()} if isinstance(value, dict) else {}
            # *** END Handle New Aggregated State Data ***
            elif isinstance(value, (int, float, str, bool, type(None))):
                serializable_results[key] = value
            elif isinstance(value, dict):
                # General dictionary handling (excluding state_metrics)
                serializable_dict = {}
                for k, v in value.items():
                    if isinstance(v, (int, float, str, bool, type(None))):
                        serializable_dict[str(k)] = v
                    elif hasattr(v, 'to_dict'):
                        serializable_dict[str(k)] = v.to_dict()
                    else:
                        pass 
                serializable_results[key] = serializable_dict
            elif hasattr(value, 'to_dict'): # Pandas objects
                serializable_results[key] = value.to_dict()
            else:
                 pass # Skip other non-serializable types
        
        serializable_results['config'] = config_data
        
        return f"Completed {num_careers} career simulations!", serializable_results
    
    except Exception as e:
        import traceback
        print(f"Error during simulation or serialization: {e}")
        traceback.print_exc()
        return f"Error in simulation: {str(e)}", None

# Callback to update summary stats
@app.callback(
    Output("summary-stats", "children"),
    [Input("simulation-results-store", "data")]
)
def update_summary_stats(results):
    if not results:
        return "Run a simulation to see results"
    
    # Extract key metrics
    completion_rate = results.get('completion_rate', 0)
    dropout_rate = results.get('dropout_rate', 0)
    avg_duration = results.get('avg_duration', 0) / 12  # Convert months to years
    avg_training_cost = results.get('avg_training_cost', 0)
    avg_total_payments = results.get('avg_total_payments', 0)
    avg_net_cash_flow = results.get('avg_net_cash_flow', 0)
    avg_roi = results.get('avg_roi', 0)
    
    # Create a summary statistics card
    return html.Div([
        html.H4("Summary Statistics", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.H5("Career Outcomes", style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.P("Completion Rate:", style={'fontWeight': 'bold'}),
                        html.P(f"{completion_rate:.1f}%")
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.P("Dropout Rate:", style={'fontWeight': 'bold'}),
                        html.P(f"{dropout_rate:.1f}%")
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.P("Average Duration:", style={'fontWeight': 'bold'}),
                        html.P(f"{avg_duration:.1f} years")
                    ])
                ])
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H5("Financial Outcomes", style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.P("Avg Training Cost:", style={'fontWeight': 'bold'}),
                        html.P(f"${avg_training_cost:.2f}")
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.P("Avg Total Payments:", style={'fontWeight': 'bold'}),
                        html.P(f"${avg_total_payments:.2f}")
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.P("Avg Net Cash Flow:", style={'fontWeight': 'bold'}),
                        html.P(f"${avg_net_cash_flow:.2f}")
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.P("Avg ROI:", style={'fontWeight': 'bold'}),
                        html.P(f"{avg_roi:.1f}%")
                    ])
                ])
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '10px'})
    ])

# Callback to update overview content
@app.callback(
    Output("overview-content", "children"),
    [Input("simulation-results-store", "data")]
)
def update_overview_content(results):
    if not results:
        return "Run a simulation to see results"

    # Extract data (state_metrics should be pre-cleaned)
    # state_distribution = results.get('state_distribution', {}) # No longer used for main table
    state_metrics = results.get('state_metrics', {})
    config = results.get('config', {})

    # Calculate progression data using the helper function
    progression_data = calculate_progression_data(state_metrics, config)

    # Prepare data for state progression table
    progression_table_data = [
        {
            "State": row['state'],
            "Entered": row['entered'],
            "Completed": row['completed'],
            "Dropouts": row['dropouts'],
            "Dropout Rate (%)": f"{row['dropout_rate']:.1f}"
        }
        for row in progression_data
    ]

    # Create state progression table
    progression_table = html.Div([
        html.H5("State Progression (Based on Configured Rates)", style={'textAlign': 'center', 'marginBottom': '15px'}),
        dash_table.DataTable(
            data=progression_table_data,
            columns=[
                {"name": "State", "id": "State"},
                {"name": "Entered", "id": "Entered"},
                {"name": "Completed", "id": "Completed"},
                {"name": "Dropouts", "id": "Dropouts"},
                {"name": "Dropout Rate (%)", "id": "Dropout Rate (%)"}
            ],
            style_cell={'textAlign': 'center', 'padding': '10px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
    ], style={'marginBottom': '30px'})

    # Create state metrics table (using metrics from progression data)
    metrics_rows = [
        {
            "State": row['state'],
            "Avg Salary": f"${row['avg_salary']:.2f}",
            "Avg Payment": f"${row['avg_payment']:.2f}",
            "Active Months": row['active_months']
        }
        for row in progression_data if row['active_months'] > 0 # Only show states with activity
    ]
    # No need to sort again, as progression_data is already sorted

    metrics_table = html.Div([
        html.H5("State Financial Metrics (Average)", style={'textAlign': 'center', 'marginBottom': '15px'}),
        dash_table.DataTable(
            data=metrics_rows,
            columns=[
                {"name": "State", "id": "State"},
                {"name": "Avg Salary", "id": "Avg Salary"},
                {"name": "Avg Payment", "id": "Avg Payment"},
                {"name": "Active Months", "id": "Active Months"}
            ],
            style_cell={'textAlign': 'center', 'padding': '10px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
    ])

    # Combine elements
    return html.Div([
        html.H4("Simulation Overview", style={'textAlign': 'center', 'marginBottom': '20px'}),
        progression_table, # Show progression table instead of final distribution
        metrics_table
    ])

# Callback to update detailed data
@app.callback(
    Output("detailed-data", "children"),
    [Input("simulation-results-store", "data")]
)
def update_detailed_data(results):
    if not results:
        return "Run a simulation to see results"
    
    # Extract config data
    config = results.get('config', {})
    
    # Create config table
    config_data = [
        {"Parameter": "Basic Training Cost", "Value": f"${config.get('basic_training_cost', 0):.2f}"},
        {"Parameter": "Basic Training Dropout Rate", "Value": f"{config.get('basic_training_dropout_rate', 0)*100:.1f}%"},
        {"Parameter": "Basic Training Duration", "Value": f"{config.get('basic_training_duration', 0)} months"},
        {"Parameter": "Include Advanced Training", "Value": "Yes" if config.get('include_advanced_training', True) else "No"}
    ]
    
    if config.get('include_advanced_training', True):
        config_data.extend([
            {"Parameter": "Advanced Training Cost", "Value": f"${config.get('advanced_training_cost', 0):.2f}"},
            {"Parameter": "Advanced Training Dropout Rate", "Value": f"{config.get('advanced_training_dropout_rate', 0)*100:.1f}%"},
            {"Parameter": "Advanced Training Duration", "Value": f"{config.get('advanced_training_duration', 0)} months"}
        ])
    
    config_data.extend([
        {"Parameter": "Number of Additional Cruises", "Value": f"{config.get('num_additional_cruises', 0)}"},
        {"Parameter": "First Cruise Base Salary", "Value": f"${config.get('first_cruise_base_salary', 0):.2f}"},
        {"Parameter": "First Cruise Dropout Rate", "Value": f"{config.get('first_cruise_dropout_rate', 0)*100:.1f}%"},
        {"Parameter": "First Cruise Salary Variation", "Value": f"{config.get('first_cruise_salary_variation', 0):.1f}%"},
        {"Parameter": "First Cruise Duration", "Value": f"{config.get('first_cruise_duration', 0)} months"},
        {"Parameter": "First Cruise Payment Fraction", "Value": f"{config.get('first_cruise_payment_fraction', 0)*100:.1f}%"},
        {"Parameter": "Subsequent Cruise Dropout Rate", "Value": f"{config.get('subsequent_cruise_dropout_rate', 0)*100:.1f}%"},
        {"Parameter": "Subsequent Cruise Salary Increase", "Value": f"{config.get('subsequent_cruise_salary_increase', 0):.1f}%"},
        {"Parameter": "Subsequent Cruise Salary Variation", "Value": f"{config.get('subsequent_cruise_salary_variation', 0):.1f}%"},
        {"Parameter": "Subsequent Cruise Duration", "Value": f"{config.get('subsequent_cruise_duration', 0)} months"},
        {"Parameter": "Subsequent Cruise Payment Fraction", "Value": f"{config.get('subsequent_cruise_payment_fraction', 0)*100:.1f}%"}
    ])
    
    config_table = html.Div([
        html.H5("Simulation Configuration", style={'textAlign': 'center', 'marginBottom': '15px'}),
        dash_table.DataTable(
            data=config_data,
            columns=[
                {"name": "Parameter", "id": "Parameter"},
                {"name": "Value", "id": "Value"}
            ],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
    ], style={'marginBottom': '30px'})
    
    # Create ROI metrics table
    roi_data = [
        {"Metric": "Average Training Cost", "Value": f"${results.get('avg_training_cost', 0):.2f}"},
        {"Metric": "Average Total Payments", "Value": f"${results.get('avg_total_payments', 0):.2f}"},
        {"Metric": "Average Net Cash Flow", "Value": f"${results.get('avg_net_cash_flow', 0):.2f}"},
        {"Metric": "Average ROI", "Value": f"{results.get('avg_roi', 0):.1f}%"},
        {"Metric": "ROI Standard Deviation", "Value": f"{results.get('roi_std', 0):.1f}%"},
        {"Metric": "ROI 10th Percentile", "Value": f"{results.get('roi_10th', 0):.1f}%"},
        {"Metric": "ROI 90th Percentile", "Value": f"{results.get('roi_90th', 0):.1f}%"}
    ]
    
    roi_table = html.Div([
        html.H5("Financial Metrics", style={'textAlign': 'center', 'marginBottom': '15px'}),
        dash_table.DataTable(
            data=roi_data,
            columns=[
                {"name": "Metric", "id": "Metric"},
                {"name": "Value", "id": "Value"}
            ],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
    ])
    
    return html.Div([
        html.H4("Detailed Simulation Data", style={'textAlign': 'center', 'marginBottom': '20px'}),
        config_table,
        roi_table
    ])

# Callback to update state tracking content
@app.callback(
    Output("state-tracking-content", "children"),
    [Input("simulation-results-store", "data")]
)
def update_state_tracking(results):
    if not results:
        return "Run a simulation to see results"

    # Extract the new aggregated data and state names
    state_metrics = results.get('state_metrics', {}) # Primarily used for state names now
    state_total_costs = results.get('state_total_costs', {})
    state_total_payments = results.get('state_total_payments', {})
    state_entry_counts = results.get('state_entry_counts', {})
    config = results.get('config', {})
    num_simulations = config.get('num_students', 0)

    # --- Process Aggregated Data ---
    state_data = []
    state_names_ordered = []
    # Ensure states are processed in the correct order (0, 1, 2, ...)
    # Handle potential string keys from JSON serialization
    max_state_idx = -1
    if state_metrics:
        try:
            # Get numeric keys, find max, handle empty dict case
            numeric_keys = [int(k) for k in state_metrics.keys() if k.isdigit()]
            if numeric_keys:
                max_state_idx = max(numeric_keys)
            else: # If no numeric keys (e.g., only names were stored), handle gracefully
                 max_state_idx = -1 # Or try another way to determine max state if possible
        except ValueError:
            # Handle cases where keys might not be convertible to int
            print("Warning: Could not determine max state index from state_metrics keys.")
            max_state_idx = -1 # Fallback

    # If max_state_idx is still -1, try getting it from other dicts if they exist
    if max_state_idx == -1:
         all_keys = list(state_total_costs.keys()) + list(state_total_payments.keys()) + list(state_entry_counts.keys())
         numeric_keys = [int(k) for k in all_keys if k.isdigit()]
         if numeric_keys:
             max_state_idx = max(numeric_keys)

    if max_state_idx == -1: # Still couldn't determine, maybe no states?
        return "Could not determine state order from simulation results."

    for state_idx in range(max_state_idx + 1):
        state_idx_str = str(state_idx)

        # Get state name safely from state_metrics
        state_name = f"State {state_idx_str}" # Default
        if state_idx_str in state_metrics and isinstance(state_metrics.get(state_idx_str), dict):
            name_raw = state_metrics[state_idx_str].get('name')
            if isinstance(name_raw, str):
                state_name = name_raw
        elif state_idx_str in state_metrics and isinstance(state_metrics.get(state_idx_str), str):
             # Handle case where state_metrics[idx] is just the name string
             state_name = state_metrics[state_idx_str]

        # If advanced training is disabled, skip its state entry in the table/chart
        if "Advanced Training" in state_name and not config.get('include_advanced_training', True):
            continue

        # Get aggregated values safely (default to 0 if key missing)
        total_cost = state_total_costs.get(state_idx_str, 0.0)
        total_payment = state_total_payments.get(state_idx_str, 0.0)
        entry_count = state_entry_counts.get(state_idx_str, 0)

        # Only add state to table/chart if it was entered by at least one simulation
        if entry_count > 0:
            state_names_ordered.append(state_name)
            net_cash_flow = total_payment - total_cost

            state_data.append({
                "State": state_name,
                "Total Costs": total_cost,
                "Total Payments": total_payment,
                "Net Cash Flow": net_cash_flow,
                "Simulations Entered": entry_count,
                "Entry Rate (%)": (entry_count / num_simulations * 100) if num_simulations > 0 else 0
            })
        # If entry_count is 0, we might still want to show it with 0s, or omit it.
        # Current logic omits states never entered. Add an else block here if you want to show them.

    # --- Create Aggregated Results Table ---
    tracking_table = html.Div([
        html.H5("Aggregated Results by State (From Simulations)", style={'textAlign': 'center', 'marginBottom': '15px'}),
        dash_table.DataTable(
            id='aggregate-state-table',
            data=state_data,
            columns=[
                {"name": "State", "id": "State"},
                {"name": "Simulations Entered", "id": "Simulations Entered", 'type': 'numeric'},
                {"name": "Entry Rate (%)", "id": "Entry Rate (%)", 'type': 'numeric', 'format': {'specifier': '.1f'}},
                {"name": "Total Costs", "id": "Total Costs", 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                {"name": "Total Payments", "id": "Total Payments", 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                {"name": "Net Cash Flow", "id": "Net Cash Flow", 'type': 'numeric', 'format': {'specifier': '$,.2f'}}
            ],
            style_cell={'textAlign': 'center', 'padding': '10px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                    'if': {'filter_query': '{Net Cash Flow} < 0'},
                    'color': 'red'
                },
                {
                    'if': {'filter_query': '{Net Cash Flow} > 0'},
                    'color': 'green'
                }
            ]
        )
    ])

    # --- Calculate Overall Summary Metrics from Aggregates ---
    # Ensure we only sum states that were actually added to state_data (i.e., entered)
    overall_total_costs = sum(item["Total Costs"] for item in state_data)
    overall_total_payments = sum(item["Total Payments"] for item in state_data)
    overall_net_cash_flow = overall_total_payments - overall_total_costs

    # --- Create Overall Cash Flow Summary ---
    cash_flow_summary = html.Div([
        html.H5("Overall Cash Flow Summary (Aggregated from Simulations)", style={'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '15px'}),
        html.Div([
            html.Div([
                html.P("Total Training Costs:", style={'fontWeight': 'bold'}),
                html.P(f"${overall_total_costs:,.2f}", style={'color': 'red'}) # Use comma formatting
            ], style={'display': 'inline-block', 'width': '33%', 'textAlign': 'center'}),

            html.Div([
                html.P("Total Payments Received:", style={'fontWeight': 'bold'}),
                html.P(f"${overall_total_payments:,.2f}", style={'color': 'green'}) # Use comma formatting
            ], style={'display': 'inline-block', 'width': '33%', 'textAlign': 'center'}),

            html.Div([
                html.P("Net Cash Flow:", style={'fontWeight': 'bold'}),
                html.P(f"${overall_net_cash_flow:,.2f}",
                      style={'color': 'green' if overall_net_cash_flow >= 0 else 'red'}) # Use comma formatting
            ], style={'display': 'inline-block', 'width': '33%', 'textAlign': 'center'})
        ], style={'border': '1px solid #ddd', 'padding': '15px', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9'})
    ])

    # --- Create Aggregated Cash Flow Visualization ---
    cash_flow_fig = go.Figure()
    if state_data: # Only create chart if there is data
        state_net_flows = [item['Net Cash Flow'] for item in state_data]
        valid_state_names = [item['State'] for item in state_data] # Get names corresponding to actual data

        cash_flow_fig.add_trace(go.Bar(
            x=valid_state_names, # Use names for states that were entered
            y=state_net_flows,
            name="Total Net Cash Flow by State",
            marker_color=['red' if cf < 0 else ('green' if cf > 0 else 'grey') for cf in state_net_flows],
            text=[f"${abs(cf):,.2f}" if cf != 0 else "" for cf in state_net_flows], # Use comma formatting
            textposition='auto'
        ))

        cash_flow_fig.update_layout(
            title='Total Net Cash Flow by State (Aggregated from Simulations)',
            xaxis_title='State',
            yaxis_title='Total Net Cash Flow ($)',
            template='plotly_white'
        )
        cash_flow_graph = html.Div([
            html.H5("Cash Flow Visualization (Simulation Aggregates)", style={'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '15px'}),
            dcc.Graph(figure=cash_flow_fig)
        ])
    else:
        cash_flow_graph = html.Div([
             html.H5("Cash Flow Visualization (Simulation Aggregates)", style={'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '15px'}),
             html.P("No state data available to display chart.", style={'textAlign': 'center'})
        ])


    # --- Combine Components ---
    return html.Div([
        html.H4("State Financial Tracking (Simulation Aggregates)", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.P("This tab shows the aggregated financial results for each state across all simulations:",
               style={'marginBottom': '15px'}),
        html.Ul([
            html.Li("Simulations Entered: Number of simulations that entered this state at least once."),
            html.Li("Entry Rate (%): Percentage of total simulations that entered this state."),
            html.Li("Total Costs: Sum of all training costs attributed to this state across relevant simulations."),
            html.Li("Total Payments: Sum of all payments received while simulations were in this state."),
            html.Li("Net Cash Flow: Total Payments - Total Costs for this state.")
        ], style={'marginBottom': '20px'}),
        tracking_table,
        cash_flow_summary,
        cash_flow_graph
    ])

# Callback for the Monthly Metrics content
@app.callback(
    Output("monthly-metrics-content", "children"),
    [Input("simulation-results-store", "data")]
)
def update_monthly_metrics(results):
    if not results:
        return "Run a simulation to see results"
    
    state_metrics = results.get('state_metrics', {})
        
    # *** Start - Explicit Name Extraction Logic ***
    metrics_data = []
    for state_idx_str, metric_info in sorted(state_metrics.items(), key=lambda item: int(item[0])):
        # Default values
        state_name = f"State {state_idx_str}"
        avg_salary = 0
        avg_payment = 0
        active_months = 0

        # Try extracting data, focusing on getting the simple name string
        if isinstance(metric_info, dict):
            # Attempt to get the simple name string
            name_raw = metric_info.get('name')
            if isinstance(name_raw, str):
                state_name = name_raw
            elif isinstance(name_raw, dict) and isinstance(name_raw.get('name'), str):
                 state_name = name_raw['name'] # Handle potential nested dict { 'name': { 'name': 'Actual Name' } }
            
            # Get other metrics safely
            avg_salary = metric_info.get('avg_salary', 0)
            avg_payment = metric_info.get('avg_payment', 0)
            active_months = metric_info.get('active_months', 0)
            
            avg_salary = avg_salary if isinstance(avg_salary, (int, float)) else 0
            avg_payment = avg_payment if isinstance(avg_payment, (int, float)) else 0
            active_months = active_months if isinstance(active_months, (int, float)) else 0
            
        elif isinstance(metric_info, str):
             # Handle case where the entire metric_info is just the name string
             state_name = metric_info

        metrics_data.append({
            'state': state_name, # Ensure this holds the simple string
            'avg_monthly_salary': avg_salary,
            'avg_monthly_payment': avg_payment,
            'active_months': int(active_months)
        })
    # *** End - Explicit Name Extraction Logic ***

    # Create the table using the processed metrics_data
    salary_table = dash_table.DataTable(
        id='monthly-metrics-table',
        columns=[
            {'name': 'State', 'id': 'state'},
            {'name': 'Avg Monthly Salary', 'id': 'avg_monthly_salary', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
            {'name': 'Avg Monthly Payment', 'id': 'avg_monthly_payment', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
            {'name': 'Active Months', 'id': 'active_months', 'type': 'numeric'}
        ],
        data=metrics_data, # data should now contain correct 'state' strings
        style_cell={'textAlign': 'center', 'padding': '10px'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )
    
    # Create graphs using the processed metrics_data
    states = [item['state'] for item in metrics_data]
    salaries = [item['avg_monthly_salary'] for item in metrics_data]
    payments = [item['avg_monthly_payment'] for item in metrics_data]
    active_months_list = [item['active_months'] for item in metrics_data]
    
    comparison_fig = go.Figure()
    comparison_fig.add_trace(go.Bar(x=states, y=salaries, name='Average Monthly Salary', marker_color='#2196F3'))
    comparison_fig.add_trace(go.Bar(x=states, y=payments, name='Average Monthly Payment', marker_color='#4CAF50'))
    # ... Add Scatter trace for payment percentage ...
    payment_percentages = [(p / s * 100) if s > 0 else 0 for s, p in zip(salaries, payments)]
    comparison_fig.add_trace(go.Scatter(x=states, y=payment_percentages, name='Payment Percentage', mode='lines+markers', yaxis='y2', line=dict(color='#FFC107')))
    comparison_fig.update_layout(
        title='Monthly Salary and Payment Comparison', xaxis_title='State', yaxis_title='Amount ($)', 
        yaxis2=dict(title='Payment Percentage', overlaying='y', side='right', range=[0, 30], ticksuffix='%'),
        barmode='group', template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    months_fig = go.Figure()
    months_fig.add_trace(go.Bar(x=states, y=active_months_list, name='Active Months', marker_color='#9C27B0'))
    months_fig.update_layout(title='Active Months by State', xaxis_title='State', yaxis_title='Number of Months', template='plotly_white')

    # Return layout including the table and graphs
    return html.Div([
        html.H4("Monthly Metrics Analysis", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.P("This analysis shows the average monthly salary and payments by state:"),
        html.H5("Monthly Salary and Payment Data", style={'textAlign': 'center', 'marginBottom': '15px', 'marginTop': '20px'}),
        salary_table,
        dcc.Graph(figure=comparison_fig, style={'marginTop': '30px'}),
        dcc.Graph(figure=months_fig, style={'marginTop': '30px'}),
        # final_state_distribution can be added back if needed
    ])

# Callback for the Active Students content -> Renamed to "Progression Summary" conceptually
@app.callback(
    Output("active-students-content", "children"),
    [Input("simulation-results-store", "data")]
)
def update_active_students(results): # Function name kept for consistency, but content changed
    if not results:
        return "Run a simulation to see results"

    # Extract data
    state_metrics = results.get('state_metrics', {})
    config = results.get('config', {})

    # Calculate progression data using the helper function
    progression_data = calculate_progression_data(state_metrics, config)

    # Create data for the progression summary table
    summary_table_data = [
        {
            'State': item['state'],
            'Entered': item['entered'],
            'Completed': item['completed'],
            'Dropouts': item['dropouts'],
            '% Dropout': f"{item['dropout_rate']:.1f}%" # Show dropout rate
            # Removed Avg Salary/Payment as it's less relevant to pure progression
        }
        for item in progression_data
    ]

    # Create the progression summary table
    summary_table = dash_table.DataTable(
        id='progression-summary-table', # New ID
        columns=[
            {'name': 'State', 'id': 'State'},
            {'name': 'Entered', 'id': 'Entered', 'type': 'numeric'},
            {'name': 'Completed', 'id': 'Completed', 'type': 'numeric'},
            {'name': 'Dropouts', 'id': 'Dropouts', 'type': 'numeric'},
            {'name': '% Dropout', 'id': '% Dropout'}
        ],
        data=summary_table_data,
        style_cell={'textAlign': 'center', 'padding': '10px'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )

    # Create a Sankey diagram for visualization
    # Prepare data for Sankey
    labels = [row['state'] for row in progression_data] + [f"Dropout ({row['state']})" for row in progression_data if row['dropouts'] > 0]
    label_indices = {label: i for i, label in enumerate(labels)}

    source = []
    target = []
    value = []
    link_colors = [] # Optional: Color links

    for i, row in enumerate(progression_data):
        current_state_label = row['state']
        current_state_idx = label_indices[current_state_label]

        # Link from previous state (if exists)
        if i > 0:
             # This assumes the 'entered' of current is the 'completed' of previous
             # We actually want to link the 'entered' amount
             prev_state_label = progression_data[i-1]['state']
             prev_state_idx = label_indices[prev_state_label]
             # Link from previous completed to current entered is complex,
             # Let's simplify: Link from state to next state (completed amount) and state to dropout
        #else: # First state entry - link from a virtual 'Start' node? Let's omit for now.
             #pass # Or add a 'Start' node: labels.append('Start'); label_indices['Start'] = len(labels)-1

        # Link to next state (Completed)
        if i < len(progression_data) - 1:
            next_state_label = progression_data[i+1]['state']
            next_state_idx = label_indices[next_state_label]
            if row['completed'] > 0:
                source.append(current_state_idx)
                target.append(next_state_idx)
                value.append(row['completed'])
                link_colors.append("rgba(55, 183, 109, 0.6)") # Greenish for completion

        # Link to Dropout state for this stage
        if row['dropouts'] > 0:
            dropout_label = f"Dropout ({row['state']})"
            dropout_state_idx = label_indices[dropout_label]
            source.append(current_state_idx)
            target.append(dropout_state_idx)
            value.append(row['dropouts'])
            link_colors.append("rgba(219, 64, 82, 0.6)") # Reddish for dropouts

    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="blue" # Default node color
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors # Apply link colors
        ))])

    sankey_fig.update_layout(title_text="Student Progression Flow (Estimated)", font_size=10)


    # Keep overall summary stats (Completion/Dropout Rate)
    summary_stats_boxes = html.Div([
        html.H5("Overall Simulation Outcomes", style={'textAlign': 'center', 'marginBottom': '15px'}),
        html.Div([
             # We can still show the overall rates from the simulation results
             html.Div([
                html.H5("Completion Rate", style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div(f"{results.get('completion_rate', 0):.1f}%", style={
                    'fontSize': '24px',
                    'textAlign': 'center',
                    'color': '#4CAF50'
                })
            ], style={'width': '45%', 'display': 'inline-block', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px', 'marginRight': '10%'}),

            html.Div([
                html.H5("Dropout Rate", style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div(f"{results.get('dropout_rate', 0):.1f}%", style={
                    'fontSize': '24px',
                    'textAlign': 'center',
                    'color': '#F44336'
                })
            ], style={'width': '45%', 'display': 'inline-block', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px'})
        ], style={'marginBottom': '20px'})
    ])


    # Return combined layout
    return html.Div([
        html.H4("Progression Flow Analysis", style={'textAlign': 'center', 'marginBottom': '20px'}), # Renamed title
        summary_stats_boxes, # Keep overall summary
        dcc.Graph(figure=sankey_fig, style={'marginBottom': '30px'}), # Show Sankey diagram
        html.H5("Progression Summary Data", style={'textAlign': 'center', 'marginBottom': '15px'}), # Renamed
        html.P([
            "This table shows the estimated flow of students through the career path stages, ",
            "based on the dropout rates configured for the simulation."
        ], style={'marginBottom': '15px'}),
        summary_table # Show the new summary table
    ])

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=10000) 
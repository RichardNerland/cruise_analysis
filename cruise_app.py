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
                        
                        # Training parameters
                        html.Div([
                            html.H4("Training Parameters", style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Include Transportation and Placement:"),
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
                                html.Label("Training Cost ($):"),
                                dcc.Input(
                                    id="basic-training-cost",
                                    type="number",
                                    value=1500,
                                    min=0,
                                    step=100,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Training Dropout Rate (%):"),
                                dcc.Slider(
                                    id="basic-training-dropout-rate",
                                    min=0,
                                    max=50,
                                    step=1,
                                    value=10,
                                    marks={i: f'{i}%' for i in range(0, 51, 10)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Training Duration (months):"),
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
                            
                            # Add the transportation and placement parameters directly in the layout instead of using conditional rendering
                            html.Div([
                                html.Div([
                                    html.Label("Transportation and Placement Cost ($):"),
                                    dcc.Input(
                                        id="advanced-training-cost",
                                        type="number",
                                        value=500,
                                        min=0,
                                        step=100,
                                        style={"width": "100%"}
                                    )
                                ], style={'marginBottom': '15px'}),
                                
                                html.Div([
                                    html.Label("Transportation and Placement Dropout Rate (%):"),
                                    dcc.Slider(
                                        id="advanced-training-dropout-rate",
                                        min=0,
                                        max=50,
                                        step=1,
                                        value=15,
                                        marks={i: f'{i}%' for i in range(0, 51, 10)},
                                    )
                                ], style={'marginBottom': '15px'}),
                                
                                html.Div([
                                    html.Label("Transportation and Placement Duration (months):"),
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
                                    value=3,
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
                                    value=3,
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
                        
                        # Break parameters
                        html.Div([
                            html.H4("Break Parameters", style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Include Breaks Between Cruises:"),
                                dcc.RadioItems(
                                    id="include-breaks",
                                    options=[
                                        {'label': 'Yes', 'value': True},
                                        {'label': 'No', 'value': False}
                                    ],
                                    value=True,
                                    inline=True
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Break Duration (months):"),
                                dcc.Input(
                                    id="break-duration",
                                    type="number",
                                    value=2,
                                    min=1,
                                    max=12,
                                    step=1,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Break Dropout Rate (%):"),
                                dcc.Slider(
                                    id="break-dropout-rate",
                                    min=0,
                                    max=20,
                                    step=0.5,
                                    value=0,
                                    marks={i: f'{i}%' for i in range(0, 21, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.P("Breaks represent time between cruises with no salary or payments.",
                                  style={'fontSize': '0.85em', 'fontStyle': 'italic', 'color': '#666'})
                        ], style={'marginBottom': '20px', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px'}),
                        
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
                            ], style={'marginBottom': '15px'}),
                            
                            # Run Simulation Button
                            html.Div([
                                html.Button(
                                    "Run Simulation", 
                                    id="run-simulation", 
                                    n_clicks=0,
                                    style={
                                        'backgroundColor': '#4CAF50',
                                        'color': 'white',
                                        'padding': '10px 20px',
                                        'fontSize': '16px',
                                        'fontWeight': 'bold',
                                        'border': 'none',
                                        'borderRadius': '4px',
                                        'cursor': 'pointer',
                                        'width': '100%'
                                    }
                                ),
                                html.Div(id="loading-message", style={'marginTop': '10px', 'textAlign': 'center'})
                            ], style={'marginBottom': '20px'})
                        ], style={'marginBottom': '20px', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px'})
                    ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'backgroundColor': '#f9f9f9', 'borderRadius': '8px'}),
                    
                    # Right panel for results
                    html.Div([
                        html.H3("Results"),
                        
                        html.Div([
                            html.Div(id="summary-stats", style={'marginBottom': '20px'}),
                            
                            dcc.Tabs([
                                dcc.Tab(label='Overview', children=[
                                    html.Div(id="overview-content")
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
        
        # Break parameters
        Output("include-breaks", "value"),
        Output("break-duration", "value"),
        Output("break-dropout-rate", "value"),
        
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
        config.include_breaks,
        config.break_duration,
        config.break_dropout_rate * 100,
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
        Input("include-breaks", "value"),
        Input("break-duration", "value"),
        Input("break-dropout-rate", "value"),
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
    include_breaks,
    break_duration,
    break_dropout_rate,
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
    break_dropout_rate = break_dropout_rate / 100 if break_dropout_rate else 0.0
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
        'include_breaks': include_breaks,
        'break_duration': break_duration,
        'break_dropout_rate': break_dropout_rate,
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
            if state_name == "Training":
                dropout_rate = config.get('basic_training_dropout_rate', 0)
            elif state_name == "Transportation and placement":
                dropout_rate = config.get('advanced_training_dropout_rate', 0)
            elif state_name == "First Cruise":
                dropout_rate = config.get('first_cruise_dropout_rate', 0)
            elif "Break" in state_name:
                dropout_rate = config.get('break_dropout_rate', 0)
            elif "Cruise" in state_name:  # For subsequent cruises
                dropout_rate = config.get('subsequent_cruise_dropout_rate', 0)

            dropouts = round(entered * dropout_rate)
            completed = entered - dropouts

            # Get financial metrics for the state
            state_salary = metrics.get('avg_salary', 0)
            state_payment = metrics.get('avg_payment', 0)

            # Get state duration for converting to monthly values
            if state_name == "Training":
                state_duration = config.get('basic_training_duration', 0)
            elif state_name == "Transportation and placement":
                state_duration = config.get('advanced_training_duration', 0)
            elif state_name == "First Cruise":
                state_duration = config.get('first_cruise_duration', 0)
            elif "Break" in state_name:
                state_duration = config.get('break_duration', 0)
            else:  # Subsequent cruises
                state_duration = config.get('subsequent_cruise_duration', 0)

            # Convert state salary and payment to monthly values if duration > 0
            avg_monthly_salary = state_salary / state_duration if state_duration > 0 else 0
            avg_monthly_payment = state_payment / state_duration if state_duration > 0 else 0

            # Calculate cash flow per student for this state
            if state_name == "Training":
                cash_flow = -config.get('basic_training_cost', 0)
            elif state_name == "Transportation and placement":
                if config.get('include_advanced_training', False):
                    cash_flow = -config.get('advanced_training_cost', 0)
                else:
                    cash_flow = 0
            else:  # Cruises and breaks
                cash_flow = state_payment  # Use total state payment

            progression_data.append({
                'state': state_name,
                'entered': entered,
                'completed': completed,
                'dropouts': dropouts,
                'dropout_rate': dropout_rate * 100,  # Convert to percentage for display
                'avg_salary': avg_monthly_salary,  # Now properly monthly
                'avg_payment': avg_monthly_payment,  # Now properly monthly
                'active_months': state_duration,
                'cash_flow_per_student': cash_flow
            })
        else:
             # Handle unexpected metric format
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
        include_breaks=config_data.get('include_breaks', True),
        break_duration=config_data.get('break_duration', 2),
        break_dropout_rate=config_data.get('break_dropout_rate', 0.0),
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
                    
                    # Explicitly preserve avg_state_salary as avg_salary for the overview tab
                    if 'avg_state_salary' in metric_info:
                        cleaned_state_data['avg_salary'] = metric_info['avg_state_salary']
                    
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
    avg_monthly_irr = results.get('avg_monthly_irr')
    
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
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.P("Monthly-Based Annual IRR:", style={'fontWeight': 'bold'}),
                        html.P(f"{avg_monthly_irr:.1f}%" if avg_monthly_irr is not None else "N/A")
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

    # Extract data
    state_metrics = results.get('state_metrics', {})
    config = results.get('config', {})
    state_total_costs = results.get('state_total_costs', {})
    state_total_payments = results.get('state_total_payments', {})
    state_entry_counts = results.get('state_entry_counts', {})
    num_simulations = config.get('num_students', 0)

    # Calculate progression data using the helper function
    progression_data = calculate_progression_data(state_metrics, config)

    # Create summary stats boxes
    summary_stats_boxes = html.Div([
        html.H5("Overall Simulation Outcomes", style={'textAlign': 'center', 'marginBottom': '15px'}),
        html.Div([
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

    # Create Sankey diagram for student flow
    labels = [row['state'] for row in progression_data] + [f"Dropout ({row['state']})" for row in progression_data if row['dropouts'] > 0]
    label_indices = {label: i for i, label in enumerate(labels)}

    source = []
    target = []
    value = []
    link_colors = []

    for i, row in enumerate(progression_data):
        current_state_label = row['state']
        current_state_idx = label_indices[current_state_label]

        if i < len(progression_data) - 1:
            next_state_label = progression_data[i+1]['state']
            next_state_idx = label_indices[next_state_label]
            if row['completed'] > 0:
                source.append(current_state_idx)
                target.append(next_state_idx)
                value.append(row['completed'])
                link_colors.append("rgba(55, 183, 109, 0.6)")

        if row['dropouts'] > 0:
            dropout_label = f"Dropout ({row['state']})"
            dropout_state_idx = label_indices[dropout_label]
            source.append(current_state_idx)
            target.append(dropout_state_idx)
            value.append(row['dropouts'])
            link_colors.append("rgba(219, 64, 82, 0.6)")

    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="blue"
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        ))])

    sankey_fig.update_layout(title_text="Student Progression Flow", font_size=10)

    # Process state financial data
    state_data = []
    for state_idx_str, metrics in sorted(state_metrics.items(), key=lambda x: int(x[0])):
        if isinstance(metrics, dict):
            state_name = metrics.get('name', f"State {state_idx_str}")
            
            # Skip advanced training state if disabled
            if "Advanced Training" in state_name and not config.get('include_advanced_training', True):
                continue

            total_cost = state_total_costs.get(state_idx_str, 0.0)
            total_payment = state_total_payments.get(state_idx_str, 0.0)
            entry_count = state_entry_counts.get(state_idx_str, 0)

            if entry_count > 0:
                net_cash_flow = total_payment - total_cost
                state_data.append({
                    "State": state_name,
                    "Total Costs": total_cost,
                    "Total Payments": total_payment,
                    "Net Cash Flow": net_cash_flow,
                    "Simulations Entered": entry_count,
                    "Entry Rate (%)": (entry_count / num_simulations * 100) if num_simulations > 0 else 0
                })

    # Calculate overall financials
    overall_total_costs = sum(item["Total Costs"] for item in state_data)
    overall_total_payments = sum(item["Total Payments"] for item in state_data)
    overall_net_cash_flow = overall_total_payments - overall_total_costs

    # Create financial summary boxes
    financial_summary = html.Div([
        html.H5("Overall Financial Summary", style={'textAlign': 'center', 'marginBottom': '15px'}),
        html.Div([
            html.Div([
                html.P("Total Training Costs:", style={'fontWeight': 'bold'}),
                html.P(f"${overall_total_costs:,.2f}", style={'color': 'red'})
            ], style={'display': 'inline-block', 'width': '33%', 'textAlign': 'center'}),

            html.Div([
                html.P("Total Payments Received:", style={'fontWeight': 'bold'}),
                html.P(f"${overall_total_payments:,.2f}", style={'color': 'green'})
            ], style={'display': 'inline-block', 'width': '33%', 'textAlign': 'center'}),

            html.Div([
                html.P("Net Cash Flow:", style={'fontWeight': 'bold'}),
                html.P(f"${overall_net_cash_flow:,.2f}",
                      style={'color': 'green' if overall_net_cash_flow >= 0 else 'red'})
            ], style={'display': 'inline-block', 'width': '33%', 'textAlign': 'center'})
        ], style={'border': '1px solid #ddd', 'padding': '15px', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9'})
    ])

    # Create cash flow visualization
    cash_flow_fig = go.Figure()
    if state_data:
        state_net_flows = [item['Net Cash Flow'] for item in state_data]
        valid_state_names = [item['State'] for item in state_data]

        cash_flow_fig.add_trace(go.Bar(
            x=valid_state_names,
            y=state_net_flows,
            name="Net Cash Flow by State",
            marker_color=['red' if cf < 0 else ('green' if cf > 0 else 'grey') for cf in state_net_flows],
            text=[f"${abs(cf):,.2f}" if cf != 0 else "" for cf in state_net_flows],
            textposition='auto'
        ))

        cash_flow_fig.update_layout(
            title='Net Cash Flow by State',
            xaxis_title='State',
            yaxis_title='Net Cash Flow ($)',
            template='plotly_white'
        )

    # Create detailed state metrics table
    state_metrics_table = dash_table.DataTable(
        id='state-metrics-table',
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

    # Combine all elements
    return html.Div([
        html.H4("Simulation Overview", style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        # Top section - Summary stats
        summary_stats_boxes,
        
        # Student flow section
        html.Div([
            html.H5("Student Progression Flow", style={'textAlign': 'center', 'marginBottom': '15px'}),
            dcc.Graph(figure=sankey_fig, style={'marginBottom': '30px'})
        ]),
        
        # Financial section
        html.Div([
            financial_summary,
            html.Div([
                html.H5("Cash Flow by State", style={'textAlign': 'center', 'marginBottom': '15px'}),
                dcc.Graph(figure=cash_flow_fig, style={'marginBottom': '30px'})
            ]),
            html.Div([
                html.H5("Detailed State Metrics", style={'textAlign': 'center', 'marginBottom': '15px'}),
                html.P("This table shows detailed metrics for each state in the simulation:",
                      style={'marginBottom': '15px'}),
                html.Ul([
                    html.Li("Simulations Entered: Number of simulations that reached this state"),
                    html.Li("Entry Rate: Percentage of total simulations that entered the state"),
                    html.Li("Total Costs: Sum of all costs incurred in this state"),
                    html.Li("Total Payments: Sum of all payments received in this state"),
                    html.Li("Net Cash Flow: Difference between payments and costs")
                ], style={'marginBottom': '15px'}),
                state_metrics_table
            ])
        ])
    ])

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=10000) 
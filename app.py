import json
import pickle
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Population Dating Simulation Dashboard", page_icon="💕", layout="wide"
)


@st.cache_data
def load_simulation_data(uploaded_file) -> Optional[Dict[str, Any]]:
    """Load simulation data from uploaded file"""
    try:
        if uploaded_file.name.endswith(".pkl"):
            data = pickle.load(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            data = json.load(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload .pkl or .json files.")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def create_sample_data():
    """Create sample data for demonstration purposes"""
    timesteps = list(range(0, 101, 10))
    sample_data = {
        "population_metrics": [],
        "encounter_events": [],
        "reproduction_events": [],
        "summary": {
            "simulation_length": len(timesteps),
            "population_growth_rate": 0.15,
            "final_population": 350,
            "mating_success_rate": 0.23,
        },
    }

    # Generate sample population metrics
    for i, t in enumerate(timesteps):
        base_pop = 300 + i * 5
        sample_data["population_metrics"].append(
            {
                "timestep": t,
                "total_population": base_pop + np.random.randint(-10, 10),
                "male_count": int((base_pop * 0.48) + np.random.randint(-5, 5)),
                "female_count": int((base_pop * 0.52) + np.random.randint(-5, 5)),
                "avg_age": 28 + np.random.uniform(-2, 4),
                "age_distribution": {
                    "18-25": np.random.randint(80, 120),
                    "26-35": np.random.randint(100, 140),
                    "36-45": np.random.randint(40, 80),
                    "46+": np.random.randint(0, 20),
                },
                "avg_income": 45000 + np.random.uniform(-5000, 10000),
                "income_inequality_gini": 0.2 + np.random.uniform(-0.05, 0.1),
                "single_males": np.random.randint(20, 60),
                "single_females": np.random.randint(80, 120),
                "avg_male_selectivity": np.random.uniform(0.05, 0.2),
                "avg_female_selectivity": np.random.uniform(0.1, 0.25),
                "clustering_coefficient": np.random.uniform(0.6, 0.9),
            }
        )

    return sample_data


def display_header_metrics(data: Dict[str, Any]):
    """Display key simulation metrics in the header"""
    summary = data.get("summary", {})
    pop_metrics = data.get("population_metrics", [])

    if not pop_metrics:
        st.warning("No population metrics data available")
        return

    initial_pop = pop_metrics[0]["total_population"]
    final_pop = pop_metrics[-1]["total_population"]
    growth_rate = (
        ((final_pop - initial_pop) / initial_pop) * 100 if initial_pop > 0 else 0
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Simulation Length",
            f"{len(pop_metrics)} timesteps",
            help="Total number of simulation timesteps",
        )

    with col2:
        st.metric(
            "Population Growth",
            f"{growth_rate:.1f}%",
            delta=f"{final_pop - initial_pop:+d}",
            help="Overall population change during simulation",
        )

    with col3:
        st.metric(
            "Final Population",
            f"{final_pop:,}",
            help="Population at the end of simulation",
        )

    with col4:
        mating_success = summary.get("mating_success_rate", 0) * 100
        st.metric(
            "Mating Success Rate",
            f"{mating_success:.1f}%",
            help="Percentage of encounters that led to successful mating",
        )

    with col5:
        total_encounters = summary.get("total_encounters", 0)
        st.metric(
            "Total Encounters",
            f"{total_encounters:,}",
            help="Total number of agent encounters during simulation",
        )


def plot_population_over_time(pop_metrics: list):
    """Create population over time visualization"""
    df = pd.DataFrame(pop_metrics)

    fig = go.Figure()

    # Total population
    fig.add_trace(
        go.Scatter(
            x=df["timestep"],
            y=df["total_population"],
            mode="lines+markers",
            name="Total Population",
            line=dict(color="#1f77b4", width=3),
            hovertemplate="<b>Timestep %{x}</b><br>Population: %{y}<extra></extra>",
        )
    )

    # Male population
    fig.add_trace(
        go.Scatter(
            x=df["timestep"],
            y=df["male_count"],
            mode="lines",
            name="Male Population",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            hovertemplate="<b>Timestep %{x}</b><br>Males: %{y}<extra></extra>",
        )
    )

    # Female population
    fig.add_trace(
        go.Scatter(
            x=df["timestep"],
            y=df["female_count"],
            mode="lines",
            name="Female Population",
            line=dict(color="#d62728", width=2, dash="dash"),
            hovertemplate="<b>Timestep %{x}</b><br>Females: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Population Dynamics Over Time",
        xaxis_title="Timestep",
        yaxis_title="Population Count",
        hovermode="x unified",
        template="plotly_white",
        height=400,
    )

    return fig


def plot_age_distribution_evolution(pop_metrics: list):
    """Create age distribution evolution visualization"""
    timesteps = []
    age_groups = ["18-25", "26-35", "36-45", "46+"]
    age_data = {group: [] for group in age_groups}

    for metric in pop_metrics:
        timesteps.append(metric["timestep"])
        for group in age_groups:
            age_data[group].append(metric["age_distribution"].get(group, 0))

    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, group in enumerate(age_groups):
        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=age_data[group],
                mode="lines+markers",
                name=f"Age {group}",
                stackgroup="one",
                line=dict(color=colors[i]),
                hovertemplate=f"<b>{group} years</b><br>Timestep: %{{x}}<br>Count: %{{y}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Age Distribution Evolution",
        xaxis_title="Timestep",
        yaxis_title="Population Count",
        hovermode="x unified",
        template="plotly_white",
        height=400,
    )

    return fig


def plot_gender_ratio_and_singles(pop_metrics: list):
    """Create gender ratio and singles analysis"""
    df = pd.DataFrame(pop_metrics)

    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Gender Ratio Over Time", "Single Population Trends"),
        vertical_spacing=0.1,
    )

    # Gender ratio
    gender_ratio = df["male_count"] / df["female_count"]
    fig.add_trace(
        go.Scatter(
            x=df["timestep"],
            y=gender_ratio,
            mode="lines+markers",
            name="Male/Female Ratio",
            line=dict(color="#9467bd", width=3),
            hovertemplate="<b>Timestep %{x}</b><br>M/F Ratio: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add horizontal line at y=1 for perfect ratio
    fig.add_hline(
        y=1,
        line_dash="dash",
        line_color="gray",
        annotation_text="Perfect Balance",
        row=1,
        col=1,
    )

    # Singles trends
    fig.add_trace(
        go.Scatter(
            x=df["timestep"],
            y=df["single_males"],
            mode="lines+markers",
            name="Single Males",
            line=dict(color="#ff7f0e", width=2),
            hovertemplate="<b>Timestep %{x}</b><br>Single Males: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestep"],
            y=df["single_females"],
            mode="lines+markers",
            name="Single Females",
            line=dict(color="#d62728", width=2),
            hovertemplate="<b>Timestep %{x}</b><br>Single Females: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(height=600, template="plotly_white", hovermode="x unified")

    fig.update_xaxes(title_text="Timestep", row=2, col=1)
    fig.update_yaxes(title_text="Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    return fig


def plot_avg_age_and_income(pop_metrics: list):
    """Create average age and income trends"""
    df = pd.DataFrame(pop_metrics)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Average Age Over Time", "Average Income & Inequality"),
        specs=[[{"secondary_y": False}, {"secondary_y": True}]],
    )

    # Average age
    fig.add_trace(
        go.Scatter(
            x=df["timestep"],
            y=df["avg_age"],
            mode="lines+markers",
            name="Average Age",
            line=dict(color="#2ca02c", width=3),
            hovertemplate="<b>Timestep %{x}</b><br>Avg Age: %{y:.1f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Average income
    fig.add_trace(
        go.Scatter(
            x=df["timestep"],
            y=df["avg_income"],
            mode="lines+markers",
            name="Average Income",
            line=dict(color="#17becf", width=3),
            hovertemplate="<b>Timestep %{x}</b><br>Avg Income: $%{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Income inequality (Gini coefficient) on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=df["timestep"],
            y=df["income_inequality_gini"],
            mode="lines+markers",
            name="Gini Coefficient",
            line=dict(color="#e377c2", width=2, dash="dot"),
            yaxis="y2",
            hovertemplate="<b>Timestep %{x}</b><br>Gini: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(height=400, template="plotly_white", hovermode="x unified")

    # Update axis labels
    fig.update_xaxes(title_text="Timestep")
    fig.update_yaxes(title_text="Age (years)", row=1, col=1)
    fig.update_yaxes(title_text="Income ($)", row=1, col=2)
    fig.update_yaxes(title_text="Gini Coefficient", secondary_y=True, row=1, col=2)

    return fig


def plot_education_distribution(pop_metrics: list):
    """Create education distribution over time"""
    # Extract education data
    timesteps = []
    education_data = {}

    for metric in pop_metrics:
        timesteps.append(metric["timestep"])
        edu_dist = metric.get("education_distribution", {})

        for edu_level, count in edu_dist.items():
            if edu_level not in education_data:
                education_data[edu_level] = []
            education_data[edu_level].append(count)

    # Create stacked area chart
    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (edu_level, counts) in enumerate(education_data.items()):
        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=counts,
                mode="lines",
                stackgroup="one",
                name=edu_level.title(),
                line=dict(color=colors[i % len(colors)]),
                hovertemplate=f"<b>{edu_level.title()}</b><br>Timestep: %{{x}}<br>Count: %{{y}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Education Distribution Over Time",
        xaxis_title="Timestep",
        yaxis_title="Population Count",
        hovermode="x unified",
        template="plotly_white",
        height=400,
    )

    return fig


def plot_income_distribution_violin(pop_metrics: list):
    """Create violin plot showing income distribution evolution"""
    # Sample income distributions for different timesteps
    sample_timesteps = pop_metrics[
        :: max(1, len(pop_metrics) // 5)
    ]  # Sample 5 timesteps

    fig = go.Figure()

    for i, metric in enumerate(sample_timesteps):
        # Generate sample income distribution based on avg and gini
        avg_income = metric["avg_income"]
        gini = metric["income_inequality_gini"]

        # Create synthetic income distribution
        np.random.seed(42 + i)  # For reproducible results
        low_income = np.random.gamma(2, avg_income * 0.3, 100)
        high_income = np.random.gamma(5, avg_income * 0.8, 50)
        income_sample = np.concatenate([low_income, high_income])

        fig.add_trace(
            go.Violin(
                y=income_sample,
                x=[f"T{metric['timestep']}"] * len(income_sample),
                name=f"Timestep {metric['timestep']}",
                box_visible=True,
                meanline_visible=True,
                hovertemplate="<b>Timestep %{x}</b><br>Income: $%{y:,.0f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Income Distribution Evolution",
        xaxis_title="Timestep",
        yaxis_title="Income ($)",
        template="plotly_white",
        height=400,
    )

    return fig


def plot_selectivity_trends(pop_metrics: list):
    """Create selectivity trends visualization"""
    df = pd.DataFrame(pop_metrics)

    fig = go.Figure()

    # Male selectivity
    fig.add_trace(
        go.Scatter(
            x=df["timestep"],
            y=df["avg_male_selectivity"],
            mode="lines+markers",
            name="Male Selectivity",
            line=dict(color="#1f77b4", width=3),
            hovertemplate="<b>Timestep %{x}</b><br>Male Selectivity: %{y:.3f}<extra></extra>",
        )
    )

    # Female selectivity
    fig.add_trace(
        go.Scatter(
            x=df["timestep"],
            y=df["avg_female_selectivity"],
            mode="lines+markers",
            name="Female Selectivity",
            line=dict(color="#ff7f0e", width=3),
            hovertemplate="<b>Timestep %{x}</b><br>Female Selectivity: %{y:.3f}<extra></extra>",
        )
    )

    # Selectivity gap
    selectivity_gap = df["avg_female_selectivity"] - df["avg_male_selectivity"]
    fig.add_trace(
        go.Scatter(
            x=df["timestep"],
            y=selectivity_gap,
            mode="lines",
            name="Selectivity Gap (F-M)",
            line=dict(color="#2ca02c", width=2, dash="dot"),
            yaxis="y2",
            hovertemplate="<b>Timestep %{x}</b><br>Gap: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Selectivity Trends by Gender",
        xaxis_title="Timestep",
        yaxis_title="Selectivity Score",
        yaxis2=dict(title="Selectivity Gap", overlaying="y", side="right"),
        template="plotly_white",
        height=400,
        hovermode="x unified",
    )

    return fig


def plot_mating_success_analysis(reproduction_events: list, encounter_events: list):
    """Create mating success analysis"""
    if not reproduction_events:
        # Create placeholder chart if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No reproduction data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(
            title="Mating Success Analysis", template="plotly_white", height=400
        )
        return fig

    # Group reproduction events by timestep
    repro_by_timestep = {}
    for event in reproduction_events:
        timestep = event.get("timestep", 0)
        if timestep not in repro_by_timestep:
            repro_by_timestep[timestep] = 0
        repro_by_timestep[timestep] += 1

    # Group encounter events by timestep if available
    encounters_by_timestep = {}
    if encounter_events:
        for event in encounter_events:
            timestep = event.get("timestep", 0)
            if timestep not in encounters_by_timestep:
                encounters_by_timestep[timestep] = 0
            encounters_by_timestep[timestep] += 1

    timesteps = sorted(repro_by_timestep.keys())
    reproductions = [repro_by_timestep[t] for t in timesteps]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Successful Reproductions Over Time", "Success Rate Analysis"),
        vertical_spacing=0.1,
    )

    # Reproduction events over time
    fig.add_trace(
        go.Bar(
            x=timesteps,
            y=reproductions,
            name="Successful Reproductions",
            marker_color="#2ca02c",
            hovertemplate="<b>Timestep %{x}</b><br>Reproductions: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Success rate if encounter data is available
    if encounters_by_timestep:
        success_rates = []
        for t in timesteps:
            encounters = encounters_by_timestep.get(t, 0)
            repros = repro_by_timestep.get(t, 0)
            rate = (repros / encounters * 100) if encounters > 0 else 0
            success_rates.append(rate)

        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=success_rates,
                mode="lines+markers",
                name="Success Rate (%)",
                line=dict(color="#ff7f0e", width=3),
                hovertemplate="<b>Timestep %{x}</b><br>Success Rate: %{y:.1f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(height=600, template="plotly_white", hovermode="x unified")

    fig.update_xaxes(title_text="Timestep", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Success Rate (%)", row=2, col=1)

    return fig


def plot_age_gap_analysis(reproduction_events: list):
    """Analyze age gaps in successful reproductions"""
    if not reproduction_events:
        fig = go.Figure()
        fig.add_annotation(
            text="No reproduction data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(title="Age Gap Analysis", template="plotly_white", height=400)
        return fig

    age_gaps = [event.get("age_gap", 0) for event in reproduction_events]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Age Gap Distribution", "Age Gap Over Time"),
        specs=[[{"type": "histogram"}, {"type": "scatter"}]],
    )

    # Age gap histogram
    fig.add_trace(
        go.Histogram(
            x=age_gaps,
            nbinsx=20,
            name="Age Gap Distribution",
            marker_color="#9467bd",
            hovertemplate="Age Gap: %{x:.1f} years<br>Count: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Age gap over time
    timesteps = [event.get("timestep", 0) for event in reproduction_events]
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=age_gaps,
            mode="markers",
            name="Age Gap by Time",
            marker=dict(color="#ff7f0e", size=6, opacity=0.6),
            hovertemplate="<b>Timestep %{x}</b><br>Age Gap: %{y:.1f} years<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(height=400, template="plotly_white")

    fig.update_xaxes(title_text="Age Gap (years)", row=1, col=1)
    fig.update_xaxes(title_text="Timestep", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Age Gap (years)", row=1, col=2)

    return fig


def plot_population_density_heatmap(pop_metrics: list, timestep_idx: int = -1):
    """Create population density heatmap for a specific timestep"""
    if not pop_metrics or timestep_idx >= len(pop_metrics):
        return go.Figure()

    metric = pop_metrics[timestep_idx]
    density_map = metric.get("population_density_map", {})

    if not density_map:
        fig = go.Figure()
        fig.add_annotation(
            text="No population density data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(
            title="Population Density Heatmap", template="plotly_white", height=400
        )
        return fig

    # Parse coordinates and densities
    coords = []
    densities = []

    for coord_str, density in density_map.items():
        try:
            # Parse coordinate string like "(14, 14)"
            coord = coord_str.strip("()").split(", ")
            x, y = int(coord[0]), int(coord[1])
            coords.append((x, y))
            densities.append(density)
        except (ValueError, IndexError):
            continue

    if not coords:
        return go.Figure()

    # Create grid for heatmap
    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]

    # Create a grid
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Create density matrix
    grid_size_x = min(50, x_max - x_min + 1)
    grid_size_y = min(50, y_max - y_min + 1)

    density_matrix = np.zeros((grid_size_y, grid_size_x))

    for (x, y), density in zip(coords, densities):
        # Map to grid indices
        grid_x = min(
            int((x - x_min) * (grid_size_x - 1) / max(1, x_max - x_min)),
            grid_size_x - 1,
        )
        grid_y = min(
            int((y - y_min) * (grid_size_y - 1) / max(1, y_max - y_min)),
            grid_size_y - 1,
        )
        density_matrix[grid_y, grid_x] += density

    fig = go.Figure(
        data=go.Heatmap(
            z=density_matrix,
            colorscale="Viridis",
            hovertemplate="X: %{x}<br>Y: %{y}<br>Density: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Population Density at Timestep {metric['timestep']}",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        template="plotly_white",
        height=500,
    )

    return fig


def plot_clustering_trends(pop_metrics: list):
    """Plot clustering coefficient trends over time"""
    df = pd.DataFrame(pop_metrics)

    if "clustering_coefficient" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No clustering coefficient data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(
            title="Clustering Coefficient Trends", template="plotly_white", height=400
        )
        return fig

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestep"],
            y=df["clustering_coefficient"],
            mode="lines+markers",
            name="Clustering Coefficient",
            line=dict(color="#2ca02c", width=3),
            hovertemplate="<b>Timestep %{x}</b><br>Clustering: %{y:.3f}<extra></extra>",
        )
    )

    # Add trend line
    z = np.polyfit(df["timestep"], df["clustering_coefficient"], 1)
    p = np.poly1d(z)

    fig.add_trace(
        go.Scatter(
            x=df["timestep"],
            y=p(df["timestep"]),
            mode="lines",
            name="Trend",
            line=dict(color="red", width=2, dash="dash"),
            hovertemplate="Trend line<extra></extra>",
        )
    )

    fig.update_layout(
        title="Spatial Clustering Over Time",
        xaxis_title="Timestep",
        yaxis_title="Clustering Coefficient",
        template="plotly_white",
        height=400,
        hovermode="x unified",
    )

    return fig


def plot_assortative_mating_patterns(reproduction_events: list):
    """Analyze assortative mating patterns"""
    if not reproduction_events:
        fig = go.Figure()
        fig.add_annotation(
            text="No reproduction data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(
            title="Assortative Mating Patterns", template="plotly_white", height=400
        )
        return fig

    # Extract education and income matching data
    education_matches = []
    income_correlations = []
    age_gaps = []

    for event in reproduction_events:
        # Education homogamy
        male_edu = event.get("male_education", "unknown")
        female_edu = event.get("female_education", "unknown")
        education_matches.append(1 if male_edu == female_edu else 0)

        # Income correlation (if available)
        male_income = event.get("male_income", 0)
        female_income = event.get("female_income", 0)
        if male_income > 0 and female_income > 0:
            income_correlations.append((male_income, female_income))

        # Age gap
        age_gaps.append(event.get("age_gap", 0))

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Education Homogamy Rate",
            "Income Correlation in Couples",
            "Age Gap Distribution",
            "Mating Patterns Over Time",
        ),
        specs=[
            [{"type": "indicator"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "scatter"}],
        ],
    )

    # Education homogamy rate
    homogamy_rate = np.mean(education_matches) * 100 if education_matches else 0

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=homogamy_rate,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Education Homogamy %"},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 50], "color": "lightgray"},
                    {"range": [50, 100], "color": "gray"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        ),
        row=1,
        col=1,
    )

    # Income correlation scatter
    if income_correlations:
        male_incomes, female_incomes = zip(*income_correlations)
        fig.add_trace(
            go.Scatter(
                x=male_incomes,
                y=female_incomes,
                mode="markers",
                name="Income Correlation",
                marker=dict(color="blue", size=6, opacity=0.6),
                hovertemplate="Male Income: $%{x:,.0f}<br>Female Income: $%{y:,.0f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Add correlation line
        if len(income_correlations) > 1:
            correlation = np.corrcoef(male_incomes, female_incomes)[0, 1]
            z = np.polyfit(male_incomes, female_incomes, 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=male_incomes,
                    y=p(male_incomes),
                    mode="lines",
                    name=f"Correlation: {correlation:.3f}",
                    line=dict(color="red", width=2),
                ),
                row=1,
                col=2,
            )

    # Age gap histogram
    fig.add_trace(
        go.Histogram(
            x=age_gaps,
            nbinsx=20,
            name="Age Gap Distribution",
            marker_color="green",
            hovertemplate="Age Gap: %{x:.1f} years<br>Count: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Mating patterns over time
    timesteps = [event.get("timestep", 0) for event in reproduction_events]
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=age_gaps,
            mode="markers",
            name="Age Gaps Over Time",
            marker=dict(color="orange", size=4, opacity=0.6),
            hovertemplate="Timestep: %{x}<br>Age Gap: %{y:.1f} years<extra></extra>",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(height=600, template="plotly_white", showlegend=False)

    return fig


def plot_agent_lifecycle_summary(reproduction_events: list, encounter_events: list):
    """Create agent lifecycle and encounter summary"""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Reproduction Success by Age",
            "Encounter to Reproduction Rate",
            "Partner Age Preferences",
            "Success Rate by Demographics",
        ),
    )

    if reproduction_events:
        # Group reproductions by age
        male_ages = [event.get("male_age", 0) for event in reproduction_events]
        female_ages = [event.get("female_age", 0) for event in reproduction_events]

        # Reproduction success by age (histogram)
        fig.add_trace(
            go.Histogram(
                x=male_ages,
                name="Male Ages at Reproduction",
                marker_color="blue",
                opacity=0.7,
                nbinsx=15,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Histogram(
                x=female_ages,
                name="Female Ages at Reproduction",
                marker_color="red",
                opacity=0.7,
                nbinsx=15,
            ),
            row=1,
            col=1,
        )

        # Partner age preferences scatter
        fig.add_trace(
            go.Scatter(
                x=male_ages,
                y=female_ages,
                mode="markers",
                name="Age Pairs",
                marker=dict(color="purple", size=6, opacity=0.6),
                hovertemplate="Male Age: %{x}<br>Female Age: %{y}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Add diagonal line for same-age pairs
        min_age, max_age = min(male_ages + female_ages), max(male_ages + female_ages)
        fig.add_trace(
            go.Scatter(
                x=[min_age, max_age],
                y=[min_age, max_age],
                mode="lines",
                name="Same Age Line",
                line=dict(color="gray", dash="dash"),
            ),
            row=2,
            col=1,
        )

    # Encounter to reproduction conversion (if encounter data available)
    if encounter_events and reproduction_events:
        encounter_timesteps = [event.get("timestep", 0) for event in encounter_events]
        repro_timesteps = [event.get("timestep", 0) for event in reproduction_events]

        # Group by timestep
        encounter_counts = pd.Series(encounter_timesteps).value_counts().sort_index()
        repro_counts = pd.Series(repro_timesteps).value_counts().sort_index()

        # Calculate conversion rates
        conversion_rates = []
        timesteps = []

        for ts in encounter_counts.index:
            encounters = encounter_counts.get(ts, 0)
            reproductions = repro_counts.get(ts, 0)
            if encounters > 0:
                rate = (reproductions / encounters) * 100
                conversion_rates.append(rate)
                timesteps.append(ts)

        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=conversion_rates,
                mode="lines+markers",
                name="Conversion Rate",
                line=dict(color="green", width=3),
                hovertemplate="Timestep: %{x}<br>Conversion Rate: %{y:.1f}%<extra></extra>",
            ),
            row=1,
            col=2,
        )

    # Success rate by demographics (education levels)
    if reproduction_events:
        education_success = {}
        for event in reproduction_events:
            male_edu = event.get("male_education", "unknown")
            female_edu = event.get("female_education", "unknown")

            for edu in [male_edu, female_edu]:
                if edu not in education_success:
                    education_success[edu] = 0
                education_success[edu] += 1

        if education_success:
            fig.add_trace(
                go.Bar(
                    x=list(education_success.keys()),
                    y=list(education_success.values()),
                    name="Reproductions by Education",
                    marker_color="teal",
                ),
                row=2,
                col=2,
            )

    fig.update_layout(height=600, template="plotly_white")

    return fig


def create_spatial_dynamics_tab(data, filtered_metrics, time_range):
    """Spatial Dynamics Tab"""
    st.markdown("---")
    st.markdown("## 🗺️ Spatial Dynamics")

    if not filtered_metrics:
        st.warning("No data available for the selected time range")
        return

    # Population density heatmap with timestep selector
    st.markdown("#### Population Density Heatmap")

    col1, col2 = st.columns([3, 1])

    with col2:
        timestep_options = [
            (i, f"Timestep {m['timestep']}") for i, m in enumerate(filtered_metrics)
        ]
        selected_idx = st.selectbox(
            "Select Timestep",
            options=range(len(timestep_options)),
            format_func=lambda x: timestep_options[x][1],
            index=len(timestep_options) - 1,
        )

        # Animation controls
        if st.button("▶️ Animate Heatmap"):
            progress_bar = st.progress(0)
            heatmap_placeholder = st.empty()

            for i in range(len(filtered_metrics)):
                fig = plot_population_density_heatmap(filtered_metrics, i)
                heatmap_placeholder.plotly_chart(fig, use_container_width=True)
                progress_bar.progress((i + 1) / len(filtered_metrics))
                time.sleep(0.5)  # Add import time at top

    with col1:
        fig_heatmap = plot_population_density_heatmap(filtered_metrics, selected_idx)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # Clustering coefficient trends
    st.markdown("#### Spatial Clustering Analysis")
    col1, col2 = st.columns([2, 1])

    with col1:
        fig_clustering = plot_clustering_trends(filtered_metrics)
        st.plotly_chart(fig_clustering, use_container_width=True)

    with col2:
        st.markdown("#### Clustering Insights")
        if filtered_metrics:
            initial_cluster = filtered_metrics[0].get("clustering_coefficient", 0)
            final_cluster = filtered_metrics[-1].get("clustering_coefficient", 0)
            cluster_change = final_cluster - initial_cluster

            st.metric("Clustering Change", f"{cluster_change:+.3f}")
            st.metric("Final Clustering", f"{final_cluster:.3f}")

            if cluster_change > 0.05:
                st.success("🔗 Population becoming more clustered")
            elif cluster_change < -0.05:
                st.info("📡 Population becoming more dispersed")
            else:
                st.info("⚖️ Clustering relatively stable")

    # Geographic distribution analysis
    st.markdown("#### Geographic Distribution Analysis")

    if filtered_metrics:
        # Analyze population spread
        final_metric = filtered_metrics[-1]
        density_map = final_metric.get("population_density_map", {})

        if density_map:
            # Calculate distribution statistics
            coords = []
            densities = []

            for coord_str, density in density_map.items():
                try:
                    coord = coord_str.strip("()").split(", ")
                    x, y = int(coord[0]), int(coord[1])
                    coords.append((x, y))
                    densities.append(density)
                except (ValueError, IndexError):
                    continue

            if coords:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    max_density_idx = np.argmax(densities)
                    hotspot = coords[max_density_idx]
                    st.metric("Population Hotspot", f"({hotspot[0]}, {hotspot[1]})")

                with col2:
                    total_occupied_cells = len([d for d in densities if d > 0])
                    st.metric("Occupied Cells", f"{total_occupied_cells:,}")

                with col3:
                    avg_density = np.mean(densities)
                    st.metric("Average Density", f"{avg_density:.1f}")

                with col4:
                    density_std = np.std(densities)
                    st.metric("Density Std Dev", f"{density_std:.1f}")


def create_reproduction_genetics_tab(data, time_range):
    """Reproduction & Genetics Tab"""
    st.markdown("---")
    st.markdown("## 🧬 Reproduction & Genetics")

    reproduction_events = data.get("reproduction_events", [])
    encounter_events = data.get("encounter_events", [])

    # Filter events by time range
    filtered_reproductions = [
        event
        for event in reproduction_events
        if time_range[0] <= event.get("timestep", 0) <= time_range[1]
    ]

    filtered_encounters = [
        event
        for event in encounter_events
        if time_range[0] <= event.get("timestep", 0) <= time_range[1]
    ]

    if not filtered_reproductions:
        st.warning("No reproduction data available for the selected time range")
        st.info(
            "💡 This analysis requires reproduction event data with partner characteristics"
        )
        return

    # Assortative mating patterns
    st.markdown("#### Assortative Mating Patterns")
    fig_assortative = plot_assortative_mating_patterns(filtered_reproductions)
    st.plotly_chart(fig_assortative, use_container_width=True)

    # Age gap detailed analysis
    st.markdown("#### Age Gap Analysis in Successful Matings")
    col1, col2 = st.columns([2, 1])

    with col1:
        fig_age_gaps = plot_age_gap_analysis(filtered_reproductions)
        st.plotly_chart(fig_age_gaps, use_container_width=True)

    with col2:
        st.markdown("#### Age Gap Statistics")
        age_gaps = [event.get("age_gap", 0) for event in filtered_reproductions]

        if age_gaps:
            st.metric("Average Age Gap", f"{np.mean(age_gaps):.1f} years")
            st.metric("Median Age Gap", f"{np.median(age_gaps):.1f} years")
            st.metric("Max Age Gap", f"{np.max(age_gaps):.1f} years")
            st.metric("Age Gap Std Dev", f"{np.std(age_gaps):.1f} years")

    # Education and income mixing
    st.markdown("#### Education & Income Mixing Analysis")

    # Calculate homogamy rates
    education_matches = []
    income_similarities = []

    for event in filtered_reproductions:
        # Education homogamy
        male_edu = event.get("male_education", "unknown")
        female_edu = event.get("female_education", "unknown")
        if male_edu != "unknown" and female_edu != "unknown":
            education_matches.append(male_edu == female_edu)

        # Income similarity (within 20% of each other)
        male_income = event.get("male_income", 0)
        female_income = event.get("female_income", 0)
        if male_income > 0 and female_income > 0:
            similarity = abs(male_income - female_income) / max(
                male_income, female_income
            )
            income_similarities.append(similarity < 0.2)  # Within 20%

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        homogamy_rate = (np.mean(education_matches) * 100) if education_matches else 0
        st.metric("Education Homogamy", f"{homogamy_rate:.1f}%")

    with col2:
        income_similarity_rate = (
            (np.mean(income_similarities) * 100) if income_similarities else 0
        )
        st.metric("Income Similarity", f"{income_similarity_rate:.1f}%")

    with col3:
        total_reproductions = len(filtered_reproductions)
        st.metric("Total Reproductions", f"{total_reproductions:,}")

    with col4:
        if filtered_reproductions:
            avg_reproductive_age = np.mean(
                [
                    (event.get("male_age", 0) + event.get("female_age", 0)) / 2
                    for event in filtered_reproductions
                ]
            )
            st.metric("Avg Reproductive Age", f"{avg_reproductive_age:.1f} years")


def create_individual_tracking_tab(data, time_range):
    """Individual Agent Tracking Tab"""
    st.markdown("---")
    st.markdown("## 👤 Individual Agent Tracking")

    reproduction_events = data.get("reproduction_events", [])
    encounter_events = data.get("encounter_events", [])

    # Filter events by time range
    filtered_reproductions = [
        event
        for event in reproduction_events
        if time_range[0] <= event.get("timestep", 0) <= time_range[1]
    ]

    filtered_encounters = [
        event
        for event in encounter_events
        if time_range[0] <= event.get("timestep", 0) <= time_range[1]
    ]

    # Agent lifecycle visualization
    st.markdown("#### Agent Lifecycle & Success Patterns")
    fig_lifecycle = plot_agent_lifecycle_summary(
        filtered_reproductions, filtered_encounters
    )
    st.plotly_chart(fig_lifecycle, use_container_width=True)

    # Success/rejection patterns
    st.markdown("#### Success and Rejection Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Encounter Outcomes")

        if filtered_encounters and filtered_reproductions:
            total_encounters = len(filtered_encounters)
            successful_encounters = len(filtered_reproductions)
            success_rate = (
                (successful_encounters / total_encounters) * 100
                if total_encounters > 0
                else 0
            )

            st.metric("Total Encounters", f"{total_encounters:,}")
            st.metric("Successful Reproductions", f"{successful_encounters:,}")
            st.metric("Success Rate", f"{success_rate:.1f}%")

            # Success rate over time
            encounter_times = pd.Series(
                [e.get("timestep", 0) for e in filtered_encounters]
            )
            repro_times = pd.Series(
                [r.get("timestep", 0) for r in filtered_reproductions]
            )

            time_success_rates = []
            timesteps = sorted(encounter_times.unique())

            for ts in timesteps:
                encounters_at_time = (encounter_times == ts).sum()
                repros_at_time = (repro_times == ts).sum()
                rate = (
                    (repros_at_time / encounters_at_time * 100)
                    if encounters_at_time > 0
                    else 0
                )
                time_success_rates.append(rate)

            if time_success_rates:
                fig_success_time = go.Figure()
                fig_success_time.add_trace(
                    go.Scatter(
                        x=timesteps,
                        y=time_success_rates,
                        mode="lines+markers",
                        name="Success Rate Over Time",
                        line=dict(color="green", width=3),
                    )
                )

                fig_success_time.update_layout(
                    title="Mating Success Rate Over Time",
                    xaxis_title="Timestep",
                    yaxis_title="Success Rate (%)",
                    template="plotly_white",
                    height=300,
                )

                st.plotly_chart(fig_success_time, use_container_width=True)

        else:
            st.info("💡 Encounter data needed for detailed success analysis")

    with col2:
        st.markdown("##### Demographic Success Factors")

        if filtered_reproductions:
            # Success by age groups
            ages = []
            for event in filtered_reproductions:
                male_age = event.get("male_age", 0)
                female_age = event.get("female_age", 0)
                ages.extend([male_age, female_age])

            if ages:
                age_bins = ["18-25", "26-35", "36-45", "46+"]
                age_counts = [0, 0, 0, 0]

                for age in ages:
                    if 18 <= age <= 25:
                        age_counts[0] += 1
                    elif 26 <= age <= 35:
                        age_counts[1] += 1
                    elif 36 <= age <= 45:
                        age_counts[2] += 1
                    elif age > 45:
                        age_counts[3] += 1

                fig_age_success = go.Figure(
                    data=[go.Bar(x=age_bins, y=age_counts, marker_color="skyblue")]
                )

                fig_age_success.update_layout(
                    title="Reproductive Success by Age Group",
                    xaxis_title="Age Group",
                    yaxis_title="Reproduction Count",
                    template="plotly_white",
                    height=300,
                )

                st.plotly_chart(fig_age_success, use_container_width=True)

    # Individual agent spotlight (if agent IDs are available)
    st.markdown("#### Agent Spotlight")

    if filtered_reproductions:
        # Get unique agents
        agents = set()
        for event in filtered_reproductions:
            male_id = event.get("male_id", "unknown")
            female_id = event.get("female_id", "unknown")
            if male_id != "unknown":
                agents.add(("male", male_id))
            if female_id != "unknown":
                agents.add(("female", female_id))

        if agents:
            st.info(f"🔍 Found {len(agents)} unique agents in reproduction events")

            # Most successful agents
            agent_success = {}
            for event in filtered_reproductions:
                male_id = event.get("male_id", "unknown")
                female_id = event.get("female_id", "unknown")

                if male_id != "unknown":
                    agent_success[male_id] = agent_success.get(male_id, 0) + 1
                if female_id != "unknown":
                    agent_success[female_id] = agent_success.get(female_id, 0) + 1

            if agent_success:
                top_agents = sorted(
                    agent_success.items(), key=lambda x: x[1], reverse=True
                )[:10]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("##### Most Successful Agents")
                    for i, (agent_id, success_count) in enumerate(top_agents[:5], 1):
                        st.write(
                            f"{i}. Agent {agent_id}: {success_count} reproductions"
                        )

                with col2:
                    # Success distribution
                    success_counts = list(agent_success.values())

                    fig_success_dist = go.Figure(
                        data=[
                            go.Histogram(
                                x=success_counts, nbinsx=10, marker_color="lightcoral"
                            )
                        ]
                    )

                    fig_success_dist.update_layout(
                        title="Distribution of Reproductive Success",
                        xaxis_title="Number of Reproductions",
                        yaxis_title="Number of Agents",
                        template="plotly_white",
                        height=300,
                    )

                    st.plotly_chart(fig_success_dist, use_container_width=True)

        else:
            st.info("💡 Agent ID data needed for individual tracking")

    else:
        st.warning("No reproduction data available for individual tracking")


def main():
    st.title("🧬 Population Dating Simulation Dashboard")
    st.markdown("### Analyzing Mating Patterns and Population Dynamics")

    # Sidebar for data upload
    st.sidebar.header("📊 Data Configuration")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Simulation Data",
        type=["pkl", "json"],
        help="Upload your simulation data dictionary (.pkl or .json format)",
    )

    # Load data
    if uploaded_file is not None:
        data = load_simulation_data(uploaded_file)
        if data is None:
            st.stop()
        st.sidebar.success("✅ Data loaded successfully!")
    else:
        st.sidebar.info("Using sample data for demonstration")
        data = create_sample_data()

    # Time range selector
    pop_metrics = data.get("population_metrics", [])
    if pop_metrics:
        min_time, max_time = pop_metrics[0]["timestep"], pop_metrics[-1]["timestep"]

        time_range = st.sidebar.slider(
            "Select Time Range",
            min_value=min_time,
            max_value=max_time,
            value=(min_time, max_time),
            help="Filter data by timestep range",
        )

        # Filter data based on time range
        filtered_metrics = [
            m for m in pop_metrics if time_range[0] <= m["timestep"] <= time_range[1]
        ]
    else:
        filtered_metrics = []
        st.error("No population metrics data found")
        st.stop()

    # Tab selection
    st.sidebar.markdown("---")
    tab_selection = st.sidebar.selectbox(
        "Select Analysis Tab",
        [
            "Overview",
            "Population Dynamics",
            "Demographics & Socioeconomics",
            "Mating Market Analysis",
            "Spatial Dynamics",
            "Reproduction & Genetics",
            "Individual Agent Tracking",
        ],
        help="Choose which analysis to display",
    )

    # Header Section with Key Metrics
    if tab_selection == "Overview":
        st.markdown("---")
        st.markdown("## 📈 Simulation Overview")
        display_header_metrics(data)

        # Quick summary charts
        col1, col2 = st.columns(2)
        with col1:
            fig_pop = plot_population_over_time(filtered_metrics)
            st.plotly_chart(fig_pop, use_container_width=True)

        with col2:
            fig_age = plot_age_distribution_evolution(filtered_metrics)
            st.plotly_chart(fig_age, use_container_width=True)

    # Population Dynamics Tab
    elif tab_selection == "Population Dynamics":
        st.markdown("---")
        st.markdown("## 👥 Population Dynamics")

        if filtered_metrics:
            # Population over time
            col1, col2 = st.columns([2, 1])

            with col1:
                fig_pop = plot_population_over_time(filtered_metrics)
                st.plotly_chart(fig_pop, use_container_width=True)

            with col2:
                # Population summary stats
                st.markdown("#### Key Statistics")
                initial = filtered_metrics[0]
                final = filtered_metrics[-1]

                st.metric(
                    "Population Change",
                    f"{final['total_population'] - initial['total_population']:+d}",
                )
                st.metric(
                    "Gender Ratio (M/F)",
                    f"{final['male_count'] / final['female_count']:.3f}"
                    if final["female_count"] > 0
                    else "N/A",
                )
                st.metric(
                    "Avg Age Change",
                    f"{final['avg_age'] - initial['avg_age']:+.1f} years",
                )

            # Age distribution evolution
            st.markdown("#### Age Distribution Over Time")
            fig_age = plot_age_distribution_evolution(filtered_metrics)
            st.plotly_chart(fig_age, use_container_width=True)

            # Gender analysis
            st.markdown("#### Gender Dynamics & Singles Market")
            fig_gender = plot_gender_ratio_and_singles(filtered_metrics)
            st.plotly_chart(fig_gender, use_container_width=True)

            # Age and income trends
            st.markdown("#### Demographics & Economics")
            fig_demo = plot_avg_age_and_income(filtered_metrics)
            st.plotly_chart(fig_demo, use_container_width=True)

        else:
            st.warning("No data available for the selected time range")

    # Demographics & Socioeconomics Tab
    elif tab_selection == "Demographics & Socioeconomics":
        st.markdown("---")
        st.markdown("## 🎓 Demographics & Socioeconomics")

        if filtered_metrics:
            # Education distribution
            st.markdown("#### Education Level Distribution")
            col1, col2 = st.columns([2, 1])

            with col1:
                fig_edu = plot_education_distribution(filtered_metrics)
                st.plotly_chart(fig_edu, use_container_width=True)

            with col2:
                # Education summary
                st.markdown("#### Education Summary")
                final_edu = filtered_metrics[-1].get("education_distribution", {})
                total_pop = sum(final_edu.values())

                for edu_level, count in final_edu.items():
                    percentage = (count / total_pop * 100) if total_pop > 0 else 0
                    st.metric(f"{edu_level.title()}", f"{count:,} ({percentage:.1f}%)")

            # Income analysis
            st.markdown("#### Income Distribution & Inequality")
            col1, col2 = st.columns(2)

            with col1:
                fig_income_violin = plot_income_distribution_violin(filtered_metrics)
                st.plotly_chart(fig_income_violin, use_container_width=True)

            with col2:
                fig_income_trend = plot_avg_age_and_income(filtered_metrics)
                st.plotly_chart(fig_income_trend, use_container_width=True)

            # Socioeconomic metrics summary
            st.markdown("#### Socioeconomic Trends")
            initial = filtered_metrics[0]
            final = filtered_metrics[-1]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                income_change = (
                    (final["avg_income"] - initial["avg_income"])
                    / initial["avg_income"]
                ) * 100
                st.metric("Income Growth", f"{income_change:+.1f}%")

            with col2:
                gini_change = (
                    final["income_inequality_gini"] - initial["income_inequality_gini"]
                )
                st.metric(
                    "Inequality Change",
                    f"{gini_change:+.3f}",
                    help="Change in Gini coefficient (higher = more inequality)",
                )

            with col3:
                age_change = final["avg_age"] - initial["avg_age"]
                st.metric("Population Aging", f"{age_change:+.1f} years")

            with col4:
                cluster_change = final.get("clustering_coefficient", 0) - initial.get(
                    "clustering_coefficient", 0
                )
                st.metric(
                    "Clustering Change",
                    f"{cluster_change:+.3f}",
                    help="Change in spatial clustering coefficient",
                )

        else:
            st.warning("No data available for the selected time range")

    # Mating Market Analysis Tab
    elif tab_selection == "Mating Market Analysis":
        st.markdown("---")
        st.markdown("## 💕 Mating Market Analysis")

        if filtered_metrics:
            # Selectivity trends
            st.markdown("#### Selectivity Patterns by Gender")
            fig_selectivity = plot_selectivity_trends(filtered_metrics)
            st.plotly_chart(fig_selectivity, use_container_width=True)

            # Get reproduction and encounter data
            reproduction_events = data.get("reproduction_events", [])
            encounter_events = data.get("encounter_events", [])

            # Filter events by time range
            filtered_reproductions = [
                event
                for event in reproduction_events
                if time_range[0] <= event.get("timestep", 0) <= time_range[1]
            ]

            filtered_encounters = [
                event
                for event in encounter_events
                if time_range[0] <= event.get("timestep", 0) <= time_range[1]
            ]

            # Mating success analysis
            st.markdown("#### Mating Success Patterns")
            fig_success = plot_mating_success_analysis(
                filtered_reproductions, filtered_encounters
            )
            st.plotly_chart(fig_success, use_container_width=True)

            # Age gap analysis
            st.markdown("#### Age Gap Analysis in Successful Matings")
            fig_age_gaps = plot_age_gap_analysis(filtered_reproductions)
            st.plotly_chart(fig_age_gaps, use_container_width=True)

            # Market dynamics summary
            st.markdown("#### Market Dynamics Summary")
            col1, col2, col3, col4 = st.columns(4)

            initial = filtered_metrics[0]
            final = filtered_metrics[-1]

            with col1:
                single_rate_initial = (
                    initial["single_males"] + initial["single_females"]
                ) / initial["total_population"]
                single_rate_final = (
                    final["single_males"] + final["single_females"]
                ) / final["total_population"]
                single_change = (single_rate_final - single_rate_initial) * 100
                st.metric("Singles Rate Change", f"{single_change:+.1f}%")

            with col2:
                selectivity_gap_initial = (
                    initial["avg_female_selectivity"] - initial["avg_male_selectivity"]
                )
                selectivity_gap_final = (
                    final["avg_female_selectivity"] - final["avg_male_selectivity"]
                )
                gap_change = selectivity_gap_final - selectivity_gap_initial
                st.metric("Selectivity Gap Change", f"{gap_change:+.3f}")

            with col3:
                total_reproductions = len(filtered_reproductions)
                st.metric("Total Reproductions", f"{total_reproductions:,}")

            with col4:
                if filtered_reproductions:
                    avg_age_gap = np.mean(
                        [event.get("age_gap", 0) for event in filtered_reproductions]
                    )
                    st.metric("Avg Age Gap", f"{avg_age_gap:.1f} years")
                else:
                    st.metric("Avg Age Gap", "N/A")

        else:
            st.warning("No data available for the selected time range")

    elif tab_selection == "Spatial Dynamics":
        create_spatial_dynamics_tab(data, filtered_metrics, time_range)
    elif tab_selection == "Reproduction & Genetics":
        create_reproduction_genetics_tab(data, time_range)
    elif tab_selection == "Individual Agent Tracking":
        create_individual_tracking_tab(data, time_range)


if __name__ == "__main__":
    main()

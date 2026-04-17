"""
Change Management Project Insights Dashboard
==========================================

What this application does
--------------------------
This Streamlit app helps teams analyze project status and change management readiness
across five strategy dimensions:
1) Involved
2) Monitoring
3) Stakeholder Engagement
4) Transitional Strategy
5) Transformational Strategy

How to run
----------
1. Install requirements (see list near bottom of this file or use pip install streamlit pandas numpy plotly)
2. Start the app:
   streamlit run app.py

The app supports manual project entry, CSV upload, and auto-generated sample data.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Configuration and constants
# -----------------------------
RISK_MAP = {"Low": 85, "Medium": 60, "High": 30}
STATUS_OPTIONS = ["On Track", "At Risk", "Delayed"]
REQUIRED_COLUMNS = [
    "Project Name",
    "Project Manager",
    "Department",
    "Budget",
    "Timeline Progress (%)",
    "Overall Completion (%)",
    "Risk Level",
    "Number of Stakeholders",
    "Stakeholder Sentiment Score",
    "Communication Frequency",
    "Change Readiness Score",
    "Training Completion (%)",
    "Adoption Score",
    "Monitoring Score",
    "Involvement Score",
    "Transitional Readiness Score",
    "Transformational Readiness Score",
    "Status",
]


def clamp_series(series: pd.Series, low: float = 0, high: float = 100) -> pd.Series:
    """Clamp numeric series to a range."""
    return series.clip(lower=low, upper=high)


def risk_to_score(risk_level: pd.Series) -> pd.Series:
    """Map risk string labels to numeric (higher is better)."""
    return risk_level.map(RISK_MAP).fillna(50)


def parse_and_validate_csv(file) -> pd.DataFrame:
    """Read uploaded CSV and validate required columns."""
    try:
        df = pd.read_csv(file)
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(f"Unable to read CSV file: {exc}") from exc

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Uploaded CSV is missing required columns: " + ", ".join(missing)
        )

    # Clean and enforce basic numeric coercions
    numeric_cols = [
        "Budget",
        "Timeline Progress (%)",
        "Overall Completion (%)",
        "Number of Stakeholders",
        "Stakeholder Sentiment Score",
        "Communication Frequency",
        "Change Readiness Score",
        "Training Completion (%)",
        "Adoption Score",
        "Monitoring Score",
        "Involvement Score",
        "Transitional Readiness Score",
        "Transformational Readiness Score",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Project Name"]).copy()

    # Normalize text columns
    for col in ["Project Manager", "Department", "Risk Level", "Status"]:
        df[col] = df[col].astype(str).str.strip()

    return df


def generate_sample_data(n_rows: int = 150, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic portfolio data."""
    rng = np.random.default_rng(seed)

    departments = ["IT", "Finance", "HR", "Operations", "Sales", "Marketing"]
    managers = [
        "A. Johnson",
        "M. Patel",
        "R. Chen",
        "K. Smith",
        "D. Garcia",
        "L. Brown",
        "S. Kim",
    ]

    risk_levels = rng.choice(["Low", "Medium", "High"], size=n_rows, p=[0.35, 0.45, 0.20])

    timeline_progress = clamp_series(pd.Series(rng.normal(62, 20, n_rows))).round(1)
    completion = clamp_series(
        timeline_progress + pd.Series(rng.normal(0, 12, n_rows)), 0, 100
    ).round(1)

    monitoring_score = clamp_series(pd.Series(rng.normal(66, 16, n_rows))).round(1)
    involvement_score = clamp_series(pd.Series(rng.normal(63, 18, n_rows))).round(1)
    stakeholder_sentiment = clamp_series(pd.Series(rng.normal(64, 17, n_rows))).round(1)
    communication_freq = clamp_series(pd.Series(rng.normal(58, 20, n_rows))).round(1)
    change_readiness = clamp_series(pd.Series(rng.normal(61, 18, n_rows))).round(1)
    training_completion = clamp_series(pd.Series(rng.normal(57, 22, n_rows))).round(1)
    adoption_score = clamp_series(pd.Series(rng.normal(60, 20, n_rows))).round(1)
    transitional_readiness = clamp_series(pd.Series(rng.normal(59, 18, n_rows))).round(1)
    transformational_readiness = clamp_series(pd.Series(rng.normal(56, 20, n_rows))).round(1)

    kpi_consistency = clamp_series(pd.Series(rng.normal(65, 14, n_rows))).round(1)
    issue_escalation_visibility = clamp_series(pd.Series(rng.normal(62, 16, n_rows))).round(1)
    process_migration_readiness = clamp_series(pd.Series(rng.normal(58, 18, n_rows))).round(1)
    innovation_readiness = clamp_series(pd.Series(rng.normal(55, 18, n_rows))).round(1)

    budget = np.round(np.exp(rng.normal(np.log(450000), 0.75, n_rows)), 0)
    stakeholders = np.clip(rng.poisson(24, n_rows) + 5, 5, 120)

    df = pd.DataFrame(
        {
            "Project Name": [f"Project-{i:03d}" for i in range(1, n_rows + 1)],
            "Project Manager": rng.choice(managers, size=n_rows),
            "Department": rng.choice(departments, size=n_rows),
            "Budget": budget,
            "Timeline Progress (%)": timeline_progress,
            "Overall Completion (%)": completion,
            "Risk Level": risk_levels,
            "Number of Stakeholders": stakeholders,
            "Stakeholder Sentiment Score": stakeholder_sentiment,
            "Communication Frequency": communication_freq,
            "Change Readiness Score": change_readiness,
            "Training Completion (%)": training_completion,
            "Adoption Score": adoption_score,
            "Monitoring Score": monitoring_score,
            "Involvement Score": involvement_score,
            "Transitional Readiness Score": transitional_readiness,
            "Transformational Readiness Score": transformational_readiness,
            # Additional useful fields
            "KPI Consistency": kpi_consistency,
            "Issue Escalation Visibility": issue_escalation_visibility,
            "Process Migration Readiness": process_migration_readiness,
            "Innovation Readiness": innovation_readiness,
        }
    )

    df = calculate_dimension_scores(df)
    df["Status"] = df.apply(classify_project_status, axis=1)
    df["Recommendations"] = df.apply(generate_recommendations, axis=1)
    return df


def calculate_dimension_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate five normalized strategy dimension scores and overall health."""
    out = df.copy()

    # Ensure optional columns exist for upload compatibility
    for opt_col, default in {
        "KPI Consistency": out.get("Monitoring Score", pd.Series(60, index=out.index)),
        "Issue Escalation Visibility": out.get(
            "Monitoring Score", pd.Series(60, index=out.index)
        ),
        "Process Migration Readiness": out.get(
            "Transitional Readiness Score", pd.Series(55, index=out.index)
        ),
        "Innovation Readiness": out.get(
            "Transformational Readiness Score", pd.Series(55, index=out.index)
        ),
    }.items():
        if opt_col not in out.columns:
            out[opt_col] = default

    # Convert expected numeric columns
    numeric_cols = [
        "Timeline Progress (%)",
        "Overall Completion (%)",
        "Number of Stakeholders",
        "Stakeholder Sentiment Score",
        "Communication Frequency",
        "Change Readiness Score",
        "Training Completion (%)",
        "Adoption Score",
        "Monitoring Score",
        "Involvement Score",
        "Transitional Readiness Score",
        "Transformational Readiness Score",
        "KPI Consistency",
        "Issue Escalation Visibility",
        "Process Migration Readiness",
        "Innovation Readiness",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out[numeric_cols] = out[numeric_cols].fillna(out[numeric_cols].median(numeric_only=True))

    stakeholder_participation = clamp_series(
        (out["Number of Stakeholders"] / out["Number of Stakeholders"].quantile(0.95))
        * 100,
        0,
        100,
    )

    out["Involved"] = clamp_series(
        0.35 * out["Involvement Score"]
        + 0.2 * out["Communication Frequency"]
        + 0.25 * out["Training Completion (%)"]
        + 0.2 * stakeholder_participation
    )

    out["Monitoring"] = clamp_series(
        0.3 * out["Monitoring Score"]
        + 0.2 * out["Timeline Progress (%)"]
        + 0.2 * risk_to_score(out["Risk Level"])
        + 0.15 * out["KPI Consistency"]
        + 0.15 * out["Issue Escalation Visibility"]
    )

    out["Stakeholder Engagement"] = clamp_series(
        0.3 * out["Stakeholder Sentiment Score"]
        + 0.2 * out["Communication Frequency"]
        + 0.15 * stakeholder_participation
        + 0.2 * out["Involvement Score"]
        + 0.15 * out["Adoption Score"]
    )

    out["Transitional Strategy"] = clamp_series(
        0.35 * out["Change Readiness Score"]
        + 0.25 * out["Training Completion (%)"]
        + 0.2 * out["Process Migration Readiness"]
        + 0.2 * out["Transitional Readiness Score"]
    )

    out["Transformational Strategy"] = clamp_series(
        0.3 * out["Adoption Score"]
        + 0.25 * out["Innovation Readiness"]
        + 0.2 * out["Change Readiness Score"]
        + 0.25 * out["Transformational Readiness Score"]
    )

    dim_cols = [
        "Involved",
        "Monitoring",
        "Stakeholder Engagement",
        "Transitional Strategy",
        "Transformational Strategy",
    ]

    out["Overall Project Health"] = out[dim_cols].mean(axis=1).round(1)
    out[dim_cols] = out[dim_cols].round(1)
    return out


def classify_project_status(row: pd.Series) -> str:
    """Rule-based classification from health and risk."""
    health = float(row.get("Overall Project Health", 0))
    risk = str(row.get("Risk Level", "Medium"))
    completion = float(row.get("Overall Completion (%)", 0))

    if health >= 75 and risk == "Low" and completion >= 50:
        return "On Track"
    if health < 55 or risk == "High":
        return "Delayed"
    return "At Risk"


def generate_recommendations(row: pd.Series) -> str:
    """Generate recommendation text based on low-scoring dimensions."""
    recommendations: List[str] = []

    if row.get("Stakeholder Engagement", 100) < 60:
        recommendations.append(
            "Increase stakeholder touchpoints: weekly updates and targeted workshops."
        )
    if row.get("Monitoring", 100) < 60:
        recommendations.append(
            "Strengthen KPI reviews, risk governance, and issue escalation cadence."
        )
    if row.get("Transitional Strategy", 100) < 60:
        recommendations.append(
            "Expand training completion and support phased rollout playbooks."
        )
    if row.get("Transformational Strategy", 100) < 60:
        recommendations.append(
            "Align leadership on long-term adoption and organizational change roadmap."
        )
    if row.get("Involved", 100) < 60:
        recommendations.append(
            "Boost team involvement with role clarity, communication, and participation goals."
        )

    if not recommendations:
        return "Maintain momentum with monthly health checks and benefit realization tracking."

    return " ".join(recommendations)


def render_project_radar(selected_row: pd.Series) -> go.Figure:
    """Radar chart for 5 dimension scores."""
    dims = [
        "Involved",
        "Monitoring",
        "Stakeholder Engagement",
        "Transitional Strategy",
        "Transformational Strategy",
    ]
    values = [selected_row[d] for d in dims]
    values += [values[0]]
    dims += [dims[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=dims,
            fill="toself",
            name=selected_row["Project Name"],
            line=dict(color="#0068C9"),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def render_dashboard(df: pd.DataFrame) -> None:
    """Render portfolio and single-project dashboard views."""
    st.subheader("Portfolio Filters")
    c1, c2, c3, c4 = st.columns(4)

    dept_options = ["All"] + sorted(df["Department"].dropna().unique().tolist())
    status_options = ["All"] + STATUS_OPTIONS
    risk_options = ["All"] + sorted(df["Risk Level"].dropna().unique().tolist())
    manager_options = ["All"] + sorted(df["Project Manager"].dropna().unique().tolist())

    selected_dept = c1.selectbox("Department", dept_options)
    selected_status = c2.selectbox("Status", status_options)
    selected_risk = c3.selectbox("Risk Level", risk_options)
    selected_manager = c4.selectbox("Project Manager", manager_options)

    filtered = df.copy()
    if selected_dept != "All":
        filtered = filtered[filtered["Department"] == selected_dept]
    if selected_status != "All":
        filtered = filtered[filtered["Status"] == selected_status]
    if selected_risk != "All":
        filtered = filtered[filtered["Risk Level"] == selected_risk]
    if selected_manager != "All":
        filtered = filtered[filtered["Project Manager"] == selected_manager]

    if filtered.empty:
        st.warning("No projects match the selected filters. Please adjust filters.")
        return

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Projects", f"{len(filtered)}")
    k2.metric("Avg Health", f"{filtered['Overall Project Health'].mean():.1f}")
    k3.metric("Avg Completion", f"{filtered['Overall Completion (%)'].mean():.1f}%")
    k4.metric("High Risk Count", f"{(filtered['Risk Level'] == 'High').sum()}")

    st.markdown("### Portfolio Insights")

    dim_cols = [
        "Involved",
        "Monitoring",
        "Stakeholder Engagement",
        "Transitional Strategy",
        "Transformational Strategy",
    ]

    col_left, col_right = st.columns(2)

    with col_left:
        avg_dim = filtered[dim_cols].mean().reset_index()
        avg_dim.columns = ["Dimension", "Score"]
        bar_fig = px.bar(
            avg_dim,
            x="Dimension",
            y="Score",
            color="Score",
            color_continuous_scale="Blues",
            range_y=[0, 100],
            title="Average Dimension Scores",
        )
        bar_fig.update_layout(margin=dict(l=20, r=20, t=45, b=60))
        st.plotly_chart(bar_fig, use_container_width=True)

    with col_right:
        status_fig = px.pie(
            filtered,
            names="Status",
            title="Project Status Distribution",
            hole=0.45,
            color="Status",
            color_discrete_map={
                "On Track": "#2ca02c",
                "At Risk": "#ffbf00",
                "Delayed": "#d62728",
            },
        )
        status_fig.update_layout(margin=dict(l=20, r=20, t=45, b=20))
        st.plotly_chart(status_fig, use_container_width=True)

    c_scatter, c_heatmap = st.columns(2)

    with c_scatter:
        scatter_fig = px.scatter(
            filtered,
            x="Timeline Progress (%)",
            y="Change Readiness Score",
            size="Budget",
            color="Status",
            hover_data=["Project Name", "Department", "Adoption Score"],
            title="Timeline Progress vs Change Readiness",
            color_discrete_map={
                "On Track": "#2ca02c",
                "At Risk": "#ffbf00",
                "Delayed": "#d62728",
            },
        )
        scatter_fig.update_layout(margin=dict(l=20, r=20, t=45, b=20))
        st.plotly_chart(scatter_fig, use_container_width=True)

    with c_heatmap:
        corr_cols = [
            "Overall Project Health",
            "Timeline Progress (%)",
            "Overall Completion (%)",
            "Change Readiness Score",
            "Adoption Score",
            "Monitoring",
            "Stakeholder Engagement",
            "Transitional Strategy",
            "Transformational Strategy",
        ]
        corr = filtered[corr_cols].corr(numeric_only=True)
        heatmap_fig = px.imshow(
            corr,
            text_auto=True,
            title="Correlation Heatmap (Key Metrics)",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
        )
        heatmap_fig.update_layout(margin=dict(l=20, r=20, t=45, b=20))
        st.plotly_chart(heatmap_fig, use_container_width=True)

    st.markdown("### Single Project View")
    selected_project = st.selectbox(
        "Select a project",
        options=filtered["Project Name"].tolist(),
        index=0,
    )

    proj_row = filtered[filtered["Project Name"] == selected_project].iloc[0]

    p1, p2 = st.columns([1, 1])
    with p1:
        st.plotly_chart(render_project_radar(proj_row), use_container_width=True)

    with p2:
        st.markdown(
            f"""
**Project Manager:** {proj_row['Project Manager']}  
**Department:** {proj_row['Department']}  
**Risk Level:** {proj_row['Risk Level']}  
**Status:** {proj_row['Status']}  
**Overall Health:** {proj_row['Overall Project Health']:.1f}
"""
        )
        st.info(proj_row["Recommendations"])

    st.markdown("### Project Insights Table")
    table_cols = [
        "Project Name",
        "Project Manager",
        "Department",
        "Risk Level",
        "Status",
        "Overall Project Health",
        "Involved",
        "Monitoring",
        "Stakeholder Engagement",
        "Transitional Strategy",
        "Transformational Strategy",
        "Recommendations",
    ]

    display_df = filtered[table_cols].sort_values(
        by="Overall Project Health", ascending=False
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered insights as CSV",
        data=csv_bytes,
        file_name="filtered_project_insights.csv",
        mime="text/csv",
    )


def render_manual_entry_form() -> pd.DataFrame | None:
    """Manual single-project entry form in sidebar."""
    with st.sidebar.expander("Manual Project Entry", expanded=False):
        with st.form("manual_project_form", clear_on_submit=False):
            project_name = st.text_input("Project Name", value="Project-Manual-001")
            manager = st.text_input("Project Manager", value="A. Johnson")
            dept = st.selectbox(
                "Department",
                ["IT", "Finance", "HR", "Operations", "Sales", "Marketing"],
            )
            budget = st.number_input("Budget", min_value=1000.0, value=350000.0, step=1000.0)
            timeline = st.slider("Timeline Progress (%)", 0, 100, 55)
            completion = st.slider("Overall Completion (%)", 0, 100, 50)
            risk = st.selectbox("Risk Level", ["Low", "Medium", "High"])
            stakeholders = st.slider("Number of Stakeholders", 1, 200, 25)
            sentiment = st.slider("Stakeholder Sentiment Score", 0, 100, 62)
            comms = st.slider("Communication Frequency", 0, 100, 60)
            change_ready = st.slider("Change Readiness Score", 0, 100, 58)
            training = st.slider("Training Completion (%)", 0, 100, 54)
            adoption = st.slider("Adoption Score", 0, 100, 59)
            monitoring = st.slider("Monitoring Score", 0, 100, 64)
            involvement = st.slider("Involvement Score", 0, 100, 63)
            trans_ready = st.slider("Transitional Readiness Score", 0, 100, 57)
            transform_ready = st.slider("Transformational Readiness Score", 0, 100, 55)
            status = st.selectbox("Status", STATUS_OPTIONS)

            submitted = st.form_submit_button("Add Manual Project")

        if submitted:
            manual_df = pd.DataFrame(
                [
                    {
                        "Project Name": project_name,
                        "Project Manager": manager,
                        "Department": dept,
                        "Budget": budget,
                        "Timeline Progress (%)": timeline,
                        "Overall Completion (%)": completion,
                        "Risk Level": risk,
                        "Number of Stakeholders": stakeholders,
                        "Stakeholder Sentiment Score": sentiment,
                        "Communication Frequency": comms,
                        "Change Readiness Score": change_ready,
                        "Training Completion (%)": training,
                        "Adoption Score": adoption,
                        "Monitoring Score": monitoring,
                        "Involvement Score": involvement,
                        "Transitional Readiness Score": trans_ready,
                        "Transformational Readiness Score": transform_ready,
                        "Status": status,
                    }
                ]
            )
            return manual_df

    return None


def main() -> None:
    st.set_page_config(
        page_title="Change Management Project Insights",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📊 Change Management Project Insights Dashboard")
    st.write(
        "Assess project health and change readiness across Involved, Monitoring, "
        "Stakeholder Engagement, Transitional Strategy, and Transformational Strategy."
    )

    st.sidebar.header("Data Controls")
    data_mode = st.sidebar.radio(
        "Choose data source",
        ["Use Sample Dataset", "Upload CSV", "Build from Manual Entry"],
    )

    if "manual_projects" not in st.session_state:
        st.session_state.manual_projects = pd.DataFrame(columns=REQUIRED_COLUMNS)

    input_df: pd.DataFrame

    if data_mode == "Use Sample Dataset":
        rows = st.sidebar.slider("Sample rows", min_value=100, max_value=500, value=150, step=25)
        regenerate = st.sidebar.button("Generate Sample Random Dataset")
        if regenerate or "sample_df" not in st.session_state or len(st.session_state.sample_df) != rows:
            st.session_state.sample_df = generate_sample_data(n_rows=rows)
        input_df = st.session_state.sample_df.copy()

    elif data_mode == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload project CSV", type=["csv"])
        if uploaded is None:
            st.info("No CSV uploaded. Using generated sample data by default.")
            input_df = generate_sample_data(n_rows=120)
        else:
            try:
                input_df = parse_and_validate_csv(uploaded)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

    else:  # Build from Manual Entry
        st.sidebar.caption("Add projects using the manual form below.")
        manual_entry = render_manual_entry_form()
        if manual_entry is not None:
            st.session_state.manual_projects = pd.concat(
                [st.session_state.manual_projects, manual_entry], ignore_index=True
            )
            st.sidebar.success("Manual project added.")

        if st.session_state.manual_projects.empty:
            st.info("No manual projects yet. Using generated sample data as baseline.")
            input_df = generate_sample_data(n_rows=100)
        else:
            input_df = st.session_state.manual_projects.copy()

    # Ensure scoring and recommendations are always current
    scored_df = calculate_dimension_scores(input_df)
    scored_df["Status"] = scored_df.apply(classify_project_status, axis=1)
    scored_df["Recommendations"] = scored_df.apply(generate_recommendations, axis=1)

    render_dashboard(scored_df)

    with st.expander("Required Python packages", expanded=False):
        st.code(
            "\n".join(
                [
                    "streamlit>=1.35.0",
                    "pandas>=2.0.0",
                    "numpy>=1.24.0",
                    "plotly>=5.18.0",
                ]
            )
        )


if __name__ == "__main__":
    main()

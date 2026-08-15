import streamlit as st
import pandas as pd
import snowflake.connector


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Scania AI Failure Prediction",
    page_icon="🚛",
    layout="wide"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .dashboard-subtitle {
        font-size: 1rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #dddddd;
        border-radius: 10px;
        padding: 15px;
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">🚛 Scania AI Failure Prediction Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Industrial AI predictive maintenance monitoring powered by Snowflake and XGBoost'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR - SNOWFLAKE CONNECTION
# ============================================================

st.sidebar.header("Snowflake Connection")

account = st.sidebar.text_input(
    "Account Identifier",
    value="CNFOYMT-WK88091"
)

username = st.sidebar.text_input(
    "Username",
    value="MANOJ"
)

password = st.sidebar.text_input(
    "Password",
    type="password"
)

connect_button = st.sidebar.button(
    "Connect to Snowflake",
    width="stretch"
)


# ============================================================
# SNOWFLAKE CONNECTION FUNCTION
# ============================================================

def connect_to_snowflake(account, username, password):
    """
    Create a connection to Snowflake.
    """

    connection = snowflake.connector.connect(
        account=account,
        user=username,
        password=password,
        warehouse="COMPUTE_WH",
        database="SCANIA_AI_PLATFORM",
        schema="ML_FEATURES"
    )

    return connection


# ============================================================
# QUERY FUNCTION
# ============================================================

def run_query(connection, query):
    """
    Execute a Snowflake query and return a pandas DataFrame.
    """

    cursor = connection.cursor()

    try:
        cursor.execute(query)

        columns = [column[0] for column in cursor.description]

        rows = cursor.fetchall()

        dataframe = pd.DataFrame(rows, columns=columns)

        return dataframe

    finally:
        cursor.close()


# ============================================================
# MAIN DASHBOARD
# ============================================================

if connect_button:

    if not password:
        st.sidebar.error("Please enter your Snowflake password.")

    else:

        try:

            with st.spinner("Connecting to Snowflake..."):

                connection = connect_to_snowflake(
                    account,
                    username,
                    password
                )

            st.sidebar.success("Connected to Snowflake")

            # =================================================
            # LOAD DASHBOARD KPIs
            # =================================================

            dashboard_query = """
                SELECT *
                FROM SCANIA_AI_PLATFORM.ML_FEATURES.APS_FAILURE_DASHBOARD
            """

            dashboard_df = run_query(
                connection,
                dashboard_query
            )

            if dashboard_df.empty:
                st.error("No dashboard data was returned from Snowflake.")
                connection.close()
                st.stop()

            dashboard = dashboard_df.iloc[0]


            # =================================================
            # KPI VALUES
            # =================================================

            total_predictions = int(
                dashboard["TOTAL_PREDICTIONS"]
            )

            predicted_failures = int(
                dashboard["PREDICTED_FAILURES"]
            )

            predicted_non_failures = int(
                dashboard["PREDICTED_NON_FAILURES"]
            )

            avg_failure_probability = float(
                dashboard["AVG_FAILURE_PROBABILITY"]
            )

            critical_count = int(
                dashboard["CRITICAL_COUNT"]
            )

            urgent_count = int(
                dashboard["URGENT_COUNT"]
            )

            monitor_count = int(
                dashboard["MONITOR_COUNT"]
            )

            normal_count = int(
                dashboard["NORMAL_COUNT"]
            )

            action_required_percentage = float(
                dashboard["ACTION_REQUIRED_PERCENTAGE"]
            )


            # =================================================
            # KPI SECTION
            # =================================================

            st.subheader("📊 Prediction Overview")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Predictions",
                    f"{total_predictions:,}"
                )

            with col2:
                st.metric(
                    "Predicted Failures",
                    f"{predicted_failures:,}"
                )

            with col3:
                st.metric(
                    "Predicted Non-Failures",
                    f"{predicted_non_failures:,}"
                )

            with col4:
                st.metric(
                    "Avg Failure Probability",
                    f"{avg_failure_probability:.2%}"
                )


            # =================================================
            # MAINTENANCE KPI SECTION
            # =================================================

            st.subheader("🔧 Maintenance Priority")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric(
                    "🔴 Critical",
                    f"{critical_count:,}"
                )

            with col2:
                st.metric(
                    "🟠 Urgent",
                    f"{urgent_count:,}"
                )

            with col3:
                st.metric(
                    "🟡 Monitor",
                    f"{monitor_count:,}"
                )

            with col4:
                st.metric(
                    "🟢 Normal",
                    f"{normal_count:,}"
                )

            with col5:
                st.metric(
                    "⚠️ Action Required",
                    f"{action_required_percentage:.2f}%"
                )


            # =================================================
            # LOAD RISK DISTRIBUTION
            # =================================================

            risk_query = """
                SELECT
                    RISK_LEVEL,
                    PREDICTION_COUNT,
                    PERCENTAGE
                FROM SCANIA_AI_PLATFORM.ML_FEATURES.APS_FAILURE_RISK_DISTRIBUTION
                ORDER BY
                    CASE RISK_LEVEL
                        WHEN 'HIGH' THEN 1
                        WHEN 'MEDIUM' THEN 2
                        WHEN 'LOW' THEN 3
                    END
            """

            risk_df = run_query(
                connection,
                risk_query
            )


            # =================================================
            # RISK DISTRIBUTION
            # =================================================

            st.subheader("🚨 Risk Distribution")

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:

                if not risk_df.empty:

                    risk_chart = risk_df.set_index(
                        "RISK_LEVEL"
                    )[["PREDICTION_COUNT"]]

                    st.bar_chart(
                        risk_chart
                    )

            with chart_col2:

                if not risk_df.empty:

                    display_risk = risk_df.copy()

                    display_risk["PERCENTAGE"] = (
                        display_risk["PERCENTAGE"]
                        .round(2)
                    )

                    st.dataframe(
                        display_risk,
                        width="stretch",
                        hide_index=True
                    )


            # =================================================
            # LOAD HIGH-RISK PREDICTIONS
            # =================================================

            high_risk_query = """
                SELECT
                    PREDICTION_ID,
                    PREDICTION_LABEL,
                    FAILURE_PROBABILITY,
                    RISK_LEVEL,
                    CREATED_AT
                FROM SCANIA_AI_PLATFORM.ML_FEATURES.APS_FAILURE_HIGH_RISK
                ORDER BY FAILURE_PROBABILITY DESC
                LIMIT 20
            """

            high_risk_df = run_query(
                connection,
                high_risk_query
            )


            # =================================================
            # HIGH-RISK TABLE
            # =================================================

            st.subheader("🔴 Top High-Risk Predictions")

            if not high_risk_df.empty:

                high_risk_display = high_risk_df.copy()

                high_risk_display["FAILURE_PROBABILITY"] = (
                    high_risk_display["FAILURE_PROBABILITY"]
                    .astype(float)
                    .map(lambda x: f"{x:.2%}")
                )

                st.dataframe(
                    high_risk_display,
                    width="stretch",
                    hide_index=True
                )

            else:

                st.info(
                    "No high-risk predictions found."
                )


            # =================================================
            # LOAD MAINTENANCE ACTIONS
            # =================================================

            action_query = """
                SELECT
                    PREDICTION_ID,
                    PREDICTION_LABEL,
                    FAILURE_PROBABILITY,
                    RISK_LEVEL,
                    MAINTENANCE_PRIORITY,
                    RECOMMENDED_ACTION,
                    CREATED_AT
                FROM SCANIA_AI_PLATFORM.ML_FEATURES.APS_FAILURE_MAINTENANCE_ACTIONS
                ORDER BY FAILURE_PROBABILITY DESC
                LIMIT 20
            """

            action_df = run_query(
                connection,
                action_query
            )


            # =================================================
            # MAINTENANCE ACTION TABLE
            # =================================================

            st.subheader("🛠️ Recommended Maintenance Actions")

            if not action_df.empty:

                action_display = action_df.copy()

                action_display["FAILURE_PROBABILITY"] = (
                    action_display["FAILURE_PROBABILITY"]
                    .astype(float)
                    .map(lambda x: f"{x:.2%}")
                )

                st.dataframe(
                    action_display,
                    width="stretch",
                    hide_index=True
                )

            else:

                st.info(
                    "No maintenance actions available."
                )


            # =================================================
            # FOOTER
            # =================================================

            st.divider()

            st.caption(
                "Scania AI Failure Prediction | "
                "Snowflake + XGBoost + SHAP | "
                "Industrial Predictive Maintenance"
            )


            # =================================================
            # CLOSE CONNECTION
            # =================================================

            connection.close()

        except Exception as error:

            st.sidebar.error(
                "Snowflake connection failed."
            )

            st.error(
                f"Error: {error}"
            )
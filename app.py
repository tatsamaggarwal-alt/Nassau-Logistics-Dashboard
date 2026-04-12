import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Nassau Logistics Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}
.main {
    background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e5e7eb;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
}
h1, h2, h3 {
    color: #0f172a;
}
.section-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
    margin-bottom: 18px;
}
.small-note {
    color: #475569;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")

    df = df.dropna(subset=["Order Date", "Ship Date", "Ship Mode", "Region", "State/Province", "Product Name"])

    df["Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df = df[df["Lead Time"] >= 0].copy()

    product_factory_map = {
        "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
        "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
        "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
        "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
        "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
        "Laffy Taffy": "Sugar Shack",
        "SweeTARTS": "Sugar Shack",
        "Nerds": "Sugar Shack",
        "Fun Dip": "Sugar Shack",
        "Fizzy Lifting Drinks": "Sugar Shack",
        "Everlasting Gobstopper": "Secret Factory",
        "Hair Toffee": "The Other Factory",
        "Lickable Wallpaper": "Secret Factory",
        "Wonka Gum": "Secret Factory",
        "Kazookles": "The Other Factory"
    }

    factory_coords = {
        "Lot's O' Nuts": (32.881893, -111.768036),
        "Wicked Choccy's": (32.076176, -81.088371),
        "Sugar Shack": (48.11914, -96.18115),
        "Secret Factory": (41.446333, -90.565487),
        "The Other Factory": (35.1175, -89.971107)
    }

    df["Factory"] = df["Product Name"].map(product_factory_map).fillna("Unknown")
    df["Route"] = df["Factory"] + " → " + df["State/Province"]
    df["Delayed"] = np.where(df["Lead Time"] > 5, 1, 0)

    df["Factory_Lat"] = df["Factory"].map(lambda x: factory_coords[x][0] if x in factory_coords else np.nan)
    df["Factory_Lon"] = df["Factory"].map(lambda x: factory_coords[x][1] if x in factory_coords else np.nan)

    return df


df = load_data()

if df.empty:
    st.error("No valid data available after cleaning.")
    st.stop()

st.title("Nassau Candy Distributor")
st.subheader("Factory-to-Customer Shipping Route Efficiency Analysis")
st.caption("Executive logistics dashboard for route performance, bottleneck detection, ship mode comparison, and lead-time prediction.")

st.sidebar.header("Filters")
region_filter = st.sidebar.selectbox("Region", ["All"] + sorted(df["Region"].dropna().unique().tolist()))
state_filter = st.sidebar.selectbox("State", ["All"] + sorted(df["State/Province"].dropna().unique().tolist()))
ship_filter = st.sidebar.selectbox("Ship Mode", ["All"] + sorted(df["Ship Mode"].dropna().unique().tolist()))
lead_threshold = st.sidebar.slider("Delay Threshold (days)", 1, 15, 5)

filtered_df = df.copy()

if region_filter != "All":
    filtered_df = filtered_df[filtered_df["Region"] == region_filter]

if state_filter != "All":
    filtered_df = filtered_df[filtered_df["State/Province"] == state_filter]

if ship_filter != "All":
    filtered_df = filtered_df[filtered_df["Ship Mode"] == ship_filter]

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

filtered_df["Delayed"] = np.where(filtered_df["Lead Time"] > lead_threshold, 1, 0)

route_perf = filtered_df.groupby("Route").agg(
    avg_lead_time=("Lead Time", "mean"),
    total_shipments=("Order ID", "count"),
    lead_time_std=("Lead Time", "std"),
    delay_frequency=("Delayed", "mean")
).reset_index()

route_perf["lead_time_std"] = route_perf["lead_time_std"].fillna(0)
max_lead = route_perf["avg_lead_time"].max()
min_lead = route_perf["avg_lead_time"].min()

if max_lead == min_lead:
    route_perf["efficiency_score"] = 100
else:
    route_perf["efficiency_score"] = 100 - (
        (route_perf["avg_lead_time"] - min_lead) / (max_lead - min_lead) * 100
    )

route_perf = route_perf.sort_values(["efficiency_score", "total_shipments"], ascending=[False, False])

ship_perf = filtered_df.groupby("Ship Mode").agg(
    avg_lead_time=("Lead Time", "mean"),
    total_orders=("Order ID", "count"),
    avg_sales=("Sales", "mean") if "Sales" in filtered_df.columns else ("Lead Time", "mean")
).reset_index()

state_perf = filtered_df.groupby("State/Province").agg(
    avg_lead_time=("Lead Time", "mean"),
    total_shipments=("Order ID", "count"),
    delay_frequency=("Delayed", "mean")
).reset_index()

best_route = route_perf.iloc[0]["Route"] if not route_perf.empty else "N/A"
worst_route = route_perf.iloc[-1]["Route"] if not route_perf.empty else "N/A"
avg_lead = round(filtered_df["Lead Time"].mean(), 2)
delay_pct = round(filtered_df["Delayed"].mean() * 100, 2)
total_orders = int(filtered_df.shape[0])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Average Lead Time", f"{avg_lead} days")
c2.metric("Delay Frequency", f"{delay_pct}%")
c3.metric("Best Route", best_route)
c4.metric("Orders Analyzed", total_orders)

left, right = st.columns([1.4, 1])

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Route Efficiency Leaderboard")
    leaderboard = route_perf.copy()
    leaderboard["avg_lead_time"] = leaderboard["avg_lead_time"].round(2)
    leaderboard["lead_time_std"] = leaderboard["lead_time_std"].round(2)
    leaderboard["delay_frequency"] = (leaderboard["delay_frequency"] * 100).round(2)
    leaderboard["efficiency_score"] = leaderboard["efficiency_score"].round(2)
    st.dataframe(
        leaderboard.rename(columns={
            "avg_lead_time": "Avg Lead Time",
            "total_shipments": "Shipment Volume",
            "lead_time_std": "Variability",
            "delay_frequency": "Delay %",
            "efficiency_score": "Efficiency Score"
        }),
        use_container_width=True,
        hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Executive Summary")
    st.markdown(
        f"""
        **Current average lead time:** {avg_lead} days  
        **Delay frequency above threshold:** {delay_pct}%  
        **Most efficient route:** {best_route}  
        **Least efficient route:** {worst_route}

        The current filtered network shows a mix of efficient and underperforming routes.
        Priority attention should go to high-volume routes with above-average lead time and delay rate.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Top 10 Most Efficient Routes")
    fig_best = px.bar(
        route_perf.head(10).sort_values("efficiency_score"),
        x="efficiency_score",
        y="Route",
        orientation="h",
        text="efficiency_score"
    )
    fig_best.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_best.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_best, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Bottom 10 Least Efficient Routes")
    fig_worst = px.bar(
        route_perf.tail(10).sort_values("avg_lead_time"),
        x="avg_lead_time",
        y="Route",
        orientation="h",
        text="avg_lead_time"
    )
    fig_worst.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_worst.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_worst, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

col_c, col_d = st.columns(2)

with col_c:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Geographic Bottleneck Analysis")
    bottlenecks = state_perf[
        (state_perf["avg_lead_time"] > state_perf["avg_lead_time"].median()) &
        (state_perf["total_shipments"] > state_perf["total_shipments"].median())
    ].sort_values("avg_lead_time", ascending=False)

    if bottlenecks.empty:
        st.info("No bottleneck states detected for the selected filters.")
    else:
        st.dataframe(
            bottlenecks.rename(columns={
                "State/Province": "State",
                "avg_lead_time": "Avg Lead Time",
                "total_shipments": "Shipment Volume",
                "delay_frequency": "Delay Frequency"
            }).round(2),
            use_container_width=True,
            hide_index=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

with col_d:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Ship Mode Performance")
    fig_ship = px.bar(
        ship_perf,
        x="Ship Mode",
        y="avg_lead_time",
        text="avg_lead_time",
        color="Ship Mode"
    )
    fig_ship.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_ship.update_layout(showlegend=False, height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_ship, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("US State Lead Time Heatmap")
map_df = state_perf.copy()
fig_map = px.choropleth(
    map_df,
    locations="State/Province",
    locationmode="USA-states",
    color="avg_lead_time",
    scope="usa",
    hover_data={"total_shipments": True, "delay_frequency": ':.2%'},
    color_continuous_scale="Reds"
)
fig_map.update_layout(height=550, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_map, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Order-Level Lead Time Distribution")
fig_hist = px.histogram(
    filtered_df,
    x="Lead Time",
    nbins=25
)
fig_hist.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_hist, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("ML-Based Lead Time Prediction")

ml_df = filtered_df[["Ship Mode", "Region", "State/Province", "Lead Time"]].copy()
ml_df = pd.get_dummies(ml_df, columns=["Ship Mode", "Region", "State/Province"], drop_first=True)

X = ml_df.drop("Lead Time", axis=1)
y = ml_df["Lead Time"]

if len(X) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    pred_col1, pred_col2, pred_col3 = st.columns(3)
    pred_region = pred_col1.selectbox("Prediction Region", sorted(df["Region"].dropna().unique()), key="pred_region")
    pred_state = pred_col2.selectbox("Prediction State", sorted(df["State/Province"].dropna().unique()), key="pred_state")
    pred_ship = pred_col3.selectbox("Prediction Ship Mode", sorted(df["Ship Mode"].dropna().unique()), key="pred_ship")

    input_row = pd.DataFrame({
        "Ship Mode": [pred_ship],
        "Region": [pred_region],
        "State/Province": [pred_state]
    })

    input_row = pd.get_dummies(input_row, columns=["Ship Mode", "Region", "State/Province"])
    input_row = input_row.reindex(columns=X.columns, fill_value=0)

    predicted_lead = model.predict(input_row)[0]
    risk_label = "High Delay Risk" if predicted_lead > lead_threshold else "Likely On Time"

    r1, r2 = st.columns(2)
    r1.metric("Predicted Lead Time", f"{predicted_lead:.2f} days")
    r2.metric("Predicted Status", risk_label)
else:
    st.info("Not enough filtered data to train prediction model.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Strategic Recommendations")
st.markdown("""
1. Prioritize process review for high-volume bottleneck states.  
2. Reassess ship mode allocation where faster modes are not materially improving lead time.  
3. Focus route optimization efforts on the bottom 10 ranked routes first.  
4. Use predicted lead time as an early warning layer for route planning and service commitments.
""")
st.markdown('</div>', unsafe_allow_html=True)

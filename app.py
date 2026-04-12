import streamlit as st
import plotly.express as px
from analysis import load_and_process_data, route_analysis, ship_mode_analysis
from model import train_model

st.set_page_config(page_title="AI Logistics Dashboard", layout="wide")

# LOAD DATA
df = load_and_process_data("data.csv")

if df.empty:
    st.error("No valid data")
    st.stop()

route_df = route_analysis(df)
ship_df = ship_mode_analysis(df)

# TRAIN MODEL
model = train_model(df)

# SIDEBAR NAVIGATION
page = st.sidebar.radio("Navigation", ["Overview", "Route Analysis", "Map", "AI Prediction"])

st.title("🚀 AI Logistics Intelligence Dashboard")

# ================= OVERVIEW =================
if page == "Overview":

    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Avg Lead Time", round(df['Lead Time'].mean(), 2))
    col2.metric("Total Orders", df.shape[0])
    col3.metric("Delay %", round(df['Delayed'].mean()*100, 2))

    st.subheader("📈 Lead Time Distribution")
    fig = px.histogram(df, x='Lead Time')
    st.plotly_chart(fig, use_container_width=True)

# ================= ROUTE =================
elif page == "Route Analysis":

    st.subheader("🏆 Route Leaderboard")
    st.dataframe(route_df)

    fig = px.bar(route_df.head(10), x='efficiency', y='Route_State', orientation='h')
    st.plotly_chart(fig, use_container_width=True)

# ================= MAP =================
elif page == "Map":

    st.subheader("🌍 Factory Distribution Map")

    fig = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        hover_name="Factory",
        color="Lead Time",
        zoom=3,
        height=500
    )

    fig.update_layout(mapbox_style="carto-darkmatter")
    st.plotly_chart(fig, use_container_width=True)

# ================= AI =================
elif page == "AI Prediction":

    st.subheader("🤖 Predict Delay")

    ship_mode = st.selectbox("Ship Mode", df['Ship Mode'].unique())
    region = st.selectbox("Region", df['Region'].unique())
    lead_time = st.slider("Lead Time", 1, 30, 5)

    # Encode
    ship_mode_enc = df['Ship Mode'].astype('category').cat.codes.iloc[0]
    region_enc = df['Region'].astype('category').cat.codes.iloc[0]

    prediction = model.predict([[ship_mode_enc, region_enc, lead_time]])

    if prediction[0]:
        st.error("⚠️ High chance of delay")
    else:
        st.success("✅ Likely on time")

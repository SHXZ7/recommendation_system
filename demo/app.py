import streamlit as st
import requests

st.set_page_config(page_title="Movie Recommender Demo", layout="centered")
st.title("🎬 Real-Time Movie Recommendation System")
st.write("Personalized recommendations powered by collaborative filtering and ANN search.")

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

# Initialize data so it always exists
data = None

user_id = st.number_input(
    "Select a user ID",
    min_value=1,
    max_value=6000,
    value=1,
    step=1
)

API_URL = "http://127.0.0.1:8000/recommend"

if st.button("Get Recommendations"):
    with st.spinner("Fetching recommendations..."):
        response = requests.get(f"{API_URL}/{user_id}")

        if response.status_code == 200:
            st.session_state.recommendations = response.json()
        else:
            st.error("Failed to fetch recommendations")

# Only run this AFTER data exists
if st.session_state.recommendations is not None:
    data = st.session_state.recommendations
    st.success(f"Recommendations generated in {data['latency_ms']} ms")

    st.subheader("Recommended Movies (IDs)")
    for i, movie_id in enumerate(data["recommendations"], start=1):
        col1, col2 = st.columns([3, 1])
        col1.write(f"{i}. Movie ID: {movie_id}")

        if col2.button(
            "I watched this",
            key=f"watch_{movie_id}"
        ):
            feedback = {
                "user_id": user_id,
                "movie_id": movie_id
            }

            r = requests.post(
                "http://127.0.0.1:8000/feedback",
                json=feedback
            )

            if r.status_code == 200:
                st.success(f"Feedback sent for movie {movie_id}")
            else:
                st.error("Failed to send feedback")

    with st.expander("Show raw API response"):
        st.json(data)

    

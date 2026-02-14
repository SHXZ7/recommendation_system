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
    max_value=600000,
    value=1,
    step=1
)

API_URL = "https://shxz7-recommendation-engine.hf.space/recommend"

if st.button("Get Recommendations"):
    with st.spinner("Fetching recommendations..."):
        try:
            response = requests.get(f"{API_URL}/{user_id}", timeout=30)

            if response.status_code == 200:
                st.session_state.recommendations = response.json()
            else:
                st.error(f"Failed to fetch recommendations. Status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend API. Please ensure the Hugging Face Space is running.")
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The backend may be starting up or overloaded.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Only run this AFTER data exists
if st.session_state.recommendations is not None:
    data = st.session_state.recommendations
    st.success(f"Recommendations generated in {data['latency_ms']} ms")

    st.subheader("Recommended Movies (IDs)")
    for i, movie in enumerate(data["recommendations"], start=1):
        col1, col2 = st.columns([3, 1])
        col1.write(f"{i}. Movie ID: {movie}")

        if col2.button(
            "I watched this",
            key=f"watch_{movie['movie_id']}"
        ):
            feedback = {
                "user_id": user_id,
                "movie_id": movie["movie_id"],
                "experiment_group": data["experiment_group"]
            }

            try:
                r = requests.post(
                    "https://shxz7-recommendation-engine.hf.space/feedback",
                    json=feedback,
                    timeout=30
                )

                if r.status_code == 200:
                    st.success(f"Feedback sent for movie {movie['movie_id']}")
                else:
                    st.error("Failed to send feedback")
            except Exception as e:
                st.error(f"Error sending feedback: {str(e)}")

    with st.expander("Show raw API response"):
        st.json(data)

    

import streamlit as st
from recommender import recommend_movies, recommend_for_user, item_based_recommend, user_based_recommend

# Page Configuration
st.set_page_config(
    page_title="Personalized Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Title
st.title("🎬 Personalized Movie Recommendation System")

st.markdown("""
Welcome to the **Personalized Movie Recommendation System**.

This project provides movie recommendations using:

- 🎥 Content-Based Filtering
- 👥 Collaborative Filtering (SVD)
- 🎬 Collaborative Filtering (Item-Based)
- 👥 Collaborative Filtering (User-Based)
""")

# Sidebar
option = st.sidebar.selectbox(
    "Choose Recommendation Method",
    [
        "Content-Based Filtering",
        "Collaborative Filtering (SVD)",
        "Collaborative Filtering (Item-Based)",
        "Collaborative Filtering (User-Based)"
    ]
)

# Content-Based Section
if option == "Content-Based Filtering":

    st.header("🎥 Content-Based Recommendation")

    movie = st.text_input("Enter Movie Name")

    if st.button("Recommend"):

        recommendations = recommend_movies(movie)

        st.subheader("Recommended Movies")

        for rec in recommendations:
            st.write("⭐", rec)

# Collaborative Section (SVD)
elif option == "Collaborative Filtering (SVD)":

    st.header("👥 Collaborative Filtering")

    user_id = st.number_input(
        "Enter User ID",
        min_value=1,
        max_value=943,
        value=50
    )

    if st.button("Recommend"):

        recommendations = recommend_for_user(user_id)

        st.subheader("Recommended Movies")

        for rec in recommendations:
            st.write("⭐", rec)

# Collaborative Section (Item-Based)
elif option == "Collaborative Filtering (Item-Based)":

    st.header("🎬 Item-Based Collaborative Filtering")

    movie_id = st.number_input(
        "Enter Movie ID",
        min_value=1,
        max_value=1682,
        value=1
    )

    if st.button("Recommend"):

        recommendations = item_based_recommend(movie_id)

        st.subheader("Recommended Movies")

        for rec in recommendations:
            st.write("⭐", rec)

# Collaborative Section (User-Based)
else:

    st.header("👥 User-Based Collaborative Filtering")

    user_id = st.number_input(
        "Enter User ID",
        min_value=1,
        max_value=943,
        value=50
    )

    if st.button("Recommend"):

        recommendations = user_based_recommend(user_id)

        st.subheader("Recommended Movies")

        for rec in recommendations:
            st.write("⭐", rec)
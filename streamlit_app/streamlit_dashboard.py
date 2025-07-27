import streamlit as st
import pandas as pd
import numpy as np
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from transformers import pipeline

# Load data and vector database
movies = pd.read_csv("movies_with_emotions.csv")
#books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
#books["large_thumbnail"] = np.where(
 #   books["thumbnail"].isna(),
##    "https://via.placeholder.com/150", # Use a placeholder image
#    books["large_thumbnail"],
#)

persist_directory = "chroma_db_books2"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db_movies = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)


def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    movie_type: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 20,
) -> pd.DataFrame:
  """
    Retrieves book recommendations based on semantic similarity, category, and tone.

    Args:
        query: The user's query for book recommendations.
        category: The desired book category (e.g., "Fiction", "Nonfiction", "All").
        movie_type: The desired type ('Movie', 'TV Show', or 'All').
        tone: The desired emotional tone (e.g., "Happy", "Sad", "Angry", "Suspenseful", "Surprising", "All").
        initial_top_k: The initial number of similar books to retrieve from the vector database.
        final_top_k: The final number of recommendations to return after filtering and sorting.

    Returns:
        A pandas DataFrame containing the recommended books.
    """
  recs = db_movies.similarity_search(query, k=initial_top_k)
  movies_list = [rec.page_content.split()[0] for rec in recs]
  movies_recs = movies[movies["show_id"].isin(movies_list)].copy() # Use .copy() to avoid SettingWithCopyWarning

  if category != "All":
    movies_recs = movies_recs[movies_recs["categorie"] == category]

  if movie_type != "All":
      movies_recs = movies_recs[movies_recs["type"] == movie_type]

  if tone != "All":
      movies_recs = movies_recs[movies_recs["emotion"] == tone.lower()]

  return movies_recs.head(final_top_k)


def recommended_book(query, category, movie_type, tone):
  """
    Generates formatted recommendations for display in the Streamlit UI.

    Args:
        query: The user's query.
        category: The selected category.
        movie_type: The selected movie type.
        tone: The selected tone.

    Returns:
        A list of tuples, where each tuple contains a formatted caption and the thumbnail URL.
    """
  recommendations = retrieve_semantic_recommendations(query, category, movie_type, tone)
  results = []
  for _, row in recommendations.iterrows():
    description = row["description"]
    truncated_desc_split = description.split()
    truncated_description = " ".join(truncated_desc_split[:30]) + "..."

    authors_split = row["rating"].split(";")
    if len(authors_split) == 2:
      authors_str = f"{authors_split[0]} and {authors_split[1]}"
    elif len(authors_split) > 2:
      authors_str = ", ".join(authors_split[:-1]) + f", and {authors_split[-1]}"
    else:
      authors_str = str(row["rating"]) # Ensure authors is a string

    caption = f"{row['title']} by {authors_str}: {truncated_description}"
    results.append((caption))
  return results

# Streamlit UI
st.title("Semantic Book Recommendation System")

# Input widgets
user_query = st.text_input("Please enter a description of a book", placeholder="e.g., A story about forgiveness")

categories = ["All"] + sorted(movies["categorie"].unique().tolist())
category_dropdown = st.selectbox("Select a category", categories)

movie_types = ["All"] + sorted(movies["type"].unique().tolist())
type_dropdown = st.selectbox("Select a Type", movie_types)

tones = ["All"] + movies["emotion"].unique().tolist()
tone_dropdown = st.selectbox("Select a Tone", tones)

submit_button = st.button("Find Recommendations")

# Display recommendations
st.markdown("## Recommendations")

if submit_button and user_query:
    recommendations = recommended_book(user_query, category_dropdown, type_dropdown, tone_dropdown)
    if recommendations:
        # Display recommendations in columns
        cols = st.columns(5) # Adjust the number of columns as needed
        for i, (caption) in enumerate(recommendations):
            with cols[i % 5]: # Distribute across columns
                #st.image(caption=caption, use_container_width=True)
                st.write(caption)
    else:
        st.write("No recommendations found for your criteria.")
elif submit_button and not user_query:
    st.write("Please enter a description to get recommendations.")

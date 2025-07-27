from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

app = FastAPI()

# Configuration des chemins
DATA_DIR = "data"
DB_DIR = "db"
MOVIES_FILE = os.path.join(DATA_DIR, "movies_with_emotions.csv")
CHROMA_DB_DIR = os.path.join(DB_DIR, "chroma_db_books2")

# Chargement des données
movies = pd.read_csv(MOVIES_FILE)

# Initialisation des embeddings et de la base de données vectorielle
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db_movies = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model)

# Définition des modèles Pydantic
class RecommendationRequest(BaseModel):
    query: str
    category: Optional[str] = "All"
    movie_type: Optional[str] = "All"
    tone: Optional[str] = "All"

class MovieRecommendation(BaseModel):
    show_id: str
    title: str
    rating: Optional[str]
    duration: Optional[str]
    listed_in: str
    description: str
    year_added: Optional[str]
    emotion: Optional[str]

@app.post("/recommendations", response_model=List[MovieRecommendation])
async def get_recommendations(request: RecommendationRequest):
    # Vérification des entrées
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Recherche sémantique
    initial_top_k = 50
    recs = db_movies.similarity_search(request.query, k=initial_top_k)
    movies_list = [rec.page_content.split()[0] for rec in recs]
    movies_recs = movies[movies["show_id"].isin(movies_list)].copy()

    # Filtrage des résultats
    if request.category != "All":
        movies_recs = movies_recs[movies_recs["listed_in"].str.contains(request.category, case=False, na=False)]

    if request.movie_type != "All":
        movies_recs = movies_recs[movies_recs["type"] == request.movie_type]

    if request.tone != "All":
        movies_recs = movies_recs[movies_recs["emotion"] == request.tone.lower()]

    # Limitation du nombre de recommandations
    final_top_k = 20
    movies_recs = movies_recs.head(final_top_k)

    # Préparation des recommandations
    recommended_movies = []
    for _, row in movies_recs.iterrows():
        recommended_movies.append(
            MovieRecommendation(
                show_id=row["show_id"],
                title=row["title"],
                rating=row["rating"],
                duration=row["duration"],
                listed_in=row["listed_in"],
                description=row["description"],
                year_added=row["year_added"],
                emotion=row["emotion"]
            )
        )

    if not recommended_movies:
        raise HTTPException(status_code=404, detail="No recommendations found for the given criteria.")

    return recommended_movies
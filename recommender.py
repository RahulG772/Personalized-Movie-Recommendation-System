import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# ----------------------------
# Load Dataset
# ----------------------------

ratings = pd.read_csv(
    "dataset/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    "dataset/ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None
)

movies.columns = [
    "movie_id", "movie_title", "release_date", "video_release_date",
    "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
    "Children", "Comedy", "Crime", "Documentary", "Drama",
    "Fantasy", "Film_Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci_Fi", "Thriller",
    "War", "Western"
]

# ----------------------------
# Content-Based Filtering
# ----------------------------

genre_columns = movies.columns[5:]

genre_matrix = movies[genre_columns]

similarity_matrix = cosine_similarity(genre_matrix)

movie_indices = pd.Series(
    movies.index,
    index=movies["movie_title"]
).drop_duplicates()


def recommend_movies(movie_title, n=10):

    if movie_title not in movie_indices:
        return ["Movie not found"]

    movie_index = movie_indices[movie_title]

    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:n+1]

    recommendations = []

    for movie in similarity_scores:
        recommendations.append(
            movies.iloc[movie[0]]["movie_title"]
        )

    return recommendations


# ----------------------------
# Collaborative Filtering
# ----------------------------

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(
    ratings[["user_id", "movie_id", "rating"]],
    reader
)

trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

model = SVD()

model.fit(trainset)


def recommend_for_user(user_id, n=10):

    predictions = []

    rated_movies = ratings[
        ratings["user_id"] == user_id
    ]["movie_id"].tolist()

    for movie_id in movies["movie_id"]:

        if movie_id in rated_movies:
            continue

        pred = model.predict(user_id, movie_id)

        predictions.append(
            (movie_id, pred.est)
        )

    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    result = []

    for movie in predictions[:n]:

        movie_name = movies.loc[
            movies["movie_id"] == movie[0],
            "movie_title"
        ].values[0]

        result.append(
            f"{movie_name} (⭐ {movie[1]:.2f})"
        )

    return result

# ----------------------------
# Collaborative Filtering (User-Based & Item-Based)
# ----------------------------

user_item_matrix = ratings.pivot_table(index="user_id", columns="movie_id", values="rating")
item_matrix_filled = user_item_matrix.fillna(0)

item_similarity_df = pd.DataFrame(
    cosine_similarity(item_matrix_filled.T),
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

def item_based_recommend(movie_id, n=10):
    if movie_id not in item_similarity_df.columns:
        return ["Movie not found"]

    scores = item_similarity_df[movie_id].sort_values(ascending=False).drop(movie_id)
    top_n = scores.head(n)

    result = []
    for mid, score in top_n.items():
        name = movies.loc[movies["movie_id"] == mid, "movie_title"].values[0]
        result.append(f"{name} (Similarity: {score:.2f})")
    return result


user_similarity_df = pd.DataFrame(
    cosine_similarity(item_matrix_filled),
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

def user_based_recommend(user_id, n=10):
    if user_id not in user_similarity_df.columns:
        return ["User not found"]

    similar_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id)
    top_similar_users = similar_users.head(10).index

    avg_ratings = user_item_matrix.loc[top_similar_users].mean(axis=0).sort_values(ascending=False)

    already_rated = user_item_matrix.loc[user_id].dropna().index
    avg_ratings = avg_ratings.drop(labels=already_rated, errors="ignore")

    top_n = avg_ratings.head(n)

    result = []
    for mid, score in top_n.items():
        name = movies.loc[movies["movie_id"] == mid, "movie_title"].values[0]
        result.append(f"{name} (Avg rating: {score:.2f})")
    return result
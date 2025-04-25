import pandas as pd
import requests

TMDB_API_KEY = "ab3e3f106356dbcb70df22107bb51b09"

def get_poster_url(tmdb_id):
    if pd.isna(tmdb_id):
        return None
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w185{poster_path}"
    return None

def get_unique_genres(movies):
    print("Lấy danh sách thể loại...")
    genres_set = set()
    for genres in movies['genres'].str.split('|'):
        genres_set.update(genres)
    print("Hoàn tất lấy danh sách thể loại.")
    return sorted(list(genres_set))

def get_top_movies_by_genre(genre, movies):
    print(f"Lấy top 10 phim cho thể loại: {genre}...")
    filtered_movies = movies[movies['genres'].str.contains(genre, case=False, na=False, regex=False)]
    top_movies = filtered_movies.sort_values(by=['num_votes', 'avg_rating'], ascending=[False, False]).head(10)
    print("Hoàn tất lấy top 10 phim.")
    return top_movies[['title', "tmdbId"]]

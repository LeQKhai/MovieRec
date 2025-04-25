import pandas as pd
import re


# Hàm sửa tiêu đề phim
def fix_the_title(title):
    pattern_with_year = r'^(.+), The \((\d{4})\)$'
    pattern_without_year = r'^(.+), The$'
    match_with_year = re.match(pattern_with_year, title)
    if match_with_year:
        movie_name = match_with_year.group(1).strip()
        year = match_with_year.group(2)
        return f"The {movie_name} ({year})"
    match_without_year = re.match(pattern_without_year, title)
    if match_without_year:
        movie_name = match_without_year.group(1).strip()
        return f"The {movie_name}"
    return title
def preprocess_movies(movies):
    # Sửa tiêu đề phim
    movies['title'] = movies['title'].apply(fix_the_title)

    # Các bước tiền xử lý khác (từ mã gốc của bạn)
    movies['clean_title'] = movies['title'].apply(lambda x: re.sub("[^a-zA-Z0-9]", " ", x))
    movies['genres'] = movies['genres'].fillna("")

    return movies


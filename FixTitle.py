import pandas as pd
import re


# Hàm sửa tiêu đề phim
def fix_the_title(title):
    """
    Chuyển đổi tiêu đề phim từ dạng 'Tên phim, The (năm)' thành 'The Tên phim (năm)'.
    Nếu không khớp mẫu, giữ nguyên tiêu đề.

    Args:
        title (str): Tiêu đề phim, ví dụ: 'Avengers, The (2012)'

    Returns:
        str: Tiêu đề đã sửa, ví dụ: 'The Avengers (2012)'
    """
    # Biểu thức chính quy để khớp với 'Tên phim, The (năm)' hoặc 'Tên phim, The'
    pattern_with_year = r'^(.+), The \((\d{4})\)$'
    pattern_without_year = r'^(.+), The$'

    # Kiểm tra tiêu đề có năm
    match_with_year = re.match(pattern_with_year, title)
    if match_with_year:
        movie_name = match_with_year.group(1).strip()
        year = match_with_year.group(2)
        return f"The {movie_name} ({year})"

    # Kiểm tra tiêu đề không có năm
    match_without_year = re.match(pattern_without_year, title)
    if match_without_year:
        movie_name = match_without_year.group(1).strip()
        return f"The {movie_name}"

    # Giữ nguyên nếu không khớp
    return title


# Hàm tích hợp vào bước tiền xử lý dữ liệu
def preprocess_movies(movies):
    """
    Tiền xử lý DataFrame movies, bao gồm sửa tiêu đề phim.

    Args:
        movies (pd.DataFrame): DataFrame chứa cột 'title'

    Returns:
        pd.DataFrame: DataFrame đã được xử lý
    """
    # Sửa tiêu đề phim
    movies['title'] = movies['title'].apply(fix_the_title)

    # Các bước tiền xử lý khác (từ mã gốc của bạn)
    movies['clean_title'] = movies['title'].apply(lambda x: re.sub("[^a-zA-Z0-9]", " ", x))
    movies['genres'] = movies['genres'].fillna("")

    return movies


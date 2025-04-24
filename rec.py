import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import requests
import gdown
import zipfile  # Thay rarfile bằng zipfile

import FixEncoding
import FixTitle

DATA_DIR = "data2"
TMDB_API_KEY = "ab3e3f106356dbcb70df22107bb51b09"

# Hàm tải và giải nén data2 từ Google Drive
def download_and_extract_data():
    print("Bắt đầu kiểm tra thư mục data2...")
    if not os.path.exists(DATA_DIR):
        print(f"Thư mục {DATA_DIR} không tồn tại. Tạo thư mục...")
        os.makedirs(DATA_DIR)
        print("Tải file data2.zip từ Google Drive...")
        file_id = "1F4xsw3ybzPW-rBi4WchccNq5MQmWgMsD"  # Thay bằng FILE_ID mới của data2.zip
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "data2.zip"
        gdown.download(url, output, quiet=False)
        print("Giải nén file data2.zip...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Giải nén hoàn tất. Xóa file data2.zip...")
        os.remove(output)
    else:
        print(f"Thư mục {DATA_DIR} đã tồn tại. Bỏ qua tải và giải nén.")

@st.cache_data
def load_data():
    download_and_extract_data()
    # ... (các phần tải dữ liệu như trong mã gốc)
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    links = pd.read_csv(os.path.join(DATA_DIR, "links.csv"))
    tags = pd.read_csv(os.path.join(DATA_DIR, "tags.csv"))

    # Tiền xử lý movies, bao gồm sửa tiêu đề
    movies = FixTitle.preprocess_movies(movies)
    tags['tag'] = tags['tag'].apply(FixEncoding.fix_encoding)

    # ... (các phần gộp dữ liệu và xử lý tiếp theo như trong mã gốc)
    movies = movies.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')
    ratings_summary = ratings.groupby('movieId').agg({'rating': 'mean', 'userId': 'count'}).rename(
        columns={'rating': 'avg_rating', 'userId': 'num_votes'})
    movies = movies.merge(ratings_summary, on='movieId', how='left')
    tags['tag'] = tags['tag'].astype(str).replace('nan', '')
    tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: " ".join(x)).reset_index()
    movies = movies.merge(tags_grouped, on='movieId', how='left')
    movies['tag'] = movies['tag'].fillna("")

    return ratings, movies


# Hàm tạo và cache TF-IDF matrix
@st.cache_data
def create_tfidf_matrix(movies):
    print("Bắt đầu tạo TF-IDF matrix...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    # Sử dụng tags để tạo ma trận TF-IDF
    tfidf_matrix = vectorizer.fit_transform(movies['tag'])
    print("Hoàn tất tạo TF-IDF matrix.")
    return vectorizer, tfidf_matrix

# Hàm tìm phim tương tự bằng Content-Based Filtering
def find_similar_movies_content_based(movie_id, movies, tfidf_matrix):
    print("Tìm phim tương tự dựa trên tags...")
    # Tìm chỉ số của phim trong DataFrame
    movie_idx = movies[movies['movieId'] == movie_id].index[0]
    # Tính độ tương đồng cosine giữa phim này và tất cả phim khác
    cosine_sim = cosine_similarity(tfidf_matrix[movie_idx], tfidf_matrix).flatten()
    # Sắp xếp theo độ tương đồng, lấy top 5 (bỏ phim gốc)
    similar_indices = cosine_sim.argsort()[-6:-1][::-1]
    print("Hoàn tất tìm phim tương tự dựa trên tags.")
    return movies.iloc[similar_indices][['title', 'tmdbId']]

# Hàm làm sạch tiêu đề
def clean_title(title):
    return re.sub("[^a-zA-Z0-9]", " ", title)

# Hàm lấy URL poster từ TMDb
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

# Hàm tìm phim tương tự
def find_similar_movies(movie_id, ratings, movies):
    print(f"Tìm phim tương tự cho movie_id: {movie_id}...")
    similar_user = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    print("Lọc người dùng tương tự...")
    similar_user_recs = ratings[(ratings["userId"].isin(similar_user)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_user)
    similar_user_recs = similar_user_recs[similar_user_recs > .1]
    print("Lọc tất cả người dùng...")
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    print("Tính điểm gợi ý...")
    rec_percentages = pd.concat([similar_user_recs, all_users_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values(by="score", ascending=False)
    rec_percentages = rec_percentages[rec_percentages.index != movie_id]
    print("Hoàn tất tìm phim tương tự.")
    return rec_percentages.head(5).merge(movies, left_index=True, right_on="movieId")[["title", "tmdbId"]]


# Hàm kết hợp gợi ý từ Collaborative và Content-Based Filtering
def find_hybrid_recommendations(movie_id, ratings, movies, tfidf_matrix):
    print(f"Tìm phim tương tự cho movie_id: {movie_id} bằng Hybrid Filtering...")
    collab_recs = find_similar_movies(movie_id, ratings, movies)
    collab_recs = collab_recs.reset_index(drop=True)
    content_recs = find_similar_movies_content_based(movie_id, movies, tfidf_matrix)
    content_recs = content_recs.reset_index(drop=True)
    hybrid_recs = []
    for i in range(5):
        if i % 2 == 0 and i // 2 < len(collab_recs):
            hybrid_recs.append(collab_recs.iloc[i // 2])
        elif i // 2 < len(content_recs):
            hybrid_recs.append(content_recs.iloc[i // 2])
    hybrid_recs_df = pd.DataFrame(hybrid_recs, columns=['title', 'tmdbId'])
    print("Hoàn tất tìm phim tương tự bằng Hybrid Filtering.")
    return hybrid_recs_df
# Hàm lấy danh sách thể loại duy nhất
def get_unique_genres(movies):
    print("Lấy danh sách thể loại...")
    genres_set = set()
    for genres in movies['genres'].str.split('|'):
        genres_set.update(genres)
    print("Hoàn tất lấy danh sách thể loại.")
    return sorted(list(genres_set))

# Hàm lấy top 10 phim theo thể loại
def get_top_movies_by_genre(genre, movies):
    print(f"Lấy top 10 phim cho thể loại: {genre}...")
    filtered_movies = movies[movies['genres'].str.contains(genre, case=False, na=False, regex=False)]
    top_movies = filtered_movies.sort_values(by=['num_votes', 'avg_rating'], ascending=[False, False]).head(10)
    print("Hoàn tất lấy top 10 phim.")
    return top_movies[['title', "tmdbId"]]

# Giao diện Streamlit
def main():
    print("Khởi động ứng dụng Streamlit...")
    st.title("Movie Recommender System")

    # Load dữ liệu với caching
    with st.spinner("Đang tải dữ liệu..."):
        print("Gọi hàm load_data...")
        ratings, movies = load_data()
        print("Gọi hàm create_tfidf_matrix...")
        vectorizer, tfidf_matrix = create_tfidf_matrix(movies)

    # Khởi tạo session state để theo dõi chế độ
    print("Khởi tạo session state...")
    if 'mode' not in st.session_state:
        st.session_state.mode = "recommend"
    if 'selected_genre' not in st.session_state:
        st.session_state.selected_genre = None

    # Sidebar
    with st.sidebar:
        st.header("Tùy chọn")
        if st.button("Chọn phim theo thể loại", key="genre_button"):
            print("Chuyển sang chế độ top_by_genre...")
            st.session_state.mode = "top_by_genre"
            st.session_state.selected_genre = None

        if st.button("Gợi ý phim", key="back_to_recommend"):
            print("Chuyển sang chế độ recommend...")
            st.session_state.mode = "recommend"
            st.session_state.selected_genre = None

        if st.session_state.mode == "top_by_genre":
            print("Hiển thị danh sách thể loại...")
            genres_list = get_unique_genres(movies)
            selected_genre = st.selectbox("Chọn thể loại:", genres_list, key="genre_select")
            if selected_genre:
                print(f"Đã chọn thể loại: {selected_genre}")
                st.session_state.selected_genre = selected_genre

    # Khu vực chính
    if st.session_state.mode == "recommend":
        st.write("Bạn muốn tìm phim giống phim của bạn?")
        movie_options = [""] + sorted(movies['title'].tolist())
        selected_movie = st.selectbox("Chọn phim để xem gợi ý:", options=movie_options, index=0, key="movie_select")

        # Thêm ba nút: Collaborative, Content-Based, và Hybrid Filtering
        col1, col2, col3 = st.columns(3)
        with col1:
            collaborative_button = st.button("Gợi ý bằng Collaborative Filtering", key="collaborative_button")
        with col2:
            content_based_button = st.button("Gợi ý bằng Content-Based Filtering", key="content_based_button")
        with col3:
            hybrid_button = st.button("Gợi ý bằng Hybrid Filtering", key="hybrid_button")

        # Xử lý khi người dùng nhấn nút
        if collaborative_button or content_based_button or hybrid_button:
            if selected_movie and selected_movie in movies['title'].values:
                with st.spinner("Đang tìm phim tương tự..."):
                    print(f"Đã chọn phim: {selected_movie}")
                    movie_id = movies[movies['title'] == selected_movie]['movieId'].iloc[0]

                    # Chọn phương pháp gợi ý dựa trên nút được nhấn
                    if collaborative_button:
                        method = "Collaborative Filtering"
                        recommendations = find_similar_movies(movie_id, ratings, movies)
                    elif content_based_button:
                        method = "Content-Based Filtering"
                        recommendations = find_similar_movies_content_based(movie_id, movies, tfidf_matrix)
                    else:  # hybrid_button
                        method = "Hybrid Filtering"
                        recommendations = find_hybrid_recommendations(movie_id, ratings, movies, tfidf_matrix)

                    # Hiển thị kết quả
                    st.subheader(f"Phim tương tự với '{selected_movie}' (Phương pháp: {method}):")
                    cols = st.columns(5)
                    for i, (col, row) in enumerate(zip(cols, recommendations.itertuples())):
                        with col:
                            poster_url = get_poster_url(row.tmdbId)
                            if poster_url:
                                st.image(poster_url, width=120)
                            else:
                                st.write("(Không có poster)")
                            st.write(f"{i + 1}. {row.title}")
            else:
                st.warning("Vui lòng chọn một phim từ danh sách trước khi nhấn nút!")

    elif st.session_state.mode == "top_by_genre" and st.session_state.selected_genre:
        with st.spinner("Đang tải top phim..."):
            print(f"Tải top phim cho thể loại: {st.session_state.selected_genre}")
            top_movies = get_top_movies_by_genre(st.session_state.selected_genre, movies)
            st.subheader(f"Top 10 phim thuộc thể loại '{st.session_state.selected_genre}':")
            for row_idx in range(2):
                cols = st.columns(5)
                start_idx = row_idx * 5
                end_idx = min((row_idx + 1) * 5, len(top_movies))
                for i, (col, row) in enumerate(zip(cols, top_movies.iloc[start_idx:end_idx].itertuples())):
                    with col:
                        poster_url = get_poster_url(row.tmdbId)
                        if poster_url:
                            st.image(poster_url, width=120)
                        else:
                            st.write("(Không có poster)")
                        st.write(f"{start_idx + i + 1}. {row.title}")


if __name__ == "__main__":
    main()
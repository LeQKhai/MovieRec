import streamlit as st
from utils.data_utils import load_data, create_tfidf_matrix
from utils.recommendation import (
    find_similar_movies,
    find_similar_movies_content_based,
    find_hybrid_recommendations,
    find_movies_by_tag_exact
)
from utils.tmdb_utils import get_poster_url, get_unique_genres, get_top_movies_by_genre
from utils.text_processing import get_meaningful_tags

def main():
    print("Khởi động ứng dụng Streamlit...")
    st.title("Movie Recommender System")

    with st.spinner("Đang tải dữ liệu..."):
        print("Gọi hàm load_data...")
        ratings, movies = load_data()
        print("Gọi hàm create_tfidf_matrix...")
        vectorizer, tfidf_matrix = create_tfidf_matrix(movies)

    if 'mode' not in st.session_state:
        st.session_state.mode = "recommend"
    if 'selected_genre' not in st.session_state:
        st.session_state.selected_genre = None

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
        if st.button("Tìm phim theo tag", key="tag_button"):
            print("Chuyển sang chế độ tag_based...")
            st.session_state.mode = "tag_based"
            st.session_state.selected_genre = None

        if st.session_state.mode == "top_by_genre":
            print("Hiển thị danh sách thể loại...")
            genres_list = get_unique_genres(movies)
            selected_genre = st.selectbox("Chọn thể loại:", genres_list, key="genre_select")
            if selected_genre:
                print(f"Đã chọn thể loại: {selected_genre}")
                st.session_state.selected_genre = selected_genre

    if st.session_state.mode == "recommend":
        st.write("Bạn muốn tìm phim giống phim của bạn?")
        movie_options = [""] + sorted(movies['title'].tolist())
        selected_movie = st.selectbox("Chọn phim để xem gợi ý:", options=movie_options, index=0, key="movie_select")

        col1, col2, col3 = st.columns(3)
        with col1:
            collaborative_button = st.button("Gợi ý bằng Collaborative Filtering", key="collaborative_button")
        with col2:
            content_based_button = st.button("Gợi ý bằng Content-Based Filtering", key="content_based_button")
        with col3:
            hybrid_button = st.button("Gợi ý bằng Hybrid Filtering", key="hybrid_button")

        if collaborative_button or content_based_button or hybrid_button:
            if selected_movie and selected_movie in movies['title'].values:
                with st.spinner("Đang tìm phim tương tự..."):
                    print(f"Đã chọn phim: {selected_movie}")
                    movie_id = movies[movies['title'] == selected_movie]['movieId'].iloc[0]

                    if collaborative_button:
                        method = "Collaborative Filtering"
                        recommendations, duration = find_similar_movies(movie_id, ratings, movies)
                    elif content_based_button:
                        method = "Content-Based Filtering"
                        recommendations, duration = find_similar_movies_content_based(movie_id, movies, tfidf_matrix)
                    else:  # hybrid_button
                        method = "Hybrid Filtering"
                        recommendations, duration = find_hybrid_recommendations(movie_id, ratings, movies, tfidf_matrix)

                    st.subheader(f"Phim tương tự với '{selected_movie}' (Phương pháp: {method}):")
                    st.write(f"Thời gian xử lý: {duration:.2f} giây")
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

    elif st.session_state.mode == "tag_based":
        st.subheader("Tìm phim theo chủ đề chính xác")

        meaningful_tags = get_meaningful_tags(movies)
        st.write("Một số tag có ý nghĩa phổ biến:")

        tag_query = st.text_input(
            "Nhập chủ đề bạn quan tâm (ví dụ: vietnam war, psychological thriller):",
            key="tag_input"
        )

        if st.button("Tìm Phim", key="exact_tag_search"):
            if tag_query:
                with st.spinner(f"Đang tìm phim với tag '{tag_query}'..."):
                    recommendations, duration = find_movies_by_tag_exact(tag_query, movies, ratings)

                    if not recommendations.empty:
                        st.subheader(f"Phim có tag '{tag_query}':")
                        st.write(f"Tìm thấy {len(recommendations)} kết quả trong {duration:.2f} giây")

                        for idx, row in recommendations.iterrows():
                            with st.expander(f"{row['title']} ({row['genres'].split('|')[0]})"):
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    poster_url = get_poster_url(row.tmdbId)
                                    if poster_url:
                                        st.image(poster_url, width=150)
                                    else:
                                        st.write("Không có poster")
                                with col2:
                                    st.write(f"**Thể loại:** {row['genres']}")
                                    st.write(f"**Tags liên quan:**")
                                    tags = [t for t in row['tag'].split() if t.lower() != tag_query.lower()]
                                    st.write(", ".join(tags[:10]))
                    else:
                        st.warning(f"Không tìm thấy phim nào với tag '{tag_query}'")
                        st.info("Thử với các tag được gợi ý ở trên hoặc kiểm tra chính tả")
            else:
                st.warning("Vui lòng nhập tag để tìm kiếm")

if __name__ == "__main__":
    main()
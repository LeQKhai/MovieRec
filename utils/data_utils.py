import os
import pandas as pd
import gdown
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from utils.FixEncoding import fix_encoding
from utils.FixTitle import preprocess_movies

DATA_DIR = "Data/data2"

def download_and_extract_data():
    print("Bắt đầu kiểm tra thư mục data2...")
    if not os.path.exists(DATA_DIR):
        print(f"Thư mục {DATA_DIR} không tồn tại. Tạo thư mục...")
        os.makedirs(DATA_DIR)
        print("Tải file data2.zip từ Google Drive...")
        file_id = "1F4xsw3ybzPW-rBi4WchccNq5MQmWgMsD"
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
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    links = pd.read_csv(os.path.join(DATA_DIR, "links.csv"))
    tags = pd.read_csv(os.path.join(DATA_DIR, "tags.csv"))

    movies = preprocess_movies(movies)
    tags['tag'] = tags['tag'].apply(fix_encoding)

    movies = movies.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')
    ratings_summary = ratings.groupby('movieId').agg({'rating': 'mean', 'userId': 'count'}).rename(
        columns={'rating': 'avg_rating', 'userId': 'num_votes'})
    movies = movies.merge(ratings_summary, on='movieId', how='left')
    tags['tag'] = tags['tag'].astype(str).replace('nan', '')
    tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: " ".join(x)).reset_index()
    movies = movies.merge(tags_grouped, on='movieId', how='left')
    movies['tag'] = movies['tag'].fillna("")

    return ratings, movies

@st.cache_data
def create_tfidf_matrix(movies):
    print("Bắt đầu tạo TF-IDF matrix...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(movies['tag'])
    print("Hoàn tất tạo TF-IDF matrix.")
    return vectorizer, tfidf_matrix
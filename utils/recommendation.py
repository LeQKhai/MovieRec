import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time
import re

def find_similar_movies_content_based(movie_id, movies, tfidf_matrix):
    start_time = time.time()
    print(
        f"[Content-Based Filtering] Bắt đầu xử lý lúc {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    movie_idx = movies[movies['movieId'] == movie_id].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[movie_idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-6:-1][::-1]

    end_time = time.time()
    duration = end_time - start_time
    print(
        f"[Content-Based Filtering] Kết thúc lúc {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}. Thời gian xử lý: {duration:.2f} giây")

    return movies.iloc[similar_indices][['title', 'tmdbId']], duration

def find_similar_movies(movie_id, ratings, movies):
    start_time = time.time()
    print(
        f"[Collaborative Filtering] Bắt đầu xử lý lúc {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    similar_user = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_user)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_user)
    similar_user_recs = similar_user_recs[similar_user_recs > .1]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_users_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values(by="score", ascending=False)
    rec_percentages = rec_percentages[rec_percentages.index != movie_id]

    end_time = time.time()
    duration = end_time - start_time
    print(
        f"[Collaborative Filtering] Kết thúc lúc {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}. Thời gian xử lý: {duration:.2f} giây")

    return rec_percentages.head(5).merge(movies, left_index=True, right_on="movieId")[["title", "tmdbId"]], duration

def find_hybrid_recommendations(movie_id, ratings, movies, tfidf_matrix):
    start_time = time.time()
    print(f"[Hybrid Filtering] Bắt đầu xử lý lúc {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    collab_recs, collab_duration = find_similar_movies(movie_id, ratings, movies)
    collab_recs = collab_recs.reset_index(drop=True)
    content_recs, content_duration = find_similar_movies_content_based(movie_id, movies, tfidf_matrix)
    content_recs = content_recs.reset_index(drop=True)
    hybrid_recs = []
    for i in range(5):
        if i % 2 == 0 and i // 2 < len(collab_recs):
            hybrid_recs.append(collab_recs.iloc[i // 2])
        elif i // 2 < len(content_recs):
            hybrid_recs.append(content_recs.iloc[i // 2])
    hybrid_recs_df = pd.DataFrame(hybrid_recs, columns=['title', 'tmdbId'])

    end_time = time.time()
    duration = end_time - start_time
    print(
        f"[Hybrid Filtering] Kết thúc lúc {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}. Thời gian xử lý: {duration:.2f} giây")

    return hybrid_recs_df, duration

def find_movies_by_tag(tag_query, movies, ratings):
    start_time = time.time()
    print(f"[Tag-Based Filtering] Bắt đầu xử lý lúc {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    tag_query = tag_query.lower().strip()

    tag_synonyms = {
        'war': ['warfare', 'military', 'battle'],
        'comedy': ['funny', 'humor'],
        'sci-fi': ['scifi', 'science fiction', 'futuristic'],
        'romance': ['love', 'relationship'],
        'action': ['fight', 'adventure'],
        'drama': ['emotional', 'serious']
    }

    search_tags = [tag_query]
    for main_tag, synonyms in tag_synonyms.items():
        if tag_query == main_tag:
            search_tags.extend(synonyms)
            break
        elif tag_query in synonyms:
            search_tags.append(main_tag)
            search_tags.extend([s for s in synonyms if s != tag_query])
            break

    pattern = '|'.join([re.escape(tag) for tag in search_tags])

    tag_movies = movies[movies['tag'].str.contains(pattern, case=False, na=False, regex=True)]

    tag_movies = tag_movies.sort_values(by=['avg_rating', 'num_votes'], ascending=[False, False])

    end_time = time.time()
    duration = end_time - start_time
    print(
        f"[Tag-Based Filtering] Kết thúc lúc {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}. thời gian xử lý: {duration:.2f} giây")

    return tag_movies[['title', 'tmdbId', 'genres']].head(10), duration

def find_movies_by_tag_exact(tag_query, movies, ratings):
    start_time = time.time()

    tag_query = tag_query.lower().strip()

    pattern = r'\b' + re.escape(tag_query) + r'\b'

    exact_matches = movies[
        movies['tag'].str.contains(pattern, case=False, na=False, regex=True)
    ]

    if len(exact_matches) == 0:
        pattern = re.escape(tag_query)
        partial_matches = movies[
            movies['tag'].str.contains(pattern, case=False, na=False, regex=True)
        ]
        matched_movies = partial_matches
    else:
        matched_movies = exact_matches

    matched_movies = matched_movies.sort_values(
        by=['avg_rating', 'num_votes'],
        ascending=[False, False]
    )

    end_time = time.time()
    duration = end_time - start_time

    return matched_movies[['title', 'tmdbId', 'genres', 'tag']].head(10), duration

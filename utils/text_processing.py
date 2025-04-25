import re
from collections import Counter
from nltk.corpus import stopwords
from collections import defaultdict
import nltk

nltk.download('stopwords')

def clean_title(title):
    return re.sub("[^a-zA-Z0-9]", " ", title)

def get_popular_tags(movies, top_n=20):
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'a', 'an', 'the', 'and', 'or', 'of', 'to', 'in', 'on', 'at',
                        'is', 'are', 'was', 'were', 'it', 'that', 'this', 'these', 'those'}
    stop_words.update(custom_stopwords)

    all_tags = []
    for tags in movies['tag'].dropna().str.split():
        filtered_tags = [tag.lower() for tag in tags
                         if tag.lower() not in stop_words and len(tag) >= 3]
        all_tags.extend(filtered_tags)

    tag_mapping = {
        'sci-fi': ['scifi', 'science fiction'],
        'romance': ['romantic', 'love story'],
        'comedy': ['funny', 'humor'],
        'action': ['actionpacked', 'fight'],
        'drama': ['emotional', 'serious'],
        'war': ['warfare', 'military']
    }

    normalized_tags = []
    for tag in all_tags:
        found = False
        for normalized, variants in tag_mapping.items():
            if tag in variants:
                normalized_tags.append(normalized)
                found = True
                break
        if not found:
            normalized_tags.append(tag)

    tag_counts = Counter(normalized_tags)
    content_tags = [tag for tag, count in tag_counts.most_common(top_n * 3)
                    if not tag.isnumeric() and not tag.replace('-', '').isnumeric()]

    seen = set()
    unique_content_tags = []
    for tag in content_tags:
        if tag not in seen:
            seen.add(tag)
            unique_content_tags.append(tag)

    return unique_content_tags[:top_n]

def get_meaningful_tags(movies, min_count=5):
    all_tags = []
    for tags in movies['tag'].dropna().str.split():
        all_tags.extend([tag.lower().strip() for tag in tags])

    tag_counts = Counter(all_tags)
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'a', 'an', 'the', 'and', 'or', 'of', 'to', 'in', 'on', 'at',
                        'is', 'are', 'was', 'were', 'it', 'that', 'this', 'these', 'those',
                        'film', 'movie', 'based', 'story', 'scene'}
    stop_words.update(custom_stopwords)

    meaningful_tags = {}
    for tag, count in tag_counts.items():
        if (tag not in stop_words and
                len(tag) >= 3 and
                not tag.isdigit() and
                count >= min_count):
            meaningful_tags[tag] = count

    tag_groups = defaultdict(list)
    for tag in meaningful_tags:
        main_word = tag.split()[-1]
        tag_groups[main_word].append(tag)

    final_tags = []
    for main_word, tags in tag_groups.items():
        if len(tags) > 1:
            sorted_tags = sorted(tags, key=lambda x: meaningful_tags[x], reverse=True)
            final_tags.extend(sorted_tags[:3])
        else:
            final_tags.append(tags[0])

    final_tags = sorted(final_tags, key=lambda x: meaningful_tags[x], reverse=True)
    return final_tags[:50]

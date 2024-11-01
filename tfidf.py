import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

"""
Extracts the top N words with the highest TF-IDF scores from a specified text column in a CSV file.

Parameters:
- csv_file (str): Path to the CSV file containing the text data.
- text_column (str): Name of the column that contains the text data.
- top_n (int): Number of top words to return based on their TF-IDF scores (default is 10).

Returns:
- pd.DataFrame: A DataFrame containing the top N words and their corresponding TF-IDF scores.
"""

def get_top_tfidf_words(csv_file: str, text_column: str, top_n: int = 10):
    
    df = pd.read_csv(csv_file)
    text_data = df[text_column].dropna().tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_data)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    tfidf_df = pd.DataFrame({'word': feature_names, 'tfidf_score': tfidf_scores})
    top_words = tfidf_df.sort_values(by='tfidf_score', ascending=False).head(top_n)
    return top_words

if __name__ == "__main__":
    top_words = get_top_tfidf_words('your_file.csv', 'text_column')
    print(top_words)
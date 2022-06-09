# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 17:54:35 2022

@author: Basem Rizk
"""
import time
import nltk
import pandas as pd
from tqdm import tqdm
# from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence as FlairSentence
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from data_builder import scrape_articles
from visualizer import visualize_flair_sentiment, visualize_vader_sentiment

st = time.time()

# =============================================================================
# Scraping Articles
# =============================================================================
articles = scrape_articles()


# =============================================================================
# Preprocessing and Tokenizing
# =============================================================================
# Tokenizer that does not include puncuations, and numbers
word_tokenizer = RegexpTokenizer(r'[A-Z]\w+')

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

print('Preprocessing and Tokenizing..')
max_num_of_sentences = 0
for article in tqdm(articles):
    article['sentences'] =\
        [s for p in article['paragraphs'] for s in sent_tokenize(p)]
    article['text'] = " ".join(article['paragraphs'])
    article['words'] = word_tokenizer.tokenize(article['text'])
    # Filtering out stop words
    article['words'] =\
        [w.lower() for w in article['words'] if not w.lower() in stop_words]
    # Max num of sentences including subtitle and title
    max_num_of_sentences = max(max_num_of_sentences, len(article['sentences']) + 2)
    
# =============================================================================
# Sentiment Calculation
# =============================================================================
vader = SentimentIntensityAnalyzer()
flair_classifier = TextClassifier.load('en-sentiment')
sentiments = []
per_s_sentiments_df = pd.DataFrame()
flair_value_encoding = {'POSITIVE': 1, 'NEGATIVE': -1}
print("Calculating Sentiments..")
for i, article in enumerate(tqdm(articles)):
    row = [article['title'], article['date']]
    # Whole text of paragraph combined
    # most_common_words = [w for w, _ in FreqDist(article['words']).most_common()]
    
    # row += TextBlob(text).sentiment

    row += list(vader.polarity_scores(article['text']).values())
    
    flair_sentence = FlairSentence(article['text'])
    flair_classifier.predict(flair_sentence)
    row += [
        flair_sentence.labels[0].score,
        flair_value_encoding[flair_sentence.labels[0].value]
    ]
    
    sentiments.append(row)

    # Calculate sentiment per paragraph
    sentences = [article['title'], article['subtitle']] + article['sentences']
    per_s_sentiment = []
    for s in sentences:
        if not s:
            per_s_sentiment.append(0)
            continue
        s = FlairSentence(s)
        flair_classifier.predict(s)
        row = s.labels[0].score \
            * flair_value_encoding[s.labels[0].value]
        per_s_sentiment.append(row)
        
    # Just padding
    per_s_sentiment += [0]*(max_num_of_sentences - len(per_s_sentiment))
    per_s_sentiments_df[str(i) + "_c*v"] = per_s_sentiment

sentiments_df = pd.DataFrame(sentiments, columns = [
        'title', 'date',
        # 'tb_t_polar', 'tb_t_subj',
        'v_t_neg', 'v_t_neu', 'v_t_pos', 'v_t_compound',
        'f_t_score', 'f_t_value',
    ])    
sentiments_df['index'] = sentiments_df.index

# =============================================================================
# Visualizations
# =============================================================================
visualize_flair_sentiment(sentiments_df, per_s_sentiments_df)
visualize_vader_sentiment(sentiments_df)


et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')



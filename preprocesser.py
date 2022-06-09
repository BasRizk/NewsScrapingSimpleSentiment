# -*- coding: utf-8 -*-
"""
Preprocessing Object for tokenization

"""
import nltk
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

class PreProcessor:
    
    def __init__(self, token_regex=r'[A-Z]\w+'):    
        # Tokenizer that does not include puncuations, and numbers
        self.word_tokenizer = RegexpTokenizer(token_regex)
        
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        
    
    def tokenize(self, article, into_sentences=True, into_words=True):
        if into_sentences:
            article['sentences'] =\
                [s for p in article['paragraphs'] for s in sent_tokenize(p)]
        article['text'] = " ".join(article['paragraphs'])
        if into_words:
            article['words'] = self.word_tokenizer.tokenize(article['text'])
            # Filtering out stop words
            article['words'] =\
                [w.lower() for w in article['words'] if not w.lower() in self.stop_words]
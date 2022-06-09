# NewsScrapingSimpleSentiment
Webscraping news articles of news website and couple different off-the-shelf attempts of sentiment analysis.

> By excuting (`~ 27 seconds`) `main.py` in `spyder` IDE at a `conda` environment with `python=3.9` after ensuring `requirements.txt` are all installed, the program goes through the following procedures:


1. Webscrapes `https://www.aljazeera.com/where/mozambique` most recent `10` articles, opens them up and extracts title, subtitle, date, and paragraphs for every article and saves them in json format in `data.json`.


    ```
    {
        "retrieved at": "08/06/2022 21:18:24",
        "articles": [
            {
                "url": "https://www.aljazeera.com/news/2022/5/23/floods-hit-south-africas-kwazulu-natal-province-again",
                "title": "Floods hit South Africa’s KwaZulu-Natal province again",
                "subtitle": "The floods are ...",
                "date": "23 May 2022",
                "paragraphs": [
                    "Heavy rains in South Africa ...",
                    ...
                ]
            },
            ...
        ]
    }
    ```

2. Some preprocessing of the data including tokenization to prepare for inference by Sentiment Analyzers later; Some processing is actually applied but not used at the moment, but could be defintely used with other analyzing forms.

3. Infering and Calculating Sentiments using `Flair` and `VaderSentiment`  packing the results into DataFrames.

4. Visualizing sentiments into two different figures corresponding to the two SentimentAnalyzers models (Explanation is provided below).



## Flair Sentiment Analysis

> My aim here by developing such graph is to discover a trend of writing the article on sentence basis, on which is turning up to be a positive or negative overall. Clearly it did not work, but possibly taking in consideration a much larger number of articles and grouping articles in sizes, categories and/or even time ranges would possibly show more promising results.

The following figure is showing two plots in a sort of heat-map form:
- Top plot corresponds to sentiment per the whole text of the paragraphs combined.
- Bottom plot corresponds to sentiment throughout the whole article on a sentence base in order including the title and the subtitle (from bottom up) - the zeros here corresponds to empty units obviously due the variable size of the articles, possibly for larger analysis, one could match up similar size articles and differeniate then between the trends.
- Total Red `-1.0` corresponds to a `Negative` sentiment inference by `Flair` with a confidence level of `100%`, while total Green `1.0` would correspond to `Positive` with a confidence level of `100%`.

![Alt text](flair_sentiment.svg?raw=true "Flair Sentiment")

## Vader Sentiment Analysis

> My aim here by developing such graph is to visualize the contribution of article sentiment (proportions of sentiment) and compound anaylsis of overall put together piece.

The following figure is showing two histogram plots:
- Top is a plot of the compound sentiment of each article of the 10.
- Bottom is a stacked columns plot corresponding to proportion size of the article relative to neutrality, negativity, and positivity. 


![Alt text](vader_sentiment.svg?raw=true "Vader Sentiment")


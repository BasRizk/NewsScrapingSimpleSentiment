# -*- coding: utf-8 -*-
"""
Scraping Articles and JSONifying them

@author: Basem Rizk
"""
import json
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

requests_session = requests.Session()

def make_request(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
            AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122\
                Safari/537.36",
        "Accept-Encoding": "gzip, deflate",
        'Accept': '*/*',
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

    try:
        return requests_session.get(url, allow_redirects=True,
                                    verify=True, headers=headers, timeout=20)
    except requests.exceptions.Timeout:
        print("\n$s : timed out" % url)
        return None
    except requests.exceptions.TooManyRedirects:
        print("\n%s : too many redirects" % url)
        return None
    except requests.exceptions.RequestException:
        # catastrophic error. bail.
        return None


def _get_current_datetime():
    from datetime import datetime
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def _scrape_article(url):
    article_html = make_request(url).text
    article_soup = BeautifulSoup(article_html, 'html.parser')
    header = article_soup.main.find_all(attrs={"class": "article-header"})[0]
    title = header.h1.get_text()
    subtitle = header.p
    if subtitle:
        subtitle = subtitle.get_text()
    else:
        subtitle = ""
    date = article_soup.main.find_all(
        attrs={"class": "article-info-block"})[0].find_all(text=True)[1]
    text = article_soup.main.find_all(
        name="div", attrs={"class": "wysiwyg"})[0]
    paragraphs = [p.get_text().strip() for p in text.find_all("p")]
    paragraphs = [p for p in paragraphs if len(p) > 0]
    return {
        "url": url,
        "title": title,
        "subtitle": subtitle,
        "date": date,
        "paragraphs": paragraphs,
    }


def scrape_articles(url="https://www.aljazeera.com/where/mozambique/",
                    max_num_of_articles=10, save=True):

    def get_link(article_soup, prefix="https://www.aljazeera.com"):
        postfix = article_soup.find_all(
            attrs={"class": "u-clickable-card__link"}
        )[0].get("href")
        return prefix + postfix

    articles_json = []
    html_page = make_request(url).text
    soup = BeautifulSoup(html_page, 'html.parser')
    homepage_articles = soup.find_all(name="article")[:max_num_of_articles]

    print("Retrieving and Organizing Articles Data:")
    for article in tqdm(homepage_articles):
        articles_json.append(_scrape_article(get_link(article)))

    if save:
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump({
                "retrieved at": _get_current_datetime(),
                "articles": articles_json
            }, f, ensure_ascii=False, indent=4)
    return articles_json

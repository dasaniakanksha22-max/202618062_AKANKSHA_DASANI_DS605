# DS605 - Lab Assignment 1: Data Scraping and Preprocessing

**Course:** DS605 - Fundamentals of Machine Learning
**Name:** _<your name>_
**ID:** _<your student ID>_

## Project Overview
This project builds an end-to-end data pipeline that:
1. Scrapes book data from [books.toscrape.com](https://books.toscrape.com/) using Scrapy
2. Cleans and preprocesses the scraped data
3. Generates visualizations and a word cloud
4. Draws data-driven insights from the results

## Project Structure
```
.
├── bookscraper/              # Scrapy project
│   ├── scrapy.cfg
│   └── bookscraper/
│       ├── settings.py
│       └── spiders/
│           └── books_spider.py
├── data/
│   ├── raw_books.csv         # raw scraped output
│   └── cleaned_books.csv     # cleaned + feature-engineered
├── plots/                    # generated visualizations
├── preprocessing.py          # Task 2
├── visualize.py               # Task 3
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt

# Task 1: Scrape data
cd bookscraper
scrapy crawl books -o ../data/raw_books.csv
cd ..

# Task 2: Clean + engineer features
python preprocessing.py

# Task 3: Generate plots + word cloud
python visualize.py
```

## Task 1 — Data Scraping
- Records scraped: _<fill in after running>_
- Missing values: _<fill in — printed by preprocessing.py>_
- Duplicate UPCs found: _<fill in>_

## Task 2 — Preprocessing
Cleaned text fields, removed duplicate UPCs, handled missing descriptions,
converted price to numeric, mapped rating words to integers, extracted stock
count, and engineered: `description_word_count`, `price_band`,
`affordability_score`, `value_score`, `recommended`.

## Task 3 — Visualization
See `/plots` for:
- `price_distribution.png`
- `rating_distribution.png`
- `avg_price_by_category.png`
- `price_vs_rating.png`
- `wordcloud.png`

## Task 4 — Insights and Interpretation
_<Write your 5-7 observations here after reviewing the plots and summary
stats printed by visualize.py. Cover: relationship between price and rating,
most represented/expensive categories, which books look like best value, and
limitations of the dataset (e.g. single-source, no real customer reviews,
static site snapshot).>_

## Limitations
- Data reflects a single scrape snapshot of a demo/practice site, not a live
  retail catalog.
- No genuine customer review text exists on this site — descriptions were
  used as a proxy for the word cloud.
- Category sample sizes are uneven, which can skew average-price comparisons.

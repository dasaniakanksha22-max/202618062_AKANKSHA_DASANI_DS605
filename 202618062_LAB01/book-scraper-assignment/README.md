
# Project Overview

This project demonstrates the complete workflow of collecting, cleaning, analyzing, and visualizing data using Python and Scrapy. Book information was scraped from the Books to Scrape website, preprocessed into a clean dataset, and explored using different visualizations to identify meaningful patterns and insights.

---

# Project Structure

```
.
├── bookscraper/
│   ├── scrapy.cfg
│   └── bookscraper/
│       ├── settings.py
│       └── spiders/
│           └── books_spider.py
├── data/
│   ├── raw_books.csv
│   └── cleaned_books.csv
├── plots/
│   ├── price_distribution.png
│   ├── rating_distribution.png
│   ├── avg_price_by_category.png
│   ├── price_vs_rating.png
│   └── wordcloud.png
├── preprocessing.py
├── visualize.py
├── requirements.txt
└── README.md
```

---

# How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Task 1 – Scrape Book Data

```bash
cd bookscraper
scrapy crawl books -o ../data/raw_books.csv
cd ..
```

### Task 2 – Preprocess Data

```bash
python preprocessing.py
```

### Task 3 – Generate Visualizations

```bash
python visualize.py
```

---

# Task 1 – Data Scraping

The Scrapy spider extracts the following information for each book:

- Title
- Category
- Price
- Rating
- Availability
- Product Description
- UPC
- Number of Reviews
- Product URL

The scraped data is stored in:

```
data/raw_books.csv
```

---

# Task 2 – Data Preprocessing

The preprocessing script performs the following tasks:

- Removes unnecessary spaces and inconsistent formatting.
- Handles missing descriptions.
- Removes duplicate books using UPC.
- Converts prices into numeric values.
- Converts ratings from text into integers.
- Extracts the available stock count.
- Creates additional features:
  - description_word_count
  - price_band
  - affordability_score
  - value_score
  - recommended

The cleaned dataset is saved as:

```
data/cleaned_books.csv
```

---

# Task 3 – Visualizations

The following plots are generated inside the **plots** folder:

- Price Distribution
- Rating Distribution
- Average Price by Category
- Price vs Rating
- Word Cloud of Book Descriptions

---

# Task 4 – Insights and Interpretation

Based on the generated summary statistics and visualizations:

- A total of **100 books** were analyzed.
- The average book price is **£34.56**, while prices range from **£10.16** to **£58.11**.
- The average book rating is approximately **3 out of 5**, indicating that most books have moderate ratings.
- **Sequential Art** is the most represented category with **14 books**, followed by **Nonfiction (12 books)** and **Default (9 books)**.
- The Price vs Rating visualization shows **no strong relationship** between book price and rating. Expensive books do not necessarily receive higher ratings.
- The word cloud highlights frequently occurring words from book descriptions, giving an overview of common themes present in the dataset.


---

# Conclusion

This project successfully demonstrates a complete data scraping and preprocessing pipeline using Python and Scrapy. The collected data was cleaned, transformed, and analyzed using summary statistics and visualizations. The analysis shows that book prices vary considerably, while ratings remain fairly consistent across different price ranges, suggesting that higher prices do not necessarily indicate better-rated books. Overall, the project provided practical experience in web scraping, data preprocessing, feature engineering, exploratory data analysis, and data visualization.
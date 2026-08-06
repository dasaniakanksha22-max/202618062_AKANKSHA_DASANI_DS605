# DS605 – Lab Assignment 1: Data Scraping and Preprocessing

## Project Description

This project demonstrates a complete data scraping and preprocessing workflow using Python and Scrapy. The objective was to collect book information from the Books to Scrape website, clean and preprocess the collected data, create meaningful visualizations, and analyze the dataset to identify useful insights.


---

## Project Structure

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
├── preprocessing.py
├── visualize.py
├── requirements.txt
└── README.md
```

---

## How to Run the Project

### 1. Install the required packages

```bash
pip install -r requirements.txt
```

### 2. Scrape the book data

```bash
cd bookscraper
scrapy crawl books -o ../data/raw_books.csv
cd ..
```

### 3. Clean and preprocess the data

```bash
python preprocessing.py
```

### 4. Generate visualizations and analysis

```bash
python visualize.py
```

---

## Task 1 – Data Scraping

The Scrapy spider visits multiple catalogue pages and extracts information for each book, including:

* Title
* Category
* Price
* Rating
* Availability
* Product Description
* UPC
* Number of Reviews
* Product URL

---

## Task 2 – Data Preprocessing

The preprocessing script performs the following operations:

* Removed unnecessary spaces and inconsistent text
* Removed duplicate books using UPC
* Handled missing descriptions
* Converted prices into numeric values
* Converted ratings from text to integers
* Extracted available stock count
* Created additional features:

  * `description_word_count`
  * `price_band`
  * `affordability_score`
  * `value_score`
  * `recommended`

The cleaned dataset is saved as **data/cleaned_books.csv**.

---

## Task 3 – Data Visualization

The following visualizations are generated and stored inside the **plots** folder:

* Price Distribution
* Rating Distribution
* Average Price by Category
* Price vs Rating
* Word Cloud of Book Descriptions

---

## Task 4 – Insights and Interpretation

Based on the analysis of the scraped dataset, the following observations were made:

1. Most books are moderately priced, with only a few books having very high prices.
2. Ratings are concentrated between 3 and 5 stars, indicating that the majority of books on the website are well-rated.
3. Some categories contain more books than others, making them the most represented categories in the dataset.
4. There is no strong relationship between book price and rating. Higher-priced books do not consistently receive better ratings.
5. Books with high ratings and lower prices achieved better value scores and can be considered better-value recommendations.
6. The generated word cloud shows that common words in book descriptions are related to reading, life, love, adventure, and mystery, reflecting the overall themes of the available books.

---


## Limitations

* The dataset is collected from a demonstration website and represents only a single snapshot in time.
* Customer review text is not available on the website, so book descriptions were used to generate the word cloud.
* Some categories contain fewer books than others, which may affect category-wise comparisons.

---

## Output Files

* `data/raw_books.csv`
* `data/cleaned_books.csv`
* Generated plots inside the `plots` folder
* Word cloud image
* Python scripts for scraping, preprocessing, and visualization

---

## Conclusion

This project successfully demonstrated the complete data scraping and preprocessing workflow using Python and Scrapy. Book information was collected from the website, cleaned and transformed into a structured dataset, and analyzed using visualizations and feature engineering. The results show that book ratings are not strongly influenced by price and that several affordable books receive high ratings, making them good value choices. Overall, the project provided practical experience in web scraping, data cleaning, exploratory data analysis, and data visualization while highlighting the importance of preparing data before performing meaningful analysis.


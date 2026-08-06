import scrapy


class BooksSpider(scrapy.Spider):
    name = "books"
    start_urls = ["https://books.toscrape.com/catalogue/page-1.html"]

    RATING_MAP = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5,
    }

    def parse(self, response):
        # Visit every book on the current page
        for book in response.css("article.product_pod"):
            relative_url = book.css("h3 a::attr(href)").get()
            yield response.follow(relative_url, callback=self.parse_book)

        # Stop after page 5
        current_page = int(
            response.url.split("page-")[-1].replace(".html", "")
        )

        if current_page < 5:
            next_page = response.css("li.next a::attr(href)").get()
            if next_page:
                yield response.follow(next_page, callback=self.parse)

    def parse_book(self, response):
        def clean(text):
            return text.strip() if text else None

        rating_class = response.css("p.star-rating::attr(class)").get() or ""
        rating_word = rating_class.replace("star-rating", "").strip()

        table = {}
        for row in response.css("table.product_page tr"):
            key = clean(row.css("th::text").get())
            value = clean(row.css("td::text").get())
            table[key] = value

        category = response.xpath(
            "//ul[@class='breadcrumb']/li[3]/a/text()"
        ).get()

        description = response.css("#product_description ~ p::text").get()

        yield {
            "title": clean(response.css("div.product_main h1::text").get()),
            "category": clean(category),
            "price": clean(response.css("p.price_color::text").get()),
            "rating": rating_word,
            "rating_numeric": self.RATING_MAP.get(rating_word),
            "availability": " ".join(
                response.css("p.availability::text").getall()
            ).strip(),
            "description": clean(description) or "",
            "upc": table.get("UPC"),
            "num_reviews": table.get("Number of reviews"),
            "product_url": response.url,
        }
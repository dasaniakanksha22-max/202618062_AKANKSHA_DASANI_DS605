BOT_NAME = "bookscraper"

SPIDER_MODULES = ["bookscraper.spiders"]
NEWSPIDER_MODULE = "bookscraper.spiders"

# Be a polite scraper: identify yourself and don't hammer the server
USER_AGENT = "bookscraper (+educational lab assignment)"
ROBOTSTXT_OBEY = True
DOWNLOAD_DELAY = 0.5          # seconds between requests
CONCURRENT_REQUESTS_PER_DOMAIN = 4

# Export settings (used if you run without -o flag)
FEED_EXPORT_ENCODING = "utf-8"

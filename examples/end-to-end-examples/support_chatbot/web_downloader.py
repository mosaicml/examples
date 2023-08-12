from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

class WebDownloader:
    def __init__(self, base_url):
        self.base_url = base_url
        self.visited_links = set()  # To track visited links
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(options=options)

    def save_content_as_txt(self, url, folder="retrieval_data"):
        os.makedirs(folder, exist_ok=True)
        
        self.driver.get(url)
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "a"))
        )
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        page_text = soup.get_text()

        if "404 Not Found" in page_text or "This page does not exist" in page_text:
            print(f"Skipping {url} due to 404 error.")
            return

        cleaned_url = url.replace(":", "{colon}").replace(".", "{dot}").replace("/", "{slash}")
        filename = cleaned_url + ".txt"
        filepath = os.path.join(folder, filename)

        with open(filepath, 'w', encoding="utf-8") as file:
            file.write(page_text)

        return filepath

    def get_all_links(self, page_url):
        self.driver.get(page_url)
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "a"))
        )

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('#'):
                continue
            absolute_url = urljoin(self.base_url, href)
            if absolute_url.startswith(self.base_url) and absolute_url not in self.visited_links:
                links.add(absolute_url)
        return list(links)
    
    def close_driver(self):
        self.driver.quit()

    def download_website(self, start_page, save_folder="retrieval_data/docs_site", depth=1, max_depth=20):
        if depth > max_depth or start_page in self.visited_links:
            return
        
        self.visited_links.add(start_page)

        links = self.get_all_links(start_page)
        
        for link in links:
            print(f"Saving content from {link}...")
            self.save_content_as_txt(link, save_folder)
            
            self.download_website(link, save_folder, depth + 1, max_depth)

        if depth == 1:
            print("All done!")

if __name__ == '__main__':
    downloader = WebDownloader(base_url="https://docs.mosaicml.com")
    downloader.download_website("https://docs.mosaicml.com/projects/mcli/")
    downloader.download_website("https://docs.mosaicml.com/projects/composer/")
    downloader.download_website("https://docs.mosaicml.com/projects/streaming/")
    downloader.close_driver()

import argparse
import json
import time

from selenium import webdriver
from selenium.webdriver.common.by import By


class QuillbotScraper:
    SUPPORTED_LANGUAGES = {
        'nl': 'Dutch',
        'en': 'English (UK)',
        'fr': 'French',
        'de': 'German',
        'pt': 'Portuguese (Brazilian)',
        'es': 'Spanish',
    }

    def __init__(self, url: str = 'https://quillbot.com/grammar-check', lang: str = 'en'):
        if lang not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Language {lang} not supported by Quillbot")
        self.url = url
        self._create_driver()
        self._select_language(lang)
        self.text_area = self._find_text_area()

    def _create_driver(self):
        self.driver = webdriver.Chrome()
        self.driver.get(self.url)
        time.sleep(3)

    def _select_language(self, lang: str):
        button = self.driver.find_element(
            By.XPATH, f'//button[contains(text(), "All")]')
        button.click()
        time.sleep(1)
        button = self.driver.find_element(
            By.XPATH, f'//p[contains(text(), "{self.SUPPORTED_LANGUAGES[lang]}")]')
        button.click()
        time.sleep(1)

    def _find_text_area(self):
        return self.driver.find_element(By.XPATH, '//*[@id="grammarbot"]')

    def get_error_count(self, text):
        self.text_area.clear()
        self.text_area.send_keys(text)

        time.sleep(5)

        error_count_element = self.driver.find_element(
            By.XPATH, '//*[@id="error-count"]'
        )

        return int(error_count_element.text)

    def __del__(self):
        self.driver.quit()


def get_data():
    with open('results.json', 'r') as file:
        return json.load(file)


def dump_data(data):
    with open('results.json', 'w') as file:
        json.dump(data, file, indent=4)


def process_language(data: dict, language: str, save_interval: int):
    quillbot_scraper = QuillbotScraper(lang=language)
    count, to_processed = 0, sum([1 for d in data[language] if "quillbot_errors" not in d])

    for d in data[language]:
        if "quillbot_errors" not in d:
            d["quillbot_errors"] = quillbot_scraper.get_error_count(d["content"])
            count += 1
            if count % save_interval == 0:
                print(f"Processed {count}/{to_processed} entries for {language}, dumping results.")
                dump_data(data)
    dump_data(data)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, nargs="*")
    parser.add_argument("--save-interval", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    data = get_data()
    for lang in args.lang:
        if lang not in data:
            print(f"Selected language {lang} not found in data!")
        else:
            process_language(data, lang, args.save_interval)


if __name__ == '__main__':
    main()

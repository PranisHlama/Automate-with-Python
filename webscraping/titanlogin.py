from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from dotenv import load_dotenv
import os
import time 

service = Service('/home/pranish/Downloads/chromedriver-linux64/chromedriver')
def get_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("disable-infobars")
    options.add_argument("start-maximized")
    options.add_argument("disable-dev-shm-usage")
    options.add_argument("no-sandbox")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument("disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(service=service, options=options)
    driver.get("https://titan22.com/account/login?return_url=%2Faccount")
    return driver

def clean_text(text):
    output = float(text.split(": "))
    return output

load_dotenv()

email = os.getenv("EMAIL")
password = os.getenv("PASSWORD")

def main():
    driver = get_driver()
    driver.find_element(by="id", value="CustomerEmail").send_keys(email)
    time.sleep(1)
    driver.find_element(by="id", value="CustomerPassword").send_keys(password + Keys.RETURN)
    time.sleep(1)
    driver.find_element(by="xpath", value="/html/body/footer/div/section/div/div[1]/div[1]/div[1]/nav/ul/li[1]").click()
    time.sleep(5)
    
    print(driver.current_url)
    # return clean_text(text)
    

print(main())

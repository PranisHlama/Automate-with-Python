from selenium import webdriver
from selenium.webdriver.chrome.service import Service

from selenium.webdriver.common.keys import Keys
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
    driver.get("https://automated.pythonanywhere.com/login")
    return driver

def clean_text(text):
    output = float(text.split(": "))
    return output

def main():
    driver = get_driver()
    driver.find_element(by="id", value="id_username").send_keys("automated")
    time.sleep(1)
    driver.find_element(by="id", value="id_password").send_keys("automatedautomated" + Keys.RETURN)
    time.sleep(1)
    # driver.find_element(by="xpath", value="/html/body/div[1]/div/div/div[3]/form/button").click()
    # time.sleep(1)
    # driver.find_element(by="xpath", value="/html/body/div[1]/div/div/div[1]/div/a[1]").click()
    # time.sleep(2)

    print(driver.current_url)
    text = driver.find_element(by="xpath", value="/html/body/div[1]/div/h1[2]")
    return clean_text(text)
    

print(main())

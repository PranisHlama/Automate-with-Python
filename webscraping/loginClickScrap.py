from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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
    output = float(text.split(": ")[1])
    return output

def main():
    driver = get_driver()
    try:
        driver.find_element(by="id", value="id_username").send_keys("automated")
        driver.find_element(by="id", value="id_password").send_keys("automatedautomated" + Keys.RETURN)
        time.sleep(1)
        
        # Check for "change your password" alert and close it if present
        try:
            alert = WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.CLASS_NAME, "alert"))
            )
            close_btn = alert.find_element(By.CLASS_NAME, "close")
            close_btn.click()
            time.sleep(1)
        except Exception:
            pass  # Alert not present, continue

        driver.find_element(by="xpath", value="/html/body/nav/div/a").click()
        
        # Debug: print all h1 elements
        time.sleep(2)
        h1s = driver.find_elements(By.TAG_NAME, "h1")
        for h in h1s:
            print("Found h1:", h.text)
        
        # Try to find the first h1 with a colon (":") in its text
        element = None
        for h in h1s:
            if ":" in h.text:
                element = h
                break
        if not element:
            raise Exception("Target h1 element not found.")
        return clean_text(element.text)
    finally:
        driver.quit()

print(main())
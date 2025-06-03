from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import pandas as pd

df = pd.DataFrame(columns=['Player','Salary','Year'])

cService = webdriver.ChromeService(executable_path='/home/pranish/Downloads/chromedriver-linux64/chromedriver') 
driver = webdriver.Chrome(service=cService)

for yr in range(1990,2023):
    page_num = str(yr) + '-' + str(yr+1) +'/'
    url = 'https://hoopshype.com/salaries/players/' + page_num
    driver.get(url)
   
    players = driver.find_elements(By.XPATH, '//td[@class="name"]')
    salaries = driver.find_elements(By.XPATH, '//td[@class="hh-salaries-sorted"]')
   
    players_list = []
    for p in range(len(players)):
        players_list.append(players[p].text)
   
    salaries_list = []
    for s in range(len(salaries)):
        salaries_list.append(salaries[s].text)
   
    p = int(len(players_list)/2)+1 
    s = int(len(salaries_list)/2)-1 

    data_tuples = list(zip(players_list[p:],salaries_list[1:s]))
    temp_df = pd.DataFrame(data_tuples, columns=['Player','Salary'])
    temp_df['Year'] = yr + 1 

    print(temp_df)

    df = pd.concat([df, temp_df], ignore_index=True)
   
driver.close()
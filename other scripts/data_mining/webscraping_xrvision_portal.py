import requests
from bs4 import BeautifulSoup
import re
import datetime
import parsel
from parsel import Selector
import time
import numpy as np
import pandas as pd
import os
import glob
import pickle
import keyboard
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


username = 'user1@user.com'
password = 'Password'

# Make dataframe to store profile info
df = pd.DataFrame(columns = ['Video Links'])
dict_blanks = {'blanks': 0}

# Logging in
driver = webdriver.Chrome("D:\Daniel\PMD\scripts\data preprocessing scripts\chromedriver.exe")
driver.get('http://5.28.131.238:8080/Account/Login')
elementID = driver.find_element_by_id('Email')
elementID.send_keys(username)
elementID = driver.find_element_by_id('Password')
elementID.send_keys(password)

elementID.submit()
#time.sleep(0.5)


# Motorcycle violations link
driver.get('http://5.28.131.238:8080/xrvision/VideoAnalytics/IndexPmd?vaType=SpeedCalculation&VideoSourceGroupId=0&DateTimeRange=&CameraLocationId=0&VideoSourceId=0&SortOrder=DetectedOnDescending&SpeedViolation=&AlertDisplayType=Default&ViolationTypes=MotorcycleViolation')

driver.get('http://5.28.131.238:8080/xrvision/VideoAnalytics/IndexPmd?sortOrder=DetectedOnDescending&vaType=SpeedCalculation&videoSourceId=0&page=2&CameraLocationId=0&VideoSourceGroupId=0&alertDisplayType=Default&ViolationTypes=RidingOnPedestrianPath&violationTypes=SpeedingOnSharedPath')
local_host = 'http://5.28.131.238:8080'

start = time.time()
page_count = 2
link_lst = []

while page_count != 12:
    src = driver.page_source
    soup = BeautifulSoup(src, 'lxml')
    video_links = soup.findAll('a', href=True)
    
    for link in video_links:
        if "/xrvision/Images/VADetectionSnippet/" in link.get('href'):
            url = link.get('href')

            link_lst.append(local_host+url)
            print(url)
    
    next_page = driver.find_elements_by_link_text(f'{page_count+1}')[0]
    print(next_page)
    next_page.click()
    
    time.sleep(0.3)
    page_count+=1


print(link_lst)


for link in link_lst:
    driver.get(link)
    time.sleep(5)
    keyboard.press_and_release('ctrl+s')
    keyboard.press_and_release('enter')



end = time.time()

print(f'scraping took {end-start} seconds')
driver.close()
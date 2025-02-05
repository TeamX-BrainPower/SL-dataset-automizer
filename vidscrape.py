from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import time
import os

driver = webdriver.Chrome()
driver.get('https://www.minetegn.no/Tegnordbok-2016/tegnordbok.php')

# Wait for the list to load
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, 'li.ord'))
)

list_items = driver.find_elements(By.CSS_SELECTOR, 'li.ord')
data = []

# Write to CSV
with open('word_videos.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['word', 'video_url'])
    writer.writeheader()

    for li in list_items:
        driver.execute_script("arguments[0].scrollIntoView();", li)
        li.click()
        
        try:
            WebDriverWait(driver, 10).until(
                lambda d: d.find_element(By.ID, 'myVideo').get_attribute('src') != ''
            )
            video = driver.find_element(By.ID, 'myVideo')
            video_filename = os.path.basename(video.get_attribute('src'))
            word = li.text

            writer.writerow({'word': word, 'video_url': video_filename})
            print({'word': word, 'video_url': video_filename})

        except Exception as e:
            print(f"Error processing {li.text}: {e}")

driver.quit()
print("CSV file created successfully!")
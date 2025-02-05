from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define a function to process a single list item by index
def process_item(index):
    # Set up headless Chrome
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")        # Run headless
    options.add_argument("--disable-gpu")     # Disable GPU acceleration
    # (Optional) Other useful options:
    # options.add_argument("--no-sandbox")
    # options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=options)
    driver.get('https://www.minetegn.no/Tegnordbok-2016/tegnordbok.php')
    
    try:
        # Wait until the list of words is loaded
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'li.ord'))
        )
        list_items = driver.find_elements(By.CSS_SELECTOR, 'li.ord')
        
        if index >= len(list_items):
            raise Exception(f"Index {index} out of range. Only found {len(list_items)} items.")
        
        li = list_items[index]
        # Scroll the element into view and click it
        driver.execute_script("arguments[0].scrollIntoView();", li)
        li.click()
        
        # Wait until the video element's src attribute is non-empty
        WebDriverWait(driver, 10).until(
            lambda d: d.find_element(By.ID, 'myVideo').get_attribute('src') != ''
        )
        
        video = driver.find_element(By.ID, 'myVideo')
        video_src = video.get_attribute('src')
        video_filename = os.path.basename(video_src)
        word = li.text
        
        result = {'word': word, 'video_url': video_filename}
        print(result)
        return result
    
    except Exception as e:
        print(f"Error processing index {index}: {e}")
        return {'word': None, 'video_url': None}
    
    finally:
        driver.quit()

if __name__ == '__main__':
    # First, load the page once (with headless Chrome) to count the number of list items.
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=options)
    driver.get('https://www.minetegn.no/Tegnordbok-2016/tegnordbok.php')
    
    WebDriverWait(driver, 10).until(
         EC.presence_of_element_located((By.CSS_SELECTOR, 'li.ord'))
    )
    list_items = driver.find_elements(By.CSS_SELECTOR, 'li.ord')
    num_items = len(list_items)
    driver.quit()
    
    print(f"Found {num_items} words to process.")
    
    results = []
    max_workers = 4  # Adjust based on your machine's capability
    
    # Process each list item in parallel using a process pool
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_item, index) for index in range(num_items)]
        for future in as_completed(futures):
            results.append(future.result())
    
    # Write results to CSV
    with open('word_videos.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['word', 'video_url'])
        writer.writeheader()
        for res in results:
            if res['word'] is not None:
                writer.writerow(res)
    
    print("CSV file created successfully!")

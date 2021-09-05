from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
import csv
import time
import re

chromeOptions = webdriver.ChromeOptions()
prefs = {'safebrowsing.enabled': 'false', "download.default_directory" : r"/Users/jessica/Documents/masterproject/dataset_Sourceforge/"}
chromeOptions.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome(chrome_options=chromeOptions)

driver.maximize_window()


def downloadSW():
    page ="https://sourceforge.net/directory/os:windows/"

    for i in range(2,100):
        
        driver.get(page)
        time.sleep(2)
        blocks = driver.find_elements_by_css_selector('li[itemprop="itemListElement"]')[2:]
        print(len(blocks))
        hrefs = []
        for block in blocks:
            app_site = block.find_element_by_css_selector('a[class="button green hollow see-project"]')
            hrefs.append(app_site.get_attribute("href"))
        for href in hrefs:
            
            driver.get(href+'files/')  
            time.sleep(2)
            download_bt=driver.find_element_by_css_selector('a[class="button green big-text download with-sub-label extra-wide"]')
            size_str = download_bt.find_element_by_class_name("sub-label").text
            str_list = re.split("[()\s]", size_str)           
            if str_list[-2] == "GB":
                print("bigger then 100MB")
                continue
            elif str_list[-2] == "MB":
                if float(str_list[-3])>100:
                    print("bigger then 100MB")
                    continue
            
            # app_name = driver.find_element_by_css_selector('h1[class="has-masthead-badges long-title"]').find_element_by_tag_name('a').text
            download_bt.click()
            app_name = ' '
            print(f'Downloading... {app_name} of size {str_list[-3]} {str_list[-2]}')
            time.sleep(10)
        page = 'https://sourceforge.net/directory/os:windows/?page='+str(i)
    
    time.sleep(1)

downloadSW()
driver.close()

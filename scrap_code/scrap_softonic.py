from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
import csv
import time
import re

chromeOptions = webdriver.ChromeOptions()
prefs = {'safebrowsing.enabled': 'false', "download.default_directory" : r"/home/qian/Masterproject/dataset/softonic"}
chromeOptions.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome(chrome_options=chromeOptions)

driver.maximize_window()


def downloadSW():
    page ="https://en.softonic.com/windows/"
    categorys = ["security-privacy","games","business-productivity","internet-network","education-reference","multimedia","development","lifestyle"]
    for category in categorys:
        category_page = "{}{}:trending".format(page,category)
        for i in range(2,10):#from page 2 to page 10
            
            driver.get(category_page)
            time.sleep(2)
            #find_pos = driver.find_element_by_css_selector('article[class="app-list-item app-list-item--interactive"]')
            blocks = driver.find_elements_by_xpath("//div[@class = 'content content--category content--colored']/ul/li")
            hrefs = []
            for block in blocks:
                app_site = block.find_element_by_tag_name('a')
                href = app_site.get_attribute("href") + "download"
                hrefs.append(href)
            for href in hrefs:
                
                driver.get(href)  
                time.sleep(4)
                download_buttom=driver.find_element_by_css_selector('a[data-meta="button-download-direct"]')

                download_url = download_buttom.get_attribute("href")
                app_name = driver.find_element_by_css_selector('h1[class="app-header__name"]').find_element_by_tag_name('a').get_attribute("title")
                driver.get(download_url )#open the download page
                #download_buttom.click()
            #    size_str = download_bt.find_element_by_class_name("sub-label").text
            #     str_list = re.split("[()\s]", size_str)           
            #     if str_list[-2] == "GB":
            #         print("bigger then 100MB")
            #         continue
            #     elif str_list[-2] == "MB":
            #         if float(str_list[-3])>100:
            #             print("bigger then 100MB")
            #             continue
                
            #     # app_name = driver.find_element_by_css_selector('h1[class="has-masthead-badges long-title"]').find_element_by_tag_name('a').text
            #     download_bt.click()
                str_size = ' '
                str_unit = 'M'
                print(f'Downloading... {app_name} of size {str_size} {str_unit}')
                time.sleep(5)
            page = 'https://en.softonic.com/windows/'+str(i)
        
        time.sleep(1)

downloadSW()
driver.close()


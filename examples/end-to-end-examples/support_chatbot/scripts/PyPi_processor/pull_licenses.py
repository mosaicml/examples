import json
import os
import shutil
import requests
from bs4 import BeautifulSoup
import tarfile
import pathlib
import statistics
# import inspect
from tqdm import tqdm
# from scrape_homepage import recursively_scrape_homepage
from streaming import MDSWriter
from streaming.base.storage import OCIUploader
import ast
import re
import magic
import signal
# get URL
BASE_URL = "https://pypi.org" 


class TimeOutException(Exception):
   pass
 
def alarm_handler(signum, frame):
    print("ALARM signal received")
    raise TimeOutException()
 
def get_license(href):
    try:
        page = requests.get(BASE_URL + href + "#files",  timeout=(10, 120))
    except:
        return ""
    if page.status_code == 200:
        
        soup = BeautifulSoup(page.content, 'html.parser')
        uls = soup.find_all('ul', class_='sidebar-section__classifiers')
        if len(uls) == 0:
            return ""
        ul = uls[0]
        licenses = [] 
        for li in ul.find_all('li'):
            if '<strong>License</strong>' in str(li):
                licenses.append(li.get_text().strip()[7:].strip())        
        if len(licenses) > 0:
            return licenses[0]
        return ""
    else:
        print("404 not found " + href)
        return ""
   

def run_all():
   

    response = requests.get("https://pypi.org/simple/",  timeout=5000)
    soup = BeautifulSoup(response.content, 'html.parser')

    href_list = ['/project/' + package.get_text() + '/' for package in soup.find_all('a')]
    print(len(href_list))
    paths = ('licenses', 'oci://mosaicml-internal-datasets/mpt-swe')
    oci_uploader = OCIUploader(out=paths, keep_local=False)
    with open('licenses/pypi_licenses.jsonl', 'w') as f:
        for href in tqdm(href_list):
            license = get_license(href)
            url = BASE_URL + href
            f.write(json.dumps({
                "url": url,
                "license": license
            }) + '\n')    
            break
    oci_uploader.upload_file('pypi_licenses.jsonl')
run_all()


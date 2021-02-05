import os
import re
import subprocess
import urllib.request
from bs4 import BeautifulSoup

downloaded_docs = os.listdir('.')
downloaded_docs = [doc_name.split('.')[0] for doc_name in downloaded_docs]

# import pdb; pdb.set_trace()

list_pages = [
             'http://www.innocom.gov.cn/gqrdw/c101409/list_gsgg_l3.shtml',
             'http://www.innocom.gov.cn/gqrdw/c101409/list_gsgg_l3_2.shtml',
             'http://www.innocom.gov.cn/gqrdw/c101409/list_gsgg_l3_3.shtml'
             ]

root_url = 'http://www.innocom.gov.cn'
doc_url_header = 'http://www.innocom.gov.cn/gqrdw/c101409/'

def get_doc_urls(list_page_url):
    content = urllib.request.urlopen(list_page_url).read()
    parsed = BeautifulSoup(content, 'lxml')
    hrefs = [link['href'] for link in parsed.find_all('a') if ('c101409' in link['href'] and '20' in link['href'])]
    return hrefs

doc_page_urls = []

for list_page_url in list_pages:
    doc_page_urls += get_doc_urls(list_page_url)

for i in range(len(doc_page_urls)):
    doc_page_urls[i] = root_url + doc_page_urls[i]

doc_urls = []

def download_doc(doc_page_url):
    content = urllib.request.urlopen(doc_page_url).read()
    parsed = BeautifulSoup(content, 'lxml')
    link = [link for link in parsed.find_all('a') if ('file' in link['href'])][0]
    if link.getText() not in downloaded_docs and 'uploadfile' not in link['href']:
        with open(link.getText().split('.')[0] + link['href'][-4:], 'wb') as f:
            file_href = link['href']
            file_url = doc_page_url[:-6] + '/' + file_href[file_href.find('files'):]
            doc_content = urllib.request.urlopen(file_url).read()
            f.write(doc_content)
        print("Finished downloading " + link.getText())
    else:
        print(link.getText() + " already exists")

    fhead, ftail = link.getText().split('.')[0], link['href'][-4:]
    if 'pdf' in ftail:
        subprocess.call(['pdftotext', fhead+ftail, fhead+'.txt'])

        
for doc_page_url in doc_page_urls:
    download_doc(doc_page_url)

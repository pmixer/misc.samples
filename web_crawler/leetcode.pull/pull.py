# -*- coding: utf-8 -*-
# Author: Peter Huang, dreaming_hz@hotmail.com
# in reference to https://github.com/xapcloud/Code-Grab-from-Leetcode
# modified a lot to make it work due to updates of leetcode,like AJAX got used

import os
import re
import sys
import time
import json
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

DURATION = 2 # wait for 2 seconds due to AJAX
USERNAME = 'TODO'
PASSWORD = 'TODO'

cookie = None
s = requests.Session()
chrome_options = Options()
# chrome_options.add_argument('--headless')
# chrome_options.add_argument('--disable-gpu')
# TODO: where's your chromedriver? modify the path in below and set chrome like above
driver = webdriver.Chrome(chrome_options=chrome_options, executable_path='C:\chromedriver.exe')
driver.implicitly_wait(10)

headers_base = {
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language':'zh-CN,zh;q=0.8,en;q=0.6',
    'Connection':'keep-alive',
    'Host': 'leetcode.com',
    'Referer':'https://leetcode.com/accounts/login/',
    'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36'
}  

def login():
    '''
    login using requests module and save cookie
    '''
    url = "https://leetcode.com/accounts/login/"
    res = s.get(url=url,headers=headers_base)

    login_data={}
    login_data['remember'] ='on'
    login_data['login'] = USERNAME
    login_data['password'] = PASSWORD
    login_data['csrfmiddlewaretoken']=res.cookies['csrftoken']
    res = s.post(url,headers = headers_base,data=login_data, timeout=20)
    print('Login status: ' + str(res.status_code))
    return res.cookies

def get_bs(url):
    '''
    get the page using Chrome and parse by BeautifulSoup
    '''
    driver.get(url)
    time.sleep(DURATION) # ajax...
    bs = BeautifulSoup(driver.page_source, "html.parser")
    return bs       

def get_ac_content(problem_name):
    '''
    get the ac solution url
    '''
    url = 'https://leetcode.com/problems/'+ problem_name +'/submissions/'
    bs = get_bs(url)
    file_name_postfix = None
    # TODO, modify if you submit in more languages
    if 'python' in bs.text:
        file_name_postfix = '.py'
    elif 'mysql' in bs.text:
        file_name_postfix = ".sql"
    else:
        file_name_postfix = ".cpp"
    # hard-coded class name grasped using Chrome devtool
    ac_solutions = bs.find_all("a",{'class':'ac__1l1V'})
    if ac_solutions is None:
        return None
    # return the most recent submission
    return file_name_postfix, 'https://leetcode.com'+ ac_solutions[0]["href"]

def get_ac_code(url):
    '''
    grasp code by given submission url
    it has been saved in embeded javascript of submission page
    '''
    bs = get_bs(url)
    ac_code = bs.find_all('script')
    for i in range(len(ac_code)):
        if 'submissionCode' in ac_code[i].text:
            final_code = txt_wrap_by('submissionCode:','editCodeUrl:',ac_code[i].text)
            final_code = re.sub('/\\*.*\\*/','',final_code) # remove annotation to prevent chinese?
            break
    # in py3, if type(final_code) == 'bytes', directly decode, if it's str, do as below
    return final_code.encode('utf8').decode('unicode-escape')

def get_problem_content(name):
    '''
    Get problem description, not yet tested
    Recommend link to problem rather than steal it
    '''
    url = 'https://leetcode.com/problems/' + name
    try:
         page = driver.get(url)
         time.sleep(DURATION)
    except (requests.exceptions.ReadTimeout,requests.exceptions.ConnectTimeout):
            print('time out')
            return 'time out'
    problem_page = BeautifulSoup(driver.page_source, "html.parser")
    problem_contents = problem_page.select('.question-content')

    problem_text = ''
    if len(problem_contents) > 0:
        contents = problem_contents[0].find_all(['p','pre'])
        for ctt in contents:
            problem_text += ctt.get_text()
    page.close
    return problem_text
  
def txt_wrap_by(start_str, end, html):
    '''
    hard-coded, get the code from the js code by fetch 
    content between submissionCode: &  editCodeUrl:
    '''
    start = html.find(start_str)
    if start >= 0:
        start += len(start_str)
        end = html.find(end, start)
        if end >= 0:
            return html[start+2:end-5].strip()

def save_all(content, file_name_postfix):
    '''
    save the submission code
    '''
    name = 'leetcode_' + str(content['id'])+'_'+ str(content['name']) + '_grabbed' + file_name_postfix
    with open(name,'w', encoding='utf8') as f:
        f.write(content['code'])

# start execution   
if not os.path.exists('leetcode_submissions'):
    os.mkdir('leetcode_submissions')
os.chdir('leetcode_submissions') 

cookie = login()
raw_json = s.get(url='https://leetcode.com/api/problems/algorithms/', cookies=cookie).text
data_json = json.loads(raw_json)
problem_list = data_json['stat_status_pairs']

# Chrome login
driver.get('https://leetcode.com/accounts/login/')
time.sleep(DURATION) # to prevent from clicking the wrong div rather than login button
driver.find_element_by_id("username-input").clear()
driver.find_element_by_id("username-input").send_keys(USERNAME)
driver.find_element_by_id("password-input").clear()
driver.find_element_by_id("password-input").send_keys(PASSWORD)
time.sleep(DURATION)
driver.find_element_by_id("sign-in-button").click()
time.sleep(5)

# loop and download
for problem_json in problem_list:
    if problem_json['status'] == 'ac':
        problem_stat = problem_json['stat']
        problem_difficulty = problem_json['difficulty']['level']
        problem_paid = problem_json['paid_only']
        problem_name = problem_stat['question__title']
        problem_name_slug = problem_stat['question__title_slug'] 
        problem_id = problem_stat['question_id']
        problem_acs = problem_stat['total_acs']
        problem_submitted = problem_stat['total_submitted']

        print('writing {}:{} ...'.format(problem_id, problem_name))
        file_name_postfix, url = get_ac_content(problem_name_slug)
        if url is None:
            print('Problem {}:{} should have been submitted in the contest'.format(problem_id, problem_name))
        else:
            code = get_ac_code(url)
            # subject = get_problem_content(problem_name_slug) # do not want to fix it
            content = {'id': problem_id, 'title':problem_name,'name':problem_name_slug,'difficulty':problem_difficulty,'code':code}
            save_all(content, file_name_postfix)
        print('Done!')
        
print("ALL DONE, THX for using it!")

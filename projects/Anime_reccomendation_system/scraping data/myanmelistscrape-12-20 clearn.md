# Scraping My Anime list top anime for analysis

The goal of this project is to make first make a websraper to collect the myanime list data.  


how to scrape a table video: https://www.youtube.com/watch?v=G8ZJwhOsmTw

how to launch Jupyter notebook in browser from VS code. shortcut cmd+shift+p and type terminal. then type jupyter notebook into terminal.  
https://stackoverflow.com/questions/66627056/open-jupyternotebook-from-vscode-into-the-browser#:~:text=Type%3A%20jupyter%20notebook%20in%20vscode,in%20browser%20of%20your%20preference.


bs4 reference https://www.crummy.com/software/BeautifulSoup/bs4/doc/  


```python
# import data. 
import numpy as np
from bs4 import BeautifulSoup
import requests
import re
import requests
import lxml.html as lh
import pandas as pd
import pickle
import os 
```

### how to log in: 

refer to this youtube video for login procedues. https://www.youtube.com/watch?v=cVnYod9Fhko  
go to this website to copy paste into cell below https://curlconverter.com/  

progress check; 
1. this code works if I just change it to the next page, I still stay logged in 
2. Need to check if the session still runs if I don't change it. 


The first step in this process was to scrape the myanimelist.net to gather all the information about anime. The reaosn why myanimelist was chosen is because of it's abudance of data and the time it existed as a platform for users to track their anime watch. 

The first step was to log in to my specific account and gather all the information of my ratings. For this, the python get libaray was used and allows me to scrape the website after I logged in. 


```python
cookies = {
    'MALHLOGSESSID': '40f711a7f2899f435a08be7a2c43fbd5',
    'm_gdpr_mdl_20220817': '1',
    'MALSESSIONID': 'a0au8bvhl5ecu0ukiavl2eoa42',
    'is_logged_in': '1',
}

headers = {
    'authority': 'myanimelist.net',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-language': 'en-CA,en-GB;q=0.9,en-US;q=0.8,en;q=0.7',
    # 'cookie': 'MALHLOGSESSID=40f711a7f2899f435a08be7a2c43fbd5; m_gdpr_mdl_20220817=1; MALSESSIONID=a0au8bvhl5ecu0ukiavl2eoa42; is_logged_in=1',
    'referer': 'https://myanimelist.net/',
    'sec-ch-ua': '"Not?A_Brand";v="8", "Chromium";v="108", "Google Chrome";v="108"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
}

params = {
    '_location': 'mal_h_m',
}

response = requests.get('https://myanimelist.net/topanime.php?limit=50', params=params, cookies=cookies, headers=headers)
```

**Generate Ranking Table**

The function, generate ranking table accepts the link and other login credienials needed in order to access the log in. This would allow me to scrape the tables that have the top ranking anime. For the purpose. This data was scraped as of Dec 07,2022. As more anime get added to the system, the top rankings may flucuate but for the purpose of the project, data will be used once in concern of runtime. In the future, for this to be improved. Look into live recommednations using machine learning.  For this analysis, the top 10,000 anime by user ranking will be considered. This is because all though there are more than 20,000 anime in the entirely of the website. The lowest ranking aniime is ex-arm, having a overall rating of 2.91 and is in rank 13415. Any anime beyond that does not have enough user data.  
Would this lead to a user not being discovering a hidden gem?  

I dont think this would be the case. If a show has a public rating of below 6 it would mean that the show already lacks enough data. In other words, if not many people have heard of the show, and it's also rated bad, there is a good chance that this is a bad show. To accout for this, a score distribution can simply be plot to see the average score of the 10050 anime and adjust accordingly. 

The function works by extracting the information and then appedning it to a dataframe, which will then be stored in as a csv file for future access. 


```python

def generate_ranking_table(top_table_link, parms = params, cookies = cookies, headers = headers):
    
    response = requests.get(top_table_link, params=params, cookies=cookies, headers=headers)
    soup_login = BeautifulSoup(response.content,'html.parser')
    my_table_login = soup_login.find('table', {'class': 'top-ranking-table'})
    
    anime_overall_rank = [] 
    anime_title = [] 
    anime_link = [] 
    anime_id = [] 
    anime_public_rating = []
    anime_private_rating = []
    my_watch_status = []

    for i in my_table_login.find_all('tr',{'class': 'ranking-list'}):

        # find ranking first column 
        for j in i.find_all('td', {'class' : 'rank ac'}):
             anime_overall_rank.append(j.text.replace("\n", ""))

        for name in i.find_all('td', {'class' : 'title al va-t word-break'}):
            # anime name 
            anime_title.append(name.h3.text)
            # anime URL
            anime_link.append(name.a['href'])
            # anime unique id. 
            anime_id.append(name.a['href'].split('/')[4])   

        # find ranking first column 
        for j in i.find_all('td', {'class' : 'score ac fs14'}):
             anime_public_rating.append(j.text.replace("\n", ""))

        # find ranking first column 
        for j in i.find_all('td', {'class' : 'your-score ac fs14'}):
             anime_private_rating.append(j.text.replace("\n", "").strip())


        # find ranking first column 
        for j in i.find_all('td', {'class' : 'status'}):
             my_watch_status.append(j.text.replace("\n", "").strip())


    table_data_dic = {'Rank': anime_overall_rank, 
                     'Title' : anime_title, 
                     'link' : anime_link, 
                     'id' : anime_id, 
                     'public_score' : anime_public_rating, 
                     'prive_rating' : anime_private_rating, 
                     'watch_status' : my_watch_status} 



    top_anime_table = pd.DataFrame(data=table_data_dic)

    return top_anime_table

# generate_ranking_table('https://myanimelist.net/topanime.php?limit=50')
```


```python
# generate_ranking_table('https://myanimelist.net/topanime.php?limit=50')
```


```python
main_table_df_top_10k = generate_ranking_table('https://myanimelist.net/topanime.php')


for i in range(50,10050,50):
    link_to_anime_50_tab ='https://myanimelist.net/topanime.php?limit='+ str(i)
    next_50_table = generate_ranking_table(link_to_anime_50_tab)
    
    
    main_table_df_top_10k =main_table_df_top_10k.append(next_50_table)
    

```


```python
main_table_df_top_10k
```


```python
# import os 
# os.makedirs('/Users/rayen/Desktop/VScode', exist_ok=True) 
# main_table_df_top_10k.to_csv('/Users/rayen/Desktop/VScode/out.csv',index=False)
```


```python
# import all anime list from last section. 

top_anime_all_table = pd.read_csv('/Users/rayen/Desktop/VScode/out.csv')
top_anime_all_table.head()
```

## Scraping anime individual info

### Scraping stats page

In this section, two main things will be scraped from this entire website. The first one is going to be the anime information page. This contains information such as release date, anime genre's, production team and other relevant information. Mainly, the genere and release date will have some sort of correlation with the rating of the show. Also, production studios with higher standards / production buget may produce higher quality shows, which may influence the user's interest in the show.  

The second peice of information is the score distribution. Which can help in general data analytics and data exploration.  

A challenge for this portion was that the code was being interupted most likely due to runtime errors. In order to solve this problem, the code was broken up into smaller chunks. This data segment was then run and then pickled for storeage. 


```python
def scrape_anime_details(stats_page_link):    
    
    anime_id = int(stats_page_link.split('/')[4])
    
    stats_page_response = requests.get(stats_page_link)
    soup_login_an_ind_stat = BeautifulSoup(stats_page_response.content,'html.parser')
    
    anime_info_stats = {}

    for i in soup_login_an_ind_stat.find_all(class_ = 'spaceit_pad'): 

        dark_text = i.find(class_ = 'dark_text')

        if dark_text != None: 
            dark_title = dark_text.text
            #print(dark_title)
            light_text = i.text
            light_text = light_text.replace(str(dark_title), "")
            #print(light_text)

            anime_info_stats[dark_title] = light_text.replace("\n", "").strip().replace("  ", "")
        
    
    score_10_list = []
    score_10_count_list = [] 

    an_ind_stats_table = soup_login_an_ind_stat.find('table' , {'class' : 'score-stats'})

    regex_str = re.compile("score-label score-\d*")
    next_table = an_ind_stats_table.find_all(class_ = regex_str)

    for i in next_table:
        score_10 = score_10_list.append(int(i.text))

    for i in an_ind_stats_table.find_all('small'): 
        score_10_count = score_10_count_list.append(int(re.sub(r'[^0-9]', '', i.text)))


    score_distribution = dict(zip(score_10_list, score_10_count_list))

    return anime_id, anime_info_stats, score_distribution
```


```python
# create list of 50 anime to loop 
# this is created because of errors.

i1 = 0 
i2 = i1 +50  

listtest = [] 
while i1 < 10050: 
    listtest.append((i1,i2))
    i1 += 50  
    i2 = i1 + 50  
```


```python
# create a loop that creates anime info and score distribution 

temp_anime_info_list = {}

for start, end in listtest: 
    
    print(start, end)
    anime_acess_info = {} 

    for i in top_anime_all_table['link'][start:end]:

        anime_stats_link = (i + '/stats')
        anime_id_ind, anime_info_i, anime_score_dist_i = scrape_anime_details(anime_stats_link)

        anime_acess_info[anime_id_ind] = {'anime_info': anime_info_i ,'score_distribution': anime_score_dist_i}
    
    temp_anime_info_list[str(start) + '_' + str(end)] = anime_acess_info
        
    
```


```python

# with open('/Users/rayen/Desktop/VScode/anime_info.pkl', 'wb') as f:
#     pickle.dump(temp_anime_info_list, f)
             
with open('/Users/rayen/Desktop/VScode/anime_info.pkl', 'rb') as f:
    anime_info = pickle.load(f)
    
#anime_info
```


```python
all_anime_dict = {} 

dict_list_all = [loaded_dict1, loaded_dict2 ,loaded_dict3 ,loaded_dict4]


for load_dict_i in dict_list_all:
    
    for i in load_dict_i.values(): 

        for j in i:

            anime_id_i = j
            anime_value_i = i[j]

            all_anime_dict[anime_id_i] = anime_value_i

```


```python
# with open('/Users/rayen/Desktop/VScode/all_anime_stats_info.pkl', 'wb') as f:
#     pickle.dump(all_anime_dict, f)

with open('/Users/rayen/Desktop/VScode/all_anime_stats_info.pkl', 'rb') as f:
    all_anime_stats_info = pickle.load(f)
```


```python
all_anime_detail_info = {}
all_anime_score_dist = {}

for i in all_anime_stats_info:
    
    a_info_i = all_anime_stats_info[i]['anime_info']
    a_stat_i = all_anime_stats_info[i]['score_distribution']
    
    all_anime_detail_info[i] = a_info_i
    all_anime_score_dist[i] = a_stat_i
    
anime_info_frame = pd.DataFrame(all_anime_detail_info).transpose()
anime_stats_frame = pd.DataFrame(all_anime_score_dist).transpose()
```


```python
# anime_info_frame

# os.makedirs('/Users/rayen/Desktop/VScode', exist_ok=True) 
# anime_info_frame.to_csv('/Users/rayen/Desktop/VScode/anime_info_frame.csv',index=True)
```


```python
# anime_stats_frame

# os.makedirs('/Users/rayen/Desktop/VScode', exist_ok=True) 
# anime_info_frame.to_csv('/Users/rayen/Desktop/VScode/anime_stats_frame.csv',index=True)
```

### Imports


```python
top_anime_info_table_all = pd.read_csv('/Users/rayen/Desktop/VScode/anime_info_frame.csv')
top_anime_stat_table_all = pd.read_csv('/Users/rayen/Desktop/VScode/anime_stats_frame.csv')
```

# scraping reviews page


```python
def scrape_anime_reviews(stats_page_link):    
    
    anime_id = int(stats_page_link.split('/')[4])
    
    review_page_response = requests.get(stats_page_link)
    soup_login_an_ind_review_home = BeautifulSoup(review_page_response.content,'html.parser')
    
    review_array = [] 
    
    for i in soup_login_an_ind_review_home.find_all(class_ = 'body'):
        
        reviews_dict = {} 
        
        #reviewer 
        reviews_dict['reviewer'] = i.find(class_ = 'username').a.text
        reviews_dict['review_date'] = i.find(class_ = 'update_at').text
        reviews_dict['reviewer_profile'] = i.find(class_ = 'username').a['href']
        reviews_dict['review_full'] = i.find(class_ = 'open').a['href']
        reviews_dict['review_feeling'] = i.find(class_ = 'tags').text.replace("\n", "").replace(" ", "")
        
        review_array.append(reviews_dict)
        
        
    return anime_id , review_array
    
    
#ta, tb = scrape_anime_reviews('https://myanimelist.net/anime/5114/Fullmetal_Alchemist__Brotherhood/reviews')
```


```python
anime_reviews_dict = {}

counter = 5600 

for i in top_anime_all_table['link']:
    
    counter += 1
    
    anime_stats_link = (i + '/reviews')
    
    anime_id, review_array = scrape_anime_reviews(anime_stats_link)

    anime_reviews_dict[anime_id] = review_array
    
    if counter % 100 == 0:
        print(counter)
    

#anime_reviews_dict
```


```python
# with open('/Users/rayen/Desktop/VScode/all_anime_reviews.pkl', 'wb') as f:
#     pickle.dump(anime_reviews_dict, f)

with open('/Users/rayen/Desktop/VScode/all_anime_reviews.pkl', 'rb') as f:
    all_anime_reviews = pickle.load(f)

```


```python
all_anime_reviews_frame1 = pd.DataFrame.from_dict(all_anime_reviews, orient="index").stack().to_frame()
all_anime_reviews_frame1 = pd.DataFrame(all_anime_reviews_frame1[0].values.tolist(), index=all_anime_reviews_frame1.index)
all_anime_reviews_frame1
```

# recommendations page scrape 


```python
def scrape_anime_useerrec(stats_page_link):    
    
    anime_id = int(stats_page_link.split('/')[4])
    
    userrec_page_response = requests.get(stats_page_link)
    soup_login_an_ind_userrec_home = BeautifulSoup(userrec_page_response.content,'html.parser')
    
    
    
    rec_tab = soup_login_an_ind_userrec_home.find(class_ = 'rightside js-scrollfix-bottom-rel')
    tab1 = rec_tab.find_all('table')
    
    userrec_array = [] 

    for i in tab1: 
        
        userrec_dict = {}

        anime_i_review_link = i.find(style ="margin-bottom: 2px;").a.text
        anime_i_review_name = i.find(style ="margin-bottom: 2px;").a['href']
        anime_i_review_content = i.find(class_ ="spaceit_pad detail-user-recs-text").text
        
        anime_i_review_count = (len(i.find_all(class_ = 'borderClass bgColor1')) + len(i.find_all(class_ = 'borderClass bgColor2')))
        
        userrec_dict['review_link'] = anime_i_review_link
        userrec_dict['review_name'] = anime_i_review_name
        userrec_dict['review_content'] = anime_i_review_content
        userrec_dict['review_count'] = anime_i_review_count
        
        userrec_array.append(userrec_dict)
        

    return anime_id, userrec_array  

#userrec_id, userrec_arr = scrape_anime_useerrec('https://myanimelist.net/anime/1575/Code_Geass__Hangyaku_no_Lelouch/userrecs')
```


```python
anime_userrec_dict1 = {}

counter = 0

for i in top_anime_all_table['link']:
    
    counter += 1
    
    anime_stats_link = (i + '/userrecs')
    
    anime_id, userrec_arr = scrape_anime_useerrec(anime_stats_link)

    anime_userrec_dict1[anime_id] = userrec_arr
    
    if counter % 50 == 0:
        print(counter)
    

#anime_reviews_dict
```


```python
# with open('/Users/rayen/Desktop/VScode/anime_userrec_dict1.pkl', 'wb') as f:
#     pickle.dump(anime_userrec_dict4, f)
```

### optional selenium reccomendations page scrape


```python
import selenium

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
#chrome_options.add_argument("--disable-extensions")
#chrome_options.add_argument("--disable-gpu")
#chrome_options.add_argument("--no-sandbox") # linux only
chrome_options.add_argument("--headless")

# Establish chrome driver and go to report site URL
url = "https://myanimelist.net/reviews.php?id=740"
driver = webdriver.Chrome(options=chrome_options)
driver.get(url)


#driver.quit()
```


```python
user_review_page_response = requests.get('https://myanimelist.net/reviews.php?id=5262', params=params, cookies=cookies, headers=headers)
user_an_ind_review_page = BeautifulSoup(user_review_page_response.content,'html.parser')
```


```python
driver.find_element(By.CLASS_NAME,'num').text
driver.find_element(By.XPATH,'//*[@id="content"]/div[2]/div[2]/div[2]/div[7]/div[1]/span').text
driver.find_element(By.XPATH,'//*[@id="content"]/div[2]/div[2]/div[2]/div[7]/div[2]/span').text
driver.find_element(By.XPATH,'//*[@id="content"]/div[2]/div[2]/div[2]/div[7]/div[3]/span').text
driver.find_element(By.XPATH,'//*[@id="content"]/div[2]/div[2]/div[2]/div[8]/div[1]/span').text
```

# Details Page


```python
def scrape_anime_details_all(detail_page_link):    
    
    anime_id = int(detail_page_link.split('/')[4])
    
    review_page_response = requests.get(detail_page_link)
    soup_login_an_ind_review_home = BeautifulSoup(review_page_response.content,'html.parser')
    
    # find description 
    description = soup_login_an_ind_review_home.find('p' , {'itemprop' : 'description'})
    description_text = description.text.replace('\n','').replace('\r','').replace('[Written by MAL Rewrite]','')
    
    # find character and VA 
    character_va_table = soup_login_an_ind_review_home.find_all(class_ = 'detail-characters-list clearfix')
    character_table = soup_login_an_ind_review_home.find_all(class_ = 'h3_characters_voice_actors')
    character_va_table = soup_login_an_ind_review_home.find_all(class_ = 'va-t ar pl4 pr4')

    character_array = [] 
    va_array = [] 

    for i in character_table:
        character_array.append(i.a['href'])

    for i in character_va_table: 
        va_array.append(i.a['href'])
        
    # find staff 
    staff_table = soup_login_an_ind_review_home.find_all(class_ = 'ac borderClass')
    all_va_staff_list = []

    for i in staff_table:
        all_va_staff_list.append(i.a['href'])

    all_va_staff_series = pd.Series(all_va_staff_list)
    staff_list = list(all_va_staff_series[~all_va_staff_series.isin(character_array + va_array)])
    
    # find related anime
    my_information_table = soup_login_an_ind_review_home.find('table', {'class': 'anime_detail_related_anime'})
    
    if my_information_table is not None: 

        related_anime_list = []
        for i in my_information_table.find_all('tr'): 
            for j in (i.find_all('a')):
                related_anime_list.append(j['href'])
    
        related_anime_list_no_dup = list(pd.Series(related_anime_list).drop_duplicates())
    else: 
        related_anime_list = []
        related_anime_list_no_dup = []


    # add to dictionary
    
    all_details_dict = {}
    
    all_details_dict['description'] = description_text
    all_details_dict['characters'] = character_array
    all_details_dict['voice_actors'] = va_array
    all_details_dict['staff'] = staff_list
    all_details_dict['related_anime'] = related_anime_list_no_dup

    return anime_id, all_details_dict

```


```python
anime_details_all_dict1 = {}

counter = 0 

for i in top_anime_all_table['link']:
    
    counter += 1
    
    anime_detail_link = (i)
    print(anime_detail_link)
    
    anime_id, all_details_dict = scrape_anime_details_all(anime_detail_link)

    anime_details_all_dict1[anime_id] = all_details_dict
    
    if counter % 100 == 0:
        print(counter)
    

```


```python
# with open('/Users/rayen/Desktop/VScode/anime_details_all_dict.pkl', 'wb') as f:
#     pickle.dump(anime_details_all_dict1, f)
```

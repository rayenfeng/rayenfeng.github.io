# Webscraping User Rating Data

code by: Rayen Feng  
Date: Dec 28th 2022  

This notebook scrapes user data from myanimelist.net Then, it will store in a dataframe containing user reivew as records. 


```python
import numpy as np
from bs4 import BeautifulSoup
import requests
import re
import requests
import lxml.html as lh
import pandas as pd
import pickle
import os 
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'
```


```python
def find_user_profile_link():
    
    user_id_link = 'https://myanimelist.net/users.php'
    response = requests.get(user_id_link)
    soup_login = BeautifulSoup(response.content,'html.parser')
    
    list_of_users = [] 
    
    for i in soup_login.find_all('td',{'class': 'borderClass'}):
        list_of_users.append(i.a['href'])
        
    return list_of_users

```


```python
list_of_all_20k_users = [] 

while len(list_of_all_20k_users) < 1000:
    
    user_list = find_user_profile_link()
    list_of_all_20k_users.append(user_list)
    
```


```python
unique_user_list = pd.Series(np.array(list_of_all_20k_users).flatten()).unique()
```


```python
unique_user_series = pd.Series(unique_user_list)

unique_user_random = unique_user_series.apply(lambda x: str(x).replace('/profile/', ''))

unique_user_random
```


```python
with open('/Users/rayen/Documents/code/anime_rec_project/data_sources_pickle/all_anime_reviews.pkl', 'rb') as f:
    reviews = pickle.load(f)
```


```python
all_anime_reviews_frame = pd.DataFrame.from_dict(reviews, orient="index").stack().to_frame()
all_anime_reviews_frame = pd.DataFrame(all_anime_reviews_frame[0].values.tolist(), index=all_anime_reviews_frame.index)
all_anime_reviews_frame.head()
```


```python
reviwer_list_users = pd.Series(all_anime_reviews_frame['reviewer'].values)

unique_user_random = unique_user_series.apply(lambda x: str(x).replace('/profile/', ''))
```


```python
all_user_list = reviwer_list_users.append(unique_user_random)

final_44k_user_list = pd.Series(all_user_list.unique())
```


```python
final_44k_user_list
```


```python
# os.makedirs('/Users/rayen/Desktop/VScode', exist_ok = True) 
# final_44k_user_list.to_csv('/Users/rayen/Desktop/VScode/final_44k_user_list.csv',index=False)
```


```python
def find_two_user_score(username_1, username_2):    
    
    
    user_table = 'https://myanimelist.net/shared.php?u1={user1}&u2={user2}'.format(user1 = username_1, user2 = username_2)
    #print(user_table)
    
    user1_amime_scores = {} 
    user2_amime_scores = {} 

    response = requests.get(user_table)
    soup_login = BeautifulSoup(response.content,'html.parser')

    tables = soup_login.find_all('table')
    
    if len(tables) != 0:

        shared_animes = tables[0]
        unique_anime1 = tables[1]
        unique_anime2 = tables[2]


        ### scraped shared list 

        for i in shared_animes.find_all('tr')[1:-2]:

            # anime title 
            anime_title_shared = i.a['href']
            # first user score 
            anime_user1_score_shared = (i.find_all('td')[1].text)
            # second user score 
            anime_user2_score_shared = (i.find_all('td')[2].text)

            user1_amime_scores[anime_title_shared] = anime_user1_score_shared
            user2_amime_scores[anime_title_shared] = anime_user2_score_shared

        ### scrape anime unique to user 1: 

        for i in unique_anime1.find_all('tr')[1:-1]:
            anime_title_1 = (i.a['href'])
            anime_user1_score = (i.find_all('td')[1].text)

            user1_amime_scores[anime_title_1] = anime_user1_score

        ## scrape anime unique to user 2: 

        for i in unique_anime2.find_all('tr')[1:-1]:

            anime_title_2 = (i.a['href'])
            anime_user2_score = (i.find_all('td')[1].text)
            user2_amime_scores[anime_title_2] = anime_user2_score
        
        
    return user1_amime_scores, user2_amime_scores
```


```python
all_user_anime_dict = {} 
initial_paired_userids = zip(final_44k_user_list[::2], final_44k_user_list[1::2])

counter = 0 

for uname1, uname2 in initial_paired_userids:
    
    counter += 1 
    if counter % 100 == 0:
        print(counter)
        print(uname1, uname2 )
    
    user1_anime_list, user1_anime_list = find_two_user_score(uname1, uname2)
    
    all_user_anime_dict[uname1] = user1_anime_list
    all_user_anime_dict[uname2] = user1_anime_list
        
```


```python
# with open('/Users/rayen/Desktop/VScode/all_user_anime_dict.pkl', 'wb') as f:
#     pickle.dump(all_user_anime_dict, f)
             
```


```python
ani_user_rat_frame = pd.DataFrame.from_dict(all_user_anime_dict, orient="index").stack().to_frame()
ani_user_rat_frame = pd.DataFrame(ani_user_rat_frame[0].values.tolist(), index=ani_user_rat_frame.index)

ani_user_rat_frame
```


```python
all_user_anime_ratings = ani_user_rat_frame.reset_index(level=[0,1])
```

### all_user_anime_ratings


```python
# os.makedirs('/Users/rayen/Desktop/VScode', exist_ok = True) 
# all_user_anime_ratings.to_csv('/Users/rayen/Desktop/VScode/all_user_anime_ratings.csv',index=False)
```


```python
with open('/Users/rayen/Desktop/VScode/all_user_anime_dict.pkl', 'rb') as f:
    all_user_anime_dict = pickle.load(f)
```

## scraping more user values 


```python
user_names_private_assumed = [] 

for key, value in all_user_anime_dict.items():
    
    if value == {}: 
        
        user_names_private_assumed.append(key)
        
        
    
    
```


```python
len(all_user_anime_dict)
```


```python
len(user_names_private_assumed)
```


```python
len(all_user_anime_dict) - len(user_names_private_assumed)
```


```python
# returns true if account is public and false if the account is private. 

def find_account_public(username_1): 
    

    user_table = 'https://myanimelist.net/animelist/{username}'.format(username = username_1)


    response = requests.get(user_table)
    soup_login = BeautifulSoup(response.content,'html.parser')
    
    check_bad_result = soup_login.find('div', {'class' : 'badresult'})
    
    #print(check_bad_result)
    
    if check_bad_result == None:

        return username_1 
    
    else:
        
        return None  
```


```python
public_profile_usernames = [] 

counter = 0 

for i in user_names_private_assumed:
    
    counter += 1 
    
    if counter % 100 == 0:
        print(counter)
        print(i)

    account_status = find_account_public(i)
    public_profile_usernames.append(account_status)
    
```


```python
series_profiles = pd.Series(public_profile_usernames)
```


```python
public_users_pf = series_profiles.dropna()
```


```python
# os.makedirs('/Users/rayen/Desktop/VScode', exist_ok = True) 
# public_users_pf.to_csv('/Users/rayen/Desktop/VScode/public_users_pf_10k_add.csv',index=False)
```


```python
public_users_list = pd.read_csv('/Users/rayen/Desktop/VScode/public_users_pf_10k_add.csv')
```


```python
even_public_user = public_users_list[:-1]['0'].to_list()
even_public_user
```


```python
all_user_anime_dict2 = {} 
initial_paired_userids2 = zip(even_public_user[::2], even_public_user[1::2])

counter = 0 

for uname1, uname2 in initial_paired_userids2:
    
    counter += 1 
    if counter % 100 == 0:
        print(counter)
        print(uname1, uname2 )
    
    user1_anime_list, user2_anime_list = find_two_user_score(uname1, uname2)
    
    all_user_anime_dict2[uname1] = user1_anime_list
    all_user_anime_dict2[uname2] = user2_anime_list
```


```python
len(all_user_anime_dict2.keys())
```


```python
# with open('/Users/rayen/Desktop/VScode/all_user_anime_dict2.pkl', 'wb') as f:
#     pickle.dump(all_user_anime_dict2, f)
```


```python
ani_user_rat_frame2 = pd.DataFrame.from_dict(all_user_anime_dict2, orient="index").stack().to_frame()
ani_user_rat_frame2 = pd.DataFrame(ani_user_rat_frame2[0].values.tolist(), index = ani_user_rat_frame2.index)

ani_user_rat_frame2
```


```python
all_user_anime_ratings2 = ani_user_rat_frame2.reset_index(level=[0,1])
```


```python
all_user_anime_ratings2
```


```python
len(all_user_anime_ratings2['level_0'].unique())
```


```python
# os.makedirs('/Users/rayen/Desktop/VScode', exist_ok = True) 
# all_user_anime_ratings2.to_csv('/Users/rayen/Desktop/VScode/all_user_anime_ratings2.csv',index=False)
```


```python
all_user_anime_ratings = ani_user_rat_frame.reset_index(level=[0,1])
```


```python
# os.makedirs('/Users/rayen/Desktop/VScode', exist_ok = True) 
# all_user_anime_ratings.to_csv('/Users/rayen/Desktop/VScode/all_user_anime_ratings.csv',index=False)
```

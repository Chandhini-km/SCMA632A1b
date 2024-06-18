#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


os.chdir('C:\\Users\\Chand\\Downloads\\Assignment 1b')


# In[6]:


ipl_bbb = pd.read_csv('IPL_ball_by_ball_updated till 2024.csv',low_memory=False)


# In[7]:


ipl_salary = pd.read_excel('IPL SALARIES 2024.xlsx')


# In[8]:


ipl_salary.head(2)


# ### Arranging the Data

# In[9]:


grouped_data = ipl_bbb.groupby(['Season', 'Innings No', 'Striker','Bowler']).agg({'runs_scored': sum, 'wicket_confirmation':sum}).reset_index()


# In[10]:


player_runs = grouped_data.groupby(['Season', 'Striker'])['runs_scored'].sum().reset_index()
player_wickets = grouped_data.groupby(['Season', 'Bowler'])['wicket_confirmation'].sum().reset_index()


# In[11]:


player_runs[player_runs['Season']=='2023'].sort_values(by='runs_scored',ascending=False)


# In[ ]:





# ### Top 3 run getters and wicket takers in each ipl round

# In[12]:


top_run_getters = player_runs.groupby('Season').apply(lambda x: x.nlargest(3, 'runs_scored')).reset_index(drop=True)
bottom_wicket_takers = player_wickets.groupby('Season').apply(lambda x: x.nlargest(3, 'wicket_confirmation')).reset_index(drop=True)
print("Top Three Run Getters:")
print(top_run_getters)
print("Top Three Wicket Takers:")
print(bottom_wicket_takers)


# In[13]:


ipl_year_id = pd.DataFrame(columns=["id", "year"])
ipl_year_id["id"] = ipl_bbb["Match id"]
ipl_year_id["year"] = pd.to_datetime(ipl_bbb["Date"], dayfirst=True).dt.year


# In[14]:


#create a copy of ipl_bbbc dataframe
ipl_bbbc= ipl_bbb.copy()


# In[15]:


ipl_bbbc['year'] = pd.to_datetime(ipl_bbb["Date"], dayfirst=True).dt.year


# In[16]:


ipl_bbbc[["Match id", "year", "runs_scored","wicket_confirmation","Bowler",'Striker']].head()


# ### Fitting the most appropriate distribution

# In[17]:


import scipy.stats as st

def get_best_distribution(data):
    dist_names = ['alpha','beta','betaprime','burr12','crystalball',
                  'dgamma','dweibull','erlang','exponnorm','f','fatiguelife',
                  'gamma','gengamma','gumbel_l','johnsonsb','kappa4',
                  'lognorm','nct','norm','norminvgauss','powernorm','rice',
                  'recipinvgauss','t','trapz','truncnorm']
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)
        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))
    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value
    print("\nBest fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))
    return best_dist, best_p, params[best_dist]


# In[18]:


total_run_each_year = ipl_bbbc.groupby(["year", "Striker"])["runs_scored"].sum().reset_index()


# In[19]:


total_run_each_year


# In[20]:


total_run_each_year.sort_values(["year", "runs_scored"], ascending=False, inplace=True)
print(total_run_each_year)


# In[21]:


total_run_each_year.head(12)


# In[22]:


list_top_batsman_last_three_year = {}
for i in total_run_each_year["year"].unique()[:3]:
    list_top_batsman_last_three_year[i] = total_run_each_year[total_run_each_year.year == i][:3]["Striker"].unique().tolist()


# In[23]:


list_top_batsman_last_three_year


# In[24]:


import warnings
warnings.filterwarnings('ignore')
runs = ipl_bbbc.groupby(['Striker','Match id'])[['runs_scored']].sum().reset_index()

for key in list_top_batsman_last_three_year:
    for Striker in list_top_batsman_last_three_year[key]:
        print("************************")
        print("year:", key, " Batsman:", Striker)
        get_best_distribution(runs[runs["Striker"] == Striker]["runs_scored"])
        print("\n\n")


# In[32]:


import warnings
warnings.filterwarnings('ignore')

runs = ipl_bbbc.groupby(['Striker','Match id'])[['runs_scored']].sum().reset_index()

# Choose the bowler you want to analyze (replace with desired bowler name)
chosen_Striker = "SV Samson"  # Replace with your chosen bowler's name

print("")
print("Best fit distribution for wickets taken by:", chosen_Striker)
get_best_distribution(runs[runs["Striker"] == chosen_Striker]["runs_scored"])
print("\n\n")


# In[25]:


total_wicket_each_year = ipl_bbbc.groupby(["year", "Bowler"])["wicket_confirmation"].sum().reset_index()


# In[26]:


total_wicket_each_year.sort_values(["year", "wicket_confirmation"], ascending=False, inplace=True)
print(total_wicket_each_year)


# In[27]:


list_top_bowler_last_three_year = {}
for i in total_wicket_each_year["year"].unique()[:3]:
    list_top_bowler_last_three_year[i] = total_wicket_each_year[total_wicket_each_year.year == i][:3]["Bowler"].unique().tolist()
list_top_bowler_last_three_year


# In[28]:


import warnings
warnings.filterwarnings('ignore')
wickets = ipl_bbbc.groupby(['Bowler','Match id'])[['wicket_confirmation']].sum().reset_index()

for key in list_top_bowler_last_three_year:
    for bowler in list_top_bowler_last_three_year[key]:
        print("************************")
        print("year:", key, " Bowler:", bowler)
        get_best_distribution(wickets[wickets["Bowler"] == bowler]["wicket_confirmation"])
        print("\n\n")


# In[29]:


R2024 =total_run_each_year[total_run_each_year['year']==2024]


# In[30]:


total_run_each_year


# In[31]:


total_wicket_each_year


# In[32]:


R2024 = total_run_each_year[total_run_each_year['year']==2024]


# In[33]:


W2024 = total_wicket_each_year[total_wicket_each_year['year']==2024]


# In[34]:


R2024


# In[35]:


W2024


# In[36]:


#pip install fuzzywuzzy


# In[37]:


pip install fuzzywuzzy


# In[39]:


from fuzzywuzzy import process

# Convert to DataFrame
df_salary = ipl_salary.copy()
df_runs = R2024.copy()
df_wickets = W2024.copy()

# Function to match names
def match_names(name, names_list):
    match, score = process.extractOne(name, names_list)
    return match if score >= 80 else None  # Use a threshold score of 80

# Create a new column in df_salary with matched names from df_runs
df_salary['Matched_Player'] = df_salary['Player'].apply(lambda x: match_names(x, df_runs['Striker'].tolist()))

# Merge the DataFrames on the matched names
df_merged = pd.merge(df_salary, df_runs, left_on='Matched_Player', right_on='Striker')


# In[40]:


# Convert to DataFrame
df_salary = ipl_salary.copy()
df_runs = R2024.copy()
df_wickets = W2024.copy()

# Function to match names
def match_names(name, names_list):
    match, score = process.extractOne(name, names_list)
    return match if score >= 80 else None  # Use a threshold score of 80

# Create a new column in df_salary with matched names from df_runs
df_salary['Matched_Player'] = df_salary['Player'].apply(lambda x: match_names(x, df_wickets['Bowler'].tolist()))

# Merge the DataFrames on the matched names
df_merged1 = pd.merge(df_salary, df_wickets, left_on='Matched_Player', right_on='Bowler')


# In[41]:


df_merged1.columns


# In[42]:


df_merged.columns


# In[43]:


df_merged1.info()


# In[44]:


df_merged.info()


# In[45]:


# Calculate the correlation
correlation = df_merged['Rs'].corr(df_merged['runs_scored'])

print("Correlation between Salary and Runs:", correlation)


# In[46]:


df_merged


# In[47]:


df_merged1


# In[ ]:


merged_df


# In[ ]:


merged_df.columns.tolist()


# In[ ]:


from fuzzywuzzy import process  # Assuming fuzzywuzzy is already imported

# Filter last three years' data (assuming 'year' column exists)
last_three_years_data = merged_df[merged_df['year_x'].isin([2022, 2023, 2024])]

# Function to get top players
def get_top_players(data, n, role):
  if role == "Matched_Player":
    top_players = last_three_years_data.groupby('Matched_Player')['runs_scored'].sum().sort_values(ascending=False).head(n)
  elif role == "Bowler":
    top_players = last_three_years_data.groupby('Bowler')['wicket_confirmation'].sum().sort_values(ascending=False).head(n)
  else:
    raise ValueError("Invalid role. Choose 'Matched_Player' or 'Bowler'")
  return top_players.index.tolist()

# Get top 10 batsmen and bowlers
top_10_batsmen = get_top_players(last_three_years_data, 10, "Matched_Player")
top_10_bowlers = get_top_players(last_three_years_data, 10, "Bowler")

# Calculate average salary (assuming 'Rs_x' column holds salary information)
avg_salary_batsmen = last_three_years_data[last_three_years_data['Matched_Player'].isin(top_10_batsmen)]['Rs_x'].mean()
avg_salary_bowlers = last_three_years_data[last_three_years_data['Bowler'].isin(top_10_bowlers)]['Rs_x'].mean()

# Calculate difference
difference = avg_salary_batsmen - avg_salary_bowlers

# Print result
print("Significant Difference in Salary (Batsmen - Bowlers):", difference)


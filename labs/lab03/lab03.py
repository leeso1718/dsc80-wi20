
import os

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def car_null_hypoth():
    """
    Returns a list of valid null hypotheses.
    
    :Example:
    >>> set(car_null_hypoth()) <= set(range(1,11))
    True
    """
    
    return [3,6]


def car_alt_hypoth():
    """
    Returns a list of valid alternative hypotheses.
    
    :Example:
    >>> set(car_alt_hypoth()) <= set(range(1,11))
    True
    """
    return [1,4]


def car_test_stat():
    """
    Returns a list of valid test statistics.
    
    :Example:
    >>> set(car_test_stat()) <= set(range(1,5))
    True
    """
    return [2,4]


def car_p_value():
    """
    Returns an integer corresponding to the correct explanation.
    
    :Example:
    >>> car_p_value() in [1,2,3,4,5]
    True
    """
    return 3


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------

def clean_apps(df):
    '''
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_apps(df)
    >>> len(cleaned) == len(df)
    True
    >>> cleaned.Reviews.dtype == int
    True
    '''
    #convert Reviews to type int
    df = df.astype({'Reviews': 'int'})
    
    #strip all letters from the Size
    df['Size'] = df['Size'].apply(lambda x : (float(x.replace("M", '')) *1000) if ('M' in x) else float(x.replace("k", '')))
    
    #strip + from Installs, remove the commas, and convert it to type int
    df['Installs'] = df['Installs'].apply(lambda x : x.replace(',','').replace("+",''))
    
    #Type -> binary 1, 0
    df['Type'] = df['Type'].apply(lambda x : x.replace('Free', '1').replace('Paid','0')).astype(int)
    
    #Price -> numeric
    df['Price'] = pd.to_numeric(df['Price'].apply(lambda x : x.replace('$', '')))

    #Last Update -> int
    df['Last Updated'] = df['Last Updated'].apply(lambda x : x[-4:]).astype(int)

    return df

def store_info(cleaned):
    '''
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_apps(df)
    >>> info = store_info(cleaned)
    >>> len(info)
    4
    >>> info[2] in cleaned.Category.unique()
    True
    '''
    #installs
    over_hundred = cleaned['Last Updated'].value_counts()>100
    group = cleaned.groupby(['Last Updated']).median()
    highest_installs = group[over_hundred.sort_index()].sort_values(by = ['Rating'], ascending = False).index[0]
    
    #Content Rating
    highest_rating = cleaned.groupby(['Content Rating']).min().sort_values(by = ['Rating'],ascending = False).index[0]
    
    #Category with highest average price
    category_price = cleaned.groupby(['Category']).mean().sort_values(by = ['Price'],ascending = False).index[0]
    
    #category with lowest average rating, among apps with at least 1000 reviews
    category_rating = cleaned[cleaned['Reviews'] > 1000].groupby(['Category']).mean().sort_values(by = ['Rating']).index[0]
    
    return [highest_installs, highest_rating, category_price, category_rating]

# ---------------------------------------------------------------------
# Question 3
# ---------------------------------------------------------------------

def standard_units(any_numbers):
    "Convert any array of numbers to standard units."
    return (any_numbers['Reviews'] - (any_numbers['Reviews']).mean())/(any_numbers['Reviews']).std()

def std_reviews_by_app_cat(cleaned):
    """
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> play = pd.read_csv(fp)
    >>> clean_play = clean_apps(play)
    >>> out = std_reviews_by_app_cat(play)
    >>> set(out.columns) == set(['Category', 'Reviews'])
    True
    >>> np.all(abs(out.select_dtypes(include='number').mean()) < 10**-7)  # standard units should average to 0!
    True
    """
    
    middle_step = cleaned.groupby('Category').apply(lambda x : standard_units(x)).to_frame()
    #middle_step['Rating'].reindex(middle_step.keys())
    final_step = middle_step.reset_index(level =[0,1])[['Category', 'Reviews']]
    return final_step


def su_and_spread():
    """
    >>> out = su_and_spread()
    >>> len(out) == 2
    True
    >>> out[0].lower() in ['medical', 'family', 'equal']
    True
    >>> out[1] in ['ART_AND_DESIGN', 'AUTO_AND_VEHICLES', 'BEAUTY',\
       'BOOKS_AND_REFERENCE', 'BUSINESS', 'COMICS', 'COMMUNICATION',\
       'DATING', 'EDUCATION', 'ENTERTAINMENT', 'EVENTS', 'FINANCE',\
       'FOOD_AND_DRINK', 'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME',\
       'LIBRARIES_AND_DEMO', 'LIFESTYLE', 'GAME', 'FAMILY', 'MEDICAL',\
       'SOCIAL', 'SHOPPING', 'PHOTOGRAPHY', 'SPORTS', 'TRAVEL_AND_LOCAL',\
       'TOOLS', 'PERSONALIZATION', 'PRODUCTIVITY', 'PARENTING', 'WEATHER',\
       'VIDEO_PLAYERS', 'NEWS_AND_MAGAZINES', 'MAPS_AND_NAVIGATION']
    True
    """
    return ['equal', 'GAME']


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


def read_survey(dirname):
    """
    read_survey combines all the survey*.csv files into a singular DataFrame
    :param dirname: directory name where the survey*.csv files are
    :returns: a DataFrame containing the combined survey data
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> out = read_survey(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> len(out)
    5000
    >>> read_survey('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """
    df1 = os.listdir(dirname)
    lst = []
    for i in df1:
        dataframe = os.path.join(dirname, i)
        show = pd.read_csv(dataframe)
        show.columns = show.columns.str.lower().str.replace('_', " ")
        show = show[["first name", "last name", "current company", "job title", "email","university"]]
        lst.append(show)
    concat_all = pd.concat([lst[0], lst[1],lst[2],lst[3],lst[4]])
    return concat_all


def com_stats(df):
    """
    com_stats 
    :param df: a DataFrame containing the combined survey data
    :returns: a list containing the most common first name, job held, 
    university attended, and current company
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> df = read_survey(dirname)
    >>> out = com_stats(df)
    >>> len(out)
    4
    >>> all([isinstance(x, str) for x in out])
    True
    """
    #emails end in ".com"
    email_com = df[df['email'].str.endswith('.com', na = False)]
     
    #the most common first name
    first_name = email_com['first name'].value_counts().index[0]
    
    #job held
    job = email_com['job title'].value_counts().index[0]
    #university attended
    univ = email_com['university'].value_counts().index[0]
    
    #current company
    company = email_com['current company'].value_counts().index[0]
    
    return [first_name, job, univ, company]


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def combine_surveys(dirname):
    """
    combine_surveys takes in a directory path 
    (containing files favorite*.csv) and combines 
    all of the survey data into one DataFrame, 
    indexed by student ID (a value 0 - 1000).

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> out = combine_surveys(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> out.shape
    (1000, 6)
    >>> combine_surveys('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """
    df1 = os.listdir(dirname)
    lst = []
    for i in df1:
        dataframe = os.path.join(dirname, i)
        show = pd.read_csv(dataframe)
        show = show.set_index('id')
        lst.append(show)
    merge_all = pd.concat([lst[0], lst[1],lst[2],lst[3],lst[4], lst[5]], axis =1)
    
    return merge_all
    
def check_credit(df):
    """
    check_credit takes in a DataFrame with the 
    combined survey data and outputs a DataFrame 
    of the names of students and how many extra credit 
    points they would receive, indexed by their ID (a value 0-1000)

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> df = combine_surveys(dirname)
    >>> out = check_credit(df)
    >>> out.shape
    (1000, 2)
    """

    df['extra points'] = (df.apply(lambda x: x.count(), axis=1) > (5*0.75)).astype(int)
    at_least_one = ((df.apply(lambda x: x.count(), axis=1) > 1).value_counts().values/len(df)) > .09
    df['extra points'] = df['extra points'].apply(lambda x: 5 if x == 1 else 0).apply(lambda y : y +1 if at_least_one == True else y)
    
    return df[['extra points', 'name']]

# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------


def at_least_once(pets, procedure_history):
    """
    How many pets have procedure performed at this clinic at least once.

    :Example:
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = at_least_once(pets, procedure_history)
    >>> out < len(pets)
    True
    """
    return pets.merge(procedure_history)['PetID'].count()



def pet_name_by_owner(owners, pets):
    """
    pet names by owner

    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> out = pet_name_by_owner(owners, pets)
    >>> len(out) == len(owners)
    True
    >>> 'Sarah' in out.index
    True
    >>> 'Cookie' in out.values
    True
    """
    merge_df = pd.merge(owners, pets, on = "OwnerID")

    sr = merge_df.groupby(['OwnerID','Name_x']).apply(lambda x : combine(x)).droplevel(0)
    sr = sr.apply(lambda x: x if len(x) >1 else x[0])
    sr = sr.rename_axis("OwnerID")
    return sr

def combine (df):
    return df['Name_y'].tolist()

def total_cost_per_owner(owners, pets, procedure_history, procedure_detail):
    """
    total cost per owner

    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_detail_fp = os.path.join('data', 'pets', 'ProceduresDetails.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')

    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_detail = pd.read_csv(procedure_detail_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = total_cost_per_owner(owners, pets, procedure_history, procedure_detail)
    >>> set(out.index) <= set(owners['OwnerID'])
    True
    """
    
    procedure = pd.merge(procedure_detail, procedure_history, on = "ProcedureType")
    whole_df = pd.merge(procedure, pets, on = "PetID")
    total_price = whole_df.groupby('OwnerID').sum()['Price']
    return total_price



# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!


GRADED_FUNCTIONS = {
    'q01': [
        'car_null_hypoth', 'car_alt_hypoth',
        'car_test_stat', 'car_p_value'
    ],
    'q02': ['clean_apps', 'store_info'],
    'q03': ['std_reviews_by_app_cat','su_and_spread'],
    'q04': ['read_survey', 'com_stats'],
    'q05': ['combine_surveys', 'check_credit'],
    'q06': ['at_least_once', 'pet_name_by_owner', 'total_cost_per_owner']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" % (q, elt)
                raise Exception(stmt)

    return True

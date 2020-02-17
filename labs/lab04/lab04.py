
import os

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


def latest_login(login):
    """Calculates the latest login time for each user
    :param login: a dataframe with login information
    :return: a dataframe with latest login time for
    each user indexed by "Login Id"
    >>> fp = os.path.join('data', 'login_table.csv')
    >>> login = pd.read_csv(fp)
    >>> result = latest_login(login)
    >>> len(result)
    433
    >>> result.loc[381, "Time"].hour > 12
    True
    """
    login['Time'] = pd.to_datetime(login['Time'])
    login['Time'] = login['Time'].dt.time
    df = login.groupby('Login Id').max()
    return df

# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------


def smallest_ellapsed(login):
    """
    Calculates the the smallest time elapsed for each user.
    :param login: a dataframe with login information but without unique IDs
    :return: a dataframe, indexed by Login ID, containing 
    the smallest time elapsed for each user.
    >>> fp = os.path.join('data', 'login_table.csv')
    >>> login = pd.read_csv(fp)
    >>> result = smallest_ellapsed(login)
    >>> len(result)
    238
    >>> 18 < result.loc[1233, "Time"].days < 23
    True
    """
    #groupby diff
    login['Time'] = pd.to_datetime(login['Time'])
    #sorting mean x min o
    table = login.groupby('Login Id').apply(lambda x: x.order(ascending=False))
    smallest = table.apply(lambda x : x.diff().min())['Time'].dropna().to_frame()

    return smallest


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------


def total_seller(df):
    """
    Total for each seller
    :param df: like sales
    :return: pivot table
    >>> fp = os.path.join('data', 'sales.csv')
    >>> df = pd.read_csv(fp)
    >>> out = total_seller(df)
    >>> out.index.dtype
    dtype('O')
    >>> out["Total"].sum() < 15000
    True

    """
    table = df.set_index('Name')['Total'].to_frame()
    
    return table


def product_name(df):
    """
    :param df: like sales
    :return: pivot table
    >>> fp = os.path.join('data', 'sales.csv')
    >>> df = pd.read_csv(fp)
    >>> out = product_name(df)
    >>> out.size
    15
    >>> out.loc["pen"].isnull().sum()
    0
    """
    table = pd.pivot_table(df, values=['Total'], index=['Product'],
                    columns=['Name'], aggfunc=np.sum)
    return table

def count_product(df):
    """
    :param df: like sales
    :return: pivot table
    >>> fp = os.path.join('data', 'sales.csv')
    >>> df = pd.read_csv(fp)
    >>> out = count_product(df)
    >>> out.loc["boat"].loc["Trump"].value_counts()[0]
    6
    >>> out.size
    70
    """
    table = pd.pivot_table(df, values=['Total'], index=['Product','Name'],
                    columns=['Date'], aggfunc=np.size).fillna(0)
    return table


def total_by_month(df):
    """
    :param df: like sales
    :return: pivot table
    >>> fp = os.path.join('data', 'sales.csv')
    >>> df = pd.read_csv(fp)
    >>> out = total_by_month(df)
    >>> out["Total"]["May"].idxmax()
    ('Smith', 'book')
    >>> out.shape[1]
    5
    """
    df['Month'] = pd.to_datetime(df['Date']).apply(lambda x: '{month}'.format(month=x.month_name()))
    table = pd.pivot_table(df, values=['Total'], index=['Name','Product'],
                    columns=['Month'], aggfunc=np.size).fillna(0)

    
    return table

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------


def diff_of_means(data, col='orange'):
    """
    diff_of_means takes in a dataframe of counts 
    of skittles (like skittles) and their origin 
    and returns the absolute difference of means 
    between the number of oranges per bag from Yorkville and Waco.

    :Example:
    >>> skittles_fp = os.path.join('data', 'skittles.tsv')
    >>> skittles = pd.read_csv(skittles_fp, sep='\\t')
    >>> out = diff_of_means(skittles)
    >>> 0 <= out
    True
    """
    abs_diff = abs(data.groupby('Factory')[col].mean().diff().iloc[-1])
    
    return abs_diff


def simulate_null(data, col='orange'):
    """
    simulate_null takes in a dataframe of counts of 
    skittles (like skittles) and their origin, and 
    generates one instance of the test-statistic 
    under the null hypothesis

    :Example:
    >>> skittles_fp = os.path.join('data', 'skittles.tsv')
    >>> skittles = pd.read_csv(skittles_fp, sep='\\t')
    >>> out = simulate_null(skittles)
    >>> isinstance(out, float)
    True
    >>> 0 <= out <= 1.0
    True
    """
    data = data.set_index('Factory')
    shuffled_weights = (data[col].sample(replace=False, frac=1).reset_index('Factory'))
    test_statistic = diff_of_means(shuffled_weights, col)
    
    return test_statistic


def pval_orange(data, col='green'):
    """
    pval_orange takes in a dataframe of counts of 
    skittles (like skittles) and their origin, and 
    calculates the p-value for the permutation test 
    using 1000 trials.
    
    :Example:
    >>> skittles_fp = os.path.join('data', 'skittles.tsv')
    >>> skittles = pd.read_csv(skittles_fp, sep='\\t')
    >>> pval = pval_orange(skittles)
    >>> isinstance(pval, float)
    True
    >>> 0 <= pval <= 0.1
    True
    """
    n_repetitions = 1000

    differences = []
    for _ in range(n_repetitions):
        # shuffle the weights
        shuffled_weights = (data[col].sample(replace=False, frac=1).reset_index(drop= True))

        # put them in a table
        shuffled = (data.assign(**{'Shuffled number of orange skittles': shuffled_weights}))

        # compute the group differences
        group_means = (shuffled.groupby('Factory').mean().loc[:, 'Shuffled number of orange skittles'])
        difference = group_means.diff().abs().iloc[-1]

        # add it to the list of results
        differences.append(difference)
        
    observed_difference = diff_of_means(data, col)
    p_value = np.count_nonzero(differences >= observed_difference) / n_repetitions
    
    return p_value


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def ordered_colors():
    """
    ordered_colors returns your answer as an ordered
    list from "most different" to "least different" 
    between the two locations. You list should be a 
    hard-coded list, where each element has the 
    form (color, p-value).

    :Example:
    >>> out = ordered_colors()
    >>> len(out) == 5
    True
    >>> colors = {'green', 'orange', 'purple', 'red', 'yellow'}
    >>> set([x[0] for x in out]) == colors
    True
    >>> all([isinstance(x[1], float) for x in out])
    True
    """

    return (('purple',0.973),('green',0.477),('red',0.228),('orange',0.046),('yellow', 0.0))

    

# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------

def same_color_distribution():
    """
    same_color_distribution outputs a hard-coded tuple 
    with the p-value and whether you 'Fail to Reject' or 'Reject' 
    the null hypothesis.

    >>> out = same_color_distribution()
    >>> isinstance(out[0], float)
    True
    >>> out[1] in ['Fail to Reject', 'Reject']
    True
    """
   
   
    return [0.005, 'Reject']

# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def perm_vs_hyp():
    """
    Multiple choice response for question 8

    >>> out = perm_vs_hyp()
    >>> ans = ['P', 'H']
    >>> len(out) == 5
    True
    >>> set(out) <= set(ans)
    True
    """
    

    return ['P','H','H','P','H']


# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def after_purchase():
    """
    Multiple choice response for question 8

    >>> out = after_purchase()
    >>> ans = ['MD', 'MCAR', 'MAR', 'NI']
    >>> len(out) == 5
    True
    >>> set(out) <= set(ans)
    True
    """

    return ['NI','MD','MAR','MCAR','MAR']

# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------


def multiple_choice():
    """
    Multiple choice response for question 9

    >>> out = multiple_choice()
    >>> ans = ['MD', 'MCAR', 'MAR', 'NI']
    >>> len(out) == 5
    True
    >>> set(out) <= set(ans)
    True
    >>> out[1] in ans
    True
    """

    return ['MCAR','NI','MD','NI','MAR']

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['latest_login'],
    'q02': ['smallest_ellapsed'],
    'q03': ['total_seller', 'product_name', 'count_product', 'total_by_month'],
    'q04': ['diff_of_means', 'simulate_null', 'pval_orange'],
    'q05': ['ordered_colors'],
    'q06': ['same_color_distribution'],
    'q07': ['perm_vs_hyp'],
    'q08': ['after_purchase'],
    'q09': ['multiple_choice']
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

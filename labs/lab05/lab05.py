import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def first_round():
    """
    :return: list with two values
    >>> out = first_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] is "NR" or out[1] is "R"
    True
    """
    return [0.158, "NR"]


def second_round():
    """
    :return: list with three values
    >>> out = second_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] is "NR" or out[1] is "R"
    True
    >>> out[2] is "ND" or out[2] is "D"
    True
    """
    return [0.036, "R", "D"]


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def verify_child(heights):
    """
    Returns a series of p-values assessing the missingness
    of child-height columns on father height.

    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> heights = pd.read_csv(fp)
    >>> out = verify_child(heights)
    >>> out['child_50'] < out['child_95']
    True
    >>> out['child_5'] > out['child_50']
    True
    """
    pv_lst = []
    for col in heights.columns[-7:]:
        
        heights_copy = heights.copy()
        heights_copy['isnull'] = heights[col].isnull()
       
        #observed ks
        observed_ks, _ = ks_2samp(
        heights.loc[heights_copy['isnull'], 'father'],
        heights.loc[~heights_copy['isnull'], 'father']
        )
        #ks statistic
        n_repetitions = 100

        kslist = []
        for _ in range(n_repetitions):

            # shuffle the child height
            shuffled_height = (
                heights_copy[col]
                .sample(replace=False, frac=1)
                .reset_index(drop=True)
            )
            shuffled_null = shuffled_height.isnull()

            # 
            shuffled = (
                heights_copy
                .assign(**{'Shuffled_Height': shuffled_null})
            )

            ks, _ = ks_2samp(
                shuffled.loc[shuffled['Shuffled_Height'], 'father'],
                shuffled.loc[~shuffled['Shuffled_Height'], 'father']
            )

            # add it to the list of results
            kslist.append(ks)
        npa = np.asarray(kslist, dtype=np.float32)    
        p_value = np.count_nonzero(npa >= observed_ks) / len(kslist)
        pv_lst.append(p_value)
        
    sr = pd.Series(pv_lst, index = heights.columns[-7:])
    return sr


def missing_data_amounts():
    """
    Returns a list of multiple choice answers.

    :Example:
    >>> set(missing_data_amounts()) <= set(range(1,6))
    True
    """

    return [1,2,5]


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def cond_single_imputation(new_heights):
    """
    cond_single_imputation takes in a dataframe with columns 
    father and child (with missing values in child) and imputes 
    single-valued mean imputation of the child column, 
    conditional on father. Your function should return a Series.

    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> df['child'] = df['child_50']
    >>> out = cond_single_imputation(df)
    >>> out.isnull().sum() == 0
    True
    >>> (df.child.std() - out.std()) > 0.5
    True
    """
    new_heights = new_heights[['father', 'child_50']].rename(columns={'child_50': 'child'}).copy()
    
    new_heights['father'] =pd.qcut(new_heights['father'],4)
    
    imputation = new_heights.groupby('father').child.transform(lambda x: x.fillna(x.mean()))

    
    return imputation

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    """
    quantitative_distribution that takes in a Series and an integer 
    N > 0, and returns an array of N samples from the distribution of 
    values of the Series as described in the question.
    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> child = df['child_50']
    >>> out = quantitative_distribution(child, 100)
    >>> out.min() >= 56
    True
    >>> out.max() <= 79
    True
    >>> np.isclose(out.mean(), child.mean(), atol=1)
    True
    """
    N_lst = []
    for i in range(N):
        #frequency -> prop (normalize)
        pro_bin = np.histogram(child.dropna(), bins = 10, density = True)

        # base on prop, choose the bin
        sr = pd.Series(pro_bin[0], index = pro_bin[1][1:])
        highest_prop = sr.sort_values(ascending = False).values[0]
        indx = sr.apply(lambda x: x if x == highest_prop else np.NaN).dropna().index

        #within the bin, choose the number to represent the missing value
        imputation = np.random.choice(indx)
        
        N_lst.append(imputation)
    npa = np.asarray(N_lst, dtype=np.float32)    
    return npa


def impute_height_quant(child):
    """
    impute_height_quant takes in a Series of child heights 
    with missing values and imputes them using the scheme in
    the question.

    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> child = df['child_50']
    >>> out = impute_height_quant(child)
    >>> out.isnull().sum() == 0
    True
    >>> np.isclose(out.mean(), child.mean(), atol=0.5)
    True
    """
    lst = quantitative_distribution(child, 100)
    child = child.fillna(np.random.choice(lst))

    return child


# ---------------------------------------------------------------------
# Question # X
# ---------------------------------------------------------------------

def answers():
    """
    Returns two lists with your answers
    :return: Two lists: one with your answers to multiple choice questions
    and the second list has 6 websites that satisfy given requirements.
    >>> list1, list2 = answers()
    >>> len(list1)
    4
    >>> len(list2)
    6
    """
    
    return [1, 2, 2, 1],["*popads.net", "*qq.com", "*360.cn", "*facebook.com", "*linkedin.com", "*linkedin.com"]




# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['first_round', 'second_round'],
    'q02': ['verify_child', 'missing_data_amounts'],
    'q03': ['cond_single_imputation'],
    'q04': ['quantitative_distribution', 'impute_height_quant'],
    'q05': ['answers']
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

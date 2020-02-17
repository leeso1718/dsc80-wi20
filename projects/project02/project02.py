import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------

def get_san(infp, outfp):
    """
    get_san takes in a filepath containing all flights and a filepath where
    filtered dataset #1 is written (that is, all flights arriving or departing
    from San Diego International Airport in 2015).
    The function should return None.

    :Example:
    >>> infp = os.path.join('data', 'flights.test')
    >>> outfp = os.path.join('data', 'santest.tmp')
    >>> get_san(infp, outfp)
    >>> df = pd.read_csv(outfp)
    >>> df.shape
    (53, 31)
    >>> os.remove(outfp)
    """
    # All flights arriving or departing from San Diego International Airport in 2015.
    L = pd.read_csv(infp, chunksize=1000)
    write_header = True
    for df in L:
        filtered = df[(df['YEAR'] == 2015) & ((df['ORIGIN_AIRPORT'] == 'SAN') | (df['DESTINATION_AIRPORT'] == 'SAN'))]
        #head = pd.Series(filtered.columns).values
        filtered.to_csv(outfp, header = write_header, index = False, mode = 'a')
        write_header = False

    return None

    


def get_sw_jb(infp, outfp):
    """
    get_sw_jb takes in a filepath containing all flights and a filepath where
    filtered dataset #2 is written (that is, all flights flown by either
    JetBlue or Southwest Airline in 2015).
    The function should return None.

    :Example:
    >>> infp = os.path.join('data', 'flights.test')
    >>> outfp = os.path.join('data', 'jbswtest.tmp')
    >>> get_sw_jb(infp, outfp)
    >>> df = pd.read_csv(outfp)
    >>> df.shape
    (73, 31)
    >>> os.remove(outfp)
    """
    # All flights flown by either JetBlue or Southwest Airline in 2015.
    L = pd.read_csv(infp, chunksize=1000)
    write_header = True
    for df in L:
        filtered = df[(df['AIRLINE'] == 'B6') | (df['AIRLINE'] == 'WN')]
        filtered.to_csv(outfp, header = write_header, index = False, mode = 'a')
        write_header = False

    return None


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------

def data_kinds():
    """
    data_kinds outputs a (hard-coded) dictionary of data kinds, keyed by column
    name, with values Q, O, N (for 'Quantitative', 'Ordinal', or 'Nominal').

    :Example:
    >>> out = data_kinds()
    >>> isinstance(out, dict)
    True
    >>> set(out.values()) == {'O', 'N', 'Q'}
    True
    """
    # Q,actual amount average makes sense / O, order raking them?/ N, category
    
    dic = {}
    dic = {'YEAR': 'O', 
           'MONTH': 'O',
           'DAY': 'O',
           'DAY_OF_WEEK': 'N',
           'AIRLINE': 'N',
           'FLIGHT_NUMBER':'N',
           'TAIL_NUMBER': 'N',
           'ORIGIN_AIRPORT':'N',
           'DESTINATION_AIRPORT':'N',
           'SCHEDULED_DEPARTURE': 'Q',
           'DEPARTURE_TIME': 'Q',
           'DEPARTURE_DELAY':'Q',
           'TAXI_OUT': 'Q',
           'WHEELS_OFF': 'Q' , 
           'SCHEDULED_TIME': 'Q', 
           'ELAPSED_TIME': 'Q', 
           'AIR_TIME':'Q', 
           'DISTANCE' : 'Q',
           'WHEELS_ON' : 'Q', 
           'TAXI_IN':'Q',
           'SCHEDULED_ARRIVAL':'Q', 
           'ARRIVAL_TIME':'Q',
           'ARRIVAL_DELAY':'Q', 
           'DIVERTED': 'N', 
           'CANCELLED':'N', 
           'CANCELLATION_REASON':'N',
           'AIR_SYSTEM_DELAY':'Q', 
           'SECURITY_DELAY':'Q', 
           'AIRLINE_DELAY':'Q',
           'LATE_AIRCRAFT_DELAY':'Q', 
           'WEATHER_DELAY':'Q' 
          }
    

    return dic


def data_types():
    """
    data_types outputs a (hard-coded) dictionary of data types, keyed by column
    name, with values str, int, float.

    :Example:
    >>> out = data_types()
    >>> isinstance(out, dict)
    True
    >>> set(out.values()) == {'int', 'str', 'float', 'bool'}
    True
    """
    dic = {}
    dic = {'YEAR': 'int', 
           'MONTH': 'int',
           'DAY': 'int',
           'DAY_OF_WEEK': 'int'  ,
           'AIRLINE': 'str',
           'FLIGHT_NUMBER':'int',
           'TAIL_NUMBER': 'str',
           'ORIGIN_AIRPORT': 'str',
           'DESTINATION_AIRPORT': 'str',
           'SCHEDULED_DEPARTURE': 'int',
           'DEPARTURE_TIME': 'float',
           'DEPARTURE_DELAY': 'float',
           'TAXI_OUT': 'float',
           'WHEELS_OFF': 'float', 
           'SCHEDULED_TIME': 'int', 
           'ELAPSED_TIME': 'float', 
           'AIR_TIME': 'float', 
           'DISTANCE': 'int',
           'WHEELS_ON': 'float', 
           'TAXI_IN': 'float',
           'SCHEDULED_ARRIVAL':'int', 
           'ARRIVAL_TIME': 'float',
           'ARRIVAL_DELAY': 'float', 
           'DIVERTED':'bool', 
           'CANCELLED':'bool', 
           'CANCELLATION_REASON': 'str',
           'AIR_SYSTEM_DELAY': 'float', 
           'SECURITY_DELAY': 'float', 
           'AIRLINE_DELAY': 'float',
           'LATE_AIRCRAFT_DELAY': 'float', 
           'WEATHER_DELAY': 'float'
          }
    
    return dic


# ---------------------------------------------------------------------
# Question #3
# ---------------------------------------------------------------------

def basic_stats(flights):
    """
    basic_stats takes flights and outputs a dataframe that contains statistics
    for flights arriving/departing for SAN.
    That is, the output should have have two rows, indexed by ARRIVING and
    DEPARTING, and have the following columns:

    * number of arriving/departing flights to/from SAN (count).
    * mean flight (arrival) delay of arriving/departing flights to/from SAN
      (mean_delay).
    * median flight (arrival) delay of arriving/departing flights to/from SAN
      (median_delay).
    * the airline code of the airline with the longest flight (arrival) delay
      among all flights arriving/departing to/from SAN (airline).
    * a list of the three months with the greatest number of arriving/departing
      flights to/from SAN, sorted from greatest to least (top_months).

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = basic_stats(flights)
    >>> out.index.tolist() == ['ARRIVING', 'DEPARTING']
    True
    >>> cols = ['count', 'mean_delay', 'median_delay', 'airline', 'top_months']
    >>> out.columns.tolist() == cols
    True
    """
    #count
    arrive_cont = flights.groupby('ORIGIN_AIRPORT').count().loc['SAN'].values[0]
    depart_cont = flights.groupby('DESTINATION_AIRPORT').count().loc['SAN'].values[0]
    
    #mean_delay
    a_mean_delay = flights.groupby('ORIGIN_AIRPORT')['ARRIVAL_DELAY'].mean().loc['SAN']
    p_mean_delay = flights.groupby('DESTINATION_AIRPORT')['ARRIVAL_DELAY'].mean().loc['SAN']
    
    #median_delay
    a_median_delay = flights.groupby('ORIGIN_AIRPORT')['ARRIVAL_DELAY'].median().loc['SAN']
    p_median_delay = flights.groupby('DESTINATION_AIRPORT')['ARRIVAL_DELAY'].median().loc['SAN']
   
    #airline
    a_max = flights.groupby('ORIGIN_AIRPORT').max()
    a_airline = a_max[['AIRLINE','ARRIVAL_DELAY']].loc['SAN']
    d_max = flights.groupby('DESTINATION_AIRPORT').max()
    d_airline = d_max[['AIRLINE','ARRIVAL_DELAY']].loc['SAN']
    
    #top_months
    a_top_months = flights[flights['ORIGIN_AIRPORT']=='SAN'].groupby('MONTH')['FLIGHT_NUMBER'].count().nlargest(3).index
    d_top_months = flights[flights['DESTINATION_AIRPORT']=='SAN'].groupby('MONTH')['FLIGHT_NUMBER'].count().nlargest(3).index
    
    dic = {
        'count' : [arrive_cont, depart_cont],
        'mean_delay': [a_mean_delay, p_mean_delay],
        'median_delay': [a_median_delay,p_median_delay],
        'airline' : [a_airline, d_airline],
        'top_months': [a_top_months, d_top_months]
    }
    index = ['ARRIVING','DEPARTING']
    df = pd.DataFrame(dic, index = index)
    
    return df


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


def depart_arrive_stats(flights):
    """
    depart_arrive_stats takes in a dataframe like flights and calculates the
    following quantities in a series (with the index in parentheses):
    - The proportion of flights from/to SAN that
      leave late, but arrive early or on-time (late1).
    - The proportion of flights from/to SAN that
      leaves early, or on-time, but arrives late (late2).
    - The proportion of flights from/to SAN that
      both left late and arrived late (late3).

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = depart_arrive_stats(flights)
    >>> out.index.tolist() == ['late1', 'late2', 'late3']
    True
    >>> isinstance(out, pd.Series)
    True
    >>> out.max() < 0.30
    True
    """
    #late1, leave late but arrive on-time or early
    late1 = len(flights[(flights['DEPARTURE_DELAY']>0) & (flights['ARRIVAL_DELAY']<=0)])/len(flights)
    
    #late2
    late2 = len(flights[(flights['DEPARTURE_DELAY']<=0) & (flights['ARRIVAL_DELAY']>0)])/len(flights)
    
    #late3
    late3 = len(flights[(flights['DEPARTURE_DELAY'] >0) & (flights['ARRIVAL_DELAY']>0)])/len(flights)
    
    sr = pd.Series([late1, late2, late3], index = ['late1', 'late2','late3'])
    return sr
    
def depart_arrive_stats_by_month(flights):
    """
    depart_arrive_stats_by_month takes in a dataframe like flights and
    calculates the quantities in depart_arrive_stats, broken down by month

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = depart_arrive_stats_by_month(flights)
    >>> out.columns.tolist() == ['late1', 'late2', 'late3']
    True
    >>> set(out.index) <= set(range(1, 13))
    True
    """
    late1 = flights.groupby('MONTH').apply(lambda x: ((x['DEPARTURE_DELAY']>0) & (x['ARRIVAL_DELAY']<=0)).sum()/len(x)).to_frame('late1')
    
    late2 = flights.groupby('MONTH').apply(lambda x: ((x['DEPARTURE_DELAY']<=0) & (x['ARRIVAL_DELAY']>0)).sum()/len(x)).to_frame('late2')
    
    late3 = flights.groupby('MONTH').apply(lambda x: ((x['DEPARTURE_DELAY']>0) & (x['ARRIVAL_DELAY']>0)).sum()/len(x)).to_frame('late3')
    
    df = pd.concat([late1, late2, late3], axis = 1)

    return df


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def cnts_by_airline_dow(flights):
    """
    mean_by_airline_dow takes in a dataframe like flights and outputs a
    dataframe that answers the question:
    Given any AIRLINE and DAY_OF_WEEK, how many flights were there (in 2015)?

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = cnts_by_airline_dow(flights)
    >>> set(out.columns) == set(flights['AIRLINE'].unique())
    True
    >>> set(out.index) == set(flights['DAY_OF_WEEK'].unique())
    True
    >>> (out >= 0).all().all()
    True
    """
    table = flights.pivot_table(values = "FLIGHT_NUMBER",index = "DAY_OF_WEEK",columns = "AIRLINE",aggfunc = "count")
    return table


def mean_by_airline_dow(flights):
    """
    mean_by_airline_dow takes in a dataframe like flights and outputs a
    dataframe that answers the question:
    Given any AIRLINE and DAY_OF_WEEK, what is the average ARRIVAL_DELAY (in
    2015)?

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = mean_by_airline_dow(flights)
    >>> set(out.columns) == set(flights['AIRLINE'].unique())
    True
    >>> set(out.index) == set(flights['DAY_OF_WEEK'].unique())
    True
    """
    table = flights.pivot_table(values = "ARRIVAL_DELAY",index = "DAY_OF_WEEK",columns = "AIRLINE",aggfunc = "mean")
    
    return table


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def predict_null_arrival_delay(row):
    """
    predict_null takes in a row of the flights data (that is, a Series) and
    returns True if the ARRIVAL_DELAY is null and otherwise False.

    :param row: a Series that represents a row of `flights`
    :returns: a boolean representing when `ARRIVAL_DELAY` is null.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = flights.drop('ARRIVAL_DELAY', axis=1).apply(predict_null_arrival_delay, axis=1)
    >>> set(out.unique()) - set([True, False]) == set()
    True
    """
    #find the related column and value -> return True or False depends on the value I found
    if (pd.isnull(row.get('ELAPSED_TIME'))) | (pd.isnull(row.get('AIR_TIME'))):
        return True
    return False


def predict_null_airline_delay(row):
    """
    predict_null takes in a row of the flights data (that is, a Series) and
    returns True if the AIRLINE_DELAY is null and otherwise False. Since the
    function doesn't depend on AIRLINE_DELAY, it should work a row even if that
    index is dropped.

    :param row: a Series that represents a row of `flights`
    :returns: a boolean representing when `AIRLINE_DELAY` is null.

    :Example: 
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = flights.drop('AIRLINE_DELAY', axis=1).apply(predict_null_airline_delay, axis=1)
    >>> set(out.unique()) - set([True, False]) == set()
    True
    """
    if row[(row['ARRIVAL_DELAY'] <= 0) | (row['DIVERTED'] == True) | (row['CANCELLED'] == True)]:
        return True
    else:
        return False


# ---------------------------------------------------------------------
# Question #7
# ---------------------------------------------------------------------

def perm4missing(flights, col, N):
    """
    perm4missing takes in flights, a column col, and a number N and returns the
    p-value of the test (using N simulations) that determines if
    DEPARTURE_DELAY is MAR dependent on col.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = perm4missing(flights, 'AIRLINE', 100)
    >>> 0 <= out <= 1
    True
    """
    sample_value = []
    for _ in range(N):
        # shuffle the weights
        flights['Null'] = flights["DEPARTURE_DELAY"].isnull()
        shuffled_flights = (flights[col].sample(replace=False, frac=1).reset_index(drop= True))

        # put them in a table
        shuffled = (flights.assign(**{'Shuffled col': shuffled_flights}))

        # compute the group differences
        shuffled_pivot_table = pd.pivot_table(shuffled, index='Null', columns=shuffled_flights, aggfunc='size')
        normalized = shuffled_pivot_table.apply(lambda x: x/x.sum(), axis=1)
        shuffled_tvd = normalized.diff(axis=0).iloc[-1].abs().sum()/2
    
        sample_value.append(shuffled_tvd)
    sample_series = pd.Series(sample_value)
    p_val = np.count_nonzero(sample_series >= observed_statistic(flights, col))/N
    return p_val



def dependent_cols():
    """
    dependent_cols gives a list of columns on which DEPARTURE_DELAY is MAR
    dependent on.

    :Example:
    >>> out = dependent_cols()
    >>> isinstance(out, list)
    True
    >>> cols = 'YEAR DAY_OF_WEEK AIRLINE DIVERTED CANCELLATION_REASON'.split()
    >>> set(out) <= set(cols)
    True
    """

    return ['DAY_OF_WEEK', 'AIRLINE']

def missing_types():
    """
    missing_types returns a Series
    - indexed by the following columns of flights:
    CANCELLED, CANCELLATION_REASON, TAIL_NUMBER, ARRIVAL_TIME.
    - The values contain the most-likely missingness type of each column.
    - The unique values of this Series should be MD, MCAR, MAR, MNAR, NaN.

    :param:
    :returns: A series with index and values as described above.

    :Example:
    >>> out = missing_types()
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) - set(['MD', 'MCAR', 'MAR', 'NMAR', np.NaN]) == set()
    True
    """
    data = {'CANCELLED': np.NaN , 'CANCELLATION_REASON': 'MD', 'TAIL_NUMBER': 'MCAR',
            'ARRIVAL_TIME': 'MAR'}
    return pd.Series(data)

# ---------------------------------------------------------------------
# Question #8
# ---------------------------------------------------------------------

def prop_delayed_by_airline(jb_sw):
    """
    prop_delayed_by_airline takes in a dataframe like jb_sw and returns a
    DataFrame indexed by airline that contains the proportion of each airline's
    flights that are delayed.

    :param jb_sw: a dataframe similar to jb_sw
    :returns: a dataframe as above

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=100)
    >>> out = prop_delayed_by_airline(jb_sw)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> (out >= 0).all().all() and (out <= 1).all().all()
    True
    >>> len(out.columns) == 1
    True
    """
    airport_lst = ["ABQ", "BDL", "BUR", "DCA", "MSY", "PBI", "PHX", "RNO", "SJC", "SLC"]
    airport_df = jb_sw[jb_sw["ORIGIN_AIRPORT"].isin(airport_lst)]
    filtered_df = airport_df[["ORIGIN_AIRPORT", "AIRLINE", "DEPARTURE_DELAY", "CANCELLED"]]
    
    group_airline = filtered_df.groupby('AIRLINE').agg({'ORIGIN_AIRPORT':'count'})
    cancelled_drop = filtered_df[filtered_df["CANCELLED"] == False]
    delayed_filter = cancelled_drop[cancelled_drop["DEPARTURE_DELAY"] > 0]
    group_delay = delayed_filter.groupby('AIRLINE').agg({'ORIGIN_AIRPORT': 'count'})
    
    return group_delay.div(group_airline)



def prop_delayed_by_airline_airport(jb_sw):
    """
    prop_delayed_by_airline_airport that takes in a dataframe like jb_sw and
    returns a DataFrame, with columns given by airports, indexed by airline,
    that contains the proportion of each airline's flights that are delayed at
    each airport.

    :param jb_sw: a dataframe similar to jb_sw
    :returns: a dataframe as above.

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=100)
    >>> out = prop_delayed_by_airline_airport(jb_sw)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> ((out >= 0) | (out <= 1) | (out.isnull())).all().all()
    True
    >>> len(out.columns) == 6
    True
    """
    airport_lst = ["ABQ", "BDL", "BUR", "DCA", "MSY", "PBI", "PHX", "RNO", "SJC", "SLC"]
    filtered_df = jb_sw[jb_sw["ORIGIN_AIRPORT"].isin(airport_lst)]
    filtered_copy = filtered_df.copy()

    filtered_copy["DELAYED"] = (filtered_copy["DEPARTURE_DELAY"] > 0) & (filtered_copy["CANCELLED"] == False)
    delayed_pivot = filtered_copy.pivot_table(index="AIRLINE", columns="ORIGIN_AIRPORT",values="DELAYED", aggfunc="mean")
    
    return delayed_pivot


# ---------------------------------------------------------------------
# Question #9
# ---------------------------------------------------------------------

def verify_simpson(df, group1, group2, occur):
    """
    verify_simpson verifies whether a dataset displays Simpson's Paradox.

    :param df: a dataframe
    :param group1: the first group being aggregated
    :param group2: the second group being aggregated
    :param occur: a column of df with values {0,1}, denoting
    if an event occurred.
    :returns: a boolean. True if simpson's paradox is present,
    otherwise False.

    :Example:
    >>> df = pd.DataFrame([[4,2,1], [1,2,0], [1,4,0], [4,4,1]], columns=[1,2,3])
    >>> verify_simpson(df, 1, 2, 3) in [True, False]
    True
    >>> verify_simpson(df, 1, 2, 3)
    False
    """
    group1_c = str(group1)
    group2_c = str(group2)

    df.columns = [group1_c, group2_c] + df.columns[2:].tolist()
    df[group1_c] = df[group1_c].astype(str)
    df[group2_c] = df[group2_c].astype(str)

    mean_group1 = df.groupby(group1_c)[occur].mean()
    group1_df = mean_group1.to_frame()
    sorted_sr = group1_df.sort_values(by=occur, ascending=False)
    
    max_value = sorted_sr.index[0]
    min_value = sorted_sr.index[1]
    verify = False
    
    pivot_table_sum = df.pivot_table(index=group1_c, columns=group2_c, values=occur, aggfunc="sum").transpose()
    pivot_table_count = df.pivot_table(index=group1_c, columns=group2_c, values=occur, aggfunc="count").transpose()
    div_df = pivot_table_sum.div(pivot_table_count)
    
    if ((div_df[min_value][0] > div_df[max_value][0]) & (div_df[min_value][1] > div_df[max_value][1])):
        verify = True
    
    return verify



# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def search_simpsons(jb_sw, N):
    """
    search_simpsons takes in the jb_sw dataset and a number N, and returns a
    list of N airports for which the proportion of flight delays between
    JetBlue and Southwest satisfies Simpson's Paradox.

    Only consider airports that have '3 letter codes',
    Only consider airports that have at least one JetBlue and Southwest flight.

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=1000)
    >>> pair = search_simpsons(jb_sw, 2)
    >>> len(pair) == 2
    True
    """

    return 0


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_san', 'get_sw_jb'],
    'q02': ['data_kinds', 'data_types'],
    'q03': ['basic_stats'],
    'q04': ['depart_arrive_stats', 'depart_arrive_stats_by_month'],
    'q05': ['cnts_by_airline_dow', 'mean_by_airline_dow'],
    'q06': ['predict_null_arrival_delay', 'predict_null_airline_delay'],
    'q07': ['perm4missing', 'dependent_cols', 'missing_types'],
    'q08': ['prop_delayed_by_airline', 'prop_delayed_by_airline_airport'],
    'q09': ['verify_simpson'],
    'q10': ['search_simpsons']
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
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True

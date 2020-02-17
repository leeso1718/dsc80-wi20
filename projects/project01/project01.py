import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    '''
    get_assignment_names takes in a dataframe like grades and returns 
    a dictionary with the following structure:

    The keys are the general areas of the syllabus: lab, project, 
    midterm, final, disc, checkpoint

    The values are lists that contain the assignment names of that type. 
    For example the lab assignments all have names of the form labXX where XX 
    is a zero-padded two digit number. See the doctests for more details.    

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    '''
    
    assignment_names = {
        "lab": parse_names(grades.columns, "lab"),
        "project": parse_names(grades.columns, "project"),
        "midterm": parse_names(grades.columns, "Midterm"),
        "final": parse_names(grades.columns, "Final"),
        "disc": parse_names(grades.columns, "discussion"),
        "checkpoint": parse_names(grades.columns, "checkpoint"),
    }
    return assignment_names

def parse_names(data, key):
    labs = data[data.str.find(key) != -1]
    # Calc stopping index
    index_stop = labs[0].find(key)+2+len(key)
    # Don't get last two digits if not assignments
    if key == "Final" or key == "Midterm":
        index_stop -= 2
    return labs.str[0:index_stop].unique().tolist()


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def projects_total(grades):
    '''
    projects_total that takes in grades and computes the total project grade
    for the quarter according to the syllabus. 
    The output Series should contain values between 0 and 1.
    
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''
    
    names = get_assignment_names(grades)
    project_grades = grades[names['project']]
    project_related = grades[grades.columns[grades.columns.str.find("project") != -1]]
    # Initialize percentage to 0
    project_grade_percentage = np.zeros(project_grades.shape[0])
    # Iterate through each project
    for project_name in project_grades.columns:
        # check for fr
        project_fr_name = project_name + "_free_response"
        # compute project
        project_grade_student = np.array(project_related[project_name].fillna(0).values)
        project_grade_final = np.array(project_related[project_name + " - Max Points"].values)
        if project_fr_name in grades.columns:
            # add fr score
            project_grade_student += np.array(project_related[project_fr_name].fillna(0).values)
            project_grade_final += np.array(project_related[project_fr_name + " - Max Points"].values)
            
        # add it to total project
        project_grade_percentage += project_grade_student / project_grade_final
              
    return pd.Series(project_grade_percentage / len(project_grades.columns))


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------


def last_minute_submissions(grades):
    """
    last_minute_submissions takes in the dataframe 
    grades and a Series indexed by lab assignment that 
    contains the number of submissions that were turned 
    in on time by the student, yet marked 'late' by Gradescope.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1,10)])
    True
    >>> (out > 0).sum()
    8
    """

    lab_related = grades[grades.columns[grades.columns.str.find("lab") != -1]]
    lab_lateness = lab_related[lab_related.columns[lab_related.columns.str.contains("Lateness")]]
    threshold = 3600 * 6
    sr = lab_lateness.apply(lambda x : (x.str.split(":").apply(lambda y: int(y[0])*3600+int(y[1])*60+int(y[2]))), axis=0)
    not_late = sr.applymap(lambda x : x < threshold)
    marked_late = sr.applymap(lambda x: x > 0)
    sr = (not_late & marked_late).sum()
    sr.index = sr.index.str[:5]
    return sr
    

# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------

def lateness_penalty(col):
    """
    lateness_penalty takes in a 'lateness' column and returns 
    a column of penalties according to the syllabus.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.8, 0.5}
    True
    """
        
    one_week = 24 * 7 * 3600
    two_weeks = one_week * 2
    penalty = np.zeros(len(col))
    late_in_sec = col.str.split(":").apply(lambda y: int(y[0])*3600+int(y[1])*60+int(y[2]))
    penalty += (late_in_sec == 0) * 1.0
    penalty += np.array(late_in_sec > 0) * np.array(one_week >= late_in_sec) * 0.9
    penalty += np.array(late_in_sec > one_week) * np.array(two_weeks >= late_in_sec) * 0.8
    penalty += (late_in_sec > two_weeks) * 0.5
    return penalty

# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def process_labs(grades):
    """
    process_labs that takes in a dataframe like grades and returns
    a dataframe of processed lab scores. The output should:
      * share the same index as grades,
      * have columns given by the lab assignment names (e.g. lab01,...lab10)
      * have values representing the lab grades for each assignment, 
        adjusted for Lateness and scaled to a score between 0 and 1.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1,10)]
    True
    >>> np.all((0.65 <= out.mean()) & (out.mean() <= 0.90))
    True
    """
    lab_related = grades[get_assignment_names(grades)['lab']]
    reduced = pd.DataFrame([], columns=lab_related.columns)
    for lab_col in lab_related:
        factor = lateness_penalty(grades[lab_col + " - Lateness (H:M:S)"])
        max_points = grades[lab_col + " - Max Points"]
        reduced[lab_col]=pd.Series(grades[lab_col].fillna(0) * factor / max_points)
    return reduced


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def lab_total(processed):
    """
    lab_total takes in dataframe of processed assignments (like the output of 
    Question 5) and computes the total lab grade for each student according to
    the syllabus (returning a Series). 
    
    Your answers should be proportions between 0 and 1.

    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """

    lowest = processed.apply(pd.Series.min, axis=1)
    total = processed.apply(pd.Series.sum,axis=1)
    lab_total = (total - lowest) / (len(processed.columns) - 1)
    return lab_total


# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def total_points(grades):
    """
    total_points takes in grades and returns the final
    course grades according to the syllabus. Course grades
    should be proportions between zero and one.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    #hi = grades[grades.columns[grades.columns.str.contains('checkpoint')]]
    #lab assignment
    lab_processed = process_labs(grades)
    lab = (lab_total(lab_processed) *0.2).values
    #project
    project_processed = helper_process(grades, 'project')
    project = (helper_total(project_processed) *0.3).values
    #check point 
    cp_processed = helper_process(grades, 'checkpoint')
    cp = (helper_total(cp_processed) *0.025).values
    # discussin
    dis_processed = helper_process(grades, 'disc')
    dis = (helper_total(dis_processed) *0.025).values
    #exam
    mid = (helper_process(grades, 'midterm')['Midterm'].values)* 0.15
    
    final = (helper_process(grades, 'final')['Final'].values)* 0.3

    
    total = lab + project + cp + dis + mid + final
    
    return total

def helper_process(grades, col):
    col_related = grades[get_assignment_names(grades)[col]]
    reduced = pd.DataFrame([], columns=col_related.columns)
    for each_col in col_related:
        #factor = lateness_penalty(grades[each_col + " - Lateness (H:M:S)"])
        max_points = grades[each_col + " - Max Points"]
        reduced[each_col]=pd.Series(grades[each_col].fillna(0) / max_points)
    return reduced

def helper_total(processed):
    total = processed.apply(pd.Series.sum,axis=1)
    final_total = total / (len(processed.columns))
    return final_total


def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.

    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """
    a = total.apply(lambda x : x >= .90)
    
    b = total.apply(lambda x : x < .90 and x >= .80)
    
    c = total.apply(lambda x : x < .80 and x >= .70)
    
    d = total.apply(lambda x : x < .70 and x >= .60)
    
    f = total.apply(lambda x : x < 0.60)
    
    total = total.replace(total[a], 'A')
    total = total.replace(total[b], 'B')
    total = total.replace(total[c], 'C')
    total = total.replace(total[d], 'D')
    total = total.replace(total[f], 'F')


    return total


def letter_proportions(grades):
    """
    letter_proportions takes in the dataframe grades 
    and outputs a Series that contains the proportion
    of the class that received each grade.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = letter_proportions(grades)
    >>> np.all(out.index == ['B', 'C', 'A', 'D', 'F'])
    True
    >>> out.sum() == 1.0
    True
    """
    letter_grades =  final_grades(pd.Series(total_points(grades)))
    proportions = letter_grades.value_counts()/len(grades)
    return proportions

# ---------------------------------------------------------------------

# Question # 8

# ---------------------------------------------------------------------


def simulate_pval(grades, N):

    """

    simulate_pval takes in the number of

    simulations N and grades and returns

    the likelihood that the grade of sophomores

    was no better on average than the class

    as a whole (i.e. calculate the p-value).


    :Example:

    >>> fp = os.path.join('data', 'grades.csv')

    >>> grades = pd.read_csv(fp)

    >>> out = simulate_pval(grades, 100)

    >>> 0 <= out <= 0.1

    True

    """

    #observed_avg

    observed_avg = total_points(grades[grades['Level'] =="SO"]).mean()

    #averages

    N_trials =  N

    averages = []

    num_sop = len(grades[grades['Level'] =="SO"])


    for i in np.arange(N_trials):

        random_sample = grades.sample(int(num_sop), replace = False)

        new_average = total_points(random_sample).mean()

        averages.append(new_average)

    

    averages = np.array(averages)

    

    p_value = np.count_nonzero(averages >= observed_avg) / N_trials

    return p_value


# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------


def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades, 
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """

    #lab assignment
    lab_processed = process_labs(grades)
    lab_processed += np.random.normal(0, 0.02, size=(lab_processed.shape))
    lab = (lab_total(lab_processed) *0.2).values
    lab = np.clip(lab, 0, 1)

    #project
    project_processed = helper_process(grades, 'project')
    project_processed += np.random.normal(0, 0.02, size=(project_processed.shape))
    project = (helper_total(project_processed) *0.3).values
    project = np.clip(project, 0, 1)

    #check point
    cp_processed = helper_process(grades, 'checkpoint')
    cp_processed += np.random.normal(0, 0.02, size=(cp_processed.shape))
    cp = (helper_total(cp_processed) *0.025).values
    cp = np.clip(cp, 0, 1)

    # discussion
    dis_processed = helper_process(grades, 'disc')
    dis_processed += np.random.normal(0, 0.02, size=(dis_processed.shape))
    dis = (helper_total(dis_processed) *0.025).values
    dis = np.clip(dis, 0, 1)
    
    #exam
    mid = (helper_process(grades, 'midterm')['Midterm'].values)* 0.15
    mid += np.random.normal(0, 0.02, size=(mid.shape))
    mid = np.clip(mid, 0, 1)
    final = (helper_process(grades, 'final')['Final'].values)* 0.3
    final += np.random.normal(0, 0.02, size=(final.shape))
    final = np.clip(final, 0, 1)
    
    total = lab + project + cp + dis + mid + final


    return total


# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def short_answer():
    """
    short_answer returns (hard-coded) answers to the 
    questions listed in the notebook. The answers should be
    given in a list with the same order as questions.

    :Example:
    >>> out = short_answer()
    >>> len(out) == 5
    True
    >>> len(out[2]) == 2
    True
    >>> 50 < out[2][0] < 100
    True
    >>> 0 < out[3] < 1
    True
    >>> isinstance(out[4], bool)
    True
    """

    return [-0.0012022836538396688, 0.049021587501757224, [53 ,  -58], 0.24485981308411214, True]

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_assignment_names'],
    'q02': ['projects_total'],
    'q03': ['last_minute_submissions'],
    'q04': ['lateness_penalty'],
    'q05': ['process_labs'],
    'q06': ['lab_total'],
    'q07': ['total_points', 'final_grades', 'letter_proportions'],
    'q08': ['simulate_pval'],
    'q09': ['total_points_with_noise'],
    'q10': ['short_answer']
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
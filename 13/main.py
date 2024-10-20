import numpy as np
import pandas as pd

def generalizeSH(SH, concept):
    """ 
    Generalizes specific hypothesis when a concept/data is given.\n
    Examples:\n
        [None, None, None]  to  ['A', 'B', 'C'] given data=['A', 'B', 'C']
        ['A', 'B', 'C']     to  ['?', 'B', '?'] given data=['D', 'B', 'A']
    """

    for count in range(len(concept)):
        if SH[count] == None:
            SH[count] = concept[count]
        elif SH[count] != concept[count]:
            SH[count] = '?'

    return SH


def matchHypothesisWithConcept(hypothesis, concept):
    """
    'hypothesis' is single general hypotheis.\n 
    'concept' is single data.\n
    Returns True if the given concept matches with the hypothesis. Else False. 
    """

    is_matching = True

    for count in range(len(concept)):
        
        if hypothesis[count] != '?' and hypothesis[count] != concept[count]:
            is_matching = False

    return is_matching


def checkValidityOfGH(GH, concepts, targets, yes):
    """
    Checks if the given concepts are are valid with all the hypotheses in GH. 
    Invalid hypothesis will be removed from GH.\n
    Examples:\n
        ['A', '?', 'C'] is valid for the data ['A', 'B', 'C', 'yes']
        ['A', '?', '?'] is not valid for the data ['D', 'B', 'C', 'no']
    """

    if GH == None: 
        return GH

    invalid_hypotheses = []

    for hypothesis in GH:

        for count in range(len(concepts)):

            c = concepts[count]
            t = targets[count]

            is_exp_no = (matchHypothesisWithConcept(hypothesis, c) == False)
        
            if (t == yes and is_exp_no) or (t != yes and not is_exp_no):
                invalid_hypotheses.append(hypothesis)
                break

    # removing invalid_hypothesis from GH
    for i in invalid_hypotheses:
        GH.remove(i)

    return GH


def initializeGH(concept, unique_values):
    """
    Given a concept, it will return the first set of general hypotheses.\n
    'unique_values' must have all possible values appearing in a column throughout the data.\n
    Example:\n
        let concept be ['A', 'D', 'E']
        let unique_values be [['A', 'B'], ['C', 'D'], ['E', 'F', 'G']]
        then returned GH will be,
            [
                ['B', '?', '?']
                ['?', 'C', '?']
                ['?', '?', 'F'], ['?', '?', 'G']
            ]
    """
    
    GH = []

    for count in range(len(concept)):

        for value in unique_values[count]:
            
            if value != concept[count]:
                hypothesis = ['?' for i in concept]
                hypothesis[count] = value
                GH.append(hypothesis)

    return GH


def specializeAHypothesis(hypothesis, unique_values):
    """
    'hypothesis' is single general hypotheis.\n 
    'concept' is single data.\n
    Returns list of all specialized hypotheses.\n
    Example 1:\n
        let hypothesis be ['?', 'C', '?']
        let unique_values be [['A', 'B'], ['C', 'D'], ['E', 'F', 'G']]
        then the returned hypotheses will be,
            [
                ['A', 'C', '?'], ['B', 'C', '?'],
                ['?', 'C', 'E'], ['?', 'C', 'F'], ['?', 'C', 'G'],
            ]\n
    Example 2:\n
        let hypothesis be ['?', 'C', 'F']
        let unique_values be [['A', 'B'], ['C', 'D'], ['E', 'F', 'G']]
        then the returned hypotheses will be,
            [
                ['A', 'C', 'F'], ['B', 'C', 'F'],
            ]
    """

    specialized_hypotheses = []

    for count in range(len(hypothesis)):
        
        if hypothesis[count] == '?':
            
            for value in unique_values[count]:
                temp_hypothesis = hypothesis.copy()
                temp_hypothesis[count] = value
                specialized_hypotheses.append(temp_hypothesis)

    return specialized_hypotheses


def generateVersionSpace(SH, GH):
    """
    Given SH and GH, this function returns the version_space and a verbose comment.
    """
    
    version_space = []
    comment = None

    if GH == None:
        comment = "Dataset has no negative examples"
        version_space = []
    elif GH != None and len(GH) == 0:
        comment = "Empty version space"
        version_space = []
    elif len(GH) == 1 and SH == GH[0]:
        comment = "Dataset is perfectly learned"
        version_space = [SH]
    else:
        comment = "Dataset is not perfectly learned"
        version_space = [SH, *GH]

        for i in GH:
            
            temp_h = i.copy()
            for count in range(len(SH)):
                
                if i[count] == '?' and SH[count] != '?':
                    temp_h[count] = SH[count]
                    if temp_h not in version_space:
                        version_space.append(temp_h)
                    temp_h = i.copy()

    return version_space, comment


def learn(concepts, targets, yes):
    """
    'concepts' must be a list of concepts.\n
    'targets' must be a list of target column values.\n
    'yes' must be the postive value of the target column.
    """

    data = pd.DataFrame(data=concepts)
    unique_values = []
    for col in data:
        unique_values.append(list(data[col].unique()))

    SH = [None for i in range(len(concepts[0]))]
    GH = None

    for i in range(len(concepts)):

        c = concepts[i]
        t = targets[i]

        if t == yes:
            SH = generalizeSH(SH, c)

        else:
            if GH == None:
                GH = initializeGH(c, unique_values)

            else:
                new_GH = []
                for hypothesis in GH:
                    if matchHypothesisWithConcept(hypothesis, c):
                        new_GH.extend(specializeAHypothesis(hypothesis, unique_values))
                    else:
                        new_GH.append(hypothesis)
                GH = new_GH

        GH = checkValidityOfGH(GH, concepts[0:i+1], targets[0:i+1], yes)

    version_space, comment = generateVersionSpace(SH, GH)

    return {
        'specific_h': SH,
        'general_h': GH,
        'version_space': version_space, 
        'comment': comment
    }


data = pd.DataFrame(data=pd.read_csv('6.csv'))

concepts = np.array(data.iloc[:,0:-1])
targets = np.array(data.iloc[:,-1])

result = learn(concepts, targets, 'Yes')

for i in result:
    print(i, ":")
    print(result[i])
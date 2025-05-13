import numpy as np
def get_quantile_threshold(alpha):
    '''
    Compute smallest n such that ceil((n+1)*(1-alpha)/n) <= 1
    '''

    n = 1
    while np.ceil((n+1)*(1-alpha)/n) > 1:
        n += 1

    return n
import numpy as np
def get_quantile_threshold(alpha):
    '''
    Compute smallest n such that ceil((n+1)*(1-alpha)/n) <= 1
    '''

    n = 1
    while np.ceil((n+1)*(1-alpha)/n) > 1:
        n += 1
    return n


def get_clustering_parameters(num_classes, n_totalcal):
    '''
    Returns a guess of good values for num_clusters and n_clustering based solely
    on the number of classes and the number of examples per class.

    This relies on two heuristics:
    1) We want at least 150 points per cluster on average
    2) We need more samples as we try to distinguish between more distributions.
    To distinguish between 2 distribution, want at least 4 samples per class.
    To distinguish between 5 distributions, want at least 10 samples per class.

    Output: n_clustering, num_clusters

    '''
    # Alias for convenience
    N = n_totalcal
    K = num_classes

    n_clustering = int(N * K / (75 + K))
    num_clusters = int(np.floor(n_clustering / 2))
    return n_clustering, num_clusters
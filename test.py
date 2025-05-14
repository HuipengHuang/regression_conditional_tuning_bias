import numpy as np
def remap_classes(labels, rare_classes):
    '''
    Exclude classes in rare_classes and remap remaining classes to be 0-indexed

    Outputs:
        - remaining_idx: Boolean array the same length as labels. Entry i is True
        iff labels[i] is not in rare_classes 
        - remapped_labels: Array that only contains the entries of labels that are 
        not in rare_classes (in order) 
        - remapping: Dict mapping old class index to new class index

    '''
    remaining_idx = ~np.isin(labels, rare_classes)

    remaining_labels = labels[remaining_idx]
    remapped_labels = np.zeros(remaining_labels.shape, dtype=int)
    new_idx = 0
    remapping = {}
    for i in range(len(remaining_labels)):
        if remaining_labels[i] in remapping:
            remapped_labels[i] = remapping[remaining_labels[i]]
        else:
            remapped_labels[i] = new_idx
            remapping[remaining_labels[i]] = new_idx
            new_idx += 1
    return remaining_idx, remapped_labels, remapping

original_labels = np.array([0, 1, 2, 3, 1, 0, 4])
rare_classes = [2, 4]

print(remap_classes(original_labels, rare_classes))
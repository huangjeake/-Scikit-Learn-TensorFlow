import numpy as np

class ProjectModel(object):
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        pass
        for filename in filenames:
            data, labels = filename
            all_data.append(data)
            all_labels.append(labels)
            self._data = np.vstack(all_data)

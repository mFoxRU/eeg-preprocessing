__author__ = 'mFoxRU'

from abc import abstractmethod
import numpy as np
from scipy import stats


class AbstractWilc(object):
    """
    Usage:
        First subclass it and implement format_data and write_output methods
        Then fill it with data using add_data method
        After it is filled you can either manually compare different datasets
    using calc_wilc or compare each dataset against all others using
    cals_wilc_each_vs_all

    """

    @abstractmethod
    def format_data(self, data):
        """
        Returns a data, formatted as a NxM numpy.array, where
        N is number of channels. N must be same for all datasets.
        :param data: Input data of any reasonable format
        :return: 1d or 2d array of specters
        :rtype: numpy.ndarray
        """
        pass

    @abstractmethod
    def write_output(self, order, first_class_label, second_class_label):
        """
        Writes output
        :param order: List with ordered indexes
        :type order: list
        :param first_class_label: Label for the first class
        :type first_class_label: str
        :param second_class_label: Label for the second class
        :type second_class_label: str
        """
        pass

    def __init__(self, name, simple_naming=False, channels_names=None):
        """Constructor for AbstractWilc"""
        self._name = name
        self._simple_naming = simple_naming
        self.channels_names = channels_names
        self._datasets = {}

    def calc_wilc(self, first_classes, second_classes):
        """
        Calculate Wilcoxon rank-sum for two classes
        :param first_classes: Dataset label of iterable of labels
        :param second_classes: Dataset label of iterable of labels
        """
        # Notes for porting to py3: in py2 str has no __iter__
        if not hasattr(first_classes, '__iter__'):
            first_classes = (first_classes,)
        self._check_keys(first_classes)

        if not hasattr(second_classes, '__iter__'):
            second_classes = (second_classes,)
        self._check_keys(second_classes)

        # Prepare datasets
        first_dataset = np.vstack([self._datasets[label]
                                   for label in first_classes])
        second_dataset = np.vstack([self._datasets[label]
                                    for label in second_classes])

        # Calculate wilx and sort
        ranks = []
        for ch in xrange(first_dataset.shape[1]):
            z, p = stats.ranksums(first_dataset[:, ch],
                                  second_dataset[:, ch])
            ranks.append((ch, abs(z), p))
        ranks.sort(key=lambda x: x[1])

        # Prepare labels
        first_label, second_label = first_classes[0], second_classes[0]
        if len(first_classes) > 1:
            if self._simple_naming:
                first_label = 'Main'
            else:
                first_label = '_'.join(first_classes)
        if len(second_classes) > 1:
            if self._simple_naming:
                second_label = 'Other'
            else:
                second_label = '_'.join(second_classes)
        # Write output
        self.write_output(ranks, first_label, second_label)

    def cals_wilc_each_vs_all(self):
        """
        Calculate Wilcoxon rank-sum for each dataset vs all others
        """
        if len(self._datasets) < 2:
            raise Exception('Less when 2 datasets')
        elif len(self._datasets) == 2:
            self.calc_wilc(self._datasets.keys()[0], self._datasets.keys()[1])
        else:
            for first_class in self._datasets.keys():
                second_class = self._datasets.keys()
                second_class.remove(first_class)
                self.calc_wilc(first_class, second_class)

    def add_data(self, label, data):
        """
        Adds a dataset for later calculation use.
        :param label: dataset label
        :type label: str
        :param data: dataset passed to format_data method
        :raise KeyError:
        """
        if label in self._datasets:
            raise KeyError('Key %s already used' % str(label))
        formatted_data = self.format_data(data)
        # ToDo: Mb add check on same channels count
        self._datasets[str(label)] = formatted_data

    def remove_data(self, label):
        """
        Removes dataset from datasets
        :param label: Dataset label
        :type label: str
        """
        try:
            self._datasets.pop(label)
        except KeyError:
            raise KeyError('No dataset associated with key %s' % str(label))

    def get_datasets_labels(self):
        """
        :return: List of datasets labels
        :rtype: list
        """
        return self._datasets.keys()

    def _check_keys(self, classes):
        for key in classes:
            if not key in self._datasets:
                raise KeyError('No dataset associated with key %s'
                               % str(key))
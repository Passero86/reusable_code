import numpy as np
from scipy.sparse import coo_matrix
from scipy.misc import comb
from math import log
from munkres import Munkres


class Metrics:
    """
    Evaluation metrics include:
        accuracy;
        f_measure;
        purity;
        rand_index;
        adjusted_rand_score;
        mutual_info_score;
        normalized_mutual_info_score;

    For all metric functions, the inputs are:
        :param: labels_true (type list): the ground truth of cluster assignment. Each element denotes an item's ground truth cluster_id.
        :param: labels_pred (type list): the predicted cluster assignments. Each element denotes an item's predicted cluster_id.

    """

    def check_clusterings(self, labels_true, labels_pred):
        """Check that the two clusterings matching 1D integer arrays"""
        labels_true = np.asarray(labels_true)
        labels_pred = np.asarray(labels_pred)

        # input checks
        if labels_true.ndim != 1:
            raise ValueError(
                "labels_true must be 1D: shape is %r" % (labels_true.shape,))
        if labels_pred.ndim != 1:
            raise ValueError(
                "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
        if labels_true.shape != labels_pred.shape:
            raise ValueError(
                "labels_true and labels_pred must have same size, got %d and %d" %
                (labels_true.shape[0], labels_pred.shape[0]))
        return labels_true, labels_pred

    def contingency_matrix(self, labels_true, labels_pred, eps=None):
        """Build a contengency matrix describing the relationship between labels.
        Parameters
        ----------
        labels_true : int array, shape = [n_samples]
            Ground truth class labels to be used as a reference
        labels_pred : array, shape = [n_samples]
            Cluster labels to evaluate
        eps: None or float
            If a float, that value is added to all values in the contingency
            matrix. This helps to stop NaN propogation.
            If ``None``, nothing is adjusted.
        Returns
        -------
        contingency: array, shape=[n_classes_true, n_classes_pred]
            Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
            true class :math:`i` and in predicted class :math:`j`. If
            ``eps is None``, the dtype of this array will be integer. If ``eps`` is
            given, the dtype will be float.
        """
        classes, class_idx = np.unique(labels_true, return_inverse=True)
        clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
        n_classes = classes.shape[0]
        n_clusters = clusters.shape[0]
        # Using coo_matrix to accelerate simple histogram calculation,
        # i.e. bins are consecutive integers
        # Currently, coo_matrix is faster than histogram2d for simple cases
        contingency = coo_matrix((np.ones(class_idx.shape[0]),
                                  (class_idx, cluster_idx)),
                                 shape=(n_classes, n_clusters),
                                 dtype=np.int).toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
        return contingency

    def comb2(self, n):
        # the exact version is faster for k == 2: use it by default globally in
        # this module instead of the float approximate variant
        return comb(n, 2, exact=1)

    def get_map_pairs(self, labels_true, labels_pred):
        """
            Given the groundtruth labels and predicted labels, get the best mapping pairs by Munkres algorithm.
        """
        labels_true, labels_pred = self.check_clusterings(
            labels_true, labels_pred)
        classes = np.unique(labels_true)
        clusters = np.unique(labels_pred)
        # Special limit cases: no clustering since the data is not split;
        # or trivial clustering where each document is assigned a unique cluster.
        # These are perfect matches hence return 1.0.
        if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
                or classes.shape[0] == clusters.shape[0] == len(labels_true)):
            return 1.0

        # Type: <type 'numpy.ndarray'>:rows are clusters, cols are classes
        contingency = self.contingency_matrix(labels_true, labels_pred)
        contingency = -contingency
        contingency = contingency.tolist()
        m = Munkres()  # Best mapping by using Kuhn-Munkres algorithm
        # best match to find the minimum cost
        map_pairs = m.compute(contingency)
        return map_pairs

    def entropy(self, labels):
        """Calculates the entropy for a labeling."""
        if len(labels) == 0:
            return 1.0
        label_idx = np.unique(labels, return_inverse=True)[1]
        pi = np.bincount(label_idx).astype(np.float)
        pi = pi[pi > 0]
        pi_sum = np.sum(pi)
        # log(a / b) should be calculated as log(a) - log(b) for
        # possible loss of precision
        return -np.sum((pi / pi_sum) * (np.log(pi) - log(pi_sum)))

    def accuracy(self, labels_true, labels_pred):
        labels_true, labels_pred = self.check_clusterings(
            labels_true, labels_pred)
        n_samples = labels_true.shape[0]
        classes = np.unique(labels_true)
        clusters = np.unique(labels_pred)
        # Special limit cases: no clustering since the data is not split;
        # or trivial clustering where each document is assigned a unique cluster.
        # These are perfect matches hence return 1.0.
        if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
                or classes.shape[0] == clusters.shape[0] == len(labels_true)):
            return 1.0

        # Type: <type 'numpy.ndarray'>:rows are clusters, cols are classes
        contingency = self.contingency_matrix(labels_true, labels_pred)
        contingency = -contingency
        contingency = contingency.tolist()
        m = Munkres()  # Best mapping by using Kuhn-Munkres algorithm
        # best match to find the minimum cost
        map_pairs = m.compute(contingency)
        sum_value = 0
        for key, value in map_pairs:
            sum_value = sum_value + contingency[key][value]

        return float(-sum_value) / n_samples

    def f_measure(self, labels_true, labels_pred):  # Return the F1 score
        labels_true, labels_pred = self.check_clusterings(
            labels_true, labels_pred)
        classes = np.unique(labels_true)
        clusters = np.unique(labels_pred)
        # Special limit cases: no clustering since the data is not split;
        # or trivial clustering where each document is assigned a unique cluster.
        # These are perfect matches hence return 1.0.
        if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
                or classes.shape[0] == clusters.shape[0] == len(labels_true)):
            return 1.0

        contingency = self.contingency_matrix(labels_true, labels_pred)
        # Compute the ARI using the contingency data
        TP_plus_FP = sum(self.comb2(n_c)
                         for n_c in contingency.sum(axis=1))  # TP+FP

        TP_plus_FN = sum(self.comb2(n_k)
                         for n_k in contingency.sum(axis=0))  # TP+FN

        TP = sum(self.comb2(n_ij) for n_ij in contingency.flatten())  # TP

        P = float(TP) / TP_plus_FP
        R = float(TP) / TP_plus_FN

        return 2 * P * R / (P + R)

    def purity(self, labels_true, labels_pred):
        labels_true, labels_pred = self.check_clusterings(
            labels_true, labels_pred)
        n_samples = labels_true.shape[0]
        classes = np.unique(labels_true)
        clusters = np.unique(labels_pred)
        # Special limit cases: no clustering since the data is not split;
        # or trivial clustering where each document is assigned a unique cluster.
        # These are perfect matches hence return 1.0.
        if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
                or classes.shape[0] == clusters.shape[0] == len(labels_true)):
            return 1.0

        # Type: <type 'numpy.ndarray'>:rows are clusters, cols are classes
        contingency = self.contingency_matrix(labels_true, labels_pred)

        cluster_number = contingency.shape[0]
        sum_ = 0
        for k in range(0, cluster_number):
            row = contingency[k, :]
            max_ = np.max(row)
            sum_ += max_
        return float(sum_) / n_samples

    def rand_index(self, labels_true, labels_pred):
        labels_true, labels_pred = self.check_clusterings(
            labels_true, labels_pred)
        n_samples = labels_true.shape[0]
        classes = np.unique(labels_true)
        clusters = np.unique(labels_pred)
        # Special limit cases: no clustering since the data is not split;
        # or trivial clustering where each document is assigned a unique cluster.
        # These are perfect matches hence return 1.0.
        if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
                or classes.shape[0] == clusters.shape[0] == len(labels_true)):
            return 1.0

        contingency = self.contingency_matrix(labels_true, labels_pred)
        # Compute the ARI using the contingency data
        TP_plus_FP = sum(self.comb2(n_c)
                         for n_c in contingency.sum(axis=1))  # TP+FP

        TP_plus_FN = sum(self.comb2(n_k)
                         for n_k in contingency.sum(axis=0))  # TP+FN

        TP = sum(self.comb2(n_ij) for n_ij in contingency.flatten())  # TP
        FP = TP_plus_FP - TP
        FN = TP_plus_FN - TP
        sum_all = self.comb2(n_samples)
        TN = sum_all - TP - FP - FN

        return float(TP + TN) / (sum_all)

    def adjusted_rand_score(self, labels_true, labels_pred):
        labels_true, labels_pred = self.check_clusterings(
            labels_true, labels_pred)
        n_samples = labels_true.shape[0]
        classes = np.unique(labels_true)
        clusters = np.unique(labels_pred)
        # Special limit cases: no clustering since the data is not split;
        # or trivial clustering where each document is assigned a unique cluster.
        # These are perfect matches hence return 1.0.
        if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
                or classes.shape[0] == clusters.shape[0] == len(labels_true)):
            return 1.0

        contingency = self.contingency_matrix(labels_true, labels_pred)
        # Compute the ARI using the contingency data
        sum_comb_c = sum(self.comb2(n_c)
                         for n_c in contingency.sum(axis=1))  # TP+FP
        sum_comb_k = sum(self.comb2(n_k)
                         for n_k in contingency.sum(axis=0))  # TP+FN
        sum_comb = sum(self.comb2(n_ij)
                       for n_ij in contingency.flatten())  # TP
        prod_comb = (sum_comb_c * sum_comb_k) / float(comb(n_samples, 2))
        mean_comb = (sum_comb_k + sum_comb_c) / 2.

        return ((sum_comb - prod_comb) / (mean_comb - prod_comb))

    def mutual_info_score(self, labels_true, labels_pred, contingency=None):
        """Mutual Information between two clusterings
        See also
        --------
        adjusted_mutual_info_score: Adjusted against chance Mutual Information
        normalized_mutual_info_score: Normalized Mutual Information
        """
        if contingency is None:
            labels_true, labels_pred = self.check_clusterings(
                labels_true, labels_pred)
            contingency = self.contingency_matrix(labels_true, labels_pred)
        contingency = np.array(contingency, dtype='float')
        contingency_sum = np.sum(contingency)
        pi = np.sum(contingency, axis=1)
        pj = np.sum(contingency, axis=0)
        outer = np.outer(pi, pj)
        nnz = contingency != 0.0
        # normalized contingency
        contingency_nm = contingency[nnz]
        log_contingency_nm = np.log(contingency_nm)
        contingency_nm /= contingency_sum
        # log(a / b) should be calculated as log(a) - log(b) for
        # possible loss of precision
        log_outer = -np.log(outer[nnz]) + log(pi.sum()) + log(pj.sum())
        mi = (contingency_nm * (log_contingency_nm - log(contingency_sum))
              + contingency_nm * log_outer)
        return mi.sum()

    def normalized_mutual_info_score(self, labels_true, labels_pred):
        labels_true, labels_pred = self.check_clusterings(
            labels_true, labels_pred)
        classes = np.unique(labels_true)
        clusters = np.unique(labels_pred)
        # Special limit cases: no clustering since the data is not split.
        # This is a perfect match hence return 1.0.
        if (classes.shape[0] == clusters.shape[0] == 1
                or classes.shape[0] == clusters.shape[0] == 0):
            return 1.0
        contingency = self.contingency_matrix(labels_true, labels_pred)
        contingency = np.array(contingency, dtype='float')
        # Calculate the MI for the two clusterings
        mi = self.mutual_info_score(labels_true, labels_pred,
                                    contingency=contingency)
        # Calculate the expected value for the mutual information
        # Calculate entropy for each labeling
        h_true, h_pred = self.entropy(labels_true), self.entropy(labels_pred)
        nmi = mi / max(np.sqrt(h_true * h_pred), 1e-10)
        return nmi

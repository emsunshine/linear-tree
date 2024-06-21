import copy
import json
import torch

from .lineartree import (
    LinearTreeRegressor,
    LinearTreeClassifier,
    LinearBoostRegressor,
    LinearBoostClassifier,
    LinearForestRegressor,
    LinearForestClassifier,
)
from .linear_combinations import LinearCombinations

default_args = {
    'LCs': None,
    'num_terms': None,
    'symmetrize': True,
    'tol_decimals': 4,
    'torch_device': None,
    'max_index': None,
}

class HyperplaneMixin():
    """Automatically take hyperplanes of features
    
    A Mixin for sklearn-like models to automatically take hyperplanes of features
    before doing anything with them. The implemented functions should cover most
    sklearn-like models.

    Parameters
    ----------
    hyperplane_weights : tensor of float, default=None
        Tensor of hyperplane weights to use. If None, will auto-generate.

    num_terms : int, default=None
        Maximum number of terms to use if auto-generating hyperplane weights.

    symmetrize : bool, defualt = True
        Whether or not to take the symmetries of all hyperplane weights 
        (reflect to all possible combinations of axes).
        Highly recommended, unless you are interested in very specific hyperplanes
        whose weights you provide with the hyperplane_weights parameter.

    tol_decimals : int, default = 4
        How many decimals to consider when down-selecting to unique hyperplane weights

    torch_device : int, default = None
        torch device for any tensors generated for the tree.
        Should be the same device that your data will be on.

    max_hp_weight : int, default = 3
        Highest weight considered when auto-generating hyperplane weights.
        Has no effect if hyperplane_weights are provided.
        See .linear_combinations.generate_planes_to_index() for more info

    """
    def __init__(
            self,
            hyperplane_weights,
            num_terms,
            symmetrize,
            tol_decimals,
            torch_device,
            max_hp_weight,
            ):
        self.hyperplane_weights = hyperplane_weights
        self.num_terms = num_terms
        self.symmetrize = symmetrize
        self.tol_decimals = tol_decimals
        self.torch_device = torch_device
        self.max_hp_weight = max_hp_weight

        self.linear_combinations_transform = LinearCombinations(
            LCs = hyperplane_weights,
            num_terms = num_terms,
            symmetrize = symmetrize,
            tol_decimals = tol_decimals,
            torch_device = torch_device,
            max_hp_weight = max_hp_weight,
        )

    def do_lcs(self, X):
        return self.linear_combinations_transform.transform(X)
    
    def fit(self, X, y, *args):
        if self.linear_features is None:
            self.linear_features = torch.arange(start = 0, end=len(X[0]), device = self.torch_device, dtype = torch.int)

        X = self.do_lcs(X)
        return super().fit(X, y, *args)
    
    def predict(self, X, *args):
        X = self.do_lcs(X)
        return super().predict(X, *args)
    
    def apply(self, X, *args):
        X = self.do_lcs(X)
        return super().apply(X, *args)
    
    def decicion_path(self, X, *args):
        X = self.do_lcs(X)
        return super().decision_path(X, *args)
    
    def predict_proba(self, X, *args):
        X = self.do_lcs(X)
        return super().predict_proba(X, *args)
    
    def predict_log_proba(self, X, *args):
        X = self.do_lcs(X)
        return super().predict_log_proba(X, *args)
    
    def decision_function(self, X, *args):
        X = self.do_lcs(X)
        return super().decision_function(X, *args)
    
    def score(self, X, *args):
        X = self.do_lcs(X)
        return super().score(X, *args)
    
    def write_to_json(self, filename):
        out = copy.deepcopy(super().summary())

        for key, node in out.items():
            if 'col' in node.keys():
                node['col'] = node['col'].item()
                node['th'] = node['th'].item()

                node['models'] = [linear_model.__dict__ for linear_model in node['models']]
                for linear_model in node['models']:
                    linear_model['params'] = linear_model['params'].tolist()

            else:
                node['models'] = node['models'].__dict__

            node['loss'] = node['loss'].item()
            node['samples'] = node['samples'].item()

    
        out['LC_matrix'] = self.linear_combinations_transform.final_matrix.tolist()
        out['type'] = str(super())[36:-9]

        with open(filename, 'w') as outfile:
            json.dump(out, outfile)
        

class HyperplaneTreeRegressor(HyperplaneMixin, LinearTreeRegressor):
    """A Hyperplane Tree Regressor.

    A Hyperplane Tree Regressor is a meta-estimator that combine the learning
    ability of Decision Tree and the predictive power of Linear Models.
    Like in tree-based algorithms, the received data are splitted according
    simple decision rules. The goodness of slits is evaluated in gain terms
    fitting linear models in each node. This implies that the models in the
    leaves are linear instead of constant approximations like in classical
    Decision Tree.

    Parameters
    ----------
    criterion : {"mse", "rmse", "mae", "poisson", "msle"}, default="mse"
        The function to measure the quality of a split. "poisson"
        requires ``y >= 0``.

    max_depth : int, default=torch.inf
        The maximum depth of the tree considering only the splitting nodes.
        A higher value implies a higher training time.
        Can be used to regularize the tree.

    min_samples_split : int or float, default=6
        The minimum number of samples required to split an internal node.
        The minimum valid number of samples in each node is 6.
        A lower value implies a higher training time.
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=0.1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least `min_samples_leaf` training samples in each of the left and
        right branches.
        The minimum valid number of samples in each leaf is 3.
        Can be used to regularize the tree
        A lower value implies a higher training time.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    max_bins : int, default=25
        The maximum number of bins to use to search the optimal split in each
        feature. Features with a small number of unique values may use less than
        ``max_bins`` bins. Must be 3 or greater.
        A higher value implies a higher training time but increases expressivity.
        Values in the range of 10-120 are recommended.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

    categorical_features : tensor of int, default=None
        Indicates the categorical features.
        All categorical indices must be in `[0, n_features)`.
        Categorical features are used for splits but are not used in
        model fitting.
        More categorical features imply a higher training time.
        - None : no feature will be considered categorical.
        - integer array-like : integer indices indicating categorical
          features.
        - integer : integer index indicating a categorical
          feature.

    split_features : tensor of int, default=None
        Defines which features can be used to split on.
        All split feature indices must be in `[0, n_features)`.
        - None : All features will be used for splitting.
        - integer array-like : integer indices indicating splitting features.
        - integer : integer index indicating a single splitting feature.

    linear_features : tensor of int, default=None
        Defines which features are used for the linear model in the leaves.
        All linear feature indices must be in `[0, n_features)`.
        - None : All features except those in `categorical_features`
          will be used in the leaf models.
        - integer array-like : integer indices indicating features to
          be used in the leaf models.
        - integer : integer index indicating a single feature to be used
          in the leaf models.

    n_jobs : int, default=None
        The number of jobs to run in parallel for model fitting.
        ``None`` means 1 using one processor. ``-1`` means using all
        processors.

    hyperplane_weights : tensor of float, default=None
        Tensor of hyperplane weights to use. If None, will auto-generate.

    num_terms : int, default=None
        Maximum number of terms to use if auto-generating hyperplane weights.

    symmetrize : bool, defualt = True
        Whether or not to take the symmetries of all hyperplane weights 
        (reflect to all possible combinations of axes).
        Highly recommended, unless you are interested in very specific hyperplanes
        whose weights you provide with the hyperplane_weights parameter.

    tol_decimals : int, default = 4
        How many decimals to consider when down-selecting to unique hyperplane weights

    torch_device : int, default = None
        torch device for any tensors generated for the tree.
        Should be the same device that your data will be on.

    max_hp_weight : int, default = 3
        Highest weight considered when auto-generating hyperplane weights.
        Has no effect if hyperplane_weights are provided.
        See .linear_combinations.generate_planes_to_index() for more info


    Attributes
    ----------
    n_features_in_ : int
        The number of features when :meth:`fit` is performed.

    feature_importances_ : ndarray of shape (n_features, )
        Normalized total reduction of criteria by splitting features.

    n_targets_ : int
        The number of targets when :meth:`fit` is performed.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from hyperplanetree import HyperplaneTreeRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=4,s
    ...                        n_informative=2, n_targets=1,
    ...                        random_state=0, shuffle=False)
    >>> regr = HyperplaneTreeRegressor()
    >>> regr.fit(X, y)
    """
    def __init__(self,
        hyperplane_weights = None,
        num_terms = None,
        symmetrize = True,
        tol_decimals = 4,
        torch_device = None,
        max_hp_weight = 3,
        **kwargs
        ):

        LinearTreeRegressor.__init__(self, **kwargs)
        HyperplaneMixin.__init__(
            self,
            hyperplane_weights = hyperplane_weights,
            num_terms = num_terms,
            symmetrize = symmetrize,
            tol_decimals = tol_decimals,
            torch_device = torch_device,
            max_hp_weight = max_hp_weight,
            )


class HyperplaneTreeClassifier(HyperplaneMixin, LinearTreeClassifier):
    """A Hyperplane Tree Classifier.

    A Hyperplane Tree Classifier is a meta-estimator that combine the learning
    ability of Decision Tree and the predictive power of Linear Models.
    Like in tree-based algorithms, the received data are splitted according
    simple decision rules. The goodness of slits is evaluated in gain terms
    fitting linear models in each node. This implies that the models in the
    leaves are linear instead of constant approximations like in classical
    Decision Tree.

    Parameters
    ----------
    criterion : {"hamming", "crossentropy"}, default="hamming"
        The function to measure the quality of a split. `"crossentropy"`
        can be used only if `base_estimator` has `predict_proba` method.

    max_depth : int, default=torch.inf
        The maximum depth of the tree considering only the splitting nodes.
        A higher value implies a higher training time.

    min_samples_split : int or float, default=6
        The minimum number of samples required to split an internal node.
        The minimum valid number of samples in each node is 6.
        A lower value implies a higher training time.
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=0.1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least `min_samples_leaf` training samples in each of the left and
        right branches.
        The minimum valid number of samples in each leaf is 3.
        A lower value implies a higher training time.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

     max_bins : int, default=25
        The maximum number of bins to use to search the optimal split in each
        feature. Features with a small number of unique values may use less than
        ``max_bins`` bins. Must be 3 or greater.
        A higher value implies a higher training time but increases expressivity.
        Values in the range of 10-120 are recommended.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

    categorical_features : tensor of int, default=None
        Indicates the categorical features.
        All categorical indices must be in `[0, n_features)`.
        Categorical features are used for splits but are not used in
        model fitting.
        More categorical features imply a higher training time.
        - None : no feature will be considered categorical.
        - integer array-like : integer indices indicating categorical
          features.
        - integer : integer index indicating a categorical
          feature.

    split_features : tensor of int, default=None
        Defines which features can be used to split on.
        All split feature indices must be in `[0, n_features)`.
        - None : All features will be used for splitting.
        - integer array-like : integer indices indicating splitting features.
        - integer : integer index indicating a single splitting feature.

    linear_features : tensor of int, default=None
        Defines which features are used for the linear model in the leaves.
        All linear feature indices must be in `[0, n_features)`.
        - None : All features except those in `categorical_features`
          will be used in the leaf models.
        - integer array-like : integer indices indicating features to
          be used in the leaf models.
        - integer : integer index indicating a single feature to be used
          in the leaf models.

    n_jobs : int, default=None
        The number of jobs to run in parallel for model fitting.
        ``None`` means 1 using one processor. ``-1`` means using all
        processors.

    hyperplane_weights : tensor of float, default=None
        Tensor of hyperplane weights to use. If None, will auto-generate.

    num_terms : int, default=None
        Maximum number of terms to use if auto-generating hyperplane weights.

    symmetrize : bool, defualt = True
        Whether or not to take the symmetries of all hyperplane weights 
        (reflect to all possible combinations of axes).
        Highly recommended, unless you are interested in very specific hyperplanes
        whose weights you provide with the hyperplane_weights parameter.

    tol_decimals : int, default = 4
        How many decimals to consider when down-selecting to unique hyperplane weights

    torch_device : int, default = None
        torch device for any tensors generated for the tree.
        Should be the same device that your data will be on.

    max_hp_weight : int, default = 3
        Highest weight considered when auto-generating hyperplane weights.
        Has no effect if hyperplane_weights are provided.
        See .linear_combinations.generate_planes_to_index() for more info

    Attributes
    ----------
    n_features_in_ : int
        The number of features when :meth:`fit` is performed.

    feature_importances_ : ndarray of shape (n_features, )
        Normalized total reduction of criteria by splitting features.

    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    Examples
    """
    def __init__(self,
        hyperplane_weights = None,
        num_terms = None,
        symmetrize = True,
        tol_decimals = 4,
        torch_device = None,
        max_hp_weight = 3,
        **kwargs
        ):

        LinearTreeClassifier.__init__(self, **kwargs)
        HyperplaneMixin.__init__(
            self,
            hyperplane_weights = hyperplane_weights,
            num_terms = num_terms,
            symmetrize = symmetrize,
            tol_decimals = tol_decimals,
            torch_device = torch_device,
            max_hp_weight = max_hp_weight,
            )



class HyperplaneBoostRegressor(HyperplaneMixin, LinearBoostRegressor):
    """A Linear Boosting Regressor.

    A Linear Boosting Regressor is an iterative meta-estimator that starts
    with a linear regressor, and model the residuals through Hyperplane Trees.
    At each iteration, the path leading to highest error (i.e. the worst leaf)
    is added as a new binary variable to the base model. This kind of Linear
    Boosting can be considered as an improvement over general linear models
    since it enables incorporating non-linear features by residuals modeling.

    Parameters
    ----------
    loss : {"linear", "square", "absolute", "exponential"}, default="linear"
        The function used to calculate the residuals of each sample.

    n_estimators : int, default=10
        The number of boosting stages to perform. It corresponds to the number
        of the new features generated.

    max_depth : int, default=torch.inf
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    hyperplane_weights : tensor of float, default=None
        Tensor of hyperplane weights to use. If None, will auto-generate.

    num_terms : int, default=None
        Maximum number of terms to use if auto-generating hyperplane weights.

    symmetrize : bool, defualt = True
        Whether or not to take the symmetries of all hyperplane weights 
        (reflect to all possible combinations of axes).
        Highly recommended, unless you are interested in very specific hyperplanes
        whose weights you provide with the hyperplane_weights parameter.

    tol_decimals : int, default = 4
        How many decimals to consider when down-selecting to unique hyperplane weights

    torch_device : int, default = None
        torch device for any tensors generated for the tree.
        Should be the same device that your data will be on.

    max_hp_weight : int, default = 3
        Highest weight considered when auto-generating hyperplane weights.
        Has no effect if hyperplane_weights are provided.
        See .linear_combinations.generate_planes_to_index() for more info

    Attributes
    ----------
    n_features_in_ : int
        The number of features when :meth:`fit` is performed.

    n_features_out_ : int
        The total number of features used to fit the base estimator in the
        last iteration. The number of output features is equal to the sum
        of n_features_in_ and n_estimators.

    coef_ : array of shape (n_features_out_, ) or (n_targets, n_features_out_)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this is a
        2D array of shape (n_targets, n_features_out_), while if only one target
        is passed, this is a 1D array of length n_features_out_.

    intercept_ : float or array of shape (n_targets, )
        Independent term in the linear model. Set to 0 if `fit_intercept = False`
        in `base_estimator`
"""

    def __init__(self,
        hyperplane_weights = None,
        num_terms = None,
        symmetrize = True,
        tol_decimals = 4,
        torch_device = None,
        max_hp_weight = 3,
        **kwargs
        ):

        LinearBoostRegressor.__init__(self, **kwargs)
        HyperplaneMixin.__init__(
            self,
            hyperplane_weights = hyperplane_weights,
            num_terms = num_terms,
            symmetrize = symmetrize,
            tol_decimals = tol_decimals,
            torch_device = torch_device,
            max_hp_weight = max_hp_weight,
            )


class HyperplaneBoostClassifier(HyperplaneMixin, LinearBoostClassifier):
    """A Linear Boosting Classifier.

    A Linear Boosting Classifier is an iterative meta-estimator that starts
    with a linear classifier, and model the residuals through Hyperplane Trees.
    At each iteration, the path leading to highest error (i.e. the worst leaf)
    is added as a new binary variable to the base model. This kind of Linear
    Boosting can be considered as an improvement over general linear models
    since it enables incorporating non-linear features by residuals modeling.

    Parameters
    ----------
    loss : {"hamming", "entropy"}, default="entropy"
        The function used to calculate the residuals of each sample.
        `"entropy"` can be used only if `base_estimator` has `predict_proba`
        method.

    n_estimators : int, default=10
        The number of boosting stages to perform. It corresponds to the number
        of the new features generated.

    max_depth : int, default=torch.inf
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    hyperplane_weights : tensor of float, default=None
        Tensor of hyperplane weights to use. If None, will auto-generate.

    num_terms : int, default=None
        Maximum number of terms to use if auto-generating hyperplane weights.

    symmetrize : bool, defualt = True
        Whether or not to take the symmetries of all hyperplane weights 
        (reflect to all possible combinations of axes).
        Highly recommended, unless you are interested in very specific hyperplanes
        whose weights you provide with the hyperplane_weights parameter.

    tol_decimals : int, default = 4
        How many decimals to consider when down-selecting to unique hyperplane weights

    torch_device : int, default = None
        torch device for any tensors generated for the tree.
        Should be the same device that your data will be on.

    max_hp_weight : int, default = 3
        Highest weight considered when auto-generating hyperplane weights.
        Has no effect if hyperplane_weights are provided.
        See .linear_combinations.generate_planes_to_index() for more info

    Attributes
    ----------
    n_features_in_ : int
        The number of features when :meth:`fit` is performed.

    n_features_out_ : int
        The total number of features used to fit the base estimator in the
        last iteration. The number of output features is equal to the sum
        of n_features_in_ and n_estimators.

    coef_ : ndarray of shape (1, n_features_out_) or (n_classes, n_features_out_)
        Coefficient of the features in the decision function.

    intercept_ : float or array of shape (n_classes, )
        Independent term in the linear model. Set to 0 if `fit_intercept = False`
        in `base_estimator`

    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.
    """
    def __init__(self,
        hyperplane_weights = None,
        num_terms = None,
        symmetrize = True,
        tol_decimals = 4,
        torch_device = None,
        max_hp_weight = 3,
        **kwargs
        ):

        LinearBoostClassifier.__init__(self, **kwargs)
        HyperplaneMixin.__init__(
            self,
            hyperplane_weights = hyperplane_weights,
            num_terms = num_terms,
            symmetrize = symmetrize,
            tol_decimals = tol_decimals,
            torch_device = torch_device,
            max_hp_weight = max_hp_weight,
            )


class HyperplaneForestRegressor(HyperplaneMixin, LinearForestRegressor):
    """"A Hyperplane Forest Regressor.

    Linear forests generalizes the well known random forests by combining
    linear models with the same random forests with hyperplane splits.
    The key idea of linear forests is to use the strength of linear models
    to improve the nonparametric learning ability of tree-based algorithms.
    Firstly, a linear model is fitted on the whole dataset, then a random
    forest is trained on the same dataset but using the residuals of the
    previous steps as target. The final predictions are the sum of the raw
    linear predictions and the residuals modeled by the random forest.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        Only available if bootstrap=True.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0, 1]`.

    hyperplane_weights : tensor of float, default=None
        Tensor of hyperplane weights to use. If None, will auto-generate.

    num_terms : int, default=None
        Maximum number of terms to use if auto-generating hyperplane weights.

    symmetrize : bool, defualt = True
        Whether or not to take the symmetries of all hyperplane weights 
        (reflect to all possible combinations of axes).
        Highly recommended, unless you are interested in very specific hyperplanes
        whose weights you provide with the hyperplane_weights parameter.

    tol_decimals : int, default = 4
        How many decimals to consider when down-selecting to unique hyperplane weights

    torch_device : int, default = None
        torch device for any tensors generated for the tree.
        Should be the same device that your data will be on.

    max_hp_weight : int, default = 3
        Highest weight considered when auto-generating hyperplane weights.
        Has no effect if hyperplane_weights are provided.
        See .linear_combinations.generate_planes_to_index() for more info

    Attributes
    ----------
    n_features_in_ : int
        The number of features when :meth:`fit` is performed.

    feature_importances_ : ndarray of shape (n_features, )
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this is a
        2D array of shape (n_targets, n_features), while if only one target
        is passed, this is a 1D array of length n_features.

    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model. Set to 0 if `fit_intercept = False`
        in `base_estimator`.

    base_estimator_ : object
        A fitted linear model instance.

    forest_estimator_ : object
        A fitted random forest instance.
    """
    def __init__(self,
        hyperplane_weights = None,
        num_terms = None,
        symmetrize = True,
        tol_decimals = 4,
        torch_device = None,
        max_hp_weight = 3,
        **kwargs
        ):
    

        LinearForestRegressor.__init__(self, **kwargs)
        HyperplaneMixin.__init__(
            self,
            hyperplane_weights = hyperplane_weights,
            num_terms = num_terms,
            symmetrize = symmetrize,
            tol_decimals = tol_decimals,
            torch_device = torch_device,
            max_hp_weight = max_hp_weight,
            )


class HyperplaneForestClassifier(HyperplaneMixin, LinearForestClassifier):
    """"A Linear Forest Classifier.

    Linear forests generalizes the well known random forests by combining
    linear models with the same random forests.
    The key idea of linear forests is to use the strength of linear models
    to improve the nonparametric learning ability of tree-based algorithms.
    Firstly, a linear model is fitted on the whole dataset, then a random
    forest is trained on the same dataset but using the residuals of the
    previous steps as target. The final predictions are the sum of the raw
    linear predictions and the residuals modeled by the random forest.

    For classification tasks the same approach used in regression context
    is adopted. The binary targets are transformed into logits using the
    inverse sigmoid function. A linear regression is fitted. A random forest
    regressor is trained to approximate the residulas from logits and linear
    predictions. Finally the sigmoid of the combinded predictions are taken
    to obtain probabilities.
    The multi-label scenario is carried out using OneVsRestClassifier.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        Only available if bootstrap=True.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0, 1]`.

        hyperplane_weights : tensor of float, default=None
        Tensor of hyperplane weights to use. If None, will auto-generate.

    num_terms : int, default=None
        Maximum number of terms to use if auto-generating hyperplane weights.

    symmetrize : bool, defualt = True
        Whether or not to take the symmetries of all hyperplane weights 
        (reflect to all possible combinations of axes).
        Highly recommended, unless you are interested in very specific hyperplanes
        whose weights you provide with the hyperplane_weights parameter.

    tol_decimals : int, default = 4
        How many decimals to consider when down-selecting to unique hyperplane weights

    torch_device : int, default = None
        torch device for any tensors generated for the tree.
        Should be the same device that your data will be on.

    max_hp_weight : int, default = 3
        Highest weight considered when auto-generating hyperplane weights.
        Has no effect if hyperplane_weights are provided.
        See .linear_combinations.generate_planes_to_index() for more info

    Attributes
    ----------
    n_features_in_ : int
        The number of features when :meth:`fit` is performed.

    feature_importances_ : ndarray of shape (n_features, )
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

    coef_ : ndarray of shape (1, n_features_out_)
        Coefficient of the features in the decision function.

    intercept_ : float
        Independent term in the linear model. Set to 0 if `fit_intercept = False`
        in `base_estimator`.

    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    base_estimator_ : object
        A fitted linear model instance.

    forest_estimator_ : object
        A fitted random forest instance.
    """
        
    def __init__(self,
        hyperplane_weights = None,
        num_terms = None,
        symmetrize = True,
        tol_decimals = 4,
        torch_device = None,
        max_hp_weight = 3,
        **kwargs
        ):

        LinearForestClassifier.__init__(self, **kwargs)
        HyperplaneMixin.__init__(
            self,
            hyperplane_weights = hyperplane_weights,
            num_terms = num_terms,
            symmetrize = symmetrize,
            tol_decimals = tol_decimals,
            torch_device = torch_device,
            max_hp_weight = max_hp_weight,
            )

import copy
import json
import torch
import inspect

from .utils import recursive_to_device

from .lineartree import (
    LinearTreeRegressor,
    LinearTreeClassifier,
    LinearBoostRegressor,
    LinearBoostClassifier,
    LinearForestRegressor,
    LinearForestClassifier,
)
from .linear_combinations import LinearCombinations


class HyperplaneMixin():
    """Automatically take hyperplanes of features

    A Mixin for sklearn-like models to automatically take hyperplanes of features
    before doing anything with them. The implemented functions should cover most
    sklearn-like models.

    Parameters
    ----------
    """
    parameter_docstring = \
    """
    hyperplane_weights : tensor of float, default=None
        Tensor of hyperplane weights to use. If None, will auto-generate.

    num_terms : int, default=None
        Maximum number of terms to use if auto-generating hyperplane weights.

    do_symmetrize : bool, defualt = True
        Whether or not to take the symmetries of all hyperplane weights 
        (reflect to all possible combinations of axes).
        Highly recommended, unless you are interested in very specific hyperplanes
        whose weights you provide with the hyperplane_weights parameter.

    do_scaling: bool (default: True)
        Automatically scale weights in LCs to correspond to maximum and minimum values in data

    tol_decimals : int, default = 4
        How many decimals to consider when down-selecting to unique hyperplane weights

    torch_device : int, default = None
        torch device for any tensors generated for the tree.
        Should be the same device that your data will be on.

    max_hp_weight : int, default = 3
        Highest weight considered when auto-generating hyperplane weights.
        Has no effect if hyperplane_weights are provided.
        See .linear_combinations.generate_planes_to_index() for more info

    disable_tqdm : bool, default = False
        Disable progress bars powered by TQDM
    """
    __doc__ += parameter_docstring
    
    def __init__(
            self,
            hyperplane_weights = None,
            num_terms = 2,
            do_symmetrize = True,
            do_scaling = True,
            tol_decimals = 4,
            torch_device = 'cpu',
            max_hp_weight = 1,
            disable_tqdm = False,
            ):
                
        self.hyperplane_weights = hyperplane_weights
        self.num_terms = num_terms
        self.do_symmetrize = do_symmetrize
        self.tol_decimals = tol_decimals
        self.torch_device = torch_device
        self.max_hp_weight = max_hp_weight
        self.do_scaling = do_scaling

        self.disable_tqdm = disable_tqdm

        self.linear_combinations_transform = LinearCombinations(
            LCs = hyperplane_weights,
            num_terms = num_terms,
            do_symmetrize = do_symmetrize,
            tol_decimals = tol_decimals,
            torch_device = torch_device,
            max_hp_weight = max_hp_weight,
            do_scaling = do_scaling,
        )

    def do_lcs(self, X):
        return self.linear_combinations_transform.transform(X)
    
    def fit(self, X, y, *args):
        if not hasattr(self, 'linear_features') or self.linear_features is None:
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
        out = {}
        out['nodes'] = copy.deepcopy(super().summary())

        for key, node in out['nodes'].items():
            if 'col' in node.keys():
                #Splitting node
                node['col'] = node['col'].item()
                node['th'] = node['th'].item()
                del node['models']

            else:
                #leaf node
                node['model'] = node['models'].__dict__
                node['model']['params'] = node['model']['params'].to('cpu').tolist()
                del node['models']

            node['loss'] = node['loss'].item()
            node['samples'] = node['samples'].item()

    
        out['hyperplanes_final_matrix'] = self.linear_combinations_transform.final_matrix.to('cpu').tolist()
        out['type'] = str(super())[36:-9]
        out['categorical_features'] = self._categorical_features.to('cpu').tolist()
        out['linear_features'] = self._linear_features.to('cpu').tolist()
        out['split_features'] = self._split_features.to('cpu').tolist()
        out['criterion'] = self.criterion
        out['max_depth'] = self.max_depth
        out['min_samples_leaf'] = self.min_samples_leaf
        out['n_features_in'] = self.n_features_in_
        out['n_targets'] = self.n_targets_

        with open(filename, 'w') as outfile:
            json.dump(out, outfile)

    def to(self, device):
        self.torch_device = device
        return recursive_to_device(self, device)


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
    """
    __doc__ += HyperplaneMixin.parameter_docstring + LinearTreeRegressor.parameter_docstring

    def __init__(self, **kwargs):
        hp_sig = inspect.signature(HyperplaneMixin.__init__)
        lt_sig = inspect.signature(LinearTreeRegressor.__init__)

        hp_kwargs = {}
        lt_kwargs = {}

        for key, value in kwargs.items():
            if key in hp_sig.parameters.keys():
                hp_kwargs[key] = value
            elif key in lt_sig.parameters.keys():
                lt_kwargs[key] = value
            else:
                raise AttributeError(f'Unknown keyword argument: {key}')

        HyperplaneMixin.__init__(self, **hp_kwargs)
        LinearTreeRegressor.__init__(self, **lt_kwargs)

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
    """
    __doc__ += HyperplaneMixin.parameter_docstring + LinearTreeClassifier.parameter_docstring
  
    def __init__(self, **kwargs):
        hp_sig = inspect.signature(HyperplaneMixin.__init__)
        lt_sig = inspect.signature(LinearTreeClassifier.__init__)

        hp_kwargs = {}
        lt_kwargs = {}

        for key, value in kwargs.items():
            if key in hp_sig.parameters.keys():
                hp_kwargs[key] = value
            elif key in lt_sig.parameters.keys():
                lt_kwargs[key] = value
            else:
                raise AttributeError(f'Unknown keyword argument: {key}')
            
        HyperplaneMixin.__init__(self, *hp_kwargs)
        LinearTreeClassifier.__init__(self, *lt_kwargs)



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
    """
    __doc__ += HyperplaneMixin.parameter_docstring + LinearBoostRegressor.parameter_docstring

    def __init__(self, **kwargs):
        hp_sig = inspect.signature(HyperplaneMixin.__init__)
        lt_sig = inspect.signature(LinearBoostRegressor.__init__)

        hp_kwargs = {}
        lt_kwargs = {}

        for key, value in kwargs.items():
            if key in hp_sig.parameters.keys():
                hp_kwargs[key] = value
            elif key in lt_sig.parameters.keys():
                lt_kwargs[key] = value
            else:
                raise AttributeError(f'Unknown keyword argument: {key}')
            
        HyperplaneMixin.__init__(self, *hp_kwargs)
        LinearBoostRegressor.__init__(self, *lt_kwargs)


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
    """
    __doc__ += HyperplaneMixin.parameter_docstring + LinearBoostClassifier.parameter_docstring

    def __init__(self, **kwargs):
        hp_sig = inspect.signature(HyperplaneMixin.__init__)
        lt_sig = inspect.signature(LinearBoostClassifier.__init__)

        hp_kwargs = {}
        lt_kwargs = {}

        for key, value in kwargs.items():
            if key in hp_sig.parameters.keys():
                hp_kwargs[key] = value
            elif key in lt_sig.parameters.keys():
                lt_kwargs[key] = value
            else:
                raise AttributeError(f'Unknown keyword argument: {key}')
            
        HyperplaneMixin.__init__(self, *hp_kwargs)
        LinearBoostClassifier.__init__(self, *lt_kwargs)


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
    """
    __doc__ += HyperplaneMixin.parameter_docstring + LinearForestRegressor.parameter_docstring

    def __init__(self, **kwargs):
        hp_sig = inspect.signature(HyperplaneMixin.__init__)
        lt_sig = inspect.signature(LinearForestRegressor.__init__)

        hp_kwargs = {}
        lt_kwargs = {}

        for key, value in kwargs.items():
            if key in hp_sig.parameters.keys():
                hp_kwargs[key] = value
            elif key in lt_sig.parameters.keys():
                lt_kwargs[key] = value
            else:
                raise AttributeError(f'Unknown keyword argument: {key}')
            
        HyperplaneMixin.__init__(self, *hp_kwargs)
        LinearForestRegressor.__init__(self, *lt_kwargs)


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
    """
    __doc__ += HyperplaneMixin.parameter_docstring + LinearForestClassifier.parameter_docstring

    def __init__(self, **kwargs):
        hp_sig = inspect.signature(HyperplaneMixin.__init__)
        lt_sig = inspect.signature(LinearForestClassifier.__init__)

        hp_kwargs = {}
        lt_kwargs = {}

        for key, value in kwargs.items():
            if key in hp_sig.parameters.keys():
                hp_kwargs[key] = value
            elif key in lt_sig.parameters.keys():
                lt_kwargs[key] = value
            else:
                raise AttributeError(f'Unknown keyword argument: {key}')
            
        HyperplaneMixin.__init__(self, *hp_kwargs)
        LinearForestClassifier.__init__(self, *lt_kwargs)
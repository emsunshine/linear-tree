import copy
import json

from lineartree import (
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
    def __init__(
            self,
            LCs,
            num_terms,
            symmetrize,
            tol_decimals,
            torch_device,
            max_index,
            ):
        self.LCs = LCs
        self.num_terms = num_terms
        self.symmetrize = symmetrize
        self.tol_decimals = tol_decimals
        self.torch_device = torch_device
        self.max_index = max_index

        self.linear_combinations_transform = LinearCombinations(
            LCs = LCs,
            num_terms = num_terms,
            symmetrize = symmetrize,
            tol_decimals = tol_decimals,
            torch_device = torch_device,
            max_index = max_index,
        )

    def do_lcs(self, X):
        return self.linear_combinations_transform.transform(X)
    
    def fit(self, X, y, *args):
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
    def __init__(self,
        LCs = None,
        num_terms = None,
        symmetrize = True,
        tol_decimals = 4,
        torch_device = None,
        max_index = 3,
        **kwargs
        ):

        LinearTreeRegressor.__init__(self, **kwargs)
        HyperplaneMixin.__init__(self, LCs, num_terms, symmetrize, tol_decimals, torch_device, max_index)


class HyperplaneTreeClassifier(HyperplaneMixin, LinearTreeClassifier):
    def __init__(self,
        LCs = None,
        num_terms = None,
        symmetrize = True,
        tol_decimals = 4,
        torch_device = None,
        max_index = 3,
        **kwargs
        ):

        LinearTreeClassifier.__init__(self, **kwargs)
        HyperplaneMixin.__init__(self, LCs, num_terms, symmetrize, tol_decimals, torch_device, max_index)


class HyperplaneBoostRegressor(HyperplaneMixin, LinearBoostRegressor):
    def __init__(self,
        LCs = None,
        num_terms = None,
        symmetrize = True,
        tol_decimals = 4,
        torch_device = None,
        max_index = 3,
        **kwargs
        ):

        LinearBoostRegressor.__init__(self, **kwargs)
        HyperplaneMixin.__init__(self, LCs, num_terms, symmetrize, tol_decimals, torch_device, max_index)

class HyperplaneBoostClassifier(HyperplaneMixin, LinearBoostClassifier):
    def __init__(self,
        LCs = None,
        num_terms = None,
        symmetrize = True,
        tol_decimals = 4,
        torch_device = None,
        max_index = 3,
        **kwargs
        ):

        LinearBoostClassifier.__init__(self, **kwargs)
        HyperplaneMixin.__init__(self, LCs, num_terms, symmetrize, tol_decimals, torch_device, max_index)

class HyperplaneForestRegressor(HyperplaneMixin, LinearForestRegressor):
    def __init__(self,
        LCs = None,
        num_terms = None,
        symmetrize = True,
        tol_decimals = 4,
        torch_device = None,
        max_index = 3,
        **kwargs
        ):

        LinearForestRegressor.__init__(self, **kwargs)
        HyperplaneMixin.__init__(self, LCs, num_terms, symmetrize, tol_decimals, torch_device, max_index)

class HyperplaneForestClassifier(HyperplaneMixin, LinearForestClassifier):
    def __init__(self,
        LCs = None,
        num_terms = None,
        symmetrize = True,
        tol_decimals = 4,
        torch_device = None,
        max_index = 3,
        **kwargs
        ):

        LinearForestClassifier.__init__(self, **kwargs)
        HyperplaneMixin.__init__(self, LCs, num_terms, symmetrize, tol_decimals, torch_device, max_index)
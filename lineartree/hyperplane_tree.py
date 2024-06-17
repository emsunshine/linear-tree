from lineartree import (
    LinearTreeRegressor,
    LinearTreeClassifier,
    LinearBoostRegressor,
    LinearBoostClassifier,
    LinearForestRegressor,
    LinearForestClassifier,
)
from .linear_combinations import LinearCombinations

class HyperplaneMixin():
    def __init__(
            self,
            LCs,
            num_terms,
            symmetrize,
            tol_decimals,
            torch_device
            ):
        self.LCs = LCs
        self.num_terms = num_terms
        self.symmetrize = symmetrize
        self.tol_decimals = tol_decimals
        self.torch_device = torch_device
        self.linear_combinations_transform = LinearCombinations(
            LCs,
            num_terms,
            symmetrize,
            tol_decimals,
            torch_device
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
        

class HyperplaneTreeRegressor(HyperplaneMixin, LinearTreeRegressor):
    def __init__(self,
        LCs = None,
        num_terms = None,
        symmetrize = True,
        tol_decimals = 4,
        torch_device = None,
        **kwargs
        ):

        LinearTreeRegressor.__init__(self, **kwargs)
        HyperplaneMixin.__init__(self, LCs, num_terms, symmetrize, tol_decimals, torch_device)


class HyperplaneTreeClassifier(HyperplaneMixin, LinearTreeClassifier):
    def __init__(self,
        LCs = None,
        num_terms = None,
        symmetrize = True,
        tol_decimals = 4,
        torch_device = None,
        **kwargs
        ):

        LinearTreeClassifier.__init__(self, **kwargs)
        HyperplaneMixin.__init__(self, LCs, num_terms, symmetrize, tol_decimals, torch_device)


class HyperplaneBoostRegressor(HyperplaneMixin, LinearBoostRegressor):
    def __init__(self,
        LCs = None,
        num_terms = None,
        symmetrize = True,
        tol_decimals = 4,
        torch_device = None,
        **kwargs
        ):

        LinearBoostRegressor.__init__(self, **kwargs)
        HyperplaneMixin.__init__(self, LCs, num_terms, symmetrize, tol_decimals, torch_device)

class HyperplaneBoostClassifier(HyperplaneMixin, LinearBoostClassifier):
    def __init__(self,
        LCs = None,
        num_terms = None,
        symmetrize = True,
        tol_decimals = 4,
        torch_device = None,
        **kwargs
        ):

        LinearBoostClassifier.__init__(self, **kwargs)
        HyperplaneMixin.__init__(self, LCs, num_terms, symmetrize, tol_decimals, torch_device)

class HyperplaneForestRegressor(HyperplaneMixin, LinearForestRegressor):
    def __init__(self,
        LCs = None,
        num_terms = None,
        symmetrize = True,
        tol_decimals = 4,
        torch_device = None,
        **kwargs
        ):

        LinearForestRegressor.__init__(self, **kwargs)
        HyperplaneMixin.__init__(self, LCs, num_terms, symmetrize, tol_decimals, torch_device)

class HyperplaneForestClassifier(HyperplaneMixin, LinearForestClassifier):
    def __init__(self,
        LCs = None,
        num_terms = None,
        symmetrize = True,
        tol_decimals = 4,
        torch_device = None,
        **kwargs
        ):

        LinearForestClassifier.__init__(self, **kwargs)
        HyperplaneMixin.__init__(self, LCs, num_terms, symmetrize, tol_decimals, torch_device)
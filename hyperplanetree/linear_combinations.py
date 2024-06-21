import torch
import itertools
import typing

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LinearRegression

class TorchLinearRegression(LinearRegression):
    def __init__(self):
        super().__init__()
    def fit(self, x, y, sample_weight = None):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        if self.fit_intercept:
            x = torch.hstack((torch.ones((len(x),1), device = x.device), x))

        if sample_weight is None:
            self.params = torch.linalg.pinv(x.T @ x) @ (x.T @ y)
        else:
            self.params = torch.linalg.pinv(x.T @ sample_weight @ x) @ (x.T @ sample_weight @ y)

    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        if self.fit_intercept:
            x = torch.hstack((torch.ones((len(x),1), device = x.device), x))
        return x @ self.params

class LinearCombinations(TransformerMixin, BaseEstimator):
    """SKLearn transformer for finite number of linear combinations

    An SKLearn transformer for feature engineering that returns a finite
    number of linear combinations of the input features.

    Parameters
    ----------
    LCs : array_like or None
        List of all linear combination weights

    num_terms: int or None
        Number of terms in each linear combination
        If None, determined automatically from shape of LCs
        Currently, only 2 is supported

    symmetrize: bool
        Whether or not to symmetrize the input LCs. e.g. if you enter [[1, 2]]
        you will get [[1, -2], [1, -0.5], [1, 0.5], [1,2]].

    tol_decimals: int
        How many decimal places of precision. Useful when reducing degeneracies
        after symmetrization step.
    """
    def __init__(self,
                 LCs = None,
                 num_terms = None,
                 symmetrize = True,
                 tol_decimals = 4,
                 torch_device = None,
                 max_hp_weight = None,
                 ):
        
        if LCs is not None:
            assert type(LCs) == torch.Tensor
        
        # Set torch device
        if torch_device is None and LCs is not None:
            torch_device = LCs.device
        elif torch_device is None and LCs is None:
            torch_device = 'cpu'

        if LCs is None and num_terms is None:
            num_terms = 2

        if LCs is None:
            if max_hp_weight is None:
                max_hp_weight = 3

            if max_hp_weight == 0:
                LCs = torch.eye(num_terms, device = torch_device)

            else:
                LCs = generate_planes_to_index(dimension = num_terms, max_hp_weight = max_hp_weight, device = torch_device)

        if num_terms is None:
            num_terms = len(LCs[0])

        assert len(LCs[0]) == num_terms


        if symmetrize:
            # Symmetrize +/- parity
            parity_matrix = torch.Tensor(tuple(itertools.product([1, -1], repeat = num_terms))).to(torch_device)
            parity_matrix = parity_matrix[:, None, :]
            LCs = torch.reshape(LCs * parity_matrix, (-1, num_terms))

            # Symmetrize permutations
            permutations_matrix = torch.Tensor(tuple((itertools.permutations(range(num_terms))))).to(torch_device).type(torch.int)
            LCs = torch.reshape(LCs[:, permutations_matrix], (-1, num_terms))

            # Remove LCs with non-trailing zeros
            previous_was_zero = torch.zeros(len(LCs), dtype=bool, device = torch_device)
            keep = torch.ones(len(LCs), dtype=bool, device = torch_device)
            for i in range(num_terms):
                keep = torch.logical_and(torch.logical_not(torch.logical_and(previous_was_zero, LCs[:, i] != 0)), keep)
                previous_was_zero = LCs[:, i] == 0

            LCs = LCs[keep]

            # Normalize all combinations
            LCs = (LCs.T / LCs[:, 0]).T

            # Only take unique LCs
            LCs = torch.unique(torch.round(LCs.to('cpu'), decimals=tol_decimals), dim=0).to(torch_device)

        self.LCs = LCs
        self.symmetrize = symmetrize
        self.num_terms = num_terms
        self.tol_decimals = tol_decimals
        self.torch_device = torch_device
        self.final_matrix = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            
        if (self.final_matrix is None) or (X.shape[1] != len(self.final_matrix)):

            num_cols = X.shape[1]
            num_out_cols = int(num_cols + torch.math.factorial(num_cols) / (torch.math.factorial(num_cols - self.num_terms)) * len(self.LCs))

            perms = itertools.permutations(range(num_cols), self.num_terms)
            final_matrix = torch.zeros((num_out_cols, num_cols), device = X.device)
            final_matrix[:num_cols] = torch.eye(num_cols)

            for i, indices in enumerate(perms):
                for j, LC in enumerate(self.LCs):
                    final_matrix[num_cols + i*len(self.LCs) + j, indices] = LC


            # find first nonzero entry in each row and normalize
            leftmost_nonzero = torch.argmax((final_matrix != 0).type(torch.int), axis=1)
            leftmost_nonzero = final_matrix[range(len(final_matrix)), leftmost_nonzero]
            final_matrix = (final_matrix.T / leftmost_nonzero).T

            # take only unique rows in final matrix
            final_matrix = torch.unique(torch.round(final_matrix[num_cols:].to('cpu'), decimals=self.tol_decimals), dim=0).to(X.device)

            self.final_matrix = torch.zeros((num_cols+len(final_matrix), num_cols), device = X.device)
            self.final_matrix[:num_cols] = torch.eye(num_cols, device = X.device)
            self.final_matrix[num_cols:] = final_matrix

            self.final_matrix = self.final_matrix.T.type(X.dtype)
        return X @ self.final_matrix
    
def generate_angular_lcs_2d(
        divisions,
        device = 'cpu'
        ):
    """Generate lines in two dimensions with equal angle spacing

    Parameters
    ----------
        divisions : int, 1 or greater
            How many lines in each quadrant

        device : torch.device or str
            device where the tensor will be created

    Returns
    -------
    spacing : torch.Tensor 
        intended to be used as hyperplane_weights when initializing a tree
    
    """
    assert divisions >= 1

    spacing = torch.Tensor([[torch.sin(x), torch.cos(x)] for x in torch.linspace(0, torch.pi/2, divisions+2)])[1:-1]
    spacing = (spacing.T / spacing[:,0]).T.to(device)
    return spacing

def generate_planes_to_index(
        dimension: int, 
        max_hp_weight: int=3,
        device = 'cpu',
        tol_decimals: int=4
        ):
    """Generate hyperplanes based on Miller index-like system

    Generates all possible planes with integer weights up to
    and including the specified max.
    Automatically reduces degenerate weight combinations.
    Automatically normalizes so the highest magnitude weight is 1.
    Does not produce negative weights.
    It is highly recommended to use this with "symmetrize" set to True
    in your tree initialization arguments to obtain all symmetries.

    Parameters
    ----------
    dimension : int
        How many terms you want in your linear combinations

    max_hp_weight : int
        Highest possible weight in the generated planes

    Returns
    -------
    out : torch.Tensor 
        intended to be used as hyperplane_weights when initializing a tree

    Example
    -------
    dimension = 2, max_index = 3 ==>

    [
        [1.0000, 0.0000], # (1, 0) plane
        [1.0000, 0.3333], # (3, 1) plane
        [1.0000, 0.5000], # (2, 1) plane
        [1.0000, 0.6667], # (3, 2) plane
        [1.0000, 1.0000], # (1, 1) plane
    ]
    
    """

    out = itertools.combinations_with_replacement(range(max_hp_weight, -1 , -1), dimension)
    out = torch.Tensor(list(out))[:-1]
    out = (out.T / out[:, 0]).T
    out.round(decimals = tol_decimals)
    out = torch.unique(out, dim = 0).to(device)
    return out
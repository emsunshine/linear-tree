import json
import torch

from sklearn.linear_model import LinearRegression

from ._classes import Node

def recursive_to_device(_self, device):
    if isinstance(_self, torch.Tensor):
        _self = _self.to(device)

    elif isinstance(_self, tuple):
        _self = tuple((recursive_to_device(x, device) for x in _self))

    elif isinstance(_self, list):
        _self = [recursive_to_device(x, device) for x in _self]

    elif isinstance(_self, dict):
        for key, value in _self.items():
            recursive_to_device(value, device)

    elif hasattr(_self, '__dict__'):
        for key, attr in _self.__dict__.items():
            if hasattr(attr, 'to'):
                setattr(_self, key, attr.to(device))

            elif isinstance(attr, dict):
                for key2, value in attr.items():
                    recursive_to_device(value, device)

            elif isinstance(attr, list):
                setattr(_self, key, [recursive_to_device(x, device) for x in attr])

            elif isinstance(attr, tuple):
                setattr(_self, key, tuple((recursive_to_device(x, device) for x in attr)))

    return _self

class TorchLinearRegression(LinearRegression):
    def __init__(self):
        super().__init__()
    def fit(self, x, y, sample_weight = None):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        if self.fit_intercept:
            x = torch.hstack((torch.ones((len(x),1), device = x.device), x))

        if sample_weight is None:
            self.params = torch.linalg.inv(x.T @ x) @ (x.T @ y)
        else:
            self.params = torch.linalg.inv(x.T @ sample_weight @ x) @ (x.T @ sample_weight @ y)

    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        if self.fit_intercept:
            x = torch.hstack((torch.ones((len(x),1), device = x.device), x))
        return x @ self.params
    
    def to(self, device):
        return recursive_to_device(self, device)
    
    @property
    def intercept_(self):
        return self.params[0]
    
    @property
    def coef_(self):
        return self.params[1:]
    
def tree_from_json(filename, torch_device = 'cpu'):
    """Load a tree from a json file.

    Parameters
    ----------
    filename : str
        The path to the json file.

    Returns
    -------
    tree : dict
        The tree in a dictionary format.
    """

    from hyperplanetree.hyperplane_tree import HyperplaneTreeRegressor

    params = json.load(open(filename, 'r'))
    params['_leaves'] = {}
    params['_nodes'] = {}

    params['base_estimator'] = TorchLinearRegression()

    for key, value in params['nodes'].items():
        if 'model' in value:
            # Node is leaf node
            node = Node(
                id = int(key),
                model = type(params['base_estimator'])(),
                loss = torch.tensor(value['loss'], device=torch_device),
                n_samples = torch.tensor(value['samples'], device=torch_device),
                #w_loss = torch.tensor(value['w_loss'], device=torch_device)
        )
            node.model.params = torch.Tensor(value['model']['params']).to(torch_device)
            node.threshold = None
            params['_leaves'][int(key)] = node

        else:
            # Node is splitting node
            node = Node(
                id = int(key),
                model = type(params['base_estimator'])(),
                loss = torch.tensor(value['loss'], device=torch_device),
                n_samples = torch.tensor(value['samples'], device=torch_device),
                #w_loss = torch.tensor(value['w_loss'], device=torch_device),
                children = (int(value['children'][0]), int(value['children'][1])),
        )
            node.col = value['col']
            node.th = value['th']
            params['_nodes'][int(key)] = node

    for id, node in params['_nodes'].items():
        if id == '0':
            node.threshold = []

        cur_thresh = node.threshold
        new_row_L = (
            torch.tensor(node.col).to(torch_device),
            'L',
            torch.tensor(node.th).to(torch_device),
        )
        
        new_thresh_L = cur_thresh + [new_row_L]

        if node.children[0] in params['_nodes'].keys():
            params['_nodes'][node.children[0]].threshold = new_thresh_L
        elif node.children[0] in params['_leaves'].keys():
            params['_leaves'][node.children[0]].threshold = new_thresh_L

        new_row_R = (
            torch.tensor(node.col).to(torch_device),
            'R',
            torch.tensor(node.th).to(torch_device),
        )
        
        new_thresh_R = cur_thresh + [new_row_R]

        if node.children[1] in params['_nodes'].keys():
            params['_nodes'][node.children[1]].threshold = new_thresh_R
        elif node.children[1] in params['_leaves'].keys():
            params['_leaves'][node.children[1]].threshold = new_thresh_R

    tree = HyperplaneTreeRegressor(
        criterion = params['criterion'],
        max_depth = params['max_depth'],
        min_samples_leaf = params['min_samples_leaf'],
        categorical_features = torch.LongTensor(params['categorical_features']).to(torch_device),
        linear_features = torch.LongTensor(params['linear_features']).to(torch_device),
        split_features = torch.LongTensor(params['split_features']).to(torch_device),
        )
    
    tree.linear_combinations_transform.final_matrix = torch.FloatTensor(params['hyperplanes_final_matrix']).to(torch_device)
        
    tree.n_features_in_ = params['n_features_in']
    tree.n_targets_ = params['n_targets']
    tree._linear_features = torch.LongTensor(params['linear_features']).to(torch_device)

    tree._nodes = params['_nodes']
    tree._leaves = params['_leaves']

    return tree
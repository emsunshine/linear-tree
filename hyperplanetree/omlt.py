import copy
import numpy as np
import torch

from pyomo.environ import ConstraintList
from omlt.linear_tree import (
    LinearTreeDefinition,
    LinearTreeGDPFormulation,
    LinearTreeHybridBigMFormulation
)

from .utils import TorchLinearRegression

class HyperplaneTreeDefinition(LinearTreeDefinition):
    """OMLT Definition for Hyperplane Trees
    """
    def __init__(
        self,
        lt_regressor,
        input_bounds_matrix = None,
        scaling_object = None,
    ):
        fm = lt_regressor.linear_combinations_transform.final_matrix
        max_bound = torch.max(torch.abs(torch.abs(input_bounds_matrix @ fm))).item()

        input_bounds = {}
        for i, row in enumerate(fm.T):
            if i < len(input_bounds_matrix):
                input_bounds[i] = tuple(input_bounds_matrix[i].tolist())
            else:
                input_bounds[i] = (-max_bound, max_bound)

        summary = copy.deepcopy(lt_regressor.summary())
        for node in summary.values():
            if 'col' in node.keys():
                node['col'] = node['col'].item()

            if 'th' in node.keys():
                node['th'] = node['th'].item()

            if isinstance(node['models'], TorchLinearRegression):
                node['models'].params = node['models'].params.tolist() + list(np.zeros(len(input_bounds) - len(input_bounds_matrix)))

        super().__init__(
            lt_regressor = summary,
            unscaled_input_bounds = input_bounds,
            scaling_object = scaling_object,
            )
        
        self.fm = fm.numpy()
        
    @property
    def n_input(self):
        return len(self.fm)
        

class HyperplaneTreeOMLTFormulationMixin():
    """A Mixin for OMLT linear tree formulations for Hyperplane Trees
    """
    def _build_formulation(self):
        super()._build_formulation()

        self.block.hyperplane_constraints = ConstraintList()

        I = range(len(self.model_definition.fm[0])) # Columns, tree input vars
        J = range(len(self.model_definition.fm)) # Rows, block input vars

        for i in I:
            self.block.hyperplane_constraints.add(sum(self.model_definition.fm[j,i]*self.block.inputs[j] for j in J) == self.block.inputs[i])

class HyperplaneTreeGDPFormulation(HyperplaneTreeOMLTFormulationMixin, LinearTreeGDPFormulation):
    pass

class HyperplaneTreeHybridBigMFormulation(HyperplaneTreeOMLTFormulationMixin, LinearTreeHybridBigMFormulation):
    """
    Currently, this results in certain linear constraints becoming nonlinear.
    Use HyperplaneTreeGDPFormulation instead.
    """
    pass
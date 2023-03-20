import torch
from torch import nn
import numpy as np

from pmlayer.common import util
from pmlayer.torch.hierarchical_lattice_layer import MultiLinearInterpolation, LUMap


class HL(nn.Module):
    '''
    Hierarchical lattice layer

    @note Current implementation does not support decreasing function
    '''

    def __init__(self, num_input_dims, lattice_sizes, indices_increasing, decision_embed_dim):
        super().__init__()
        '''
        Parameters
        ----------
        num_input_dims : int
            The number of input features

        lattice_sizes : Tensor
            Specifies the granularity of lattice for each monotone feature
            Each number must be at least 2

        indices_increasing : list of indices
            The list of indices of monotonically increasing features.
        '''

        # create map table
        input_len = num_input_dims - len(indices_increasing)
        output_len = 1
        cols_monotone = [False] * num_input_dims
        for idx, size in zip(indices_increasing, lattice_sizes):
            cols_monotone[idx] = True
            output_len *= size.item()
        self.mli = MultiLinearInterpolation(lattice_sizes)
        self.map_table = self._create_map_table()

        # initialize neural network
        if input_len > 0:
            self.output_layer = nn.Linear(decision_embed_dim, 1)
        else:
            var = torch.sqrt(torch.full((output_len,), 2.0 / output_len))
            initial_b = torch.normal(0.0, var)
            self.b = nn.Parameter(initial_b)

        # set monotonicity
        self.cols_monotone = torch.tensor(cols_monotone)
        self.cols_monotone = self.cols_monotone.to(lattice_sizes.device)
        self.cols_non_monotone = torch.logical_not(self.cols_monotone)
        self.cols_non_monotone = self.cols_non_monotone.to(lattice_sizes.device)

    def _create_map_table(self):
        '''
        Create ret (list of LUMap) based on self.mli.mesh_size
        '''

        # create LUMap
        max_hamming_distance = torch.sum(self.mli.mesh_size).item()
        max_hamming_distance -= self.mli.mesh_size.shape[0]
        ret = []
        bt = util.create_skewed_tree(max_hamming_distance + 2)
        tree_preorder = util.traverse_preorder(bt)
        mesh_size = self.mli.mesh_size.tolist()
        for tree_node in tree_preorder:
            ret.extend(self._create_map_table_sub([], mesh_size, tree_node[1] - 1, tree_node))

        # convert coordinates in LUMap into long values
        for node in ret:
            coordinate = torch.tensor(node.coordinate, dtype=torch.long)
            coordinate = coordinate.to(self.mli.mesh_size.device)
            node.index = self.mli.coordinate2index(coordinate)
            ls = torch.tensor(list(node.lower_set), dtype=torch.long)
            ls = ls.to(self.mli.mesh_size.device)
            node.lower_index = self.mli.coordinate2index(ls)
            us = torch.tensor(list(node.upper_set), dtype=torch.long)
            us = us.to(self.mli.mesh_size.device)
            node.upper_index = self.mli.coordinate2index(us)
        return ret

    def _create_map_table_sub(self, coordinate, cols_max, residual, tree_node):
        '''
        Parameters
        ----------
        coordinate : list of integer
            Coordinates determined so far

        cols_max : list of integer
            Maximum of each dimension of coordinates

        residual : integer
            Residual that can be used to fill coordinates

        tree_node : tuple of three integers
            tree_node = (left, value, right)
        '''

        cols_index = len(coordinate)
        if cols_index >= len(cols_max):
            # do nothing if coordinate is invalid
            if residual > 0:
                return []

            # create lumap based on coordinate
            coordinate = tuple(coordinate)
            lumap = LUMap(coordinate)
            diff_u = tree_node[2] - tree_node[1]
            lumap.upper_set = self._enumerate_upper_set([], coordinate, cols_max, diff_u)
            diff_l = tree_node[1] - tree_node[0]
            lumap.lower_set = self._enumerate_lower_set([], coordinate, cols_max, diff_l)
            return [lumap]

        ret = []
        for i in range(min(cols_max[cols_index], residual + 1)):
            coordinate.append(i)
            mt = self._create_map_table_sub(coordinate, cols_max, residual - i, tree_node)
            ret.extend(mt)
            coordinate.pop()
        return ret

    def _enumerate_upper_set(self, coordinate, cols_base, cols_max, residual):
        cols_index = len(coordinate)
        if cols_index >= len(cols_max):
            if residual == 0:
                return [tuple(np.array(cols_base) + np.array(coordinate))]
            else:
                return []

        ret = []
        num = min(cols_max[cols_index] - cols_base[cols_index], residual + 1)
        for i in range(num):
            coordinate.append(i)
            ret.extend(self._enumerate_upper_set(coordinate, cols_base, cols_max, residual - i))
            coordinate.pop()
        return ret

    def _enumerate_lower_set(self, coordinate, cols_base, cols_max, residual):
        cols_index = len(coordinate)
        if cols_index >= len(cols_max):
            if residual == 0:
                return [tuple(np.array(cols_base) - np.array(coordinate))]
            else:
                return []

        ret = []
        for i in range(min(cols_base[cols_index] + 1, residual + 1)):
            coordinate.append(i)
            ret.extend(self._enumerate_lower_set(coordinate, cols_base, cols_max, residual - i))
            coordinate.pop()
        return ret

    def forward(self, x, decision_embed):
        '''
        Parameters
        ----------
        x : Tensor
            x.shape = [batch_size, len(lattice_sizes)]

        Returns
        -------
        ret : Tensor
            ret.shape = [batch_size, 1]
        '''

        # predict values associated with lattice vertices
        xn = x[:, self.cols_non_monotone]
        if xn.shape[1] > 0:
            xn = self.output_layer(decision_embed)
        else:
            b = torch.sigmoid(self.b)
            xn = torch.tile(b, (xn.shape[0], 1))
        if xn.shape[1] == 1:  # all inputs are non-monotone
            return xn

        # transform tree structure into estimated grid values
        out = torch.Tensor(xn.shape).to(x.device)
        for item in self.map_table:
            if item.lower_index is None:
                lb = torch.zeros(xn.shape[0], device=x.device)
            else:
                lb = torch.index_select(out, 1, item.lower_index)
                lb, _ = torch.max(lb, 1)
                lb = lb.view(-1)
            if item.upper_index is None:
                ub = torch.ones(xn.shape[0], device=x.device)
            else:
                ub = torch.index_select(out, 1, item.upper_index)
                ub, _ = torch.min(ub, 1)
                ub = ub.view(-1)
            out[:, item.index] = torch.lerp(lb, ub, xn[:, item.index])

        # interpolate by using the output of the neural network
        monotone_inputs = x[:, self.cols_monotone]
        ret = self.mli.interpolate(monotone_inputs, out)
        return ret.view(-1, 1)

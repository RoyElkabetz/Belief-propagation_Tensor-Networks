import numpy as np
import gPEPS as gpeps
import glassy_gPEPS as glassy
import BPupdate as bp
import DEnFG as denfg


class Tensor_Network:

    def __init__(self, tensors_list, bond_vectors_list, structure_matrix, incidence_matrix):
        self.TT = tensors_list
        self.LL = bond_vectors_list
        self.smat = structure_matrix
        self.imat = incidence_matrix
        self.tensor_counter = len(tensors_list)
        self.edge_counter = len(bond_vectors_list)


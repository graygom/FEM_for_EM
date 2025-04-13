#
# TITLE:
# AUTHOR: Hyunseung Yoo
# PURPOSE: implementing FEM
# REFERENCE: A.C.Polycarpou - Introduction to the Finite Element Method in Electromagnetics (2006)
#
#

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


#==========================================
# 1.9 FEM solution of the electrostatic BVP

class FEM_1D:

    def __init__(self):
        
        # fundamental constants
        self.ep0 = 8.85e-12        # in F/m

        # element connectivity
        self.element_connectivity_input = {}
        self.element_connectivity_cal = {}

        # number of nodes, elements
        self.num_nodes = 0
        self.num_elements = 0

        # element coefficient matrix Ke, source vector fe
        self.Ke = {}
        self.fe = {}

        # electric potential, electric field
        self.V = []
        self.E = []

        # node position
        self.X_pos_node = []
        self.X_pos_elem = []
        self.len_elem = []
        self.epr_elem = []
        self.rho_elem = []
        


    def add_element(self, length, epr, rho):
        print('running add_element(length, epr, rho) ===')

        # updating number of nodes, elements
        self.num_nodes += 1
        self.num_elements += 1

        # updating element connectivity (input) in user system
        self.element_connectivity_input[self.num_elements-1] = [self.num_nodes-1, self.num_nodes, length, epr, rho]

        # length in MKS system
        length_cal = 0.0
        for val in length.keys():
            if length[val] == 'm':
                length_cal = val
            elif length[val] == 'cm':
                length_cal = val * 1e-2
            elif length[val] == 'mm':
                length_cal = val * 1e-3
            elif length[val] == 'um':
                length_cal = val * 1e-6
            elif length[val] == 'nm':
                length_cal = val * 1e-9
            elif length[val] == 'angstrom':
                length_cal = val * 1e-10
            else:
                print(' > add_element(): length unit error...')

        # electric permittivity in MKS system
        ep_cal = self.ep0 * epr                     # in F/m

        # rho in MKS system
        rho_cal = 0.0
        for val in rho.keys():
            if rho[val] == 'C/m3':
                rho_cal = val
            elif rho[val] == 'C/mm3':
                rho_cal = val * 1e9
            elif rho[val] == 'C/um3':
                rho_cal = val * 1e18
            elif rho[val] == 'C/nm3':
                rho_cal = val * 1e27
            elif rho[val] == 'C/angstrom3':
                rho_cal = val * 1e30
            else:
                print(' > add_element(): rho unit error...')   

        # updating element connectivity (cal) in mks system        
        self.element_connectivity_cal[self.num_elements-1] = [self.num_nodes-1, self.num_nodes, length_cal, ep_cal, rho_cal]

        # adding element coefficient matrix Ke @LHS
        self.Ke[self.num_elements-1] = np.zeros([2,2], dtype=float)
        self.Ke[self.num_elements-1][0,0] = +ep_cal / length_cal
        self.Ke[self.num_elements-1][0,1] = -ep_cal / length_cal
        self.Ke[self.num_elements-1][1,0] = -ep_cal / length_cal
        self.Ke[self.num_elements-1][1,1] = +ep_cal / length_cal

        # adding source vector fe @RHS
        self.fe[self.num_elements-1] = np.zeros([2,1], dtype=float)
        self.fe[self.num_elements-1][0,0] = -length_cal * rho_cal / 2.0
        self.fe[self.num_elements-1][1,0] = -length_cal * rho_cal / 2.0


    def print_element_connectivity(self):
        print('running print_element_connectivity() ===')
        
        # user input
        print('=== user input ===')
        for num_element in self.element_connectivity_input.keys():
            output_string = ' element %3i > %s' % (num_element, self.element_connectivity_input[num_element])
            print(output_string)
        print('')

        # calculation
        print('=== calculation ===')
        for num_element in self.element_connectivity_cal.keys():
            output_string = ' element %3i > %s' % (num_element, self.element_connectivity_cal[num_element])
            print(output_string)
        print('')

        # debugging
        if False:
            print('=== element coefficient matrix Ke ===')
            for element_no in range(self.num_elements):
                print(' element %3i > ' % element_no)
                print(self.Ke[element_no])
            print('')

        # debugging
        if False:
            print('=== source vector fe ===')
            for element_no in range(self.num_elements):
                print(' element %3i > ' % element_no)
                print(self.fe[element_no])
            print('')


    def make_pos_epr_rho_node_elem(self):
        #
        X_pos_node = []
        X_pos_elem = []
        len_elem = []
        epr_elem = []
        rho_elem = []

        # making array
        for elem_cnt in range(len(self.element_connectivity_cal)):
            # first node of X_pos
            if elem_cnt == 0:
                local_node1 = self.element_connectivity_cal[elem_cnt][0]
                elem_length = self.element_connectivity_cal[elem_cnt][2]
                X_pos_node.append(local_node1*elem_length)
                
            # other nodes of X_pos
            local_node1 = self.element_connectivity_cal[elem_cnt][0]
            local_node2 = self.element_connectivity_cal[elem_cnt][1]
            elem_length = self.element_connectivity_cal[elem_cnt][2]
            len_elem.append(elem_length)

            # elements of X_pos
            X_pos_node.append(local_node2*elem_length)
            X_pos_elem.append((local_node1+local_node2)/2.0*elem_length)

            # relative epsilicon
            epr = self.element_connectivity_input[elem_cnt][3]
            epr_elem.append(epr)

            # charge density
            rho = self.element_connectivity_cal[elem_cnt][4]
            rho_elem.append(rho)

        # numpy array
        self.X_pos_node = np.array(X_pos_node, dtype=float)
        self.X_pos_elem = np.array(X_pos_elem, dtype=float)
        self.len_elem = np.array(len_elem, dtype=float)
        self.epr_elem = np.array(epr_elem, dtype=float)
        self.rho_elem = np.array(rho_elem, dtype=float)


    def make_global_matrix_system(self):
        print('running make_global_matrix_system() ===')
        
        # initializaiton of K, f
        self.K = np.zeros([self.num_nodes+1, self.num_nodes+1], dtype=float)
        self.f = np.zeros([self.num_nodes+1, 1], dtype=float)

        # checking element connectivity
        for num_element in self.element_connectivity_cal.keys():
            # nodes in each element
            node1 = self.element_connectivity_cal[num_element][0]
            node2 = self.element_connectivity_cal[num_element][1]
            
            # updating K matrix
            self.K[node1, node1] += self.Ke[num_element][0, 0]
            self.K[node1, node2] += self.Ke[num_element][0, 1]
            self.K[node2, node1] += self.Ke[num_element][1, 0]
            self.K[node2, node2] += self.Ke[num_element][1, 1]

            # updating f vector
            self.f[node1, 0] += self.fe[num_element][0, 0]
            self.f[node2, 0] += self.fe[num_element][1, 0]

            # debugging
            if False:
                print(' element %3i = global node %3i & global node %3i' % (num_element, node1, node2))

        # debugging
        if False:
            print(self.K)
            print(self.f)


    def apply_dirichlet_bc_both_sides(self, start_node_potential, end_node_potential):
        print('running apply_dirichlet_bc_both_sides() ===')
        
        # debugging
        K_old = self.K.copy() / self.K[0,0]
        f_old = self.f.copy() / self.K[0,0]
        
        # modifying K, f @start node 
        selected_node = 0
        K_new1 = K_old[(selected_node+1):,(selected_node+1):]
        f_new1 = f_old[(selected_node+1):,0] - K_old[(selected_node+1):,selected_node] * start_node_potential

        # modifying K, f @end node
        selected_node = -1
        K_new2 = K_new1[:selected_node,:selected_node]
        f_new2 = f_new1[:selected_node] - K_new1[:selected_node,selected_node] * end_node_potential

        # V (electric potential) [V]
        K_new2_det = np.linalg.det(K_new2)                      # Cramer's rule: denominator
        
        V = []
        V.append(start_node_potential)

        for col_cnt in range(K_new2.shape[0]):
            K_new3 = K_new2.copy()
            K_new3[:,col_cnt] = f_new2                          # Cramer's rule: nominator
            V.append(np.linalg.det(K_new3)/K_new2_det)          # Cramer's rule: solution = nominator / denominator

        V.append(end_node_potential)

        self.V = np.array(V)

        # E (electric field) [V/m]
        self.E = -(self.V[1:] - self.V[:-1]) / self.len_elem     
        
        # debugging
        if False:
            print(len(self.X_pos), len(self.V))
            print(K_old)
            print(f_old)
            print(K_new1)
            print(f_new1)
            print(K_new2)
            print(f_new2)


    def visualization(self):
        print('running visualization() ===')
        
        #
        fig, ax = plt.subplots(2, 2, figsize=(12,12))

        # V: electric potential 
        ax[0,0].plot(self.X_pos_node, self.V, 'o:')
        ax[0,0].grid(ls=':')
        ax[0,0].set_title('electric potential [V]')
        ax[0,0].set_xlabel('node position [m]')
        ax[0,0].set_ylabel('electric potential [V]')
        ax[0,0].set_xlim(self.X_pos_node[0], self.X_pos_node[-1])

        # E: electric field
        ax[0,1].plot(self.X_pos_elem, self.E, 'o:')
        ax[0,1].grid(ls=':')
        ax[0,1].set_title('electric field [V/m]')
        ax[0,1].set_xlabel('node position [m]')
        ax[0,1].set_ylabel('electric field [V/m]')
        ax[0,1].set_xlim(self.X_pos_node[0], self.X_pos_node[-1])
        
        # relative epsilon
        ax[1,0].plot(self.X_pos_elem, self.epr_elem, 'o:')
        ax[1,0].grid(ls=':')
        ax[1,0].set_title('relative epsilion')
        ax[1,0].set_xlabel('element centroid position [m]')
        ax[1,0].set_ylabel('relative epsilon')
        ax[1,0].set_xlim(self.X_pos_node[0], self.X_pos_node[-1])

        # charge density
        ax[1,1].plot(self.X_pos_elem, self.rho_elem, 'o:')
        ax[1,1].grid(ls=':')
        ax[1,1].set_title('charge density')
        ax[1,1].set_xlabel('element centroid position [m]')
        ax[1,1].set_ylabel('charge density [C/m3]')
        ax[1,1].set_xlim(self.X_pos_node[0], self.X_pos_node[-1])

        # 
        plt.show()


#==========================================
# MAIN
#

# user input
input_legnth = 8.0         # in cm
input_rho = 1.0e-8          # in C/m3
input_ele_num = 20          # ea

# 1D FEM
sec1_9 = FEM_1D()

# adding elements
for ele_cnt in range(input_ele_num):
    ele_length = input_legnth / input_ele_num
    ele_rho = input_rho
    sec1_9.add_element(length={ele_length:'cm'}, epr=1.0, rho={ele_rho:'C/m3'})
sec1_9.print_element_connectivity()
sec1_9.make_pos_epr_rho_node_elem()

# making global matrix system
sec1_9.make_global_matrix_system()

# applying dirichlet BC
sec1_9.apply_dirichlet_bc_both_sides(start_node_potential=1.0, end_node_potential=0.0)

# visualization
sec1_9.visualization()

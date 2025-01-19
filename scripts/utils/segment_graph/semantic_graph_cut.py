import numpy as np
import open3d as o3d
import open3d.core as o3c
import copy
import time
from utils.segment_graph.utils import *
# from utils.semantics.pano_colormap import  VALID_CLASS_IDS_GT ,color_map
from semantics import general_color_map
import maxflow

class SemanticGraphCut:
    def __init__(self, instances_info, labels_info, confidence_map, \
                 device, log_io, K = 1.0, theta = 0.5, task = 'CoCoPano'):

        self.device = device

        # load confidence map and initial label_inst_guess
        self.confidence_map = confidence_map # self.confidence_map[label_a][semantic_label][label_b]
        self.labels_info = labels_info # labels_info[label] = {'semantic': semantic_label, 'instance': instance_label, 'color': color}
        self.instances_info = instances_info # instance_info[instance_label] = {'semantic': semantic_label, 'labels': [label]}
        self.instances_labels = list(self.instances_info.keys())
        self.segs_labels = list(self.labels_info.keys())
        self.max_instance_label = max(self.instances_labels)

        # get intermediate variables
        self.labels_info_refined = copy.deepcopy(labels_info)
        self.instances_info_refined = copy.deepcopy(instances_info)

        # log
        self.log_io = log_io

        # parameter
        self.eps = 1e-3 # less than eps is regarded as 0
        # coefficient for binary term
        self.theta = theta
        self.K = K
        # SegGraph initialized!
        if self.log_io is not None:
            self.log_to_file("SemanticSegGraphCut initialized!" + '\n')

        # init semantics meta data
        general_color_map.init(task)
        self.isThing = general_color_map.IsThing
        self.color_map = general_color_map.COLOR_MAP
        self.VALID_CLASS_IDS_GT = general_color_map.VALID_CLASS_IDS
    
    def log_to_file(self, info):
        self.log_io.write(info)
    def log_list_to_file(self, infos):
        for info in infos:
            self.log_io.write(info)

    def computeLabelInfo(self, labels_info):
        semantic_seg_map = {}
        for seg_label in labels_info:
            seg_semantic = labels_info[seg_label]['semantic']

            # semantic_seg_map
            if seg_semantic in semantic_seg_map:
                semantic_seg_map[seg_semantic].add(seg_label)
            else:
                semantic_seg_map[seg_semantic] = set([seg_label])

        for seg_semantic in semantic_seg_map:
            semantic_seg_map[seg_semantic] = list(semantic_seg_map[seg_semantic])
        return semantic_seg_map

    def computeSegsConnectionConfidenceMap(self):
        self.segs_connection = {} # external confidence between segs over all semantics 
        for seg_a in self.confidence_map:
            self.segs_connection[seg_a] = {}
            seg_confidence_map = self.confidence_map[seg_a]
            for semantics_label in seg_confidence_map:
                for seg_b in seg_confidence_map[semantics_label]:
                    if seg_b in self.segs_connection[seg_a]:
                        self.segs_connection[seg_a][seg_b] += seg_confidence_map[semantics_label][seg_b]
                    else:
                        self.segs_connection[seg_a][seg_b] = seg_confidence_map[semantics_label][seg_b]
    def getSegsConnectionConfidence(self, seg_a, seg_b):
        if seg_a in self.segs_connection:
            if seg_b in self.segs_connection[seg_a]:
                return self.segs_connection[seg_a][seg_b]
        return 0.

    def computeSegNeighbors(self):
        self.segs_neighbors_map = {}
        for seg_label in self.segs_labels:
            seg_neighbors = set()
            seg_confidence_map = self.confidence_map[seg_label]
            for semantic_label in seg_confidence_map:
                seg_neighbors = seg_neighbors.union( set(seg_confidence_map[semantic_label].keys()) )

            seg_neighbors_list = list(seg_neighbors)
            if seg_label in seg_neighbors_list:
                seg_neighbors_list.remove(seg_label)

            self.segs_neighbors_map[seg_label] = seg_neighbors_list
    def getSegNeighbors(self, seg_label):
        if seg_label not in self.segs_neighbors_map:
            return []
        return self.segs_neighbors_map[seg_label]

    def queryConfidence(self, label_a, semantic_label, label_b):
        if(label_a in self.confidence_map):
            if(semantic_label in self.confidence_map[label_a]):
                if(label_b in self.confidence_map[label_a][semantic_label]):
                    return self.confidence_map[label_a][semantic_label][label_b]
        return 0.

    def getUnaryEnergy(self, seg_label, semantic_alpha):
        """ method description
        -log(p(sem)+eps)
        """
        seg_confidence_map = self.confidence_map[seg_label]
        sem_alpha_confidence = self.queryConfidence(seg_label, semantic_alpha, seg_label)

        semantic_confidence_total = 0.
        for semantic in seg_confidence_map:
            semantic_confidence_total += seg_confidence_map[semantic][seg_label]

        semantic_prob = sem_alpha_confidence/(semantic_confidence_total + self.eps)

        return -np.log( min(semantic_prob+self.eps, 1.) )

    def getBinaryEnergy(self, seg_a, sem_alpha, seg_b, sem_beta):
        """ method description
        get binary energy given internal and external confidence of two seg labels 
        """
        if sem_alpha == sem_beta:
            return 0.

        external_connection = self.getSegsConnectionConfidence(seg_a, seg_b)
        if(external_connection < self.eps):
            return 0.
        
        internal_confidence_a = self.queryConfidence(seg_a, sem_alpha, seg_a)
        internal_confidence_b = self.queryConfidence(seg_b, sem_beta, seg_b)
        
        # # binary0
        # kernal_a = internal_confidence_a/external_connection
        # kernal_b = internal_confidence_b/external_connection
        # return 0.5 * self.K * np.exp( -0.5 * (kernal_a/self.theta)**2 ) + \
        #     0.5 * self.K * np.exp( -0.5 * (kernal_b/self.theta)**2 )

        # # # binary1
        # kernal_input = 0.5*(internal_confidence_a + internal_confidence_b)/external_connection
        # return self.K * np.exp( -0.5 * (kernal_input/self.theta)**2 )

        # binary2
        kernal_input = np.sqrt(internal_confidence_a * internal_confidence_b)/external_connection
        return self.K * np.exp( -0.5 * (kernal_input/self.theta)**2 )

    def getTerminalEnergy(self, seg_label, sem_alpha, sem_beta, labels_info):
        terminal_enery_alpha = self.getUnaryEnergy(seg_label, sem_alpha)
        terminal_enery_beta = self.getUnaryEnergy(seg_label, sem_beta)

        seg_neighbors_list = self.getSegNeighbors(seg_label)
        for seg_neighbor in seg_neighbors_list:
            sem_seg_neighbor = labels_info[seg_neighbor]['semantic']
            if (sem_seg_neighbor != sem_alpha) and (sem_seg_neighbor != sem_beta):
                terminal_enery_alpha += self.getBinaryEnergy(seg_label, sem_alpha, \
                    seg_neighbor, sem_seg_neighbor)
                terminal_enery_beta += self.getBinaryEnergy(seg_label, sem_beta, \
                    seg_neighbor, sem_seg_neighbor)
        return terminal_enery_alpha, terminal_enery_beta

    def computeEnergy(self, labels_info):
        total_unary_energy_semantic = 0.
        total_binary_energy_semantic = 0.
        
        labels_list = list(labels_info)
        computed_edges = set()
        for s_a_i, seg_a in enumerate(labels_list):
            total_unary_energy_semantic += self.getUnaryEnergy(seg_a, labels_info[seg_a]['semantic'])

            seg_a_neighbors = self.getSegNeighbors(seg_a)
            for s_neighbor in seg_a_neighbors:
                neighbor_pair = ( min(seg_a, s_neighbor), max(seg_a,s_neighbor) )
                if( neighbor_pair not in computed_edges):
                    total_binary_energy_semantic += self.getBinaryEnergy(seg_a, labels_info[seg_a]['semantic'] \
                        , s_neighbor, labels_info[s_neighbor]['semantic'])
                    computed_edges.add(neighbor_pair)

        return total_unary_energy_semantic + total_binary_energy_semantic, total_unary_energy_semantic, total_binary_energy_semantic

    def alphaSwapOnce(self, sem_alpha, sem_beta, labels_info, semantic_seg_map):
        """ method description
        alpha-swap
        use labels_info, semantic_seg_map
        """
        # get segment labels of two inst label
        segs_alpha = set(semantic_seg_map[sem_alpha]) # list of seg labels 
        segs_beta = set(semantic_seg_map[sem_beta]) # list of seg labels 
        
        # check input validness
        assert( len(segs_alpha.intersection(segs_beta)) == 0)

        segs = segs_alpha.union(segs_beta)
        num_segs = len(segs)
        if (num_segs < 1):
            return None

        # get edges 
        segs_list = list(segs)
        edges_info = [] # 0-indexed
        num_edges = 0 # 0-indexed
        for s_a_i, seg_a in enumerate(segs_list):
           for s_b_i in range(s_a_i+1, num_segs): 
                seg_b = segs_list[s_b_i]
                binary_energy = self.getBinaryEnergy(seg_a, sem_alpha, seg_b, sem_beta)
                if(binary_energy > self.eps):
                    edges_info.append({'node_a_i': s_a_i, 'node_b_i': s_b_i, \
                        'edge_energy': binary_energy})
                    num_edges += 1

    # construct maxflow graph
        graph_maxflow = maxflow.Graph[float](num_segs,num_edges)
        nodes = graph_maxflow.add_nodes(num_segs)
        # add terminal edges 
        for node_i, node in enumerate(nodes):
            seg = segs_list[node_i]
            seg_t_energy_alpha, seg_t_energy_beta = \
                self.getTerminalEnergy(seg, sem_alpha, sem_beta, labels_info)
            graph_maxflow.add_tedge(node, seg_t_energy_alpha, seg_t_energy_beta)
        # add non_terminal edges 
        for edge_info in edges_info:
            node_a = nodes[edge_info['node_a_i']]
            node_b = nodes[edge_info['node_b_i']]
            binary_energy_ab = edge_info['edge_energy']
            graph_maxflow.add_edge(node_a,node_b,binary_energy_ab,binary_energy_ab)

    # min-cut
        graph_maxflow.maxflow()
        cut_segs = graph_maxflow.get_grid_segments(nodes)
        return (cut_segs, segs_list) # [1 for sem_alpha, 0 for sem_beta]

    def getSwapPairs(self, instances_info, th_confidence = 0.5):
        """ method description
        given certain semantic label pairs for alpha-swap
        at least one of pairs should be in VALID_CLASS_IDS_GT
        """
        # get all semantic label in the scene
        semantic_set = set()
        for inst_label in instances_info:
            semantic_set.add(instances_info[inst_label]['semantic'])
        semantic_list = list(semantic_set)

        sem_pairs = []
        for sem_alpha_i, sem_alpha in enumerate(semantic_list):
            for sem_beta_i in range(sem_alpha_i+1, len(semantic_list)):
                sem_beta = semantic_list[sem_beta_i]
                # only consider pairs include semantic label we care about
                if (sem_alpha not in self.VALID_CLASS_IDS_GT) and (sem_beta not in self.VALID_CLASS_IDS_GT):
                    continue
                if (not self.isThing(sem_alpha)) and (not self.isThing(sem_beta)):
                    continue 
                sem_pairs.append( (sem_alpha, sem_beta) )
        
        return sem_pairs

    # from memory_profiler import profile
    # @profile
    def regularizeSemantic(self):
        """ method description
        use alpha-swap to regularize seg graph
        """
        logs = []
        # get pairs for alpha-swap
        sem_pairs = self.getSwapPairs(self.instances_info_refined)

        # precompute useful variables 
        semantic_seg_map = self.computeLabelInfo(self.labels_info_refined)
        self.computeSegsConnectionConfidenceMap()
        self.computeSegNeighbors()
        energy_old, energy_unary_old, energy_biary_old = \
            self.computeEnergy(self.labels_info_refined)
        swap_success = False
        iteration = 0
        # log for each iteration
        log_info = "    Iteration-" + str(iteration) + "; energy "+ str(energy_old) + \
            "; unary "+ str(energy_unary_old) + "; binary "+ str(energy_biary_old) + '\n'
        logs.append(log_info)
        while(not swap_success):
            for sem_pair in sem_pairs:
                sem_alpha = sem_pair[0]
                sem_beta = sem_pair[1]
                # if (sem_alpha == 59 and sem_beta == 123) or (sem_alpha == 123 and sem_beta == 59):
                #     breakpoint = None 
                max_flow_results = self.alphaSwapOnce(sem_alpha, sem_beta, self.labels_info_refined, semantic_seg_map)
                if max_flow_results is None:
                    continue
                # get min-cut result
                cut_segs, segs_list = max_flow_results
                updated_segs_temp = {}
                
                for seg_i, seg_label in enumerate(segs_list):
                    seg_sem_old = self.labels_info_refined[seg_label]['semantic']
                    seg_sem_new = sem_alpha if(cut_segs[seg_i]) else sem_beta
                    if(seg_sem_old != seg_sem_new):
                        updated_segs_temp[seg_label] = {'prev_sem': seg_sem_old, 'curr_sem': seg_sem_new}

                if(len(updated_segs_temp) != 0):
                    labels_info_temp = copy.deepcopy(self.labels_info_refined)
                    for updated_seg in updated_segs_temp:
                        labels_info_temp[updated_seg]['semantic'] = updated_segs_temp[updated_seg]['curr_sem']
                    energy_new, energy_unary_new, energy_biary_new = \
                        self.computeEnergy(labels_info_temp)
                    # update sem labeling if new energy is lower 
                    if(energy_new < energy_old):
                        swap_success = True
                        energy_old = energy_new
                        self.labels_info_refined = labels_info_temp

                        log_info = "    swap inst " + str(sem_alpha) + " and " + str(sem_beta) + \
                            "; energy "+ str(energy_new) + "; unary "+ str(energy_unary_new) + \
                            "; binary "+ str(energy_biary_new) + '\n'
                        logs.append(log_info)

                        for updated_seg in updated_segs_temp:
                            # update semantic_seg_map
                            semantic_seg_map[updated_segs_temp[updated_seg]['curr_sem']].append(updated_seg)
                            semantic_seg_map[updated_segs_temp[updated_seg]['prev_sem']].remove(updated_seg)

                            # log update 
                            log_info = "        label " + str(updated_seg).zfill(5) + " sem from " +  str(updated_segs_temp[updated_seg]['prev_sem']).zfill(5) + " to " + \
                                str(updated_segs_temp[updated_seg]['curr_sem']).zfill(5) + '\n'
                            logs.append(log_info)
                    else:
                        del labels_info_temp

            # continue refine util local minima
            iteration += 1
            log_info = "    Iteration-" + str(iteration) + "; energy "+ str(energy_old) + \
            "; unary "+ str(energy_unary_old) + "; binary "+ str(energy_biary_old) + '\n'
            logs.append(log_info)
            if(swap_success):
                swap_success = False
            else:
                break
        # log updated labels 
        logs.append("\n" + "Final label-semnatic change" + "\n")
        updated_segs = {}
        for seg_label in self.labels_info_refined:
            seg_sem_old = self.labels_info[seg_label]['semantic']
            seg_sem_new = self.labels_info_refined[seg_label]['semantic']

            # we cannot let semantic label totally determined by its label
            if (seg_sem_new not in self.confidence_map[seg_label]) or \
                    (self.confidence_map[seg_label][seg_sem_new][seg_label] < 1.0):
                self.labels_info_refined[seg_label]['semantic'] = seg_sem_old

            if(seg_sem_old != seg_sem_new):
                updated_segs[seg_label] = {'prev_sem': seg_sem_old, 'curr_sem': seg_sem_new}

                log_info = "        label " + str(seg_label).zfill(5) + " sem from " +  str(seg_sem_old).zfill(5) + " to " + \
                    str(seg_sem_new).zfill(5) + '\n'
                logs.append(log_info)

        self.log_list_to_file(logs)

        
        for updated_seg in updated_segs:
            seg_sem_old = updated_segs[updated_seg]['prev_sem']
            seg_sem_new = updated_segs[updated_seg]['curr_sem']   

            # remove old instance labels
            inst_label_old = self.labels_info[updated_seg]['instance']
            if(inst_label_old == 0):
                self.labels_info_refined[updated_seg]['instance'] = None
                continue
            self.instances_info_refined[inst_label_old]['labels'].remove(updated_seg)
            self.labels_info_refined[updated_seg]['instance'] = None
            
            # assign exising/new instance to updated segs 
            # seg_neighbors = self.getSegNeighbors(updated_seg)
            # inst_label_new = None
            # connected_seg = None
            # max_confidence = 0.
            # for seg_neighbor in seg_neighbors:
            #     if (seg_sem_new == self.labels_info_refined[seg_neighbor]['semantic']):
            #         max_confidence = self.queryConfidence(updated_seg, seg_sem_new, seg_neighbor)
            #         connected_seg = 

        return self.instances_info_refined, self.labels_info_refined, list(updated_segs.keys())

    def generateSemanticMesh(self, labels_info_refined, label_mesh_f, out_semantic_mesh_f, reserve_faces = False): # TODO

        if not reserve_faces:
            point_dtype = o3c.float32
            color_dtype = o3c.float32
            label_mesh = o3d.io.read_point_cloud(label_mesh_f)

            label_colors = np.asarray(label_mesh.colors).astype(np.float32)
            out_semantic_colors = np.ones_like(label_colors).astype(np.float32) * 200.0/255 # initial gray

            label_colors_tensor = o3c.Tensor(label_colors, dtype=color_dtype, device=self.device)
            out_semantic_colors_tensor = o3c.Tensor(out_semantic_colors, dtype=point_dtype, device=self.device)


            for seg_label in labels_info_refined:
                # labels_info_refined[seg_label]['color'] = 

                seg_label_color = labels_info_refined[seg_label]['color']*1.0/255.0 # scale to 1
                seg_label_color_tensor = o3c.Tensor(seg_label_color, dtype=color_dtype, device=self.device)
                seg_vertice_index = ( o3c.Tensor.abs(seg_label_color_tensor - label_colors_tensor) < 1e-4 ).all(dim = 1)

                semantic_color = np.array( self.color_map[labels_info_refined[seg_label]['semantic']])*1.0/255.0 
                out_semantic_colors_tensor[seg_vertice_index] = semantic_color

            # generate mesh
            out_semantic_mesh = o3d.geometry.PointCloud()
            out_semantic_mesh.points = label_mesh.points
            out_semantic_mesh.colors = o3d.utility.Vector3dVector(out_semantic_colors_tensor.cpu().numpy())
            o3d.io.write_point_cloud(out_semantic_mesh_f, out_semantic_mesh)
        else:
            point_dtype = o3c.float32
            color_dtype = o3c.float32
            label_mesh = o3d.io.read_triangle_mesh(label_mesh_f)

            label_colors = np.asarray(label_mesh.vertex_colors).astype(np.float32)
            out_semantic_colors = np.ones_like(label_colors).astype(np.float32) * 200.0/255 # initial gray

            label_colors_tensor = o3c.Tensor(label_colors, dtype=color_dtype, device=self.device)
            out_semantic_colors_tensor = o3c.Tensor(out_semantic_colors, dtype=point_dtype, device=self.device)
            # labels_info_refined[3198]['semantic'] = 62
            # labels_info_refined[4733]['semantic'] = 59
            # labels_info_refined[4889]['semantic'] = 57
            # labels_info_refined[5302]['semantic'] = 57
            # labels_info_refined[5310]['semantic'] = 57
            # labels_info_refined[5412]['semantic'] = 57
            for seg_label in labels_info_refined:

                seg_label_color = labels_info_refined[seg_label]['color']*1.0/255.0 # scale to 1
                seg_label_color_tensor = o3c.Tensor(seg_label_color, dtype=color_dtype, device=self.device)
                seg_vertice_index = ( o3c.Tensor.abs(seg_label_color_tensor - label_colors_tensor) < 1e-4 ).all(dim = 1)

                semantic_color = np.array( self.color_map[labels_info_refined[seg_label]['semantic']])*1.0/255.0 
                out_semantic_colors_tensor[seg_vertice_index] = semantic_color

            # generate mesh
            out_semantic_mesh = copy.deepcopy(label_mesh)
            out_semantic_mesh.vertex_colors = o3d.utility.Vector3dVector(out_semantic_colors_tensor.cpu().numpy())
            o3d.io.write_triangle_mesh(out_semantic_mesh_f, out_semantic_mesh)

        return None

            
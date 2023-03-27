import numpy as np
import open3d as o3d
import open3d.core as o3c
import copy
import time
from panoptic_mapping.utils.segment_graph.utils import *
import maxflow

class SegGraphCut:
    def __init__(self, instances_info, labels_info, confidence_map, semantic_instance_map, \
        semantic_updated_segs = [], log_io = None, K = 1.0, Break_inst=0.2, Break_seg=0.5):
    # log
        self.log_io = log_io

    # parameter
        # break threshold
        self.inst_th = 3
        self.break_th_inst = Break_inst
        self.break_th_seg = Break_seg

        # eps
        self.eps = 1e-2 # less than eps is regarded as 0
        # coefficient for binary term
        self.K = K

    # load confidence map and initial label_inst_guess
        self.confidence_map = None # self.confidence_map[label_a][semantic_label][label_b]
        self.labels_info_initial = None # labels_info[label] = {'semantic': semantic_label, 'instance': instance_label, 'color': color}
        self.instances_info_initial = None # instance_info[instance_label] = {'semantic': semantic_label, 'labels': [label]}
        self.semantic_instance_map = None

        self.labels_info_initial = labels_info
        self.instances_info_initial = instances_info
        self.semantic_instance_map = semantic_instance_map
        self.confidence_map = confidence_map
        # refined segment graph
        self.instances_info_refined = copy.deepcopy(self.instances_info_initial)
        self.labels_info_refined = copy.deepcopy(self.labels_info_initial)
        # get intermediate variables
        self.instances_labels = list(self.instances_info_initial.keys())
        self.max_instance_label = max(self.instances_labels)

        self.semantic_updated_segs = semantic_updated_segs
        self.semantic_updated_segs_map = {}
        for seg_label in semantic_updated_segs:
            semantic_label = labels_info[seg_label]['semantic']
            if semantic_label in self.semantic_updated_segs_map:
                self.semantic_updated_segs_map[semantic_label].append(seg_label)
            else:
                self.semantic_updated_segs_map[semantic_label] = [seg_label]

        # device for mesh operation
        if(o3c.cuda.is_available()):
            self.device = o3c.Device("CUDA", 0)
        else:
            self.device = o3c.Device("CPU", 0)
        # inst color
        self.inst_color = InstanceColor()

        # SegGraph initialized!
        if self.log_io is not None:
            self.log_to_file("SegGraphCut initialized!" + '\n')
    
    def computeInstLabelInfo(self, labels_info):
        semantic_inst_map = {}
        semantic_seg_map = {}
        for seg_label in labels_info:
            seg_semantic = labels_info[seg_label]['semantic']
            inst_semantic = labels_info[seg_label]['instance']
            # semantic_inst_map
            if seg_semantic in semantic_inst_map:
                if(inst_semantic not in semantic_inst_map[seg_semantic]):
                    semantic_inst_map[seg_semantic].add(inst_semantic)
            else:
                semantic_inst_map[seg_semantic] = set([inst_semantic])

            # semantic_seg_map
            if seg_semantic in semantic_seg_map:
                if(seg_label not in semantic_seg_map[seg_semantic]):
                    semantic_seg_map[seg_semantic].add(seg_label)
            else:
                semantic_seg_map[seg_semantic] = set([seg_label])
        for seg_semantic in semantic_seg_map:
            semantic_inst_map[seg_semantic] = list(semantic_inst_map[seg_semantic])
            semantic_seg_map[seg_semantic] = list(semantic_seg_map[seg_semantic])
        return semantic_inst_map, semantic_seg_map

    def log_to_file(self, info):
        self.log_io.write(info)
    def log_list_to_file(self, infos):
        for info in infos:
            self.log_io.write(info)

    def isThing(self, semantic_label):
        BackgroundSemantic = 80
        return semantic_label < BackgroundSemantic

    def queryConfidence(self, label_a, semantic_label, label_b):
        if(label_a in self.confidence_map):
            if(semantic_label in self.confidence_map[label_a]):
                if(label_b in self.confidence_map[label_a][semantic_label]):
                    return self.confidence_map[label_a][semantic_label][label_b]
        return 0.

    def computeInstanceConfidence(self, inst_label, instances_info):
        label_instance_confidence_map = {} # record confidence between label and its current instance
        inst_confidence = 0. # instance confidence, defined as max external edge confidence 

        if inst_label not in instances_info:
            return inst_confidence, label_instance_confidence_map

        semantic_label = instances_info[inst_label]['semantic']
        inst_seg_labels = instances_info[inst_label]['labels']
        labels_num = len(inst_seg_labels)

        # edge condition
        if(labels_num == 0):
            return inst_confidence, label_instance_confidence_map
        if(labels_num == 1):
            inst_confidence = self.queryConfidence(inst_seg_labels[0],semantic_label,inst_seg_labels[0])
            label_instance_confidence_map[inst_seg_labels[0]] = inst_confidence
            return inst_confidence, label_instance_confidence_map

        label_instance_confidence_map = {label:0. for label in inst_seg_labels}
        for l_i in range(labels_num):
            label_a = inst_seg_labels[l_i]
            for l_j in range(l_i+1, labels_num):
                label_b = inst_seg_labels[l_j]
                confidence = self.queryConfidence(label_a,semantic_label, label_b)
                # inst confidence
                if confidence > inst_confidence:
                    inst_confidence = confidence
                # label_instance_confidence
                if confidence > label_instance_confidence_map[label_a]:
                    label_instance_confidence_map[label_a] = confidence
                if confidence > label_instance_confidence_map[label_b]:
                    label_instance_confidence_map[label_b] = confidence
        return inst_confidence, label_instance_confidence_map

    def getUnaryEnergy(self, seg_label, inst_alpha, instances_info):
        """ method description
        get unary energy given inst label, seg label and existing seg-inst label
        UnaryEnergy in [0, 1]
        """
        segs_inst_alpha = instances_info[inst_alpha]['labels']
        if(len(segs_inst_alpha) == 0):
            return 0
        semantic_alpha = instances_info[inst_alpha]['semantic']
        # assert(seg_label in segs_inst_alpha)
        internal_confidence = self.queryConfidence(seg_label, semantic_alpha, seg_label)
        
        # # 1 - ave((i_pq/i_p)) current best
        # num_neighbors_alpha = 0  
        # confidence_neighbor_ave = 0.
        # for seg_neighbor in segs_inst_alpha:
        #     confidence_neighbor_ave += ( self.queryConfidence(seg_label, semantic_alpha, seg_neighbor)/internal_confidence ) 
        #     num_neighbors_alpha += 1
        # confidence_neighbor_ave /= num_neighbors_alpha
        # return 1 - confidence_neighbor_ave

        # 1 - max(i_pq)/i_p
        # confidence_neighbor_max = 0.
        # for seg_neighbor in segs_inst_alpha:
        #     confidence_neighbor = self.queryConfidence(seg_label, semantic_alpha, seg_neighbor)
        #     confidence_neighbor_max = max(confidence_neighbor_max, confidence_neighbor)
        # return 1 - confidence_neighbor_max*1.0/internal_confidence

        # 1 - i_pq/inst_confidence
        # confidence_neighbor_max = 0.
        # for seg_neighbor in segs_inst_alpha:
        #     confidence_neighbor = self.queryConfidence(seg_label, semantic_alpha, seg_neighbor)
        #     confidence_neighbor_max = max(confidence_neighbor_max, confidence_neighbor)
        # inst_confidence, _  = self.computeInstanceConfidence(inst_alpha, instances_info)
        # if(inst_confidence < self.eps):
        #     return 0
        # return 1 - min(1, internal_confidence*1.0/inst_confidence)

        # test 1 - ave((i_pq/i_p)) current best
        num_neighbors_alpha = 0  
        confidence_neighbor_ave = 0.
        for seg_neighbor in segs_inst_alpha:
            confidence_neighbor_ave += ( self.queryConfidence(seg_label, semantic_alpha, seg_neighbor)/internal_confidence ) 
            num_neighbors_alpha += 1
        confidence_neighbor_ave /= num_neighbors_alpha
        return 1.0 / ( (confidence_neighbor_ave)**2 + self.eps )

    def getBinaryEnergy(self, seg_a, inst_a, seg_b, inst_b ,semantic_label):
        """ method description
        get binary energy given internal and external confidence of two seg labels 
        """
        if(inst_a == inst_b):
            return 0

        confidence_node_a = self.queryConfidence(seg_a,semantic_label,seg_a)
        confidence_node_b= self.queryConfidence(seg_b,semantic_label,seg_b)
        edge_confidence = self.queryConfidence(seg_a,semantic_label,seg_b)

        is_confidence_valid = (edge_confidence > self.eps) and (confidence_node_a > self.eps) \
            and (confidence_node_b > self.eps)
        
        # i_pq/sqrt(i_p * i_q)  current best
        # if(is_confidence_valid):
        #     return  self.K * (  edge_confidence / np.sqrt(confidence_node_a*confidence_node_b) )

        # # 0.5*sqrt(i_pq/i_p) + 0.5*sqrt(i_pq/i_1)
        # if(is_confidence_valid):
        #     return  0.5* self.K * (  np.sqrt(edge_confidence/confidence_node_a) + np.sqrt(edge_confidence/confidence_node_b) )

        # i_pq^2/(i_p * i_q)
        # if(is_confidence_valid):
        #     return  self.K * (  edge_confidence*edge_confidence / (confidence_node_a*confidence_node_b) )

        # test 
        if(is_confidence_valid):
            return  self.K * (  edge_confidence*edge_confidence / (confidence_node_a*confidence_node_b) )**2

        return 0

    def getTerminalEnergy(self, seg_label, inst_alpha, inst_beta, semantic_label, \
            instances_info, labels_info):
        seg_neighbors = list(self.confidence_map[seg_label][semantic_label].keys())
        if seg_label in seg_neighbors:
            seg_neighbors.remove(seg_label)

        terminal_enery_alpha = self.getUnaryEnergy(seg_label, inst_alpha, instances_info)
        terminal_enery_beta = self.getUnaryEnergy(seg_label, inst_beta, instances_info)

        for seg_neighbor in seg_neighbors:
            inst_seg_neighbor = labels_info[seg_neighbor]['instance']
            if (inst_seg_neighbor != inst_alpha) and (inst_seg_neighbor != inst_beta):
                terminal_enery_alpha += self.getBinaryEnergy(seg_label, inst_alpha, \
                    seg_neighbor, inst_seg_neighbor, semantic_label)
                terminal_enery_beta += self.getBinaryEnergy(seg_label, inst_beta, \
                    seg_neighbor, inst_seg_neighbor, semantic_label)
        return terminal_enery_alpha, terminal_enery_beta

    def computeEnergy(self, semantic_label, instances_info, labels_info, connected_segs):
        assert(self.isThing(semantic_label))

        total_unary_energy_semantic = 0.
        total_binary_energy_semantic = 0.

        for s_a_i, seg_a in enumerate(connected_segs):
            inst_seg_a = labels_info[seg_a]['instance']
            if inst_seg_a == 0:
                continue
            total_unary_energy_semantic += self.getUnaryEnergy(seg_a, inst_seg_a, instances_info)
            for s_b_i in range(s_a_i+1, len(connected_segs)):
                seg_b = connected_segs[s_b_i]
                inst_seg_b = labels_info[seg_b]['instance']
                if inst_seg_b == 0:
                    continue
                total_binary_energy_semantic += self.getBinaryEnergy(seg_a, inst_seg_a, seg_b, inst_seg_b, semantic_label)
        return total_unary_energy_semantic + total_binary_energy_semantic, total_unary_energy_semantic, total_binary_energy_semantic

    def alphaSwapOnce(self, inst_alpha, inst_beta, instances_info, labels_info):
        """ method description
        alpha-swap
        use self.instances_info_refined and self.labels_info_refined
        """
    # get segment labels of two inst label
        segs_alpha = set(instances_info[inst_alpha]['labels']) # list of seg labels 
        semantic_alpha = instances_info[inst_alpha]['semantic']
        segs_beta = set(instances_info[inst_beta]['labels']) # list of seg labels 
        semantic_beta = instances_info[inst_beta]['semantic']
        
    # check input validness
        assert( len(segs_alpha.intersection(segs_beta)) == 0)
        segs = segs_alpha.union(segs_beta)
        num_segs = len(segs)
        is_segs = (num_segs >= 1)
        is_semantic_consistant = (semantic_alpha == semantic_beta)
        if (not is_segs) or (not is_semantic_consistant):
            return None

    # get edges 
        segs_list = list(segs)
        edges_info = [] # 0-indexed
        num_edges = 0 # 0-indexed
        pairs_to_edges_id = {} # 0-indexed
        for s_a_i, seg_a in enumerate(segs_list):
           for s_b_i in range(s_a_i+1, num_segs): 
                seg_b = segs_list[s_b_i]
                binary_energy = self.getBinaryEnergy(seg_a, inst_alpha, seg_b, inst_beta, semantic_alpha)
                if(binary_energy > self.eps):
                    edges_info.append({'node_a_i': s_a_i, 'node_b_i': s_b_i, \
                        'edge_energy': binary_energy, 'edge_id': num_edges})
                    num_edges += 1

    # construct maxflow graph
        graph_maxflow = maxflow.Graph[float](num_segs,num_edges)
        nodes = graph_maxflow.add_nodes(num_segs)
        # add terminal edges 
        for node_i, node in enumerate(nodes):
            seg = segs_list[node_i]
            seg_t_energy_alpha, seg_t_energy_beta = \
                self.getTerminalEnergy(seg, inst_alpha, inst_beta, semantic_alpha, \
                    instances_info, labels_info)
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
        return (cut_segs, segs_list) # [1 for inst_alpha, 0 for inst_beta]

    def breakWeakConnection(self, semantic_label, instances_info, labels_info, \
            semantic_instance_map, segs_semantic_update):
        """ method description
        break all weak inst_label connection in instances_info, labels_info
        """
        instance_list = semantic_instance_map[semantic_label]
        breaked_labels = [] # list of disconneccted seg_label
        # loop over instances
        for inst_label in instance_list:
            # break all label-inst links if links are not strong enough
            if inst_label == 0 or inst_label not in instances_info:
                continue
            inst_info = instances_info[inst_label]
            inst_seg_labels = copy.deepcopy(inst_info['labels'])
            inst_semantic_label = inst_info['semantic']
            inst_confidence, label_instance_confidence_map =  self.computeInstanceConfidence(inst_label, instances_info)
            if inst_confidence < self.inst_th:
                continue
            for seg_label in inst_seg_labels:
                is_weak_link = (label_instance_confidence_map[seg_label] < self.break_th_inst*inst_confidence) or \
                    (label_instance_confidence_map[seg_label] < self.break_th_seg*self.queryConfidence(seg_label, semantic_label, seg_label))
                if is_weak_link:
                    # seg_label_confidence = self.queryConfidence(seg_label, inst_semantic_label, seg_label)
                    breaked_labels.append(seg_label)
                    # update insts_info and labels_info and assign new inst
                    instances_info[inst_label]['labels'].remove(seg_label)

                    self.max_instance_label += 1
                    new_inst_label = self.max_instance_label
                    labels_info[seg_label]['instance'] = new_inst_label
                    self.instances_labels.append(new_inst_label)
                    instances_info[new_inst_label] = {'semantic': semantic_label, 'labels': [seg_label]}
        for break_label in breaked_labels:
            semantic_instance_map[semantic_label].append(labels_info[seg_label]['instance'])
        # loop over segs with semantics updated 
        for seg_label in segs_semantic_update:
            self.max_instance_label += 1
            new_inst_label = self.max_instance_label
            labels_info[seg_label]['instance'] = new_inst_label
            semantic_label = labels_info[seg_label]['semantic']
            self.instances_labels.append(new_inst_label)
            instances_info[new_inst_label] = {'semantic': semantic_label, 'labels': [seg_label]}
            breaked_labels.append(seg_label)
        # self.log_io.write(" Getting ", )
        return breaked_labels

    def getSwapPairs(self, semantic, instances_info,labels_info, semantic_instance_map, th_confidence = 3):
        """ method description
        given certain semantic label, return inst pairs for alpha-swap
        use breadth first search to get sets of connected instances
        """
        if(not self.isThing(semantic)):
            return []
        
        inst_list = list(semantic_instance_map[semantic])
        if 0 in inst_list:
            inst_list.remove(0)

        assigned_insts = set()
        queried_segs = set()
        connected_inst_sets = []
        for inst_i,inst_label in enumerate(inst_list):
            if inst_label in assigned_insts or inst_label == 0 or inst_label not in instances_info:
                continue

            assigned_insts.add(inst_label)
            connected_inst_set = [inst_label]
            insts_to_query = [inst_label]
            while(len(insts_to_query) > 0):
                inst_to_query = insts_to_query[0]
                inst_segs = instances_info[inst_label]['labels']
                for seg_label in inst_segs:
                    connected_segs = set(self.confidence_map[seg_label][semantic].keys())
                    queried_segs.add(seg_label)
                    connected_segs_no_quried = connected_segs.difference(queried_segs)
                    for connected_seg in connected_segs_no_quried:
                        is_not_inst_assigned = labels_info[connected_seg]['instance'] not in assigned_insts
                        if (is_not_inst_assigned):
                            is_connected = (labels_info[connected_seg]['semantic'] == semantic) and \
                                (self.confidence_map[seg_label][semantic][connected_seg] > th_confidence)
                            inst_neighbor = labels_info[connected_seg]['instance']
                            is_inst_valid = (inst_neighbor!=0)
                            if(is_connected and is_inst_valid):
                                assigned_insts.add(inst_neighbor)
                                insts_to_query.append(inst_neighbor)
                                connected_inst_set.append(inst_neighbor)
                insts_to_query.remove(inst_to_query)
            if( len(connected_inst_set) > 0):
                connected_inst_sets.append(connected_inst_set)
    # record connected_inst_sets
        self.connected_inst_sets = connected_inst_sets
    # get inst_pairs
        semantic_inst_pairs = []
        for connected_inst_set in connected_inst_sets:
            semantic_inst_pairs_set = []
            for inst_alpha_i, inst_alpha in enumerate(connected_inst_set):
                for inst_beta_i in range(inst_alpha_i+1, len(connected_inst_set)):
                    inst_beta = connected_inst_set[inst_beta_i]
                    semantic_inst_pairs_set.append((inst_alpha, inst_beta))
            semantic_inst_pairs.append(semantic_inst_pairs_set)
        return semantic_inst_pairs

    def regularizeSegGraph(self):
        """ method description
        use alpha-swap to regularize seg graph
        """
        logs = []
    # compute usefule variables
        semantic_instance_map, semantic_seg_map = self.computeInstLabelInfo(self.labels_info_refined)
        breaked_labels_len_total = 0
    # loop over all thing semantics for alpha-swap
        for semantic_label in semantic_instance_map:
        # skip stuff 
            if(not self.isThing(semantic_label)):
                continue
            instance_list = list(semantic_instance_map[semantic_label])
            # log for semantic label
            log_info = "\n" + "Refining semantic " + str(semantic_label) + \
                " with " + str(len(instance_list)) + ' sets\n'
            logs.append(log_info)
        # continue refine in each connected_inst_set util local minima
            # break weak links and assign new inst labels
            breaked_labels = self.breakWeakConnection(semantic_label, self.instances_info_refined, self.labels_info_refined, \
                semantic_instance_map, self.semantic_updated_segs)
            # log for breaked segments
            log_info = "  Break " + str(len(breaked_labels)) +  " segs " + '\n'
            breaked_labels_len_total += len(breaked_labels)
            logs.append(log_info)

            inst_pairs = self.getSwapPairs(semantic_label, self.instances_info_refined, self.labels_info_refined, semantic_instance_map)
            for inst_set_i, inst_connected_pairs in enumerate(inst_pairs):
                # log for connected sets
                if( len(inst_connected_pairs) == 0 ):
                    continue
            
                log_info = "  Refining set " + str(inst_set_i) + \
                    " with " + str(len(inst_connected_pairs)) + ' pairs\n'
                logs.append(log_info)

                connected_segs = set()
                for inst_pair in inst_connected_pairs:
                    connected_segs = connected_segs.union(set(self.instances_info_refined[inst_pair[0]]['labels']))
                    connected_segs = connected_segs.union(set(self.instances_info_refined[inst_pair[1]]['labels']))
                if(semantic_label == 56):
                    breakpoint = None
                energy_old, energy_unary_old, energy_biary_old = \
                    self.computeEnergy(semantic_label, self.instances_info_refined, \
                    self.labels_info_refined, list(connected_segs))
                swap_success = False
                iteration = 0
                # log for each iteration
                log_info = "    Iteration-" + str(iteration) + "; energy "+ str(energy_old) + \
                    "; unary "+ str(energy_unary_old) + "; binary "+ str(energy_biary_old) + '\n'
                logs.append(log_info)
                while(not swap_success):
                # loop over pairs
                    for inst_pair in inst_connected_pairs:
                        inst_alpha = inst_pair[0]
                        inst_beta = inst_pair[1]
                        if(inst_alpha == 0 or inst_beta == 0):
                            continue
                    # alpha-swap
                        max_flow_results = self.alphaSwapOnce(inst_alpha, inst_beta, self.instances_info_refined, self.labels_info_refined)
                        if max_flow_results is None:
                            continue
                    # get new label and inst info
                        cut_segs, segs_list = max_flow_results
                        labels_info = copy.deepcopy(self.labels_info_refined)
                        instances_info = copy.deepcopy(self.instances_info_refined)
                        updated_segs_temp = {}
                        for seg_i, seg_label in enumerate(segs_list):
                            seg_inst_label_old = labels_info[seg_label]['instance']
                            seg_inst_label_new = inst_alpha if(cut_segs[seg_i]) else inst_beta
                            if(seg_inst_label_old != seg_inst_label_new):
                                updated_segs_temp[seg_label] = {'prev_inst': seg_inst_label_old, \
                                    'curr_inst': seg_inst_label_new}
                                labels_info[seg_label]['instance'] = seg_inst_label_new
                                instances_info[seg_inst_label_old]['labels'].remove(seg_label)
                                instances_info[seg_inst_label_new]['labels'].append(seg_label)
                        if(len(updated_segs_temp) == 0):
                            continue
                    # estimate energy 
                        energy_new, energy_unary_new, energy_biary_new  = \
                            self.computeEnergy(semantic_label, instances_info, labels_info, list(connected_segs))
                        log_info = "    swap inst " + str(inst_alpha) + " and " + str(inst_beta) + \
                            "; energy "+ str(energy_new) + "; unary "+ str(energy_unary_new) + \
                            "; binary "+ str(energy_biary_new) + '\n'
                        logs.append(log_info)
                        for updated_seg in updated_segs_temp:
                            updated_seg_info = updated_segs_temp[updated_seg]
                            log_info = "        label " + str(updated_seg).zfill(5) + " inst from " +  str(updated_seg_info['prev_inst']).zfill(5) + " to " + \
                                str(updated_seg_info['curr_inst']).zfill(5) + '\n'
                            logs.append(log_info)
                        if( energy_new < energy_old):
                            swap_success = True
                            energy_old = energy_new
                            self.labels_info_refined = labels_info
                            self.instances_info_refined = instances_info
                # continue refine util local minima
                    if(swap_success):
                        swap_success = False
                    else:
                        break
                    iteration += 1
                    log_info = "    Iteration-" + str(iteration) + "; energy "+ str(energy_old) + \
                    "; unary "+ str(energy_unary_old) + "; binary "+ str(energy_biary_old) + '\n'
                    logs.append(log_info)
        # log updated labels 
        logs.append("\n" + "Final label change" + "\n")
        for semantic_label in semantic_seg_map:
            if(not self.isThing(semantic_label)):
                continue
            logs.append("     semantic " + str(semantic_label) + "\n")
            for seg_label in semantic_seg_map[semantic_label]:
                seg_inst_label_old = self.labels_info_initial[seg_label]['instance']
                seg_inst_label_new = self.labels_info_refined[seg_label]['instance']
                if(seg_inst_label_old != seg_inst_label_new):
                    log_info = "        label " + str(seg_label).zfill(5) + " inst from " +  str(seg_inst_label_old).zfill(5) + " to " + \
                        str(seg_inst_label_new).zfill(5) + '\n'
                    logs.append(log_info)
        
        self.log_list_to_file(logs)
        self.log_io.write("\n Total new instances: "+str(breaked_labels_len_total)+"\n")
        return self.instances_info_refined, self.labels_info_refined

    def generateMesh(self, instances_info_refined, label_mesh_f, out_inst_mesh_f): # TODO
        point_dtype = o3c.float32
        color_dtype = o3c.float32
        label_mesh = o3d.io.read_point_cloud(label_mesh_f)

        label_colors = np.asarray(label_mesh.colors).astype(np.float32)
        out_inst_colors = np.ones_like(label_colors).astype(np.float32) * 200.0/255 # initial gray

        label_colors_tensor = o3c.Tensor(label_colors, dtype=color_dtype, device=self.device)
        out_inst_colors_tensor = o3c.Tensor(out_inst_colors, dtype=point_dtype, device=self.device)

        for inst_label in instances_info_refined:
            inst_seg_labels =  instances_info_refined[inst_label]['labels']
            inst_color = self.inst_color.getColor(inst_label)*1.0/255.0 # scale to 1
            inst_color_tensor = o3c.Tensor(inst_color, dtype=color_dtype, device=self.device)
            # paint for instance mesh 
            for seg_label in inst_seg_labels:
                assert(seg_label in self.labels_info_refined)
                seg_label_color = self.labels_info_refined[seg_label]['color']*1.0/255.0 # scale to 1
                seg_label_color_tensor = o3c.Tensor(seg_label_color, dtype=color_dtype, device=self.device)
                seg_vertice_index = ( o3c.Tensor.abs(seg_label_color_tensor - label_colors_tensor) < 1e-4 ).all(dim = 1)
                out_inst_colors_tensor[seg_vertice_index] = inst_color_tensor

        # generate mesh
        out_inst_mesh = o3d.geometry.PointCloud()
        out_inst_mesh.points = label_mesh.points
        out_inst_mesh.colors = o3d.utility.Vector3dVector(out_inst_colors_tensor.cpu().numpy())
        o3d.io.write_point_cloud(out_inst_mesh_f, out_inst_mesh)
        return None
import numpy as np
import open3d as o3d
import open3d.core as o3c
import copy
import pickle
import os

from utils.segment_graph.utils import *
from semantics import general_color_map
class SegGraphSpecific:
    def __init__(self, instances_info, labels_info, confidence_map, semantic_instance_map, \
        semantic_updated_segs = [],  log_io = None,  task = "CoCoPano", \
            device = o3c.Device("CPU", 0), break_weak_connection = True, parameter = None,
            parameter_class_specific_file = None):

        # hyper-parameter
        self.connect_threshold_label_side, self.connect_threshold_high_label_side, \
        self.connect_threshold_inst_side, self.connect_threshold_between_edges = \
            (0.1 , 0.7, 0.05, 0.7)

        if (parameter is not None) and (len(parameter)==4):
            self.connect_threshold_label_side, self.connect_threshold_high_label_side, \
            self.connect_threshold_inst_side, self.connect_threshold_between_edges = \
                tuple(parameter)
            
        self.params_specific = {}
        if (parameter_class_specific_file is not None) and \
            (os.path.isfile(parameter_class_specific_file)):
            with open(parameter_class_specific_file, 'rb') as f:
                self.params_specific = pickle.load(f)

        # semantics meta data
        general_color_map.init(task)
        self.background_semantic_label = general_color_map.BackGroundID
        self.isThing = general_color_map.IsThing
        # log
        self.log_io = log_io
        # load confidence map and initial label_inst_guess
        self.instances_info_initial, self.labels_info_initial, self.confidence_map = \
            instances_info, labels_info,  confidence_map
        self.semantic_instance_map = semantic_instance_map

        self.semantic_updated_segs = semantic_updated_segs
        self.semantic_updated_segs_map = {}
        for seg_label in semantic_updated_segs:
            semantic_label = labels_info[seg_label]['semantic']
            if semantic_label in self.semantic_updated_segs_map:
                self.semantic_updated_segs_map[semantic_label].append(seg_label)
            else:
                self.semantic_updated_segs_map[semantic_label] = [seg_label]

        self.instances_labels = list(self.instances_info_initial.keys())
        self.max_instance_label = max(self.instances_labels)

        self.device = device

        # inst color
        self.inst_color = InstanceColor()

        # whether break weak links or just update semantic changed segments 
        self.break_weak_connection = break_weak_connection

        # SegGraph initialized!
        if self.log_io is not None:
            self.log_to_file("SegGraph initialized!" + '\n')
    
    def log_to_file(self, info):
        self.log_io.write(info)

    def queryConfidence(self, label_a, semantic_label, label_b):
        if(label_a in self.confidence_map):
            if(semantic_label in self.confidence_map[label_a]):
                if(label_b in self.confidence_map[label_a][semantic_label]):
                    return self.confidence_map[label_a][semantic_label][label_b]
        return 0.

    def computeMergeInstances(self):
        merge_instances = {}
        for semantic_label in self.semantic_instances_map:
            instances = self.semantic_instances_map[semantic_label]
            instances_num = len(instances)
            if(instances_num < 2 ):
                continue

            index_assigned_instances = set()
            # breadth-first to aggregate instance
            for i in range(instances_num):
                if(i in index_assigned_instances):
                    continue
                index_assigned_instances.add(i)
                insatance_query = instances[i]
                for j in range(i+1, instances_num):
                    if(j in index_assigned_instances):
                        continue
                    instance_to_check = instances[j]
                    if(insatance_query.isInstanceConnected(instance_to_check)):
                        index_assigned_instances.add(j)
                        if(insatance_query.id_ in merge_instances):
                            merge_instances[insatance_query.id_].append(instance_to_check.id_)
                        else:
                            merge_instances[insatance_query.id_] =[instance_to_check.id_]
        return merge_instances

    def computeInstanceConfidence(self, inst_label, insts_info):
        label_instance_confidence_map = {} # record confidence between label and its current instance
        inst_confidence = 0. # instance confidence, defined as max external edge confidence 
        if inst_label == 0:
            return 0., None
        semantic_label = insts_info[inst_label]['semantic']
        inst_seg_labels = insts_info[inst_label]['labels']
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

    def repeatInstanceMap(self):
        instances_info_refined =copy.deepcopy(self.instances_info_initial)
        labels_updated = {}
        return instances_info_refined, labels_updated


    def refineLIMC3BreakAllReunion(self, inst_break_threshold = 0.99):
        """ method description
        use confidence of method3; confidence of method3 is lower than that of method0;
        so maybe lower down the inst_confidence_threshold and lower down the 
        break_threshold; connect_threshold_label_sidel; connect_threshold_inst_side and 
        connect_threshold_high_label_side more 
        loop over all semantics: 
            1. first break all label-inst links if links are not strong enough;
            2. then loop over all breaked segment labels in order of label internal confidence
                Re-assign breaked segments labels to existing instance if
                    i) seg_label connected with current broken seg label has instance
                &&  [ ii) seg_label-instance confidence > connect_threshold_inst_side * instance confidence
                    or    iii) seg_label-instance confidence > connect_threshold_high_label_side * seg_label confidence ]
                &&  iv) seg_label-instance confidence > connect_threshold_label_side * seg_label confidence
                Or create a new instance for current broken seg label
        """
        # break parameter 
        inst_confidence_threshold = 3

        # output variables
        instances_info_refined =copy.deepcopy(self.instances_info_initial)
        labels_info_refined = copy.deepcopy(self.labels_info_initial)
        labels_updated = {}
        # loop over all semantics
        for semantic_label in self.semantic_instance_map:
            break_threshold = inst_break_threshold # max_link_between_label_inst / inst_confidence 
            # max edge connection parameter
            connect_threshold_label_side = self.connect_threshold_label_side
            connect_threshold_high_label_side = self.connect_threshold_high_label_side
            connect_threshold_inst_side = self.connect_threshold_inst_side
            # edge connection parameter
            connect_threshold_between_edges = self.connect_threshold_between_edges

            if('TH_break' in self.params_specific):
                if(semantic_label in self.params_specific['TH_break']):
                    break_threshold = self.params_specific['TH_break'][semantic_label]
            if('TH_label' in self.params_specific):
                if(semantic_label in self.params_specific['TH_label']):
                    connect_threshold_label_side = self.params_specific['TH_label'][semantic_label]
            if('TH_label_high' in self.params_specific):
                if(semantic_label in self.params_specific['TH_label_high']):
                    connect_threshold_high_label_side = self.params_specific['TH_label_high'][semantic_label]
            if('TH_inst' in self.params_specific):
                if(semantic_label in self.params_specific['TH_inst']):
                    connect_threshold_inst_side = self.params_specific['TH_inst'][semantic_label]
            if('TH_edges' in self.params_specific):
                if(semantic_label in self.params_specific['TH_edges']):
                    connect_threshold_between_edges = self.params_specific['TH_edges'][semantic_label]


            log_info = "using parameter for sem " + str(semantic_label) + ": " + \
                str(break_threshold) + "  " + \
                str(connect_threshold_label_side) + "_" + \
                str(connect_threshold_high_label_side) + "_" + \
                str(connect_threshold_inst_side) + "_" + \
                str(connect_threshold_between_edges) + '\n'
            self.log_to_file(log_info)
            breaked_labels_tuple = [] # list of tuple(seg_label, label_confidence)
            # loop over updated_segs_semantic to find semantically updated segs from regularizaiton
            updated_segs_semantic = []
            if semantic_label in self.semantic_updated_segs_map:
                updated_segs_semantic = self.semantic_updated_segs_map[semantic_label]
            for seg_label in updated_segs_semantic:
                seg_label_confidence = self.queryConfidence(seg_label, semantic_label, seg_label)
                breaked_labels_tuple.append((seg_label, seg_label_confidence))

            instance_list = self.semantic_instance_map[semantic_label]
            # loop over instances and find weakly connected segs
            if self.break_weak_connection:
                for inst_label in instance_list:
                    # 1. break all label-inst links if links are not strong enough
                    inst_info = self.instances_info_initial[inst_label]
                    inst_seg_labels = inst_info['labels']
                    inst_semantic_label = inst_info['semantic']
                    inst_confidence, label_instance_confidence_map =  self.computeInstanceConfidence(inst_label, self.instances_info_initial)
                    if inst_confidence < inst_confidence_threshold:
                        continue
                    for seg_label in inst_seg_labels:
                        is_weak_link = (label_instance_confidence_map[seg_label] < break_threshold*inst_confidence)
                        if is_weak_link:
                            seg_label_confidence = self.queryConfidence(seg_label, inst_semantic_label, seg_label)
                            breaked_labels_tuple.append((seg_label, seg_label_confidence))
            # 2. re-assign instance labels for those breaked segments
            breaked_labels_tuple.sort(key = lambda l: l[1], reverse=True)
            # 2-1. remove related initial guess of breaked seg labels
            for breaked_label_tuple in breaked_labels_tuple:
                breaked_label = breaked_label_tuple[0]
                initial_inst_label = labels_info_refined[breaked_label]['instance']
                if initial_inst_label is not None:
                    instances_info_refined[initial_inst_label]['labels'].remove(breaked_label)
                    labels_info_refined[breaked_label]['instance'] = None
            # 2-2. re-assign instance labels
            # iterate all labels to be assigned
            for breaked_label_tuple in breaked_labels_tuple:
                breaked_label = breaked_label_tuple[0]
                breaked_label_confidence = breaked_label_tuple[1]
                semantic_label = labels_info_refined[breaked_label]['semantic'] 

                breaked_label_confidence_map = self.confidence_map[breaked_label][semantic_label]

                # iterate all connected labels for re-assignment
                max_connect_edge = [None, 0.]  # (label, confidence)
                labels_with_inst = []
                connected_edge_with_inst = [] # [ (label, confidence)]

                # find strongest connection
                for seg_label_query in breaked_label_confidence_map:
                    if seg_label_query == breaked_label:
                        continue
                    semantic_seg_query = self.labels_info_initial[seg_label_query]['semantic']
                    # if semantic label not equal, don't consider
                    if semantic_label != semantic_seg_query:
                        continue
                    connect_confidence = breaked_label_confidence_map[seg_label_query]
                    if(connect_confidence > max_connect_edge[1]):
                        max_connect_edge[1] = connect_confidence
                        max_connect_edge[0] = seg_label_query

                    is_exising_instance = (labels_info_refined[seg_label_query]['instance'] is not None)
                    if(is_exising_instance):
                        labels_with_inst.append(seg_label_query)
                    
                # if strongest connection
                if (max_connect_edge[0] is not None):
                    connected_seg_label = max_connect_edge[0]
                    max_connect_confidence = max_connect_edge[1]
                    is_exising_instance_max_edge = (labels_info_refined[connected_seg_label]['instance'] is not None)
                    is_strong_link_label_side = max_connect_confidence > connect_threshold_label_side * breaked_label_confidence
                    is_extra_strong_link_label_side = max_connect_confidence > connect_threshold_high_label_side * breaked_label_confidence

                    is_strong_link_inst_side = False
                    if is_exising_instance_max_edge:
                        exising_instance_label = labels_info_refined[connected_seg_label]['instance']
                        exising_instance_confidence, _ = self.computeInstanceConfidence(exising_instance_label, instances_info_refined)
                        is_strong_link_inst_side = max_connect_confidence > connect_threshold_inst_side * exising_instance_confidence

                    is_new_inst = True
                    # if connected label with maximum connection has instance label
                    if(is_exising_instance_max_edge and (is_strong_link_inst_side or is_extra_strong_link_label_side) and is_strong_link_label_side):
                        is_new_inst = False  # assign segments to existing instance
                        reconnected_inst_label = labels_info_refined[connected_seg_label]['instance']
                        if reconnected_inst_label in instances_info_refined:
                            labels_info_refined[breaked_label]['instance'] = reconnected_inst_label
                            instances_info_refined[reconnected_inst_label]['labels'].append(breaked_label)
                    else:
                        if (not is_exising_instance_max_edge):
                            # find strong enough edges with label with inst  label
                            breaked_label_with_inst_confidence_map = \
                                [[label, breaked_label_confidence_map[label]] for label in labels_with_inst]
                            breaked_label_with_inst_confidence_map.sort(key = lambda l: l[1], reverse=True)
                            # get through those edges and assign inst if possible 
                            for seg_label_tuple in breaked_label_with_inst_confidence_map:
                                edge_label = seg_label_tuple[0]
                                edge_confidence = seg_label_tuple[1]
                                is_strong_max_ege_with_inst = (edge_confidence > connect_threshold_between_edges*max_connect_confidence)
                                if(is_strong_max_ege_with_inst):
                                    reconnected_inst_label = labels_info_refined[edge_label]['instance']
                                    if reconnected_inst_label in instances_info_refined:
                                        is_new_inst = False # assign segments to existing instance
                                        labels_info_refined[breaked_label]['instance'] = reconnected_inst_label
                                        instances_info_refined[reconnected_inst_label]['labels'].append(breaked_label)
                                        break
                        if(is_new_inst ):
                            # create new instance
                            self.max_instance_label += 1
                            new_inst_label = self.max_instance_label
                            labels_info_refined[breaked_label]['instance'] = new_inst_label
                            instances_info_refined[new_inst_label] = {'semantic': semantic_label, 'labels': [breaked_label]}
                # record re-assignment
                if labels_info_refined[breaked_label]['instance'] != self.labels_info_initial[breaked_label]['instance']:
                    labels_updated[breaked_label] = {'original_inst': self.labels_info_initial[breaked_label]['instance'], \
                        'current_inst': labels_info_refined[breaked_label]['instance']}

        inst_weak_list = {}
        # check for inst prediction with weak confidence
        for instance_l in instances_info_refined:
            instance_info = instances_info_refined[instance_l]
            if (len(instance_info['labels'])!=0) and (self.isThing(instance_info['semantic'])):
                inst_confidence, _ = self.computeInstanceConfidence(instance_l, instances_info_refined)
                if(inst_confidence < 3.0):
                    inst_weak_list[instance_l] = inst_confidence
        # remove inst prediction with weak confidence
        for instance_weak in inst_weak_list:
            instances_info_refined[instance_weak]['semantic'] = self.background_semantic_label
            for seg_label in instances_info_refined[instance_weak]['labels']:
                labels_info_refined[seg_label]['semantic'] = self.background_semantic_label


        return instances_info_refined, labels_info_refined, labels_updated

    def generateInstanceMesh(self, instances_info_refined, labels_updated, label_mesh_f, out_inst_mesh_f): # TODO
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
                assert(seg_label in self.labels_info_initial)
                seg_label_color = self.labels_info_initial[seg_label]['color']*1.0/255.0 # scale to 1
                seg_label_color_tensor = o3c.Tensor(seg_label_color, dtype=color_dtype, device=self.device)
                seg_vertice_index = ( o3c.Tensor.abs(seg_label_color_tensor - label_colors_tensor) < 1e-4 ).all(dim = 1)
                out_inst_colors_tensor[seg_vertice_index] = inst_color_tensor
                # log updated segments
                if(seg_label in  labels_updated):
                    label_vert_num = o3c.Tensor.sum(seg_vertice_index.to(o3c.int32))
                    if(label_vert_num > 100):
                        semantic_label = self.labels_info_initial[seg_label]['semantic']
                        log_info = "label " + str(seg_label).zfill(5) + " semantic_label: " + str(semantic_label) + \
                            " inst from " +  str(labels_updated[seg_label]['original_inst']).zfill(5) + " to " + \
                            str(labels_updated[seg_label]['current_inst']).zfill(5) + '\n'
                        self.log_io.write(log_info)

        # generate mesh
        out_inst_mesh = o3d.geometry.PointCloud()
        out_inst_mesh.points = label_mesh.points
        out_inst_mesh.colors = o3d.utility.Vector3dVector(out_inst_colors_tensor.cpu().numpy())
        o3d.io.write_point_cloud(out_inst_mesh_f, out_inst_mesh)
        return None

    def generateInstanceMeshVis(self, labels_updated, inst_mesh_f, label_mesh_f, out_inst_mesh_f): # TODO
        point_dtype = o3c.float32
        color_dtype = o3c.float32
        label_mesh = o3d.io.read_triangle_mesh(label_mesh_f)
        instance_mesh = o3d.io.read_triangle_mesh(inst_mesh_f)

        label_colors = np.asarray(label_mesh.vertex_colors).astype(np.float32)
        out_inst_colors = np.asarray(instance_mesh.vertex_colors).astype(np.float32)

        label_colors_tensor = o3c.Tensor(label_colors, dtype=color_dtype, device=self.device)
        out_inst_colors_tensor = o3c.Tensor(out_inst_colors, dtype=point_dtype, device=self.device)

        for seg_label in labels_updated:
            assert(seg_label in self.labels_info_initial)
            seg_inst_label = labels_updated[seg_label]['current_inst']

            inst_color = np.array(list(self.inst_color.getColor(seg_inst_label) ))*1.0/255.0 # scale to 1
            inst_color_tensor = o3c.Tensor(inst_color, dtype=color_dtype, device=self.device)

            seg_label_color = self.labels_info_initial[seg_label]['color']*1.0/255.0 # scale to 1
            seg_label_color_tensor = o3c.Tensor(seg_label_color, dtype=color_dtype, device=self.device)
            seg_vertice_index = ( o3c.Tensor.abs(seg_label_color_tensor - label_colors_tensor) < 1e-4 ).all(dim = 1)
            out_inst_colors_tensor[seg_vertice_index] = inst_color_tensor

        # generate mesh
        out_inst_mesh = copy.deepcopy(instance_mesh)
        out_inst_mesh.vertex_colors = o3d.utility.Vector3dVector(out_inst_colors_tensor.cpu().numpy())
        o3d.io.write_triangle_mesh(out_inst_mesh_f, out_inst_mesh)
        return None
import numpy as np
import open3d as o3d
import open3d.core as o3c
import copy

def loadConfidenceMap(confidence_file):
    confidence_map = {}  # label-sem-label-confidence

    confidence_raw = np.loadtxt(confidence_file, comments="#")
    labels_unique = np.unique(confidence_raw[:,0])

    confidence_map = {}
    for label in labels_unique:
        confidence_map[label] = {}
        sem_label_confidence_map = confidence_raw[ confidence_raw[:,0] == label ]
        semantic_unique = np.unique( sem_label_confidence_map[:, 1] )
        for semantic in semantic_unique:
            confidence_map[label][semantic] = {}
            label_confidence_map = sem_label_confidence_map[ sem_label_confidence_map[:, 1] == semantic ]
            connected_labels_unique = np.unique( label_confidence_map[:, 2] )
            assert(label_confidence_map.shape[0] == connected_labels_unique.shape[0])
            for connected_label_confidnece in label_confidence_map:
                connected_label = connected_label_confidnece[2]
                confidence = connected_label_confidnece[3]
                confidence_map[label][semantic][connected_label] = confidence
    return confidence_map

def loadLabelInitualGuess(initial_guess_file):
    labels_info_raw = np.loadtxt(initial_guess_file, comments="#")

    labels_info = {}
    instance_info = {}
    for label_info in labels_info_raw:
        label = int(label_info[0])
        semantic_label = int(label_info[1])
        instance_label = int(label_info[2])
        color = label_info[3:].astype(int)
        # record label information
        labels_info[label] = {'semantic': semantic_label, 'instance': instance_label, 'color': color}
        # record instance information
        if(instance_label != 0):
            if(instance_label not in instance_info):
                instance_info[instance_label] = {'semantic': semantic_label, 'labels': [label]}
            else:
                assert(instance_info[instance_label]['semantic'] == semantic_label)
                instance_info[instance_label]['labels'].append(label)
    return labels_info, instance_info

class InstanceColor: # TODO
    def __init__(self):
        self.instance = set()
        self.assigned_colors = set()
        self.instance_colors = {}

    def getColor(self, instance_label):
        if(instance_label in self.instance_colors):
            return self.instance_colors[instance_label]
        else:
            fresh_color = self.getFreshColor()
            self.instance_colors[instance_label] = fresh_color
            self.assigned_colors.add(fresh_color)
            return np.array(fresh_color)

    def getFreshColor(self):
        return tuple(np.random.choice(range(256), size=3))

class Instance:
    def __init__(self, instance_id, semantic, labels, confidence_map):
        self.id_ = instance_id
        self.labels_ = list(labels)
        self.semantic = semantic

        self.confidence_map = confidence_map # use referrence
        self.confidence = 0
        self.is_confidence_update = False

    def queryConfidence(self, label_a, label_b):
        if(label_a in self.confidence_map):
            if(self.semantic in self.confidence_map[label_a]):
                if(label_b in self.confidence_map[label_a][self.semantic]):
                    return self.confidence_map[label_a][self.semantic][label_b]
        return 0

    def computeInstanceConfidence(self):
        # use maximum of external edges between labels as confidence
        
        labels_num = len(self.labels_)
        if(labels_num == 1):
            label = self.labels_[0]
            self.confidence = self.queryConfidence(label,label)
            return
        for l_i in range(labels_num):
            label_a = self.labels_[l_i]
            for l_j in range(l_i+1, labels_num):
                label_b = self.labels_[l_j]
                confidence = self.queryConfidence(label_a,label_b)
                if confidence > self.confidence:
                    self.confidence = confidence
        self.is_confidence_update = True
    def updateConfidence(self):
        if(not self.is_confidence_update):
            self.computeInstanceConfidence()

    def isInstanceConnected(self, instance):
        # check whether other instance could be merged into this instance
        min_inst_confidence = 10
        min_merge_ratio = 0.5
        self.updateConfidence()
        instance.updateConfidence()

        if(self.semantic != instance.semantic):
            return False
        if(self.confidence < min_inst_confidence or instance.confidence < min_inst_confidence):
            return False

        max_instance_connection = 0
        for label_a in self.labels_:
            for label_b in instance.labels_:
                instance_connection = self.queryConfidence(label_a,label_b)
                if(instance_connection > max_instance_connection):
                    max_instance_connection = instance_connection
        if max_instance_connection > self.confidence * min_merge_ratio:
            return True

    def mergeInstance(self, instance_list):
        assert( len(instance_list) > 0 )
        for instance in instance_list:
            self.labels_.update(instance.labels_)
        self.updateConfidence()

class SegGraph:
    def __init__(self, confidence_file, initial_guess_file, log_io,  BackgroundSemLabel = 80):
        # log
        self.log_io = log_io
        # load confidence map and initial label_inst_guess
        self.confidence_map = loadConfidenceMap(confidence_file)
        self.labels_info_initial, self.instances_info_initial = \
            loadLabelInitualGuess(initial_guess_file)
        self.background_semantic_label = BackgroundSemLabel

        # seperate instances into groups according to semantics 
        self.semantic_instances_map = {}
        for instance_label in self.instances_info_initial.keys():

            instance_info = self.instances_info_initial[instance_label]
            semantic_label = instance_info['semantic']
            if(semantic_label == self.background_semantic_label):
                continue
            labels = instance_info['labels']
            instance = Instance(instance_label, semantic_label, labels, self.confidence_map)
            if semantic_label in self.semantic_instances_map:
                self.semantic_instances_map[semantic_label].append(instance)
            else:
                self.semantic_instances_map[semantic_label] = [instance]

        # device for mesh operation
        if(o3c.cuda.is_available()):
            self.device = o3c.Device("CUDA", 0)
        else:
            self.device = o3c.Device("CPU", 0)
        # inst color
        self.inst_color = InstanceColor()

        # SegGraph initialized!
        self.log_to_file("SegGraph initialized!")
    
    def log_to_file(self, info):
        self.log_io.write(info)

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

    def refineLabelInstanceMap(self):
        instances_info_refined =copy.deepcopy(self.instances_info_initial)
        labels_updated = {}
        return instances_info_refined, labels_updated

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
                assert(seg_label in self.labels_info_initial)
                seg_label_color = self.labels_info_initial[seg_label]['color']*1.0/255.0 # scale to 1
                seg_label_color_tensor = o3c.Tensor(seg_label_color, dtype=color_dtype, device=self.device)
                seg_vertice_index = ( o3c.Tensor.abs(seg_label_color_tensor - label_colors_tensor) < 1e-4 ).all(dim = 1)
                out_inst_colors_tensor[seg_vertice_index] = inst_color_tensor

        # generate mesh
        out_inst_mesh = o3d.geometry.PointCloud()
        out_inst_mesh.points = label_mesh.points
        out_inst_mesh.colors = o3d.utility.Vector3dVector(out_inst_colors_tensor.cpu().numpy())
        o3d.io.write_point_cloud(out_inst_mesh_f, out_inst_mesh)
        return None

confidence_file = "/home/yang/big_ssd/results/032/PanoSem2LSISegGraph/log/ConfidenceMap.txt"
initial_guess_file = "/home/yang/big_ssd/results/032/PanoSem2LSISegGraph/log/LabelInitialGuess.txt"
label_mesh_f = "/home/yang/big_ssd/results/032/PanoSem2LSISegGraph/label_mesh_.ply"
out_inst_mesh_f = "/home/yang/big_ssd/results/032/PanoSem2LSISegGraphRepeat/instance_mesh_.ply"


seg_graph = SegGraph(confidence_file, initial_guess_file)
instances_info_refined,labels_updated  = seg_graph.refineLabelInstanceMap()
seg_graph.generateMesh(instances_info_refined, label_mesh_f, out_inst_mesh_f)
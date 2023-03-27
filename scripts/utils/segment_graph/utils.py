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

def loadLabelInitualGuess(initial_guess_file, confidence_file):
    labels_info_raw = np.loadtxt(initial_guess_file, comments="#")

    labels_info = {}
    instance_info = {}
    semantic_instance_map = {}
    for label_info in labels_info_raw:
        label = int(label_info[0])
        semantic_label = int(label_info[1])
        instance_label = int(label_info[2])
        color = label_info[3:].astype(int)
        
        # record label information
        labels_info[label] = {'semantic': semantic_label, 'instance': instance_label, 'color': color}
        
        if(instance_label != 0):

            # record instance information
            if(instance_label not in instance_info):
                instance_info[instance_label] = {'semantic': semantic_label, 'labels': [label]}
            else:
                assert(instance_info[instance_label]['semantic'] == semantic_label)
                instance_info[instance_label]['labels'].append(label)
            # record semantic-instance-map
            if(semantic_label not in semantic_instance_map):
                semantic_instance_map[semantic_label] = set([instance_label])
            else:
                semantic_instance_map[semantic_label].add(instance_label)

    confidence_raw = np.loadtxt(confidence_file, comments="#")

    confidence_map = {}
    for label in list(labels_info.keys()):
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
                if connected_label in labels_info:
                    confidence = connected_label_confidnece[3]
                    confidence_map[label][semantic][connected_label] = confidence
    return labels_info, instance_info, semantic_instance_map, confidence_map

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
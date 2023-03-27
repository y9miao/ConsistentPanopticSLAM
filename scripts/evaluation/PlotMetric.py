import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
sys.path.append("/home/yang/toolbox/evaluation")

MetriccFolder = "/home/yang/toolbox/evaluation/MetricNumbers/Seqs74/results"
task = "CoCoPano"
ov_ths= [50, 75]

# class names
labels = None
if(task == "CoCo"):
	from EvaluationMy.SemanticEvaluation.color_maps import CLASS_LABELS_GT,VALID_CLASS_IDS_GT,VALID_CLASSID2COLOR
	labels = CLASS_LABELS_GT
elif(task == "CoCoPano"):
	from EvaluationMy.SemanticEvaluation.color_maps import CLASS_LABELS_GT,VALID_CLASS_IDS_GT,VALID_CLASSID2COLOR
	labels = CLASS_LABELS_GT

class SemMetricPLot:
	def __init__(self, MetriccFolder, Methods, MethodNames, Metrics, PlotsFolder, ov_ths, labels):
		MethodsFolders = [os.path.join(MetriccFolder, method_name) for method_name in Methods]
		self.method_folders = MethodsFolders
		self.methods = Methods
		self.method_names = MethodNames
		self.metrics = Metrics
		self.plots_folder = PlotsFolder
		self.ov_ths = ov_ths
		self.labels = labels
		self.metric_numbers = {}

	def plot_panoptic_quality(self):
		panoptic_plot_folder = os.path.join(self.plots_folder, "panoptic_quality")
		if not os.path.exists(panoptic_plot_folder):
			os.makedirs(panoptic_plot_folder)
		MetricNameMap = {"PQ": "Panoptic Quality", "SQ": "Segmentation Quality", "RQ": "Recognition Quality" }
		labels_pq = self.labels + ['ave_class', 'ave_instance']

		panoptic_methods = []
		for m_i, method_folder in enumerate(self.method_folders):
			panoptic_quality_pkl = os.path.join(method_folder, 'panoptic_quality', "panoptic_quality.pkl" )
			panoptic_quality = None
			with open(panoptic_quality_pkl, 'rb') as f:
				panoptic_quality = pickle.load(f)
				panoptic_methods.append(panoptic_quality)
		# read in panoptic quality
		for ov_th in self.ov_ths:
			for item in ["PQ", "SQ", "RQ"]:
				panoptic_values = [] # m x l, m for different methods, l for different semantic classes
				for m_i,method_name in enumerate(self.method_names):
					panoptic_values_m = []
					for label_name in labels_pq:
						if label_name in panoptic_methods[m_i][ov_th] and item in panoptic_methods[m_i][ov_th][label_name]:
							panoptic_values_m.append(panoptic_methods[m_i][ov_th][label_name][item])
						else:
							panoptic_values_m.append(0)
					panoptic_values.append(panoptic_values_m)
				
				file = os.path.join(panoptic_plot_folder, item+"_"+str(ov_th)+".png")
				metric_title = item+" (IOU "+str(ov_th)+")"
				metric_values = np.array(panoptic_values).reshape(len(panoptic_methods), len(labels_pq))
				metric_name = MetricNameMap[item]
				self.plotBar( labels_pq, metric_title, metric_values, metric_name, self.method_names, file)

				txt_file = os.path.join(panoptic_plot_folder, item+"_"+str(ov_th)+".txt")
				# save log txt
				self.logTxt(metric_values, labels_pq, self.method_names, txt_file)
		
		return None

	def plot_mAP(self):
		mAP_plot_folder = os.path.join(self.plots_folder, "mAP")
		if not os.path.exists(mAP_plot_folder):
			os.makedirs(mAP_plot_folder)
		labels_map = self.labels + ['ave_class', 'ave_instance']
		mAP_methods = []
		for m_i, method_folder in enumerate(self.method_folders):
			mAP_pkl = os.path.join(method_folder,'mAP', "mAP.pkl" )
			mAP = None
			with open(mAP_pkl, 'rb') as f:
				mAP = pickle.load(f)
				mAP_methods.append(mAP)
		# read in mAP
		for ov_th in ["ap50%", "ap"]:
			mAP_values = [] # m x l, m for different methods, l for different semantic classes
			for m_i,method_name in enumerate(self.method_names):
				mAP_values_m = []
				for label_name in labels_map:
					if label_name in mAP_methods[m_i]:
						mAP_values_m.append(mAP_methods[m_i][label_name][ov_th])
					else:
						mAP_values_m.append(0)
				mAP_values.append(mAP_values_m)
			
			file = os.path.join(mAP_plot_folder, "mAP_"+str(ov_th)+".png")

			metric_title = None
			if ov_th == "ap50%":
				metric_title = "mAP (IOU "+str(50)+")"
			else:
				metric_title = "mAP (IOU "+str(75)+")"
			metric_values = np.array(mAP_values).reshape(len(mAP_values), len(labels_map))
			metric_name = "mAP"
			self.plotBar( labels_map, metric_title, metric_values, metric_name, self.method_names, file)

			txt_file = os.path.join(mAP_plot_folder, "mAP_"+str(ov_th)+".txt")
			# save log txt
			self.logTxt(metric_values, labels_map,self.method_names, txt_file)
		return None

	def plot_matched_inst_num(self):
		inst_num_plot_folder = os.path.join(self.plots_folder, "matched_num")
		if not os.path.exists(inst_num_plot_folder):
			os.makedirs(inst_num_plot_folder)
		# read in inst_num
		inst_num_methods = []
		labels_inst_num = self.labels+['total']
		for m_i, method_folder in enumerate(self.method_folders):
			inst_num_pkl = os.path.join(method_folder,'matched_num', "matched_num.pkl" )
			inst_num = None
			with open(inst_num_pkl, 'rb') as f:
				inst_num = pickle.load(f)
				inst_num_methods.append(inst_num)
		for ov_th in [50, 75]:
			inst_nums = [] # m x (l+1), m for different methods, l for different semantic classes
			for m_i,method_name in enumerate(self.method_names):
				total_m = 0
				inst_num_m = []
				for label_name in self.labels:
					if label_name in inst_num_methods[m_i][ov_th]:
						inst_num_m.append(inst_num_methods[m_i][ov_th][label_name])
						total_m += inst_num_methods[m_i][ov_th][label_name]
					else:
						inst_num_m.append(0)
				inst_num_m.append(total_m)
				inst_nums.append(inst_num_m)

			file = os.path.join(inst_num_plot_folder, "instance_num_"+str(ov_th)+".png")
			metric_title = None
			if ov_th == 50:
				metric_title = "num of matched instance (IOU "+str(50)+")"
			else:
				metric_title = "num of matched instance (IOU "+str(75)+")"
			metric_values = np.array(inst_nums).reshape(len(inst_nums), len(labels_inst_num))
			metric_name = "num of matched instance"
			self.plotBar( labels_inst_num, metric_title, metric_values, metric_name, self.method_names, file)

			txt_file = os.path.join(inst_num_plot_folder, "instance_num_"+str(ov_th)+".txt")
			# save log txt
			self.logTxt(metric_values, labels_inst_num, self.method_names, txt_file)
		return None

	def plot_chamfer(self):
		chamfer_plot_folder = os.path.join(self.plots_folder, "chamfer")
		if not os.path.exists(chamfer_plot_folder):
			os.makedirs(chamfer_plot_folder)

		labels_chamfer = self.labels+ ['ave_class', 'ave_instance']
		chamfer_methods = []
		for m_i, method_folder in enumerate(self.method_folders):
			chamfer_pkl = os.path.join(method_folder, 'chamfer', "chamfer.pkl" )
			chamfer = None
			with open(chamfer_pkl, 'rb') as f:
				chamfer = pickle.load(f)
				chamfer_methods.append(chamfer)
		# read in chamfer distance
		for ov_th in [50, 75]:
			for item in ['pred2gt', 'gt2pred']:
				chamfer_values = [] # m x l, m for different methods, l for different semantic classes
				for m_i,method_name in enumerate(self.method_names):
					chamfer_values_m = []
					for label_name in labels_chamfer:
						if label_name in chamfer_methods[m_i][ov_th][item]:
							chamfer_values_m.append(chamfer_methods[m_i][ov_th][item][label_name])
						else:
							chamfer_values_m.append(0)
					chamfer_values.append(chamfer_values_m)
				
				file = os.path.join(chamfer_plot_folder, "chamfer_"+item+"_"+str(ov_th)+".png")
				item_name = None
				if item == 'pred2gt':
					item_name = "inaccuracy"
				else:
					item_name = "incompleteness"
				metric_title = "Chamfer "+ item_name +" (IOU "+str(ov_th)+")"
				metric_values = np.array(chamfer_values).reshape(len(chamfer_values), len(labels_chamfer)) * 1e4 # transform to cm^2
				metric_name = "Chamfer "+ item_name
				self.plotBar( labels_chamfer, metric_title, metric_values, metric_name, self.method_names, file)

				txt_file = os.path.join(chamfer_plot_folder, item_name+"_"+str(ov_th)+".txt")
				# save log txt
				self.logTxt(metric_values, labels_chamfer, self.method_names, txt_file)
		return None

	def plotBar(self, labels, metric_title, metric_values, metric_name, method_names, fig_path):
		# metric_values m x l, m for different methods, l for different semantic classes
		x = np.arange(len(labels))
		fig, ax = plt.subplots(figsize=(12, 9))
		ax.tick_params(axis='both', which='major', labelsize=14)

		num_methods = metric_values.shape[0]
		num_labels = metric_values.shape[1]
		bar_width = min(0.08, 1.0/(num_methods*2) )
		bars = {}

		for m_i in range(num_methods):
			bar_shift = m_i-num_methods/2.0
			bars[m_i] = ax.bar(x + bar_width*bar_shift, metric_values[m_i], bar_width, label=method_names[m_i])

		ax.set_ylabel(metric_name, fontsize=16)
		ax.set_xlabel('semantic class', fontsize=16)
		ax.set_title(metric_title)
		ax.set_xticks(x)
		ax.set_xticklabels(labels)
		ax.legend(loc='upper left', fontsize=12)
		fig.tight_layout()
		fig.savefig(fig_path, bbox_inches='tight')

		return None
	def logTxt(self, metric_values, labels, method_names, txt_file):
		_SPLITTER = ",  "
		method_names_len_list = [len(method_name) for method_name in method_names]
		labels_len_list = [len(label) for label in labels]
		longest_method_len = max(method_names_len_list)
		longest_label_len = max(labels_len_list)
		head_string = [' '.ljust(longest_method_len)]
		for label in labels:
			head_string.append(str(label).ljust(longest_label_len))

		with open(txt_file, 'w') as f:
			f.write(_SPLITTER.join(head_string) + '\n')

			for m_i,method in enumerate(method_names):
				value_string_list = [str(method).ljust(longest_method_len)]
				for l_i, label in enumerate(labels):
					value = metric_values[m_i][l_i]
					value_str = str(value).ljust(longest_label_len)[:longest_label_len]
					value_string_list.append(value_str)
				f.write(_SPLITTER.join(value_string_list) + '\n') 
					
		return None

# PlotsFolder = "/home/yang/toolbox/evaluation/MetricNumbers/Seqs74/plots/plots_8_noweight"
# Methods = ["same_ros", "2D_original", "2D_confidence", "2D_confidence_SemMerging", "2D_SemMerge2", "2D_confidence_SemMerging_Label", "2D_original_instance"]
# MethodNames = ["1.original", "2.2D_confidence", "3.semMergeCount", "4.semMergeCount+max", "5. semMergeQuery+max", "6. 4.+label_confi", "7. 2.+instance_confi"]
# Metrics = ["mAP", "chamfer", "panoptic_quality"]

# PlotsName = "PanopticCompre"
# PlotsFolder = os.path.join( "/home/yang/toolbox/evaluation/MetricNumbers/Seqs74/plots", PlotsName )
# Methods = ["same_ros2_order_debug",  "SemMergeQuery", "SemMergeQuery2", "ClassSemInstanceCount","ClassInstanceSemCount", "Sem2CSICount", \
# 			"Pano","PanoSim","PanoSim2",  "PanoSemInstCount",  "PanoInstSemCount", "PanoSem2CSICount" ]
			
# MethodNames = ["1.Voxblox++", "2.SemMergeQuery", "3.SemMergeQuery2","4.ClassSemInstCount", "5.ClassInstSemCount", "3.+4.", \
# 				"1.Pano++","2.PanoSem","3.PanoSem2", "4.PanoSemInstCount", "5.PanoInstSemCount", "3.+4."]
# Metrics = ["mAP", "chamfer", "panoptic_quality", 'matched_num']

# PlotsName = "SegGraphCompre"
# PlotsFolder = os.path.join( "/home/yang/toolbox/evaluation/MetricNumbers/Seqs74/plots", PlotsName )
# Methods = ["same_ros2_order_debug", "PanoSim2", "PanoSem2CSICount", "PanoSem2SG_0_0.12", "PanoSem2SG_0_0.2", "PanoSem2SG_0_0.3", \
# 		 "PanoSem2SG_1_0.12", "PanoSem2SG_1_0.2", "PanoSem2SG_1_0.3"]
			
# MethodNames = ["1.Voxblox++", "2.PanoSem2", "3.PanoSem2LSI." , "4. SG0_0.12", "5. SG0_0.2", "6. SG0_0.3", \
# 		"7. SG1_0.12", "8. SG1_0.2", "9. SG1_0.3"]
# Metrics = ["mAP", "chamfer", "panoptic_quality", 'matched_num']

# PlotsName = "SegGraphCompare"
# PlotsFolder = os.path.join( "/home/yang/toolbox/evaluation/MetricNumbers/Seqs74/plots", PlotsName )
# Methods = ["PanoSem2CSICount", "PanoSem2LSISegGraphBreak", "PanoSem2LSISegGraphBreak4", "LSISegGraph/Confidence0BreakAll", \
# 			"LSISegGraph/Confidence0BreakAllCrazy", "LSISegGraph/Confidence1Break4", "LSISegGraph/Confidence1BreakAllCrazy",
# 			"LSISegGraph/Confidence2Break4", "LSISegGraph/Confidence2BreakAll", "LSISegGraph/Confidence3", "LSISegGraph/Confidence3BreakAllCrazy", \
# 			"LSISegGraph/Confidence3BreakReunion", "LSISegGraph/Sem3Confidence3", "LSISegGraph/Sem2Confidence3Fast", "LSISegGraph/Sem2Confidence3FastBreakReunion"]
			
# MethodNames = ["1.PanoSem2CSICount", "2.PanoSem2LSISegGraphBreak.",  "3.PanoSem2LSISegGraphBreak4.", "PanoSem2LSISegGraphBreakAll", \
# 		"PanoSem2LSISegGraphBreakAllC0Crazy", "4.SegGraphC1B4.", "PanoSem2LSISegGraphBreakAllC1Crazy" ,
# 		"5. SegGraphC2B4", "6. SegGraphC2BAll", "7. SegGraphC3", "SegGraphBreakAllC3Crazy", "SegGraphBreakC3Reunion", "8.Sem3Confidence3", \
# 			"9.Sem2Confidence3Fast", "9-b Sem2Confidence3FastBreakReunion"]

# PlotsName = "SegGraphCompare"
# PlotsFolder = os.path.join( "/home/yang/toolbox/evaluation/MetricNumbers/Seqs74/plots", PlotsName )
# Methods = ["PanoSem2CSICount","LSISegGraph/Confidence3", "LSISegGraph/Confidence3BreakReunion","LSISegGraph/Sem2Confidence3Fast", \
# 			"LSISegGraph/GraphCut1UnaryOverAveNeighor", "LSISegGraph/GraphCut04UnaryOverAveNeighor", \
# 				"LSISegGraph/GraphCut01UnaryOverAveNeighor", "LSISegGraph/GraphCutTest"]
			
# MethodNames = ["1.PanoSem2CSICount", "2.Confidence3.",  "3.Confidence3BreakReunion.", "4.Sem2Confidence3Fast", \
# 		 "6.GraphCut1UnaryOverAveNeighor", "7. GraphCut04UnaryOverAveNeighor", \
# 			"8. GraphCut01UnaryOverAveNeighor", "9.GraphCutTest"]

# PlotsName = "C3Sem2FastSemGraphCutTest"
# PlotsFolder = os.path.join( "/home/yang/toolbox/evaluation/MetricNumbers/Seqs74/plots", PlotsName )
# Methods = ["Pano" , "LSISegGraph/Confidence3", "LSISegGraph/Confidence3BreakReunion"]
# Method_Tests = []
# K_list = [1., 8., 15., 20.]
# theta_list = [0.05, 0.1, 0.5, 0.7]
# for K in K_list:
# 	for theta in theta_list:
# 		Method = "LSISegGraph/C3Sem2FastSemGraphCutTest" + "/Test_" + str(K) + "_" + str(theta)
# 		Method_Tests.append(Method)
# Methods.extend(Method_Tests)

# PlotsName = "C3Sem2FastSemGraphCutTestB2"
# PlotsFolder = os.path.join( "/home/yang/toolbox/evaluation/MetricNumbers/Seqs74/plots", PlotsName )
# Methods = ["Pano" , "LSISegGraph/Confidence3", "LSISegGraph/Confidence3BreakReunion"]
# Method_Tests = []
# K_list = [1., 5., 10., 15.]
# theta_list = [0.01, 0.1, 0.5, 1., 5.]
# for K in K_list:
# 	for theta in theta_list:
# 		Method = "LSISegGraph/C3Sem2FastSemGraphCutTestB2" + "/Test_" + str(K) + "_" + str(theta)
# 		Method_Tests.append(Method)
# Methods.extend(Method_Tests)
			
# MethodNames = ["1.Pano", "2.PanoSem2LSI.",  "3.Confidence3BreakReunion."]
# MethodNames_Tests = []

# PlotsName = "FinalCompare"
# PlotsFolder = os.path.join( "/home/yang/toolbox/evaluation/MetricNumbers/Seqs74/plots", PlotsName )

# Methods = [
# 	"2D_confidence", \
# 	"Pano" , \
# 	"PanoSemInstCount",\
# 	"PanoSem2LSISegGraph", \
# 	"V_SegGraph/Sem0C4FastB2_Break015_0.5", \
# 	"V_SegGraph/Sem0C4_Break115_0.5", \
# 	"V_25SegGraph/Sem2C4_Break_0_15_0.5", \
# 	"V_25SegGraph/Sem2C4_OnlyBreak", \
# 	"V_SegGraph/Sem0C4FastB2_Break115_0.5", \
# 	"V_25SegGraph/Sem2Confidence4FastTest/Test_15.0_0.5"
# 	]

# MethodNames = [
# 	"1.Voxbloxpp", 
# 	"2.Voxbloxpp with Pano", 
# 	"3.Modified Voxblox.", \
# 	"4.Modified V. with C2", \
# 	"5.Modified V. with C4", \
# 	"6.Modified V. with C5", \
# 	"7.Modified V. with C2, C4", \
# 	"8.Modified V. with C2, C5", \
# 	"9.Modified V. with C4, C5", \
# 	"10.Final Version(Modified V. with C2, C4, C5)"
# 	]

# Metrics = ["mAP", "chamfer", "panoptic_quality", 'matched_num']
# metric_plot = SemMetricPLot(MetriccFolder, Methods, MethodNames, Metrics, PlotsFolder, ov_ths, labels)
# metric_plot.plot_panoptic_quality()
# metric_plot.plot_mAP()
# metric_plot.plot_chamfer()
# metric_plot.plot_matched_inst_num()

PlotsName = "InstGraphCutTune"
PlotsFolder = os.path.join( "/home/yang/toolbox/evaluation/MetricNumbers/Seqs74/plots", PlotsName )


Methods = ["V_25SegGraph/Sem2Confidence4FastTest/Test_15.0_0.5"]
MethodNames = ["Final"]
K_inst_list = [10]
Break_th_inst_list = [0.05, 0.1, 0.2, 0.3]
Break_th_label_list = [0.4, 0.5, 0.6, 0.7]

for K_inst in K_inst_list:
	for Break_th_inst in Break_th_inst_list:
		for Break_th_label in Break_th_label_list:	
			params_params = "_".join([str(K_inst), str(Break_th_inst), str(Break_th_label)])
			MethodName = "InstGraphCutTest/Test_" + str(K_inst)+str(Break_th_inst)+str(Break_th_label)
			Methods.append(MethodName)
			MethodNames.append("Test_" + str(K_inst)+str(Break_th_inst)+str(Break_th_label))

Metrics = ["mAP", "chamfer", "panoptic_quality", 'matched_num']
metric_plot = SemMetricPLot(MetriccFolder, Methods, MethodNames, Metrics, PlotsFolder, ov_ths, labels)
metric_plot.plot_panoptic_quality()
metric_plot.plot_mAP()
metric_plot.plot_chamfer()
metric_plot.plot_matched_inst_num()
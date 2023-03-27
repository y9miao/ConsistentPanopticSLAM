import subprocess
import os
import math
import numpy as np
import pickle
import copy


class SemMetric:

	def __init__(self, output_folder):

		self.mAP = {}
		self.chamfer_distance = { 
				50:{'pred2gt': {}, 'gt2pred': {} }, 
				75:{'pred2gt': {}, 'gt2pred': {} }	}
		self.panoptic_quality = {50:{}, 75: {}}
				
		self.class_weight_mAP = {}
		self.class_weight_chamfer = {50:{}, 75: {}}
		self.panoptic_weight = {50:{}, 75: {}}
		self.matched_instance_num = {50:{}, 75: {}}
		self.gt_count = {}

		self.seq_mAP = {}
		self.seq_chamfer = {}
		self.seq_panoptic = {}
		self.seq_matched_instance_num = {}

		self.output_folder = output_folder
		self.mAP_folder = os.path.join(output_folder, "mAP")
		self.chamfer_folder = os.path.join(output_folder, "chamfer")
		self.panoptic_quality_folder = os.path.join(output_folder, "panoptic_quality")
		self.matched_num_folder = os.path.join(output_folder, "matched_num")

		self.seq_mAP_folder = os.path.join(self.mAP_folder, "seqs")
		self.seq_chamfer_folder = os.path.join(self.chamfer_folder, "seqs")
		self.seq_panoptic_quality_folder = os.path.join(self.panoptic_quality_folder, "seqs")
		self.seq_matched_num_folder = os.path.join(self.matched_num_folder, "seqs")

		self.class_names = None

		if not os.path.exists(self.seq_mAP_folder):
			os.makedirs(self.seq_mAP_folder)
		if not os.path.exists(self.seq_chamfer_folder):
			os.makedirs(self.seq_chamfer_folder)
		if not os.path.exists(self.seq_panoptic_quality_folder):
			os.makedirs(self.seq_panoptic_quality_folder)
		if not os.path.exists(self.seq_matched_num_folder):
			os.makedirs(self.seq_matched_num_folder)
		return None

	# insert matched instance number
	def insert_matchedNum(self, matched_instance_num, gt_class_counts, seq_num):
		class_names = list(gt_class_counts.keys())
		class_names.sort()
		for class_name in gt_class_counts.keys():
			if class_name not in self.gt_count:
				self.gt_count[class_name] = copy.deepcopy(gt_class_counts[class_name])
			else:
				self.gt_count[class_name] += gt_class_counts[class_name]
			for ov_th in [50, 75]:
				if class_name not in self.matched_instance_num[ov_th]:
					self.matched_instance_num[ov_th][class_name] = \
						copy.deepcopy(matched_instance_num[ov_th][class_name])
				else:
					self.matched_instance_num[ov_th][class_name] += \
						copy.deepcopy(matched_instance_num[ov_th][class_name])	
		self.seq_matched_instance_num[seq_num] = copy.deepcopy(matched_instance_num)
		if self.class_names is None:
			self.class_names = copy.deepcopy(class_names)

	def write_seq_matchedNum(self, gt_class_counts, seq_num):
		_SPLITTER = ",  "
		matchedNum_txt = os.path.join(self.seq_matched_num_folder, seq_num+".txt")
		with open(matchedNum_txt, 'w') as f:
			head_string = ['class'.ljust(8), 'gt_count'.ljust(8),'ov_50'.ljust(6),'ov_75'.ljust(6) ]
			f.write(_SPLITTER.join(head_string) + '\n')
			class_names = list(gt_class_counts.keys())
			class_names.sort()
			for class_name in class_names:
				gt_count_str = format(gt_class_counts[class_name]).ljust(8)[:8]
				matcthed_num_ov50 = format(self.seq_matched_instance_num[seq_num][50][class_name]).ljust(6)[:6]
				matcthed_num_ov75 = format(self.seq_matched_instance_num[seq_num][75][class_name]).ljust(6)[:6]
				value_string = [class_name.ljust(8), gt_count_str, matcthed_num_ov50, matcthed_num_ov75]
				f.write(_SPLITTER.join(value_string) + '\n') 
		return None

	# insert mAP of the scene to compute weighted average mAP
	def insert_mAP(self, mAP, gt_class_counts, seq_num):
		for class_name in mAP['classes'].keys():
			# if the scene contains no certain class
			if gt_class_counts[class_name] == 0:
				continue
			# if the first meet of the class
			if class_name not in self.mAP:
				self.mAP[class_name] = copy.deepcopy( mAP['classes'][class_name] )
				self.class_weight_mAP[class_name] = copy.deepcopy( gt_class_counts[class_name] )
				continue
			weight_cur = self.class_weight_mAP[class_name]
			weight_insert = gt_class_counts[class_name]
			weight_update = weight_cur+weight_insert
			for ap in mAP['classes'][class_name].keys():
				mAP_cur = self.mAP[class_name][ap]
				mAP_insert = copy.deepcopy( mAP['classes'][class_name][ap] )
				mAP_updated = (mAP_cur*weight_cur+mAP_insert*weight_insert)/weight_update

				self.mAP[class_name][ap] = mAP_updated
			self.class_weight_mAP[class_name] = weight_update

		# record sequence mAP
		self.seq_mAP[seq_num] = {}
		for class_name in mAP['classes'].keys():
			self.seq_mAP[seq_num][class_name] = copy.deepcopy( mAP['classes'][class_name] )
		return None
	def write_seq_mAP(self, avgs, seq_num):
		_SPLITTER = ",  "
		mAP_txt = os.path.join(self.seq_mAP_folder, seq_num+".txt")
		with open(mAP_txt, 'w') as f:
			f.write(_SPLITTER.join(['class'.ljust(5), 'ap75'.ljust(10), 
					'ap50'.ljust(10),'ap25'.ljust(10) ]) + '\n')
			class_names = list(avgs["classes"].keys())
			class_names.sort()
			for class_name in class_names:
				ap = format(avgs["classes"][class_name]["ap"], '.10f').ljust(10)[:10]
				ap50 = format(avgs["classes"][class_name]["ap50%"], '.10f').ljust(10)[:10]
				ap25 = format(avgs["classes"][class_name]["ap25%"], '.10f').ljust(10)[:10]
				f.write(_SPLITTER.join([str(x) for x in [str(class_name).ljust(5), ap, ap50,ap25]]) + '\n') 
		return None
	def insert_chamfer(self, chamfer_distance_class, gt_class_counts, seq_num):
		for class_name in chamfer_distance_class[50]['gt2pred']:
			# if the scene contains no certain class
			if gt_class_counts[class_name] == 0:
				continue
			for ap in [50,75]:
				# if no match 
				if(np.isnan(chamfer_distance_class[ap]['pred2gt'][class_name])):
					continue
				# if the first meet of the class
				if class_name not in self.class_weight_chamfer[ap]:
					self.class_weight_chamfer[ap][class_name] = \
						copy.deepcopy(chamfer_distance_class[ap]['count'][class_name])
					for chamfer_item in ['pred2gt', 'gt2pred']:
						self.chamfer_distance[ap][chamfer_item][class_name] = \
							copy.deepcopy(chamfer_distance_class[ap][chamfer_item][class_name])
					continue
				weight_cur = self.class_weight_chamfer[ap][class_name]
				weight_insert = chamfer_distance_class[ap]['count'][class_name]
				weight_update = weight_cur+weight_insert
				for chamfer_item in ['pred2gt', 'gt2pred']:
					chamfer_cur = self.chamfer_distance[ap][chamfer_item][class_name]
					chamfer_insert = chamfer_distance_class[ap][chamfer_item][class_name]
					chamfer_updated = (chamfer_cur*weight_cur+chamfer_insert*weight_insert)/weight_update

					self.chamfer_distance[ap][chamfer_item][class_name] = chamfer_updated
				self.class_weight_chamfer[ap][class_name] = weight_update
		# record sequence chamfer
		self.seq_chamfer[seq_num] = { 
				50:{'pred2gt': {}, 'gt2pred': {} }, 
				75:{'pred2gt': {}, 'gt2pred': {} }	}
		for class_name in chamfer_distance_class[50]['gt2pred']:
			for ap in [50,75]:
				for chamfer_item in ['pred2gt', 'gt2pred']:
					self.seq_chamfer[seq_num][ap][chamfer_item][class_name] = \
						copy.deepcopy(chamfer_distance_class[ap][chamfer_item][class_name])
		return None
	def insert_panoptic(self, panoptic_quality, gt_class_counts, seq_num):
		class_names = list(gt_class_counts.keys())
		class_names.sort()
		for ov_th in [50,75]:	
			for class_name in class_names:
				# if the scene contains no certain class

				# if the first meet of the class
				if class_name not in self.panoptic_weight[ov_th]:
					self.panoptic_weight[ov_th][class_name] = {}
					self.panoptic_quality[ov_th][class_name] = {}
				if gt_class_counts[class_name] == 0:
					continue
				for item in ["PQ", "SQ", "RQ"]:
					if(np.isnan(panoptic_quality[ov_th][class_name][item])):
						continue
					weight_cur_item = None
					if item == 'SQ':
						weight_cur_item = panoptic_quality[ov_th]['count'][class_name]
					else:
						weight_cur_item = gt_class_counts[class_name]
					if weight_cur_item == 0:
						continue

					if item not in self.panoptic_weight[ov_th][class_name]:
						self.panoptic_weight[ov_th][class_name][item] = \
							copy.deepcopy(weight_cur_item)
						self.panoptic_quality[ov_th][class_name][item] = \
							copy.deepcopy(panoptic_quality[ov_th][class_name][item])
						continue
					weight_cur = self.panoptic_weight[ov_th][class_name][item]
					weight_insert = weight_cur_item
					weight_update = weight_cur+weight_insert
					
					quality_cur = self.panoptic_quality[ov_th][class_name][item]
					quality_insert = panoptic_quality[ov_th][class_name][item]
					quality_updated = (quality_cur*weight_cur+quality_insert*weight_insert)/weight_update

					self.panoptic_quality[ov_th][class_name][item] = quality_updated
					self.panoptic_weight[ov_th][class_name][item] = weight_update
		# record sequence panoptic quality
		self.seq_panoptic[seq_num] = {50:{}, 75: {}}
		for ov_th in [50,75]:	
			for class_name in class_names:
				# for item in ["PQ", "SQ", "RQ"]:
				self.seq_panoptic[seq_num][ov_th][class_name] = \
					copy.deepcopy(panoptic_quality[ov_th][class_name])
		return None
	def write_seq_quality(self, panoptic_quality, seq_num):
		_SPLITTER = ",  "
		panoptic_txt = os.path.join(self.seq_panoptic_quality_folder, seq_num+".txt")
		# save panoptic quality results
		with open(panoptic_txt, 'w') as f:
			f.write(_SPLITTER.join(['class'.ljust(12), 
					'PQ_50'.ljust(6),'SQ_50'.ljust(6),'RQ_50'.ljust(6),
					'PQ_75'.ljust(6),'SQ_75'.ljust(6),'RQ_75'.ljust(6) ]) + '\n')
			for class_name in self.class_names :

				panoptic_quality_strings = []
				for ov_th in [50,75]:
					if(class_name not in panoptic_quality[ov_th]):
						continue
					for item in ["PQ", "SQ", "RQ"]:
						value_string = None
						if item not in panoptic_quality[ov_th][class_name]:
							value_string = format(np.nan, '.6f').ljust(6)[:6]
						else:
							value_string = format(panoptic_quality[ov_th][class_name][item], '.6f').ljust(6)[:6]
						panoptic_quality_strings.append(value_string)
				output_strings = [str(class_name).ljust(12)]+panoptic_quality_strings
				f.write(_SPLITTER.join([str(x) for x in output_strings]) + '\n')  
	def write_seq_chamfer(self, chamfer, seq_num):
		_SPLITTER = ",  "
		chamfer_txt = os.path.join(self.seq_chamfer_folder, seq_num+".txt")
		# save chamfer distance results
		with open(chamfer_txt, 'w') as f:
			f.write(_SPLITTER.join(['class'.ljust(12),
					'pred_ov50'.ljust(8),'gt_ov50'.ljust(8),'pred_ov75'.ljust(8),'gt_ov75'.ljust(8) ]) + '\n')
			for class_name in self.class_names :
				pred_ov50 = format(chamfer[50]['pred2gt'][class_name]*1e3, '.10f').ljust(8)[:8]
				pred_ov75 = format(chamfer[75]['pred2gt'][class_name]*1e3, '.10f').ljust(8)[:8]
				gt_ov50 = format(chamfer[50]['gt2pred'][class_name]*1e3, '.10f').ljust(8)[:8]
				gt_ov75 = format(chamfer[75]['gt2pred'][class_name]*1e3, '.10f').ljust(8)[:8]

				f.write(_SPLITTER.join([str(x) for x in [str(class_name).ljust(12), 
					pred_ov50,gt_ov50 ,pred_ov75, gt_ov75]]) + '\n')  
	def write_results(self):
		_SPLITTER = ', '
		sequence_nums = list(self.seq_mAP.keys())
		sequence_nums.sort()
		class_names_per_class = self.class_names 
		# saved matched instance num
		# write sequence-wise result
		class_names = self.class_names 
		for ov_th in [50, 75]:
			seq_matched_num_txt = os.path.join(self.matched_num_folder, 'seq_matched_num_'+str(ov_th)+'.txt')
			with open(seq_matched_num_txt, 'w') as f:
				f.write(_SPLITTER.join(['seqs'.ljust(5)]+[str(class_name).ljust(8) for class_name in class_names_per_class]) + '\n')
				for seq_i, seq_num in enumerate(sequence_nums):
					seq_matched_instance_string = [str(seq_num).ljust(5)[:5]]
					for class_i, class_name in enumerate(class_names_per_class):
						value = self.seq_matched_instance_num[seq_num][ov_th][class_name]
						value_string = format(value).ljust(8)[:8]
						seq_matched_instance_string.append(value_string)
					f.write(_SPLITTER.join(seq_matched_instance_string) + '\n')

		# write total results
		matched_num_txt = os.path.join(self.matched_num_folder, 'matched_num.txt')
		with open(matched_num_txt, 'w') as f:
			head_string = ['class'.ljust(8), 'gt_count'.ljust(8),'ov_50'.ljust(6),'ov_75'.ljust(6) ]
			f.write(_SPLITTER.join(head_string) + '\n')
			for class_name in class_names_per_class:
				gt_count_str = format(self.gt_count[class_name]).ljust(8)[:8]
				matcthed_num_ov50 = format(self.matched_instance_num[50][class_name]).ljust(6)[:6]
				matcthed_num_ov75 = format(self.matched_instance_num[75][class_name]).ljust(6)[:6]
				value_string = [class_name.ljust(8), gt_count_str, matcthed_num_ov50, matcthed_num_ov75]
				f.write(_SPLITTER.join(value_string) + '\n') 
		matched_num_pkl = os.path.join(self.matched_num_folder, 'matched_num.pkl')
		with open(matched_num_pkl, 'wb') as f:
			pickle.dump(self.matched_instance_num, f)
		# save mAP results
		# write sequence-wise result
		seq_mAP_arrs = {}
		seq_mAP_arrs_pkl = os.path.join(self.mAP_folder, 'mAP_seq.pkl')
		for ap in ["ap50%", "ap"]:
			mAP_seq_ap_txt = os.path.join(self.mAP_folder, 'mAP_seq_'+str(ap)+'.txt')
			seq_mAP_ap_arr = np.zeros((len(sequence_nums), len(class_names_per_class)))
			with open(mAP_seq_ap_txt, 'w') as f:
				f.write(_SPLITTER.join(['seqs'.ljust(5)]+[str(class_name).ljust(8) for class_name in class_names_per_class]) + '\n')
				for seq_i, seq_num in enumerate(sequence_nums):
					seq_mAP_string = [str(seq_num).ljust(5)[:5]]
					for class_i, class_name in enumerate(class_names_per_class):
						value = self.seq_mAP[seq_num][class_name][ap]
						seq_mAP_ap_arr[seq_i,class_i] = value
						value_string = format(value, '.10').ljust(8)[:8]
						seq_mAP_string.append(value_string)
					f.write(_SPLITTER.join(seq_mAP_string) + '\n')
			seq_mAP_arrs[ap] = copy.deepcopy( seq_mAP_ap_arr) 
		with open(seq_mAP_arrs_pkl, 'wb') as f:
			pickle.dump(seq_mAP_arrs, f)	

		# averaged mAP
		total_75_per_class = np.nan
		total_50_per_class = np.nan
		total_75_per_instance = np.nan
		total_50_per_instance = np.nan
		count_class_total = 0
		count_instance_total = 0

		mAP_txt = os.path.join(self.mAP_folder, 'mAP.txt')
		with open(mAP_txt, 'w') as f:
			f.write(_SPLITTER.join(['class'.ljust(12), 'ap50'.ljust(8),'ap75'.ljust(8),'gt_count'.ljust(8)]) + '\n')
			class_names = list(self.mAP.keys())
			class_names.sort()
			for class_name in class_names:
				ap50 = self.mAP[class_name]["ap50%"]
				ap75 = self.mAP[class_name]["ap"]
				ap50_str = format(ap50, '.10f').ljust(8)[:8]
				ap75_str = format(ap75, '.10f').ljust(8)[:8]
				count = 0 if class_name not in self.class_weight_mAP else self.class_weight_mAP[class_name]
				count_str = format(count).ljust(8)[:8]				
				f.write(_SPLITTER.join([str(x) for x in [str(class_name).ljust(12), ap50_str, ap75_str, count_str]]) + '\n')   

				if class_name not in self.class_weight_mAP:
					continue
				# average map over all classes and instances
				weight_per_class = 1
				weight_per_instance = self.class_weight_mAP[class_name]
				if not np.isnan(ap50):
					total_50_per_class = ap50*weight_per_class if np.isnan(total_50_per_class) else (total_50_per_class+ap50*weight_per_class)
					total_50_per_instance = ap50*weight_per_instance if np.isnan(total_50_per_instance) else (total_50_per_instance+ap50*weight_per_instance)
				if not np.isnan(ap75):
					total_75_per_class = ap75*weight_per_class if np.isnan(total_75_per_class) else (total_75_per_class+ap75*weight_per_class)
					total_75_per_instance = ap75*weight_per_instance if np.isnan(total_75_per_instance) else (total_75_per_instance+ap75*weight_per_instance)
				count_class_total += weight_per_class
				count_instance_total += weight_per_instance
				breakpoint = None
			if count_class_total!=0:
				total_50_per_class = total_50_per_class/count_class_total
				total_75_per_class = total_75_per_class/count_class_total
			if count_instance_total!=0:
				total_50_per_instance = total_50_per_instance/count_instance_total
				total_75_per_instance = total_75_per_instance/count_instance_total
			# save average results
			av_per_class_ov50_str = format(total_50_per_class, '.10f').ljust(8)[:8]
			av_per_class_ov75_str = format(total_75_per_class, '.10f').ljust(8)[:8]
			av_per_instance_ov50_str = format(total_50_per_instance, '.10f').ljust(8)[:8]
			av_per_instance_ov75_str = format(total_75_per_instance, '.10f').ljust(8)[:8]
			class_count_str = format(count_class_total).ljust(8)[:8]
			inst_count_str = format(count_instance_total).ljust(8)[:8]
			
			av_str_per_class = [str(x) for x in ["ave_class".ljust(12), av_per_class_ov50_str, av_per_class_ov75_str, class_count_str]]
			av_str_per_instance = [str(x) for x in ["ave_instance".ljust(12), av_per_instance_ov50_str, av_per_instance_ov75_str, inst_count_str]]
			f.write(_SPLITTER.join(av_str_per_class) + '\n')  
			f.write(_SPLITTER.join(av_str_per_instance) + '\n')  
		self.mAP["ave_class"] = {"ap50%":total_50_per_class, "ap":total_75_per_class }
		self.mAP["ave_instance"] = {"ap50%":total_50_per_instance, "ap":total_75_per_instance }
		mAP_pkl = os.path.join(self.mAP_folder, 'mAP.pkl')
		with open(mAP_pkl, 'wb') as f:
			pickle.dump(self.mAP, f)	
				
		# save chamfer distance results
		# write sequence-wise result
		seq_chamfer_arrs = {}
		for ov in [50, 75]:
			seq_chamfer_arrs[ov] = {}
			for item in ['pred2gt', 'gt2pred']:
				seq_chamfer_arr = np.zeros((len(sequence_nums), len(class_names_per_class)))
				chamfer_seq_txt = os.path.join(self.chamfer_folder, 'chamfer_seq_'+str(ov)+"_"+str(item)+'.txt')
				with open(chamfer_seq_txt, 'w') as f:
					f.write(_SPLITTER.join(['seqs'.ljust(5)]+[str(class_name).ljust(8) for class_name in class_names_per_class]) + '\n')
					for seq_i, seq_num in enumerate(sequence_nums):
						seq_chamfer_string = [str(seq_num).ljust(5)[:5]]
						for class_i, class_name in enumerate(class_names_per_class):
							value = self.seq_chamfer[seq_num][ov][item][class_name]
							seq_chamfer_arr[seq_i,class_i] = value
							value_string = format(value*1e3, '.10').ljust(8)[:8]
							seq_chamfer_string.append(value_string)
						f.write(_SPLITTER.join(seq_chamfer_string) + '\n')
				seq_chamfer_arrs[ov][item] = copy.deepcopy(seq_chamfer_arr)
		chamfer_seq_pkl = os.path.join(self.chamfer_folder, 'chamfer_seq.pkl')
		with open(chamfer_seq_pkl, 'wb') as f:
			pickle.dump(seq_chamfer_arrs, f)	

		# averaged chamfer
		pred_ov50_per_class = np.nan
		pred_ov75_per_class = np.nan
		gt_ov50_per_class= np.nan
		gt_ov75_per_class = np.nan
		pred_ov50_per_inst = np.nan
		pred_ov75_per_inst = np.nan
		gt_ov50_per_inst = np.nan
		gt_ov75_per_inst = np.nan

		count_class_total_ov50 = 0
		count_class_total_ov75 = 0
		count_matched_ov50_total = 0
		count_matched_ov75_total = 0

		chamfer_txt = os.path.join(self.chamfer_folder, 'chamfer.txt')
		with open(chamfer_txt, 'w') as f:
			f.write(_SPLITTER.join(['class'.ljust(12), 
					'pred_ov50'.ljust(8),'gt_ov50'.ljust(8),'matched50'.ljust(10),\
						'pred_ov75'.ljust(8),'gt_ov75'.ljust(8),'matched75'.ljust(10) ]) + '\n')
			class_names = self.class_names 
			class_names.sort()
			for class_name in class_names:
				pred_ov50 = np.nan
				pred_ov75 = np.nan
				gt_ov50 = np.nan
				gt_ov75 = np.nan
				class_count= 1
				matched_count_ov50 = 0
				matched_count_ov75 = 0
				# for IOU threshold 50%
				if class_name in self.chamfer_distance[50]['pred2gt']:
					pred_ov50 = self.chamfer_distance[50]['pred2gt'][class_name]
					gt_ov50 = self.chamfer_distance[50]['gt2pred'][class_name]
					# average chamfer distance over all matched instances 
					matched_count_ov50 = self.class_weight_chamfer[50][class_name]		
					pred_ov50_per_inst = pred_ov50*matched_count_ov50 if np.isnan(pred_ov50_per_inst) \
						else (pred_ov50_per_inst+pred_ov50*matched_count_ov50)
					gt_ov50_per_inst = gt_ov50*matched_count_ov50 if np.isnan(gt_ov50_per_inst) \
						else (gt_ov50_per_inst+gt_ov50*matched_count_ov50)
					count_matched_ov50_total += matched_count_ov50
					# average chamfer distance over all classes 
					pred_ov50_per_class = pred_ov50*class_count if np.isnan(pred_ov50_per_class) \
						else (pred_ov50_per_class+pred_ov50*class_count)
					gt_ov50_per_class = gt_ov50*class_count if np.isnan(gt_ov50_per_class) \
						else (gt_ov50_per_class+gt_ov50*class_count)
					count_class_total_ov50 += class_count
				# for IOU threshold 75%
				if class_name in self.chamfer_distance[75]['pred2gt']:
					pred_ov75 = self.chamfer_distance[75]['pred2gt'][class_name]
					gt_ov75 = self.chamfer_distance[75]['gt2pred'][class_name]
					# average chamfer distance over all matched instances  
					matched_count_ov75 = self.class_weight_chamfer[75][class_name]
					pred_ov75_per_inst = pred_ov75*matched_count_ov75 if np.isnan(pred_ov75_per_inst) \
						else (pred_ov75_per_inst+pred_ov75*matched_count_ov75)
					gt_ov75_per_inst = gt_ov75*matched_count_ov75 if np.isnan(gt_ov75_per_inst) \
						else (gt_ov75_per_inst+gt_ov75*matched_count_ov75)
					count_matched_ov75_total += matched_count_ov75
					# average chamfer distance over all classes 
					pred_ov75_per_class = pred_ov75*class_count if np.isnan(pred_ov75_per_class) \
						else (pred_ov75_per_class+pred_ov75*class_count)
					gt_ov75_per_class = gt_ov75*class_count if np.isnan(gt_ov75_per_class) \
						else (gt_ov75_per_class+gt_ov75*class_count)
					count_class_total_ov75 += class_count
				pred_ov50_str = format(pred_ov50*1e3, '.10f').ljust(8)[:8] # save chamfer distance in mm
				gt_ov50_str = format(gt_ov50*1e3, '.10f').ljust(8)[:8]
				pred_ov75_str = format(pred_ov75*1e3, '.10f').ljust(8)[:8]
				gt_ov75_str = format(gt_ov75*1e3, '.10f').ljust(8)[:8]
				class_count_50_str = format(matched_count_ov50).ljust(10)[:10]
				class_count_75_str = format(matched_count_ov75).ljust(10)[:10]
				chamfer_class_str = [str(class_name).ljust(12), pred_ov50_str,gt_ov50_str,class_count_50_str \
						 ,pred_ov75_str, gt_ov75_str, class_count_75_str]
				f.write(_SPLITTER.join(chamfer_class_str) + '\n')  

			# normalize chamfer distance over all instances  
			if(count_matched_ov50_total!=0):
				pred_ov50_per_inst = pred_ov50_per_inst*1.0/count_matched_ov50_total
				gt_ov50_per_inst = gt_ov50_per_inst*1.0/count_matched_ov50_total
			if(count_matched_ov75_total!=0):
				pred_ov75_per_inst = pred_ov75_per_inst*1.0/count_matched_ov75_total
				gt_ov75_per_inst = gt_ov75_per_inst*1.0/count_matched_ov75_total
			# normalize chamfer distance over all classes 
			if(count_class_total_ov50!=0):
				pred_ov50_per_class = pred_ov50_per_class*1.0/count_class_total_ov50
				gt_ov50_per_class = gt_ov50_per_class*1.0/count_class_total_ov50
			if(count_class_total_ov75!=0):
				pred_ov75_per_class = pred_ov75_per_class*1.0/count_class_total_ov75
				gt_ov75_per_class = gt_ov75_per_class*1.0/count_class_total_ov75

			pred_ov50_per_inst_str = format(pred_ov50_per_inst*1e3, '.10f').ljust(8)[:8]
			gt_ov50_per_inst_str = format(gt_ov50_per_inst*1e3, '.10f').ljust(8)[:8]
			pred_ov75_per_inst_str = format(pred_ov75_per_inst*1e3, '.10f').ljust(8)[:8]
			gt_ov75_per_inst_str = format(gt_ov75_per_inst*1e3, '.10f').ljust(8)[:8]
			matched_ov50_total_str = format(count_matched_ov50_total).ljust(10)[:10]
			matched_ov75_total_str = format(count_matched_ov75_total).ljust(10)[:10]
			ave_per_inst_str = [str("ave_per_inst").ljust(12), pred_ov50_per_inst_str, gt_ov50_per_inst_str, \
				matched_ov50_total_str, pred_ov75_per_inst_str, gt_ov75_per_inst_str, matched_ov75_total_str]
			f.write(_SPLITTER.join(ave_per_inst_str) + '\n')  
			pred_ov50_per_class_str = format(pred_ov50_per_class*1e3, '.10f').ljust(8)[:8]
			gt_ov50_per_class_str = format(gt_ov50_per_class*1e3, '.10f').ljust(8)[:8]
			pred_ov75_per_class_str = format(pred_ov75_per_class*1e3, '.10f').ljust(8)[:8]
			gt_ov75_per_class_str = format(gt_ov75_per_class*1e3, '.10f').ljust(8)[:8]
			class_ov50_total_str = format(count_class_total_ov50).ljust(10)[:10]
			class_ov75_total_str = format(count_class_total_ov75).ljust(10)[:10]
			ave_per_class_str = [str("ave_per_class").ljust(12), pred_ov50_per_class_str, gt_ov50_per_class_str, \
				class_ov50_total_str, pred_ov75_per_class_str, gt_ov75_per_class_str, class_ov75_total_str]
			f.write(_SPLITTER.join(ave_per_class_str) + '\n')  
		chamfer_pkl = os.path.join(self.chamfer_folder, 'chamfer.pkl')
		self.chamfer_distance[50]['pred2gt']['ave_instance'] = pred_ov50_per_inst
		self.chamfer_distance[50]['gt2pred']['ave_instance'] = gt_ov50_per_inst
		self.chamfer_distance[75]['pred2gt']['ave_instance'] = pred_ov75_per_inst
		self.chamfer_distance[75]['gt2pred']['ave_instance'] = gt_ov75_per_inst
		self.chamfer_distance[50]['pred2gt']['ave_class'] = pred_ov50_per_class
		self.chamfer_distance[50]['gt2pred']['ave_class'] = gt_ov50_per_class
		self.chamfer_distance[75]['pred2gt']['ave_class'] = pred_ov75_per_class
		self.chamfer_distance[75]['gt2pred']['ave_class'] = gt_ov75_per_class
		with open(chamfer_pkl, 'wb') as f:
			pickle.dump(self.chamfer_distance, f)

		# save panoptic segmentation 
		# write sequence-wise result
		seq_panoptic_arrs = {}
		seq_panoptic_arrs_pkl = os.path.join(self.panoptic_quality_folder, "panoptic_seq.pkl")
		for ov in [50, 75]:
			seq_panoptic_arrs[ov] = {}
			for  item in ["PQ", "SQ", "RQ"]:
				seq_panoptic_txt = os.path.join(self.panoptic_quality_folder, \
					"panoptic_seq_"+ str(ov)+"_"+ str(item)+".txt")
				seq_panoptic_arr = np.zeros((len(sequence_nums), len(class_names_per_class)))
				with open(seq_panoptic_txt, 'w') as f:
					f.write(_SPLITTER.join(['seqs'.ljust(5)]+[str(class_name).ljust(8) for class_name in class_names_per_class]) + '\n')
					for seq_i, seq_num in enumerate(sequence_nums):
						
						seq_panoptic_string = [str(seq_num).ljust(5)[:5]]
						for class_i, class_name in enumerate(class_names_per_class):
							value = self.seq_panoptic[seq_num][ov][class_name][item]
							seq_panoptic_arr[seq_i, class_i]
							value_string = format(value, '.10').ljust(8)[:8]
							seq_panoptic_string.append(value_string)
						f.write(_SPLITTER.join(seq_panoptic_string) + '\n')
				seq_panoptic_arrs[ov][item] = copy.deepcopy(seq_panoptic_arr)
		with open(seq_panoptic_arrs_pkl, 'wb') as f:
			pickle.dump(seq_panoptic_arrs, f)

		# averaged panoptic quality
		panoptic_ave_per_class = {50:{"PQ":np.nan, "SQ":np.nan, "RQ":np.nan}, 75:{"PQ":np.nan, "SQ":np.nan, "RQ":np.nan}}
		PQ_count_per_class = {50:{"PQ":0, "SQ":0, "RQ":0}, 75:{"PQ":0, "SQ":0, "RQ":0}}
		panoptic_ave_per_inst = {50:{"PQ":np.nan, "SQ":np.nan, "RQ":np.nan}, 75:{"PQ":np.nan, "SQ":np.nan, "RQ":np.nan}}
		PQ_count_per_inst = {50:{"PQ":0, "SQ":0, "RQ":0}, 75:{"PQ":0, "SQ":0, "RQ":0}}

		panoptic_txt = os.path.join(self.panoptic_quality_folder, 'panoptic_quality.txt')
		with open(panoptic_txt, 'w') as f:
			f.write(_SPLITTER.join(['class'.ljust(12), 
					'PQ_50'.ljust(6),'SQ_50'.ljust(6),'RQ_50'.ljust(6),'PQcount'.ljust(8),
					'PQ_75'.ljust(6),'SQ_75'.ljust(6),'RQ_75'.ljust(6),'PQcount'.ljust(8), ]) + '\n')
			for class_name in class_names_per_class:
				panoptic_quality_strings = []
				for ov_th in [50,75]:
					PQ_class_weight = 1
					PQ_inst_weight = 0
					PQ_weight = 0
					for item in ["PQ", "SQ", "RQ"]:
						value_string = None
						if item not in self.panoptic_weight[ov_th][class_name]:
							value_string = format(np.nan, '.6f').ljust(6)[:6]
						else:
							value = self.panoptic_quality[ov_th][class_name][item]
							value_string = format(value, '.6f').ljust(6)[:6]
							# weight = self.panoptic_weight[ov_th][class_name][item]
							PQ_inst_weight = self.panoptic_weight[ov_th][class_name][item]
							if item == "PQ":
								PQ_weight = PQ_inst_weight
							if not np.isnan(value): 
								# average panoptic quality over all instance
								panoptic_ave_per_inst[ov_th][item] = value*PQ_inst_weight if np.isnan(panoptic_ave_per_inst[ov_th][item]) \
									else (panoptic_ave_per_inst[ov_th][item]+value*PQ_inst_weight)
								PQ_count_per_inst[ov_th][item] += PQ_inst_weight								
								# average panoptic quality over all classes
								panoptic_ave_per_class[ov_th][item] = value*PQ_class_weight if np.isnan(panoptic_ave_per_class[ov_th][item]) \
									else (panoptic_ave_per_class[ov_th][item]+value*PQ_class_weight)
								PQ_count_per_class[ov_th][item] += PQ_class_weight
						panoptic_quality_strings.append(value_string)
					PQ_weight_string =  format(PQ_weight).ljust(8)[:8]
					panoptic_quality_strings.append(PQ_weight_string)
				output_strings = [str(class_name).ljust(12)]+panoptic_quality_strings
				f.write(_SPLITTER.join(output_strings) + '\n')  
			ave_PQ_inst_strings = [str("ave_instance").ljust(12)]
			ave_PQ_class_strings = [str("ave_class").ljust(12)]
			for ov_th in [50,75]:
				for item in ["PQ", "SQ", "RQ"]:
					# average over instance
					value = panoptic_ave_per_inst[ov_th][item]
					if PQ_count_per_inst[ov_th][item]!=0:
						panoptic_ave_per_inst[ov_th][item] = value/PQ_count_per_inst[ov_th][item]
						value_string = format(panoptic_ave_per_inst[ov_th][item], '.6f').ljust(6)[:6]
						ave_PQ_inst_strings.append(value_string)
					else:
						value_string = format(np.nan, '.6f').ljust(6)[:6]
						ave_PQ_inst_strings.append(value_string)
					# average over class
					value = panoptic_ave_per_class[ov_th][item]
					if PQ_count_per_class[ov_th][item]!=0:
						panoptic_ave_per_class[ov_th][item] = value/PQ_count_per_class[ov_th][item]
						value_string = format(panoptic_ave_per_class[ov_th][item], '.6f').ljust(6)[:6]
						ave_PQ_class_strings.append(value_string)
					else:
						value_string = format(np.nan, '.6f').ljust(6)[:6]
						ave_PQ_class_strings.append(value_string)
				PQ_weight_inst = PQ_count_per_inst[ov_th]["PQ"]
				PQ_weight_inst_string =  format(PQ_weight_inst).ljust(8)[:8]
				ave_PQ_inst_strings.append(PQ_weight_inst_string)
				PQ_weight_class = PQ_count_per_class[ov_th]["PQ"]
				PQ_weight_class_string =  format(PQ_weight_class).ljust(8)[:8]
				ave_PQ_class_strings.append(PQ_weight_class_string)
				self.panoptic_quality[ov_th]["ave_instance"] = copy.deepcopy(panoptic_ave_per_inst[ov_th]) 
				self.panoptic_quality[ov_th]["ave_class"] = copy.deepcopy(panoptic_ave_per_class[ov_th]) 
			f.write(_SPLITTER.join(ave_PQ_inst_strings) + '\n')  
			f.write(_SPLITTER.join(ave_PQ_class_strings) + '\n')  
		panoptic_pkl = os.path.join(self.panoptic_quality_folder, 'panoptic_quality.pkl')
		with open(panoptic_pkl, 'wb') as f:
			pickle.dump(self.panoptic_quality, f)

		test_panoptic = None
		with open(panoptic_pkl, 'rb') as f:
			test_panoptic = pickle.load(f)
		return None

def evaluateBatch(MetricFolder, ResultsSeqsFolder, Seqs_nums, MethodName ):
	ResultsFolders = [os.path.join(ResultsSeqsFolder, scene_num, MethodName) for scene_num in Seqs_nums]
	EvaluationFolders = [ os.path.join(result_folder, "geo_sem_eval") for result_folder in ResultsFolders]

	OutPutFolder = os.path.join(MetricFolder,MethodName)
	ave_metric = SemMetric(OutPutFolder)

	import pickle

	for seq_i, evaluation_folder in enumerate(EvaluationFolders):
		seq_num = Seqs_nums[seq_i]
		if seq_num == '231':
			continue
		gt_class_counts = None
		gt_class_counts_pkl = os.path.join(evaluation_folder, "gt_class_counts.pkl")
		with open(gt_class_counts_pkl, 'rb') as f:
			gt_class_counts = pickle.load(f)

		matched_instance_num = None
		matched_instance_num_pkl = os.path.join(evaluation_folder, "matched_instance_num.pkl")
		with open(matched_instance_num_pkl, 'rb') as f:
			matched_instance_num = pickle.load(f)
		ave_metric.insert_matchedNum(matched_instance_num, gt_class_counts, seq_num)
		ave_metric.write_seq_matchedNum(gt_class_counts, seq_num)

		chamfer_distance_class = None
		chamfer_pkl = os.path.join(evaluation_folder, "chamfer.pkl")
		with open(chamfer_pkl, 'rb') as f:
			chamfer_distance_class = pickle.load(f)
		ave_metric.insert_chamfer(chamfer_distance_class, gt_class_counts, seq_num)
		ave_metric.write_seq_chamfer(chamfer_distance_class, Seqs_nums[seq_i])

		mAP = None
		mAP_pkl = os.path.join(evaluation_folder, "mAP.pkl")
		with open(mAP_pkl, 'rb') as f:
			mAP = pickle.load(f)
		ave_metric.insert_mAP(mAP, gt_class_counts ,seq_num)
		ave_metric.write_seq_mAP(mAP,Seqs_nums[seq_i])

		panoptic_quality = None
		panoptic_quality_pkl = os.path.join(evaluation_folder, "panoptic_quality.pkl")
		with open(panoptic_quality_pkl, 'rb') as f:
			panoptic_quality = pickle.load(f)
		ave_metric.insert_panoptic(panoptic_quality, gt_class_counts, seq_num)
		ave_metric.write_seq_quality(panoptic_quality, Seqs_nums[seq_i])	
		breakpoint = None
	ave_metric.write_results()

MetricFolder = "/home/yang/toolbox/evaluation/MetricNumbers/Seqs74/results"
ResultsSeqsFolder = "/home/yang/big_ssd/results"
# ResultsSeqsFolder = "/home/yang/990Pro/results_estimated_pose"
Seqs_nums = os.listdir(ResultsSeqsFolder)
Seqs_nums.sort()

# MethodNames = [
# 	"Estimate/Sem2Confidence4Fast", "Estimate/FinalVersion"
# ]
MethodNames = [
	"V_25SegGraph/Replicate"
]
Task = "CoCoPano"
UseSemantic = 1

# MethodNames = [
# 	"2D_confidence", \
# 	"Pano" , \
# 	"PanoSemInstCount",\
# 	"PanoSem2LSISegGraph", \
# 	"V_SegGraph/Sem0C4FastB2_Break015_0.5", \
# 	"V_SegGraph/Sem0C4_Break115_0.5", \
# 	"V_25SegGraph/Sem2C4_Break_0_15_0.5", \
# 	"V_25SegGraph/Sem2C4_OnlyBreak", \
# 	"V_SegGraph/Sem0C4FastB2_Break115_0.5", \
# 	"V_25SegGraph/Sem2Confidence4FastTest/Test_15.0_0.5"]
# for MethodName in MethodNames:	
# 	evaluateBatch(MetricFolder, ResultsSeqsFolder, Seqs_nums, MethodName)

# MethodNames = []
# K_inst_list = [0.1, 1.0, 10]
# Break_th_inst_list = [0.05, 0.1, 0.2, 0.3]
# Break_th_label_list = [0.4, 0.5, 0.6, 0.7]
# for K_inst in K_inst_list:
# 	for Break_th_inst in Break_th_inst_list:
# 		for Break_th_label in Break_th_label_list:	
# 			params_params = "_".join([str(K_inst), str(Break_th_inst), str(Break_th_label)])
# 			MethodName = "InstGraphCutTest/Test_" + str(K_inst)+str(Break_th_inst)+str(Break_th_label)
# 			MethodNames.append(MethodName)
# for MethodName in MethodNames:	
# 	evaluateBatch(MetricFolder, ResultsSeqsFolder, Seqs_nums, MethodName)

# MethodNames = [
# 	"V_25SegGraph/Sem2Confidence4Fast"
# ]

for MethodName in MethodNames:	
	evaluateBatch(MetricFolder, ResultsSeqsFolder, Seqs_nums, MethodName)
import pandas as pd 	 
import numpy as np 		 	
import statistics as s
import matplotlib.pyplot as plt 	#Data Visualization

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

def Describe_Data(dataset,datasetx):
	print(datasetx.describe())
	print()
	print(dataset['Name or Symbol'].describe())
	
def Merge(dataset):
	dataset['Name or Symbol'] = dataset['Synonym Symbol'].astype(str)+ " " + dataset['National Common Name'].astype(str)
	dataset['Name or Symbol'] = dataset["Name or Symbol"].str.strip('nan')
	dataset.drop(dataset.columns[[1, 3]], inplace = True, axis = 1)

	dataset['Name or Symbol'] = dataset['Name or Symbol'].str.strip()

	dataset['Family_1'] = dataset['Family']
	dataset.drop(dataset.columns[[2]], inplace=True, axis = 1)
	dataset['Family'] = dataset['Family_1']
	dataset.drop(dataset.columns[[3]], inplace=True,axis = 1)

	return dataset

def label_encode(dataset):

	#Converting Categorical Data into Numerical Values 

	dataset['Symbol_code'] = dataset["Symbol"].astype("category").cat.codes
	dataset['Family_code'] = dataset["Family"].astype("category").cat.codes

	return dataset

def Plot_Fig(dataset):

	fig = plt.figure()
	# X= dataset.iloc[:,:].values
	plt.hist(dataset["Family"], bins=200)
	# plt.xlabel("Family")
	plt.show()
	
####### FIMs ################	

def hot_encode(x): 
	if(x == '0'): 
		return 0
	else: 
		return 1

def TransactionEncoder(dataset):

	Dataset_Encoded = []
	ListDicts = dataset.T.to_dict().values() # Convert Dataframe to list of dicts

	UniqueValues, UniqueValKeys = GetUniqueValues(dataset)

	for data in ListDicts:
		EncodedData = []
		for val, key in zip(UniqueValues, UniqueValKeys):
			if val == data[key]:
				EncodedData.append(True)
			else:
				EncodedData.append(False)
		Dataset_Encoded.append(EncodedData)
	
	Dataset_Encoded = pd.DataFrame(Dataset_Encoded, columns=UniqueValues)

	return Dataset_Encoded

def GetUniqueValues(dataset):
	UniqueValues = []
	UniqueValKeys = []
	for key in dataset.keys():
		uniquevals = dataset[key].unique()
		for uval in uniquevals:
			if uval not in UniqueValues:
				UniqueValues.append(uval)
				UniqueValKeys.append(key)
	return UniqueValues, UniqueValKeys


def Apriori_FPGROWTH(dataset):

	uniqueFam = dataset.Family.unique()

	for i in range(len(uniqueFam)):		#Pruning Data to Reduce Lagging in Mining as one go.. Divided Data based on Family
		prunedata = (dataset[dataset['Family'] == uniqueFam[i]].groupby(['Symbol', 'Scientific Name with Author'])['Name or Symbol'].sum().unstack().reset_index().fillna('0') .set_index('Symbol')) 
		pruneencode = prunedata.applymap(hot_encode)
		prunedata = pruneencode


		print('\n-------------------------------------------------------------------------------------------------')
		print("FOR FAMILY: ",uniqueFam[i])

		frq_items = Apriori(prunedata)
		print(frq_items)

		print("RULES: -----------------------\n")
		rules = Rules_Generation(prunedata, frq_items)
		print('\n\n')

		frq_items = FP_Growth(prunedata)
		print(frq_items)

		print("RULES: -----------------------\n")
		rules = Rules_Generation(prunedata, frq_items)
		print('\n')

def OtherAlgos(dataset):
	uniqueFam = dataset.Family.unique()

	for i in range(len(uniqueFam)):
		prunedata = (dataset[dataset['Family'] == uniqueFam[i]])
		EncodedData = TransactionEncoder(prunedata)
		prunedata = EncodedData

		print('\n-------------------------------------------------------------------------------------------------')
		print("FOR FAMILY: ",uniqueFam[i])

		frq_items = AClose(prunedata)
		print(frq_items)

		frq_items = CHARM(prunedata)
		print(frq_items)
		
		frq_items = MAFIA(prunedata)
		print(frq_items)
			
		frq_items = PincerSearch(prunedata)
		print(frq_items)

		frq_items = LApriori(prunedata)
		print(frq_items)

		frq_items = LFPGrowth(prunedata)
		print(frq_items)



def Charm_Property(nodes, new_nodes, X, item, c_item, min_support):		#For Charm Algorithm
	if len(X[1]) >= min_support:
		if item[1] == c_item[1]:
			nodes.remove(c_item)
			for i in range(len(nodes)):
				if nodes[i] == item:
					nodes[i] == X
		elif item[1].issubset(c_item[1]):
			for i in range(len(nodes)):
				if nodes[i] == item:
					nodes[i] == X
		elif c_item[1].issubset(item[1]):
			nodes.remove(c_item)
			new_nodes.append(X)
		elif item[1] != c_item[1]:
			new_nodes.append(X)

def Charm_Extend(nodes, CFI, min_support):
	for item in nodes:
		new_nodes = []
		X = item
		for c_item in nodes:
			if len(item[1]) > len(c_item[1]):
				X = [item[0].union(c_item[0]), item[1].intersection(c_item[1])]
				Charm_Property(nodes, new_nodes, X, item, c_item, min_support)

		if new_nodes != []:
			new_nodes.sort(key = lambda x:len(x[1]))
			Charm_Extend(new_nodes, CFI, min_support)
		if X not in CFI:
			CFI.append(X)
	return(CFI)


def SelfJoin(Set, sizelimit=None):			#For AClose and Pincer Search
	JoinedSet = []
	JoinedSetDicts = []


	for i in range(len(Set)):
		for j in range(i+1, len(Set)):
			val = {}
			for x, y in zip(Set[i], Set[j]):
				val[x] = True
				val[y] = True
			if sizelimit == None or sizelimit >= len(val.keys()):
				if val not in JoinedSetDicts:
					JoinedSetDicts.append(val)
					JoinedSet.append(list(val.keys()))
			
	return JoinedSet

def genMFCS(MFCS, Items_Si):				#For Pincer Search Algorithm
	MFCS = MFCS.copy()

	for inf_items in Items_Si:
		for MFCS_item in MFCS.copy():
			
			#if infrequent is subset of MFCS
			if all(s_item in MFCS_item for s_item in inf_items):
				MFCS.remove(MFCS_item)

				for item in inf_items:
					updateMFCS_item = MFCS_item.copy()
					updateMFCS_item.remove(item)

					if not any(all(s_item in Rem_MFCS for s_item in updateMFCS_item) for Rem_MFCS in MFCS):
						MFCS.append(updateMFCS_item)
	return MFCS

class MafiaTreeNode:						#For Mafia Algorithm
	def __init__(self, head, tail, supportCount=None):
		self.head = head
		self.tail = tail.copy()
		self.supportCount = supportCount

def compVertBitmap(itemset, bitMap):
	if len(itemset) == 1:
		item = str(itemset[0])
		return bitMap[item]

	else:
		last_item = str(itemset[-1])
		return compVertBitmap(itemset[:-1], bitMap) & bitMap[last_item]


def countSupp(itemset, bitMap):
	
	itemset_map = compVertBitmap(itemset, bitMap)
	itemset_supp_count = np.count_nonzero(itemset_map)

	return itemset_supp_count

def MafiaRun(currentMFNode, MFI, bitMap, min_support):
	
	#Head Union Tail Pruning (HUT)------>
	HUT = currentMFNode.head + tuple(currentMFNode.tail)
	# HUT = currentMFNode.head.append(currentMFNode.tail)

	#If HUT is in MFI -> Stop Searching nd return
	if any(all(item in mfi for item in HUT) for mfi in MFI):
		return
	
	#Count Support of all children
	nodeChild_supportCount = [(item, countSupp(currentMFNode.head + (item,), bitMap) ) for item in currentMFNode.tail]

	#Extract frequent Children of node and support count
	nodeFreqChildCount = [(item, support_count) for item, support_count in nodeChild_supportCount if support_count >= min_support]

	node_childEqualParent = []	# items in tail with support count equal to that of parent
	node_tail_suppCount = []	# items in node tail sorted by Decreasing Support

	for item, support_count in nodeFreqChildCount:
		if support_count == currentMFNode.supportCount:
			node_childEqualParent.append(item)
		else:
			node_tail_suppCount.append((item, support_count))

	#Sort items in the trimmed tail by increasing support:
	node_tail_suppCount.sort(key=lambda x:x[1])
	node_tail_items = [item for item, support in node_tail_suppCount]

	currentMFNode.head += tuple(node_childEqualParent)
	currentMFNode.tail = node_tail_items

	is_leaf = not bool(currentMFNode.tail)

	for i, item in enumerate(currentMFNode.tail):
		new_node_head = currentMFNode.head + (item,)
		new_node_tail = currentMFNode.tail[i+1:]
		new_node_supportCount = node_tail_suppCount[i][1]

		new_node = MafiaTreeNode(new_node_head, new_node_tail, new_node_supportCount)

		MafiaRun(new_node, MFI, bitMap, min_support)

	if is_leaf and currentMFNode.head and not any(all(item in mfi for item in currentMFNode.head) for mfi in MFI):
		MFI.append(set(currentMFNode.head))


def Apriori(PrunedData):	
	
	print("Minimum Support = 25%")
	frq_items = apriori(PrunedData, min_support = 0.25, use_colnames = True)

	return frq_items
	

def FP_Growth(PrunedData):		

	print("Minimum Support = 25%")
	frq_items = fpgrowth(PrunedData, min_support = 0.25, use_colnames = True)

	return frq_items

def AClose(PrunedData):

	min_support = 0.25

	min_support = min_support * len(PrunedData.index)

	print("Minimum Support = ", min_support)
	min_itemset_length = 1

	CFI = []
	DataList = PrunedData.T.to_dict().values() 

	# Count Items
	C1_Item = PrunedData.keys()
	C1_Count = []
	for item in C1_Item:
		C1_Count.append(sum(PrunedData[item]))
	
	# Prune Level 1
	L1_Item = []
	L1_Count = []
	for item, support in zip(C1_Item, C1_Count):
		if support >= min_support:
			L1_Item.append([item])
			L1_Count.append(support)

	# Keep Pruning Till Empty
	Lk_Item = L1_Item
	Lk_Count = L1_Count
	k = 1
	while(len(Lk_Item) > 0):
		k += 1

		# Add previous Values to CFI if superset and remove subsets
		currentCFI = CFI.copy()
		for item in Lk_Item:
			for cfi in CFI:
				if set(cfi).issubset(set(item)): # Check if subset
					if cfi in currentCFI:
						currentCFI.remove(cfi)
			if min_itemset_length <= len(item):
				currentCFI.append(item)
				
		CFI = currentCFI

		# Self-Join
		Ck_Item = SelfJoin(Lk_Item, sizelimit=k)

		# Count Supports of Ck_Item
		Ck_Count = [0] * len(Ck_Item)
		for data in DataList:
			for i in range(len(Ck_Item)):
				isItemPresent = True
				for val in Ck_Item[i]:
					if data[val] == False:
						isItemPresent = False
						break
				if isItemPresent:
					Ck_Count[i] += 1
				
		
		# Frequent Itemset of Level k
		Lk_Item = []
		for item, count in zip(Ck_Item, Ck_Count):
			if count >= min_support:
				Lk_Item.append(item)

		# print(CFI)
	
	return CFI


def CHARM(PrunedData):
	min_support = 0.25

	min_support = min_support * len(PrunedData.index)
	print("Minimum Support = ", min_support)

	DataList = PrunedData.T.to_dict().values() # Convert Dataframe to list of dicts
	Items = PrunedData.keys()

	Data = PrunedData.values.tolist()

	nodes =[]
	CFI = []
	t = set()
	for i in range(len(Items)):
		t = set()
		for j in range(len(Data)):
			if Data[j][i] == 1:
				t = t.union({j})
		if len(t)>= min_support:
			nodes.append([{Items[i]},t])
	nodes.sort(key = lambda x:len(x[1]))

	Charm_Extend(nodes, CFI, min_support)

	return CFI

def MAFIA(PrunedData):
	min_support = 0.25

	min_support = min_support * len(PrunedData.index)

	print("Minimum Support = ", min_support)

	MFI = []
	Items = PrunedData.keys()
	Itemlist = []
	for item in Items:
		Itemlist.append(item)

	items_vertical_bitmaps = {item:np.array(PrunedData[item]) for item in Items}

	root = tuple()
	MFRoot = MafiaTreeNode(root, Itemlist)	#Creates a root Node 

	MafiaRun(MFRoot, MFI, items_vertical_bitmaps, min_support)

	# print(MFI)
	return MFI

def PincerSearch(PrunedData):
	
	min_support = 0.25

	min_support = min_support * len(PrunedData.index)

	print("Minimum Support = ", min_support)

	DataList = PrunedData.T.to_dict().values() # Convert Dataframe to list of dicts
	Items = PrunedData.keys()

	Ck_Count = []

	MFS = []
	MFCS = [[item] for item in Items]

	Lk_Item = []
	Sk_Item = []
	k = 1
	
	Ck_Item = [[item] for item in Items]

	while(len(Ck_Item) > 0):
		# Count Supports of Ck_Item and MFCS
		Ck_Count = [0] * len(Ck_Item)
		for data in DataList:
			for i in range(len(Ck_Item)):
				isItemPresent = True
				for val in Ck_Item[i]:
					if data[val] == False:
						isItemPresent = False
						break
				if isItemPresent == True:
					Ck_Count[i] += 1
		
		MFCSCount = [0] * len(MFCS)
		for data in DataList:
			for i in range(len(MFCS)):
				isItemPresent = True
				for val in MFCS[i]:
					if data[val] == False:
						isItemPresent = False
						break
				if isItemPresent == True:
					MFCSCount[i] += 1

		#Update MFS: MFS U {freqitems in MFCS}
		for itemset, support in zip(MFCS, MFCSCount):
			if ((support >= min_support) and (itemset not in MFS)):
				MFS.extend(itemset)

		# Infrequent sets
		Lk_Item = []
		Sk_Item = []
		for item, count in zip(Ck_Item, Ck_Count):
			if count >= min_support:
				Lk_Item.append(item)
			else:
				Sk_Item.append(item)
		#update MFCS
		MFCS = genMFCS(MFCS, Sk_Item)
		
		# Prune LK that are subsets of MFS
		Lk = Lk_Item.copy()
		for item in Lk_Item.copy():
			if any(all(l_item in MFSitem for l_item in item) for MFSitem in MFS):
				Lk.remove(item)
		Lk_Item = Lk

		# Self-Join
		Ck_Item = SelfJoin(Lk_Item, sizelimit=k)

		# #Recover:
		Ck = Ck_Item.copy()
		Lk = Lk_Item.copy()

		for item in Lk_Item.copy():
			if any(all(l_item in m_item for l_item in item) for m_item in MFS):
				for i in range(k, len(m_item)):
					Ck.append(item)

		Ck_Item = Ck

		#Prune Ck+1 that are not in MFCS
		Ck = Ck_Item.copy()
		for item in Ck_Item.copy():
			if not any(all(c_item in MFCSitem for c_item in item) for MFCSitem in MFCS):
				Ck.remove(item)
		Ck_Item = Ck

		k += 1
	# print(MFS)
	
	return MFS

def LApriori(PrunedData):

	print("Minimum Support = 25%")
	frq_items = apriori(PrunedData, min_support = 0.25, use_colnames = True)

	FI = []
	for itemset in frq_items['itemsets']:
		FI.append(list(itemset))
	maxlen = 0
	for fi in FI:
		if len(fi) == maxlen:
			LFI.append(fi)
		elif len(fi) > maxlen:
			LFI = []
			LFI.append(fi)
			maxlen = len(fi)

	return LFI

def LFPGrowth(PrunedData):

	print("Minimum Support = 25%")
	frq_items = fpgrowth(PrunedData, min_support = 0.25, use_colnames = True)

	FI = []
	for itemset in frq_items['itemsets']:
		FI.append(list(itemset))
	maxlen = 0
	for fi in FI:
		if len(fi) == maxlen:
			LFI.append(fi)
		elif len(fi) > maxlen:
			LFI = []
			LFI.append(fi)
			maxlen = len(fi)

	return LFI


#RULES GENERATION AND MEASURES__________________________________

def Rules_Generation(dataset, frq_items):
	rules = []
	Association_Rules = []
	if(len(frq_items) != 0 and len(frq_items) < 100):		#Choosen 100 as max limit mainly to prevent lagging in System
		rules = association_rules(frq_items, metric ="confidence", min_threshold = 1) 
		assn_rule = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
		# print(rules)
		
		antecedent = assn_rule['antecedents']
		consequent = assn_rule['consequents']
		confidence = assn_rule['confidence']
		lift = assn_rule['lift']
		conviction = assn_rule['conviction']
		leverage = assn_rule['leverage']


		antecedent = [list(x) for x in antecedent]
		consequent = [list(x) for x in consequent]
		confidence = [str(x) for x in confidence]
		lift = [str(x) for x in lift]
		conviction = [str(x) for x in conviction]
		leverage = [str(x) for x in leverage]

		# print(support)
		for i in range(len(antecedent)):
			print("{} : {} => {}" .format(i, antecedent[i], consequent[i]))
			print("Confidence = {}, Lift = {}, Conviction = {}, leverage = {}".format(confidence[i], lift[i], conviction[i], leverage[i]))
			Association_Rules.append((antecedent[i], consequent[i]))

	
	return Association_Rules



###############################
def DriverFunction(dataset_file):
	dataset_en = pd.read_csv(dataset_file)
	dataset = pd.read_csv(dataset_file)

	#PreProcess:
	dataset = Merge(dataset)		#for removing null (merge 2 columns)
	dataset_en = label_encode(dataset_en)	#for Visualizing
	Describe_Data(dataset, dataset_en)

	#Visualize:
	Mode = s.mode(dataset_en['Family'])
	print("Mode of Family: ", Mode)
	Plot_Fig(dataset_en)					#Bar Plots 

	#Mining:
	Apriori_FPGROWTH(dataset)		#FIM
	OtherAlgos(dataset)				#CFI, MFI, LFI and Rules Generation

datasetfile = 'stateDownload.csv'
DriverFunction(datasetfile)
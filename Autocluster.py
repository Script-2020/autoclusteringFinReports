from math import ceil
import multiprocessing as mp
import ray
import gc
import numpy as np
import os
import pickle
import tqdm
import time
import copy 

from util.NLP import NLP
from util.FileManager import FileManager
from util.Logger import Logger
from util.ChartDrawer import ChartDrawer

from nlpde import FDExt
from collections import Counter

@ray.remote
def ray_remote_processAutoclusteringBatch(document_info):
    return processAutoclusteringBatch(document_info)

def processAutoclusteringBatch(document_info):
    document_list = document_info['dataset']
    batch_number = document_info['batch_number']
    thresholdSimilarity = document_info['thresholdSimilarity']  
    full_computation = document_info['full_computation']
    dicClusters = document_info['dicClusters']
    features_weights = document_info['features_weights']

    if dicClusters is None: dicClusters = {}
    oLogger = Logger()
    oLogger.printLog('Start procesing batch', batch_number,  'with', len(document_list))

    dicDocuments = {}
    for i, document in enumerate(document_list):
        dicClusters, dicDocuments = setDocumentInCluster(document, dicClusters, dicDocuments, thresholdSimilarity, full_computation=full_computation, features_weights=features_weights)

    oLogger.getElapsedTime()
    oLogger.printLog('Finished batch', batch_number, 'with', len(document_list),  'documents and', len(dicClusters), 'clusters')

    return {'clusters': dicClusters, 'documents': dicDocuments}

@ray.remote
def ray_remote_getAdjacencyMatrixOfClusters(infoset, filter_out=True):
    return getAdjacencyMatrixOfClusters(infoset, filter_out)

def getAdjacencyMatrixOfClusters(infoset, filter_out=True):
    dicClusters = infoset['dicClusters']
    merging_threshold = infoset['merging_threshold']
    short_path = infoset['short_path']
    features_weights = infoset['features_weights']
    sensitivity_increase = 0.05

    indexes_dic = [item for item in dicClusters]
    max_index = max(len(dicClusters), max(indexes_dic)+1)
    min_index = min(indexes_dic)
    total_clusters = max_index - min_index

    size = (total_clusters, total_clusters)
    adjMatrix = np.zeros(size)
    listClusters = [dicClusters[i+min_index] for i in range(max_index) if dicClusters.get(i+min_index) is not None]
    for iClusterA in range(0, len(listClusters)):
        clusterA = listClusters[iClusterA]
        if clusterA is not None:
            for iClusterB in range(iClusterA+1, len(listClusters)):
                clusterB = listClusters[iClusterB]
                if clusterB is not None:
                    no_featuresA = len (clusterA['features'])
                    no_featuresB = len (clusterB['features'])
                    if short_path and min(no_featuresA, no_featuresB) / max(no_featuresA,no_featuresB) < merging_threshold:
                        distance = 0
                    else:
                        merging_threshold_upd = merging_threshold
                        min_no_feats = min(no_featuresA, no_featuresB)
                        if min_no_feats < 10:
                            merging_threshold_upd *= (1+sensitivity_increase)
                        distance, coincidentFeatures = getSimilarityScore(clusterA['features'], clusterB['features'], features_weights, level=None)
                        if distance < merging_threshold_upd and filter_out:
                            distance = 0
                    adjMatrix[clusterA['cluster_id']-min_index][clusterB['cluster_id']-min_index] = distance
                    adjMatrix[clusterB['cluster_id']-min_index][clusterA['cluster_id']-min_index] = distance

    return adjMatrix, min_index

@ray.remote
def ray_remote_get_working_dataset(dataDirectory, fileName, file_limit, onlyLastPresentedDocument):
    return ray_remote_get_working_dataset(dataDirectory, fileName, file_limit, onlyLastPresentedDocument)

def getSoftValueSimilarity(featuresA, featuresB, feature):
    new_features = {}
    b_candidates = {item:1 for item in featuresB if feature.split("_")[0] in item and abs(featuresA[feature]['value']- featuresB[item]['value'])<=5}
    if len(b_candidates) >0:
        new_features[feature] = featuresA[feature]['contribution']          
        for new_feature in b_candidates: 
            new_features[new_feature] = featuresB[new_feature]['contribution'] 
    return new_features 

def getSimilarityScore(featuresA, featuresB, features_weights, level):
    if len(featuresA) == 0:
        return -1
    documentContribution = 0
    coincidentFeatures = {}
    totalWeight_A = 0 
    for feature in featuresA:
        if level is None or featuresA[feature]['level']==level: 
            featureWeight = features_weights.get(featuresA[feature]['type'])
            if featureWeight is None:
                featureWeight = 1
            totalWeight_A += featureWeight
            if featuresB.get(feature) is not None:
                coincidentFeatures[feature] = 1
                documentContribution += featuresA[feature]['contribution'] * featureWeight
            
            if featuresA[feature]['type'] in ['position', 'size']: 
                added_features = getSoftValueSimilarity(featuresA, featuresB, feature)
                if len(added_features)>0:
                    for added_feature in added_features:
                        if coincidentFeatures.get(added_feature) is None:
                            coincidentFeatures[added_feature] = 1
                            documentContribution += added_features[added_feature] * featureWeight 
                                
    totalWeight_B = 0
    for feature in featuresB:
        if level is None or featuresB[feature]['level']==level:
            featureWeight = features_weights.get(featuresB[feature]['type'])
            if featureWeight is None:
                featureWeight = 1
            totalWeight_B += featureWeight

    averageLen = (totalWeight_B + totalWeight_A) / 2

    averageContribution = documentContribution / averageLen
    return averageContribution, coincidentFeatures

def setDocumentInCluster(newDocument, dicClusters, dicDocuments, threshold, full_computation = False, features_weights = None):
    if dicClusters is None:
        dicClusters = {}
        dicDocuments = {}
        currentMaxCluster = 0
    else:
        clusterIdList = [int(clusterId) for clusterId in dicClusters]
        if len(clusterIdList) > 0:
            currentMaxCluster = max(clusterIdList)
        else:
            currentMaxCluster = 0

    cluster_found = False
    min_cluster_id = -1
    min_document_id = -1
    min_prospect_doc = None

    featuresNewDocument = Autocluster.extractFeatures(newDocument)
    if len(featuresNewDocument) > 5:
        min_distance = 0
        min_temp_cluster = {}
        for i_c, clusterId in enumerate(dicClusters):
            if clusterId > 0:
                featuresCluster = dicClusters[clusterId]['features']
                distance, coincidentFeatures = getSimilarityScore(featuresCluster, featuresNewDocument, features_weights, level='cluster')
                if distance >= threshold:
                    cluster_found = True
                    cohesion_ratio = dicClusters[clusterId]['affinity_ratio']
                    temp_cluster = Autocluster.updateFeatureCohesionRatios(copy.deepcopy(dicClusters[clusterId]), featuresNewDocument, newDocument[1][0]['file_name'], coincidentFeatures, level='cluster')
                    cohesion_ratio -= temp_cluster['affinity_ratio']
                    prospect_doc = {'cluster': clusterId, 'admission_distance': distance,
                                    'affinity_ratio_delta': cohesion_ratio,
                                    'total_features': len(featuresNewDocument), 'features': featuresNewDocument,
                                    'company_id': newDocument[1][0]['company_id'],
                                    'company_name': newDocument[1][0]['company_name'],  
                                    'company_industry': newDocument[1][0]['company_industry'],
                                    'language': newDocument[1][0]['language'],
                                    'year': newDocument[1][0]['document_year'],
                                    'document_link': newDocument[1][0]['document_link'],
                                    'document_type': newDocument[1][0]['document_type'],
                                    'file_name':newDocument[1][0]['file_name'],
                                    'total_features_cluster': len(dicClusters[clusterId]['features'])}

                    if not full_computation:
                        dicClusters[clusterId] = copy.deepcopy(temp_cluster)
                        dicDocuments[newDocument[1][0]["file_name"]] = prospect_doc
                        dicClusters[clusterId]["documents"].extend([newDocument[1][0]['file_name']])
                        break
                    else:
                        if distance >= min_distance:
                            cluster_found = True
                            min_distance = distance
                            min_temp_cluster = copy.deepcopy(temp_cluster)
                            min_cluster_id = clusterId
                            min_document_id = newDocument[1][0]["file_name"]
                            min_prospect_doc = prospect_doc

        if full_computation and cluster_found:
            dicClusters[min_cluster_id] = min_temp_cluster 
            dicDocuments[min_document_id] = min_prospect_doc
            dicClusters[min_cluster_id]["documents"].extend([newDocument[1][0]['file_name']])

        if not cluster_found:
            currentMaxCluster += 1
            dicClusters[currentMaxCluster] = Autocluster.updateFeatureCohesionRatios(None, featuresNewDocument,  newDocument[1][0]['file_name'], None, level='cluster')
            dicDocuments[newDocument[1][0]["file_name"]] = {'cluster': currentMaxCluster, 'admission_distance': 1,
                                                        'affinity_ratio_delta': 0,
                                                        'total_features': len(featuresNewDocument),
                                                        'features': featuresNewDocument,
                                                        'company_id': newDocument[1][0]['company_id'],
                                                        'company_name': newDocument[1][0]['company_name'],  
                                                        'company_industry': newDocument[1][0]['company_industry'],
                                                        'language': newDocument[1][0]['language'],
                                                        'year': newDocument[1][0]['document_year'],
                                                        'document_link': newDocument[1][0]['document_link'],
                                                        'document_type': newDocument[1][0]['document_type'],
                                                        'file_name':newDocument[1][0]['file_name'],
                                                        'total_features_cluster': len(dicClusters[currentMaxCluster]['features'])}

            dicClusters[currentMaxCluster]["documents"] = [newDocument[1][0]['file_name']]
            dicClusters[currentMaxCluster]["cluster_id"] = currentMaxCluster 
    else:
        dicDocuments[newDocument[1][0]["file_name"]] = {'cluster': 0, 'admission_distance': 0,
                                                    'affinity_ratio_delta': 0,
                                                    'total_features': len(featuresNewDocument),
                                                    'features': featuresNewDocument,
                                                    'company_id': newDocument[1][0]['company_id'],
                                                    'company_name': newDocument[1][0]['company_name'],  
                                                    'company_industry': newDocument[1][0]['company_industry'],
                                                    'language': newDocument[1][0]['language'],
                                                    'year': newDocument[1][0]['document_year'],
                                                    'document_link': newDocument[1][0]['document_link'],
                                                    'document_type': newDocument[1][0]['document_type'],
                                                    'file_name':newDocument[1][0]['file_name'],
                                                    'total_features_cluster': 0}
        if dicClusters.get(0) is None:
            dicClusters[0] = {}
            dicClusters[0]["documents"] = []
            dicClusters[0]['affinity_ratio'] = 0
            dicClusters[0]['average_features'] = 0
            dicClusters[0]['total_features'] = 0
            dicClusters[0]['cluster_id'] = 0
            dicClusters[0]['features'] = []
        dicClusters[0]["documents"].extend([newDocument[1][0]['file_name']])
        dicClusters[0]["numberDocuments"] = len(dicClusters[0]["documents"])

    return dicClusters, dicDocuments


class Autocluster:

    def __init__(self, action,  output_dir, model_dir, use_case_name, data_directory = None, 
                thr_similarity=None, thr_clustering_merging=None, thr_subclustering_merging=None, full_computation_ph1=None,short_path_ph_merging=None,
                using_mp=None, number_workers =None, batch_size=None, file_limit=None, 
                dataset_perc=None, filter_type_doc =None, only_last_presented_doc =None, min_year = None):
        self.logger = Logger()
        assert os.path.isdir(output_dir)
        assert os.path.isdir(model_dir)
        if data_directory is not None: assert os.path.isdir(data_directory)
        
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.use_case_name = use_case_name
        if data_directory is not None: self.data_directory = data_directory
        self.results_report = {}
        self.using_mp = using_mp

        if action == 'train' or action == 'train_evaluate':
            self.only_last_presented_doc = True
            self.clusters_model_filename = os.path.join(self.output_dir, use_case_name, 'clusters_model.ac')
            self.documents_set_filename = os.path.join(self.output_dir, use_case_name, 'document_set.ac')
            self.training_args_filename = os.path.join(self.output_dir, use_case_name, 'training_args.bin')
            self.dataset_perc = dataset_perc
            self.features_weights = {'title': 1, 'subtitle': 2, 'text': 1, 'page': 1, 'enumerator': 4}
            load_model = False
        elif action == 'append' or action == 'load':
            self.only_last_presented_doc = only_last_presented_doc
            self.clusters_model_filename = os.path.join(self.model_dir, use_case_name, 'clusters_model.ac')
            self.documents_set_filename = os.path.join(self.model_dir, use_case_name, 'document_set.ac')
            self.training_args_filename = os.path.join(self.model_dir, use_case_name, 'training_args.bin')
            load_model = True

        if load_model:
            self.cluster_set = pickle.load(open(self.clusters_model_filename, 'rb'))
            self.documents_set = pickle.load(open(self.documents_set_filename, 'rb'))
            training_args = pickle.load(open(self.training_args_filename, 'rb'))
            self.features_weights = training_args['features_weights']
            self.data_directory = training_args['data_directory']
            self.thr_similarity = training_args['thr_similarity']
            self.thr_clustering_merging = training_args['thr_clustering_merging']
            self.thr_subclustering_merging = training_args['thr_subclustering_merging']
            self.full_computation_ph1 = training_args['full_computation_ph1']
            self.short_path_ph_merging = training_args['short_path_ph_merging'] 
            self.filter_type_doc = training_args['filter_type_doc'] 
            
            self.number_workers = number_workers
            self.batch_size = batch_size
            self.file_limit = file_limit
            self.min_year = min_year
        else:
            self.clusters_set = {}
            self.documents_set = {}
            self.data_directory = data_directory
            self.thr_similarity = thr_similarity
            self.thr_clustering_merging = thr_clustering_merging
            self.thr_subclustering_merging = thr_subclustering_merging
            self.full_computation_ph1 = full_computation_ph1
            self.number_workers = number_workers
            self.batch_size = batch_size
            self.file_limit = file_limit
            self.short_path_ph_merging = short_path_ph_merging
            self.min_year = min_year
            self.filter_type_doc = filter_type_doc 

        output_dir = os.path.join(output_dir, self.use_case_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    

    def clusterAndSaveModel(self):
        dicClusters = None
        dicDocuments = None
        
        #LOAD DATA
        oFDExt = FDExt(self.data_directory, self.output_dir) 
        oFDExt.loadDataset(filter_last_doc=self.only_last_presented_doc, filter_type_doc=self.filter_type_doc)
        documentDataset = oFDExt.getTextListByDocument(self.dataset_perc, group_by_page=True)

        #STATS
        number_companies_total = len({documentDataset[document_id][1][0]['company_id'] for document_id in documentDataset if documentDataset[document_id][1][0].get('company_id') is not None })
        number_documents_total = len(documentDataset)
        self.results_report['data_working_total'] = number_documents_total
        self.results_report['data_working_companies'] = number_companies_total
        Logger.printLog('Working data ' + str(number_documents_total) + ' companies ' + str(number_companies_total))

        #RUN CLUSTERING ALGORITHM
        dicClusters, dicDocuments = self.doClustering(documentDataset, dicClusters, dicDocuments, self.thr_similarity, self.thr_clustering_merging, self.thr_subclustering_merging,
                                    full_computation=self.full_computation_ph1, number_workers=self.number_workers,batch_size=self.batch_size,
                                                      short_path_merging=self.short_path_ph_merging, using_mp=self.using_mp)

        
        #append a indicator that the document was used during training
        for document_id in dicDocuments:
            dicDocuments[document_id]['train'] = True

        reportExcel = [dicDocuments[documentId] for cluster_id in dicClusters if dicClusters[cluster_id] is not None and cluster_id >0 for documentId in dicClusters[cluster_id]['documents']]
        
        resultFileLocation =  os.path.join(self.output_dir, self.use_case_name, self.use_case_name + '_results' + str(self.thr_similarity)+'_'+str(len(documentDataset)))

        FileManager.saveDictToExcel(resultFileLocation, reportExcel)
        
        dicDocumentDataset = {documentDataset[item][1][0]['file_name']: documentDataset[item] for item in documentDataset}
        highConfidentClusters, clustersList = Autocluster.getHighConfidentClusters(dicClusters, dicDocuments, dicDocumentDataset, self.thr_similarity)
        ChartDrawer.drawTreeMap(highConfidentClusters, dicClusters, total_documents=len(documentDataset),  output_dir=os.path.join(self.output_dir, self.use_case_name))

        self.logger.printLog('Total cluster 0: ', len(dicClusters[0]['documents']), no_time=True)
        self.results_report['cluster_0_docs'] = len(dicClusters[0]['documents'])

        self.cluster_set = dicClusters
        self.documents_set = dicDocuments
        self.save_model()

        self.results_report['high_conf_clusters'] = len(highConfidentClusters)
        self.results_report['high_conf_num_docs'] = sum([item[1]['numberDocuments'] for item in highConfidentClusters])
        self.results_report['high_conf_clusters_ratio_docs']= 0 if self.results_report['high_conf_num_docs'] == 0 else self.results_report['high_conf_num_docs']/self.results_report['high_conf_clusters']

        #SAVING RESULTS
        result_stats_fileLocation = os.path.join(self.output_dir, self.use_case_name, self.use_case_name + '_results_performance')
        FileManager.saveJSON(result_stats_fileLocation, self.results_report)
        print('Saving performance results in ', result_stats_fileLocation)

        print('Building relations')
        
        resultFileLocation_relations =  os.path.join(self.output_dir, self.use_case_name + '_results' + str(self.thr_similarity)+'_'+str(len(documentDataset))+'_relations')
        reportExcelRelations = [dicDocuments[cluster_key] for cluster_key in dicDocuments]
        FileManager.saveDictToExcel(resultFileLocation_relations, reportExcelRelations)
        
        
        return highConfidentClusters, dicClusters, dicDocuments

   
    @staticmethod
    def companyAccuracyClustering(dicDocuments):
        dicCompanies = {dicDocuments[item]['company_id']:{} for item in dicDocuments}
        for i, key_company in enumerate(dicCompanies):
            document_list = {dicDocuments[item]['file_name']:dicDocuments[item] for item in dicDocuments if dicDocuments[item]['company_id']==key_company}
            cluster_list = {document_list[item]['cluster']:dicDocuments[item]['file_name'] for item in document_list if document_list[item]['cluster'] > 0}
            subcluster_list = {str(document_list[item]['cluster']) + "_" + str(document_list[item]['subcluster']): dicDocuments[item]['file_name'] for item in document_list if document_list[item]['cluster'] > 0}
            number_docs = len(document_list)
            number_clusters = len(cluster_list)
            number_subclusters = len(subcluster_list)
            dicCompanies[key_company]['number_documents'] = number_docs 

            if number_clusters > 0:
                dicCompanies[key_company]['ratio_cluster'] = number_docs / number_clusters
            if number_subclusters > 0:
                dicCompanies[key_company]['ratio_subcluster'] = number_docs / number_subclusters

        return dicCompanies

    def trainClustering(self, dataset, trueLabels, thresholdSimilarity = 0.85, minYear = 2014, thresholdClusterMerging=0.80, thresholdSubClustering = 0.8, checkSubCluster = False,  full_computation=False):
        dicClusters = None
        dicDocuments = None
        dicClusters, dicDocuments = self.doClustering(dataset, dicClusters, dicDocuments, thresholdSimilarity, minYear, thresholdClusterMerging, thresholdSubClustering, full_computation=full_computation)
        
        predictedClusters = {}
        numberOfDocsPredicted = 0
        for clusterKey in dicClusters:
            if not checkSubCluster:
                if predictedClusters.get(clusterKey) is None:
                    predictedClusters[clusterKey] = []
                    for documentId in dicClusters[clusterKey]['documents']:
                        numberOfDocsPredicted += 1
                        predictedClusters[clusterKey].append({'file_name': documentId})
            else:
                for subclusterKey in dicClusters[clusterKey]['subclusters']:
                    for documentId in dicClusters[clusterKey]['subclusters'][subclusterKey]['documents']:
                        subClusterKey2 = clusterKey * 1000 + subclusterKey
                        if predictedClusters.get(subClusterKey2) is None:
                            predictedClusters[subClusterKey2] = []
                        numberOfDocsPredicted += 1
                        predictedClusters[subClusterKey2].append({'file_name': documentId})

        trueClusters = {}
        numberOfDocsTrue = 0
        for documentKey in trueLabels:
            if checkSubCluster:
                clusterKey = trueLabels[documentKey]['cluster'] * 1000 + trueLabels[documentKey]['subcluster']
            else:
                clusterKey = trueLabels[documentKey]['cluster'] 
            if trueClusters.get(clusterKey) is None:
                trueClusters[clusterKey] = []
            trueClusters[clusterKey].append({'file_name': documentKey})
            numberOfDocsTrue += 1
        index, indexPerDocument = Autocluster.getRandIndex(predictedClusters, trueClusters)
        print("Rand Index : " + str(index))
        return dicClusters, dicDocuments

    def doClustering(self, documentDataset, dicClusters, dicDocuments, thresholdSimilarity = 0.85, thresholdClusterMerging=0.80, thresholdSubClustering = 0.8, full_computation=False, number_workers=None, batch_size=None, short_path_merging=None, using_mp=None):

        #PHASE 1: AUTOCLUSTERING
        self.logger.printLog('I. Starting autoclustering phase', no_time=True)
        documentDataset, dicClusters, dicDocuments = self.startAutoClustering(documentDataset, dicClusters, dicDocuments, thresholdSimilarity,  full_computation=full_computation, number_workers=number_workers, batch_size=batch_size, using_mp=using_mp)

        self.results_report['time_ph1_min'] = self.logger.getElapsedTime()
        self.results_report['clusters_ph1']= len(dicClusters)
        clustersbg2 = {item:dicClusters[item] for item in dicClusters if dicClusters[item]['numberDocuments']>1 and dicClusters[item]['cluster_id']>0}
        self.results_report['ph1_clusters_bt2']= len(clustersbg2)
        self.results_report['ph1_num_docs_bt2'] = sum([dicClusters[item]['numberDocuments'] for item in clustersbg2])
        self.results_report['ph1_ratio_docs'] = 0 if self.results_report['ph1_num_docs_bt2'] == 0 else self.results_report['ph1_num_docs_bt2'] / self.results_report['ph1_clusters_bt2']

        #PHASE 2: CLEANING
        self.logger.printLog('II. Starting cleaning phase', no_time=True)
        dicClusters = self.forgetLessFrequentFeatures(dicClusters)
        self.results_report['time_ph2_min'] = self.logger.getElapsedTime()

        #PHASE 3: MERGING
        self.logger.printLog('III. Starting merging phase', no_time=True)
        dicClusters, dicDocuments = self.mergeClustersByMatrix(dicClusters, dicDocuments, thresholdClusterMerging, number_workers=number_workers, short_path_merging=short_path_merging, using_mp=using_mp)
        self.results_report['time_ph3_min'] = self.logger.getElapsedTime()

        #PHASE 4: SUBCLUSTERING
        self.logger.printLog('IV. Starting subclustering phase', no_time=True)
        dicClusters, dicDocuments = self.createSubClusters(dicClusters, dicDocuments, thresholdSubClustering)
        self.results_report['time_ph5_min'] = self.logger.getElapsedTime()

        self.results_report['total_clusters']=len(dicClusters)
        clustersbg2 = {item:dicClusters[item] for item in dicClusters if dicClusters[item]['numberDocuments']>1  and dicClusters[item]['cluster_id']>0}
        clusters1 = {item:dicClusters[item] for item in dicClusters if dicClusters[item]['numberDocuments']==1  and dicClusters[item]['cluster_id']>0}
        self.results_report['total_clusters_bt2'] = len(clustersbg2)
        self.results_report['total_num_docs_bt2'] = sum([dicClusters[item]['numberDocuments'] for item in clustersbg2])
        self.results_report['total_num_docs_1'] = sum([dicClusters[item]['numberDocuments'] for item in clusters1])
        self.results_report['total_ratio_docs'] = 0 if self.results_report['total_num_docs_bt2'] ==0 else self.results_report['total_num_docs_bt2']/self.results_report['total_clusters_bt2']

        return dicClusters, dicDocuments


    def startAutoClustering(self, documentDataset, dicClusters, dicDocuments, thresholdSimilarity, full_computation=False, number_workers=10, batch_size=None, using_mp=None):
        #Insert document in Cluster
        if number_workers > 1 and not using_mp :
            if os.environ.get("ip_head") is not None:
                ray.init(address=os.environ["ip_head"], num_cpus = number_workers)
            else:
                ray.init(num_cpus = number_workers)
            print("Nodes in the Ray cluster:", ray.nodes())
            
        
        data_batch = []
        if batch_size is None:
            batch_size = ceil(len(documentDataset)/number_workers)
        document_list = []
        if dicClusters is None or len(dicClusters) == 0:
            dicClusters = {}
            max_cluster_id_sent = 0
        else:
            max_cluster_id_sent = max({item:0 for item in dicClusters})

        if dicDocuments is None:
            dicDocuments = {}

        max_id_iteration = 0
        if number_workers ==1:
            self.logger.printLog('Executing sequential work', no_time=True)
        else:
            self.logger.printLog('Number of workers:', number_workers, 'having each', batch_size, no_time=True)
        i = -1
        documents_no_company = 0
        for document_id in tqdm.tqdm(documentDataset):
            i += 1
            document = documentDataset[document_id]
            if document[1][0].get('company_id') is None:
                documents_no_company +=1
                continue
            
            if number_workers > 1:
                document_list.append(document) 
                if len(document_list) == batch_size or i == len(documentDataset)-1: 
                    data_batch.append({'batch_number': len(data_batch),'dataset': document_list, 'thresholdSimilarity': thresholdSimilarity,
                                       'full_computation': full_computation,'dicClusters': dicClusters, 'features_weights': self.features_weights})
                    document_list = []
                    if len(data_batch) == number_workers or i == len(documentDataset)-1:
                        self.logger.printLog('Starting distribution', no_time = True)
                        if using_mp:
                            oPool = mp.Pool(number_workers)
                            results = oPool.map(processAutoclusteringBatch, data_batch)
                            oPool.close()
                        else:
                            futures = [ray_remote_processAutoclusteringBatch.remote(l_batch) for l_batch in data_batch]
                            results = ray.get(futures)
                        
                        del data_batch
                        data_batch = []
                        self.logger.printLog('Collecting data from workers ', len(results), no_time = True)
                        for k, result in enumerate(results):
                            self.logger.printLog('Working with batch ', k, no_time = True)
                            dicClusters_result = result['clusters']
                            dicDocuments_result = result['documents']
                            dicDocuments.update(dicDocuments_result)

                            #cluster 0
                            for d_cluster in dicClusters_result:
                                if d_cluster ==0:
                                    if dicClusters.get(0) is None:
                                        dicClusters[0] = dicClusters_result[0]
                                    else:
                                        old_docs = {item:0 for item in dicClusters[0]['documents']}
                                        new_docs = [item for item in dicClusters_result[0]['documents'] if old_docs.get(item) is None]
                                        dicClusters[0]['documents'].extend(new_docs)
                                        dicClusters[0]['numberDocuments'] += len(new_docs)
                                elif d_cluster <= max_cluster_id_sent:
                                        old_docs = {item: 0 for item in dicClusters[d_cluster]['documents']}
                                        new_docs = [item for item in dicClusters_result[d_cluster]['documents'] if old_docs.get(item) is None]
                                        if len(new_docs)>0:
                                            for doc_id in new_docs:
                                                distance, coincidentFeatures = getSimilarityScore(dicClusters[d_cluster]['features'], dicDocuments[doc_id]['features'], self.features_weights, level='cluster')
                                                temp = Autocluster.updateFeatureCohesionRatios(dicClusters[d_cluster], dicDocuments[doc_id]['features'],  document_id, coincidentFeatures, level='cluster')
                                                dicClusters[d_cluster]['affinity_ratio'] = temp['affinity_ratio']
                                                dicClusters[d_cluster]['numberDocuments'] = temp['numberDocuments']
                                                dicClusters[d_cluster]['documents'].append(doc_id)

                            cluster_dic_temp = {dicClusters_result[item]['cluster_id']:item for item in dicClusters_result}

                            for d_cluster in dicClusters_result:
                                if d_cluster > max_cluster_id_sent and d_cluster >0:
                                    if dicClusters.get(d_cluster) is None:                                      #This is only for the first group's result
                                        dicClusters.update({d_cluster:dicClusters_result[d_cluster]})
                                        max_id_iteration = max({item:0 for item in dicClusters})
                                    else:                                                                       #This is for the second and later group's result
                                        doc_list_temp = dicClusters_result[d_cluster]['documents'] 
                                        max_id_iteration += 1

                                        if dicClusters.get(max_id_iteration) is None:
                                            dicClusters[max_id_iteration] = copy.deepcopy(dicClusters_result[d_cluster])
                                            dicClusters[max_id_iteration]['cluster_id'] = max_id_iteration
                                            for i_doc, d_document in enumerate(doc_list_temp):
                                                dicDocuments[d_document]['cluster'] = max_id_iteration
                                        else:
                                            print('Cluster already exists, having the max dict ', max_id_iteration)


                            number_docs_dic = len(dicDocuments)
                            number_docs_clu = sum([len(dicClusters[item]['documents']) for item in dicClusters])
                            if number_docs_dic != number_docs_clu:
                                self.logger.printLog('Error. Documents not added', no_time=True, logging_level='WARNING')

                        max_cluster_id_sent = max_id_iteration
                        if not using_mp: del futures
                        del results 
                        time.sleep(10)
                        gc.collect()
            else:
                dicClusters, dicDocuments = setDocumentInCluster(document, dicClusters, dicDocuments, thresholdSimilarity, full_computation=full_computation, features_weights=self.features_weights)

        if documents_no_company>0:
            print('Documents without company info: ' + str(documents_no_company))

        return documentDataset, dicClusters, dicDocuments

    def calculateAffinityRatio(self, cluster):
        document_dic = {}
        if cluster['average_features'] >0:
            for feature in cluster['features']:
                for document in cluster['features'][feature]['documents']:
                    if document_dic.get(document) is None:
                        document_dic[document] = 1
                    else:
                        document_dic[document] += 1
            
            average_affinity = 0
            for document in document_dic:
                average_affinity += document_dic[document]/cluster['average_features']

            return average_affinity / len(document_dic)
        return 0

    def forgetLessFrequentFeatures(self, dicClusters, report = True):
        number_features_cleaned = 0
        number_features_total = 0
        number_cluster_more_4_docs = 0
        for cluster in dicClusters:
            if dicClusters.get(cluster) is not None:
                number_documents = dicClusters[cluster]['numberDocuments']
                current_features = dicClusters[cluster]['features'] 
                if number_documents > 4 and len(current_features) > 0:
                    minF = 1/number_documents if number_documents < 5 else 2/number_documents
                    number_cluster_more_4_docs += 1
                    number_features_total += len(current_features)
                    filteredFeatures = {item: current_features[item] for item in current_features if current_features[item]['contribution'] > minF}
                    number_features_cleaned += len(current_features) - len(filteredFeatures)
                    if number_features_cleaned > 0:
                        dicClusters[cluster]['features'] = filteredFeatures
                        dicClusters[cluster]['affinity_ratio'] = self.calculateAffinityRatio(dicClusters[cluster]) 

        if report:
            self.results_report['clean_features'] = number_features_total
            self.results_report['clean_features_cleaned'] = number_features_cleaned
            self.results_report['clean_clusters'] = number_cluster_more_4_docs
            self.results_report['clean_ratio'] = 0
            if number_features_total > 0:
                self.results_report['clean_ratio'] = number_features_cleaned / number_features_total
            Logger.printLog('clean clusters', self.results_report['clean_clusters'], no_time=True)
            Logger.printLog('clean features', self.results_report['clean_features'], no_time=True)
            Logger.printLog('clean features cleaned', self.results_report['clean_features_cleaned'], no_time=True)
            Logger.printLog('pruning %', int(self.results_report['clean_ratio']*100), no_time=True)
        return dicClusters

     

    @staticmethod
    def getHighConfidentClusters(dicClusters, dicDocuments, dicDocumentDataset, thresholdSimilarity, minDocumentsPerCluster=1):
        highConfidentClusters = [(item, dicClusters[item]) for item in dicClusters if  dicClusters[item] is not None and dicClusters[item]['affinity_ratio'] >= thresholdSimilarity and dicClusters[item]['numberDocuments'] > minDocumentsPerCluster]
        clustersList = {}
        for cluster in highConfidentClusters:
            clustersList[cluster[0]] = [{'cluster_id': dicDocuments[item]['cluster'],
                                         'affinity_ratio': cluster[1]['affinity_ratio'],
                                         'admission_distance': dicDocuments[dicDocumentDataset[item][1][0]['file_name']]['admission_distance'],
                                         'affinity_ratio_delta': dicDocuments[dicDocumentDataset[item][1][0]['file_name']][ 'affinity_ratio_delta'],
                                         'total_features_cluster': len(cluster[1]['features']),
                                         'total_features_document':  dicDocuments[dicDocumentDataset[item][1][0]['file_name']]['total_features'],
                                         'company_id': dicDocumentDataset[item][1][0]['company_id'],
                                         'company_name': dicDocumentDataset[item][1][0]['company_name'],
                                         'file_name': dicDocumentDataset[item][1][0]['file_name'],
                                         'year': dicDocumentDataset[item][1][0]['document_year'],
                                         'document_link': dicDocumentDataset[item][1][0]['document_link'],
                                         'language': dicDocumentDataset[item][1][0]['language'],
                                         'document_type': dicDocumentDataset[item][1][0]['document_type'],
                                         'company_industry': dicDocumentDataset[item][1][0]['company_industry'],
                                         'number_documents': cluster[1]['numberDocuments'] if cluster[1].get('numberDocuments') is not None else 0,
                                         'relatedClusters':cluster[1]['relatedClusters'] if cluster[1].get('relatedClusters') is not None else ''}
                                        for item in dicDocuments if dicDocuments[item]['cluster'] == cluster[0]]

        return highConfidentClusters, clustersList

    
    @staticmethod
    def getRandIndex(predictedClusters, trueClusters):
        predictedDocs = {document['document_id']:itemKey for itemKey in predictedClusters for document in predictedClusters[itemKey]}
        trueDocs = {document['document_id']:itemKey for itemKey in trueClusters for document in trueClusters[itemKey]}

        documentDictionaryA = {document['document_id']: 1 for item in predictedClusters for document in predictedClusters[item]}
        documentDictionaryB = {document['document_id']: 1 for item in trueClusters for document in trueClusters[item]}
        documentDictionaryA.update(documentDictionaryB)
        documentDictionary = {item: i for i, item in enumerate(documentDictionaryA)}

        predictedMatrix, documentDictionary = Autocluster.getPairMatrix(predictedClusters, documentDictionary)
        documentDictionaryIndex = {documentDictionary[documentKey]:documentKey for documentKey in documentDictionary}
        trueMatrix, documentDictionary = Autocluster.getPairMatrix(trueClusters, documentDictionary)
        indexPerDocument = []
        for rowItem in range(len( predictedMatrix)):
            totalDocuments = 0
            coincidences = 0 
            randIndex = 0
            for columnItem in range (len(predictedMatrix[0])):
                sum = predictedMatrix[rowItem][columnItem] +  trueMatrix[rowItem][columnItem]
                if sum == 1:
                    totalDocuments +=1
                elif sum ==2:
                    totalDocuments +=1
                    coincidences += 1
            if coincidences >0:
                randIndex = coincidences / totalDocuments
            indexPerDocument.append({'document_id': documentDictionaryIndex[rowItem], 'totalDocuments' : totalDocuments, 'coincidences': coincidences, 'randIndex' : randIndex})

        globalIndex = np.sum([item['randIndex'] for item in indexPerDocument]) / len(indexPerDocument)
        return globalIndex, indexPerDocument


    @staticmethod
    def getPairMatrix(clusterList, documentDictionary):
        documentMatrix = np.zeros((len(documentDictionary), len(documentDictionary)))
        for idCluster in clusterList:
            docsInCluster = clusterList[idCluster]
            for i in range(len(docsInCluster)):
                for j in range(i , len(docsInCluster)):
                    documentMatrix[documentDictionary[docsInCluster[i]['document_id']], documentDictionary[docsInCluster[j]['document_id']]] = 1
                    documentMatrix[documentDictionary[docsInCluster[j]['document_id']], documentDictionary[docsInCluster[i]['document_id']]] = 1
        return documentMatrix, documentDictionary

    def mergeClustersByMatrix(self, dicClusters, dicDocuments, thresholdMerging, numberOfMaxIterations=10, number_workers=None, short_path_merging=None, matrix_size = 3000, using_mp=None):
        if short_path_merging:
            self.logger.printLog('Using short path in merging.', no_time=True)
        else:
            self.logger.printLog('Using Long path in merging.', no_time=True)

        i_merge_group = 0
        numberOfIterations = 0
        dicClusters_batch_set = []

        if number_workers > 1:
            numberBatchs = ceil(len(dicClusters) / matrix_size)
        else:
            numberBatchs = 1

        while numberOfIterations < numberOfMaxIterations:
            numberOfIterations += 1
            numberOfClusters = len([item for item in dicClusters if dicClusters.get(item) is not None])
            self.logger.printLog('--Starting merging iteration', numberOfIterations, 'out of', numberOfMaxIterations, 'with' , numberOfClusters , 'clusters.', no_time=True)
            number_groups_no_updated = 0
            clusterIdList = [item for item in dicClusters if item >0 and dicClusters.get(item) is not None]
            merging_batch_total = ceil(len(clusterIdList) / numberBatchs)
            if number_workers == 1:
                infoset = {'dicClusters': dicClusters, 'merging_threshold': thresholdMerging, 'short_path': short_path_merging, 'features_weights': self.features_weights}
                adjMatrix_set = [getAdjacencyMatrixOfClusters(infoset)]
            elif numberOfIterations == numberOfMaxIterations - 2 and merging_batch_total > matrix_size:
                infoset = {'dicClusters': dicClusters, 'merging_threshold': thresholdMerging, 'short_path': short_path_merging, 'features_weights': self.features_weights}
                adjMatrix_set = [getAdjacencyMatrixOfClusters(infoset)]
            else:
                for i_merge_group in range(numberBatchs):
                    dicCluster_batch_tmp = {item:dicClusters[item] for item in clusterIdList[merging_batch_total*i_merge_group:merging_batch_total*(i_merge_group+1)]}
                    dicCluster_batch = {'dicClusters': dicCluster_batch_tmp, 'merging_threshold': thresholdMerging, 'short_path': short_path_merging, 'features_weights': self.features_weights}
                    if len(dicCluster_batch)>0:
                        dicClusters_batch_set.append(dicCluster_batch)
                        dicCluster_batch = []
                    else:
                        break
                if using_mp:
                    oPool = mp.Pool(number_workers)
                    adjMatrix_set = oPool.map(getAdjacencyMatrixOfClusters, dicClusters_batch_set)
                else:
                    futures =[ray_remote_getAdjacencyMatrixOfClusters.remote(l_batch) for l_batch in dicClusters_batch_set]
                    adjMatrix_set = ray.get(futures)

            dicClusters_batch_set = []
            for set_adjMatrix in adjMatrix_set:
                min_index = set_adjMatrix[1]
                adjMatrix = set_adjMatrix[0]
                highConfidentMerging = [(i+min_index, j+min_index, cell, i, j) for i, row in enumerate(adjMatrix) for j, cell in enumerate(row) if cell >= thresholdMerging and i > j]
                set_adjMatrix = None
                processedPairs = {}
                self.logger.printLog('-----Found ', len(highConfidentMerging), ' high confident merging pairs.', no_time=True)
                if len(highConfidentMerging)>0:
                    for pair in highConfidentMerging:
                        if pair[0] not in processedPairs:
                            listPairs = Autocluster.getListOfSimilarClusters(initialClusterId=pair[0], clusterPairList=highConfidentMerging, listAccumulated=None)
                            if len(listPairs) > 0:
                                clustersToMerge = [dicClusters[cluster_id] for cluster_id in listPairs] 
                                idsToUpdate = [idCluster['cluster_id'] for idCluster in clustersToMerge]

                                clustersToMerge2 = Autocluster.convergeMultipleClusters(clustersToMerge)
                                dicClusters[idsToUpdate[0]] = clustersToMerge2[0]
                                for posCluster in range(1, len(idsToUpdate)):
                                    dicClusters[idsToUpdate[posCluster]] = None
                                for documentId in dicClusters[idsToUpdate[0]]['documents']:
                                    dicDocuments[documentId]['cluster'] = idsToUpdate[0] 
                                processedPairs.update({item:1 for item in idsToUpdate})
                else:
                    number_groups_no_updated += 1

            if number_groups_no_updated == numberBatchs:
                break
            else:
                numberBatchs = numberBatchs - 2 if numberBatchs > 2 else 1

            self.logger.printLog('--Finishing merging iteration with ', len(dicClusters), ' clusters.', no_time=True)

        total_clean = len(dicClusters)
        dicClusters = {item:dicClusters[item] for item in dicClusters if dicClusters.get(item) is not None}
        total_clean -= len(dicClusters)
        dicClusters = self.forgetLessFrequentFeatures(dicClusters, report=False)
        self.logger.printLog('----Finishing merging ', i_merge_group, 'removing ', total_clean, ' clusters.', no_time=True)
        self.results_report['merging_clusters'] = len(dicClusters)
        self.results_report['merging_documents'] = sum([dicClusters[item]['numberDocuments'] for item in dicClusters])

        return dicClusters, dicDocuments


    def createSubClusters(self, dicClusters, dicDocuments, thresholdSubClustering):
        #Created subclusters are not having all the corresponding documents on it
        bigClustersKeys = [key for key in dicClusters]
        for bigClusterIndex in bigClustersKeys:
            bigCluster = dicClusters[bigClusterIndex]
            if bigCluster is not None:
                subDicClusters = None 
                #Insert document in subcluster
                for documentKey in bigCluster['documents']: 
                    if dicDocuments[documentKey]['cluster'] >0:
                        subDicClusters, dicDocuments = self.setDocumentInSubCluster(dicDocuments[documentKey], subDicClusters, dicDocuments, thresholdSubClustering)
            
                if subDicClusters is not None and len(subDicClusters)>0:
                    dicClusters[bigClusterIndex]['subclusters'] = subDicClusters
        return dicClusters, dicDocuments

    @staticmethod
    def getListOfSimilarClusters(initialClusterId, clusterPairList, listAccumulated):
        if listAccumulated is None:
            listAccumulated = {initialClusterId:1}
        currentNumberOfItems = len(listAccumulated)
        coincidences = [item[1] if item[0] == initialClusterId else item[0] for item in clusterPairList if (item[0] == initialClusterId or item[1] == initialClusterId)]
        listAccumulated.update({item:1 for i,item in enumerate(coincidences)})
        if len(listAccumulated) == currentNumberOfItems or len(coincidences) == 0:
            return listAccumulated
        else:
            for idCluster in coincidences:
                listAccumulated = Autocluster.getListOfSimilarClusters(idCluster, clusterPairList, listAccumulated)
        return listAccumulated

    @staticmethod
    def convergeMultipleClusters(clustersToMerge):
        totalFeatures = {}
        listOfDocs = []
        totalDocs = 0 
        average_features = 0
        for cluster in clustersToMerge:
            totalDocs += cluster['numberDocuments']
            listOfDocs.extend(cluster['documents']) 
            for feature in cluster['features']:
                if totalFeatures.get(feature) is None:
                    totalFeatures[feature] =  cluster['features'][feature]
                else:
                    totalFeatures[feature]['documents'].extend(cluster['features'][feature]['documents'])
            
            average_features += cluster['average_features']*cluster['numberDocuments']

        for feature in totalFeatures:
            totalFeatures[feature]['contribution'] = len(totalFeatures[feature]['documents'])/ totalDocs
            
        average_features = average_features / len(listOfDocs)
        clustersToMerge[0]['documents'] = listOfDocs
        clustersToMerge[0]['numberDocuments'] = len(listOfDocs)
        clustersToMerge[0]['features'] = totalFeatures
        clustersToMerge[0]['average_features'] = average_features

        for clusterPos in range(1,len(clustersToMerge)):
            clustersToMerge[clusterPos] = None
        return clustersToMerge


    def setDocumentInSubCluster(self, newDocument, dicSubClusters, dicDocuments, threshold):
        cluster_found = False
        distanceCandidate = 0
        features_candidate = {}
        if dicSubClusters is None:
            dicSubClusters = {} 
            currentMaxCluster = -1
        else:
            clusterIdList = [int(clusterId) for clusterId in dicSubClusters]
            currentMaxCluster = max(clusterIdList)

        featuresNewDocument = newDocument['features']
        
        for subClusterId in dicSubClusters: 
            featuresCluster =  dicSubClusters[subClusterId]['features']
            distance, coincidentFeatures = getSimilarityScore(featuresCluster, featuresNewDocument, self.features_weights, level='subcluster')
            if distance >= threshold and distance > distanceCandidate:
                cluster_found = True
                subclusterCandidate = subClusterId
                distanceCandidate = distance
                features_candidate = copy.deepcopy(coincidentFeatures)

        if cluster_found:
            cohesion_ratio= dicSubClusters[subclusterCandidate]['affinity_ratio']
            dicSubClusters[subclusterCandidate] = Autocluster.updateFeatureCohesionRatios(dicSubClusters[subclusterCandidate], featuresNewDocument, newDocument["file_name"], features_candidate, level='subcluster')
            cohesion_ratio -= dicSubClusters[subclusterCandidate]['affinity_ratio']
            dicDocuments[newDocument["file_name"]]['subcluster'] = subclusterCandidate
            dicDocuments[newDocument["file_name"]]['subcluster_admission_distance']= distanceCandidate
            dicDocuments[newDocument["file_name"]]['subcluster_affinity_ratio_delta']= cohesion_ratio
            dicDocuments[newDocument["file_name"]]['subcluster_total_features']= len(dicSubClusters[subclusterCandidate]['features'])
            dicDocuments[newDocument["file_name"]]['subcluster_features']= featuresNewDocument
            dicSubClusters[subclusterCandidate]["documents"].extend([newDocument['file_name']])                        

        if not cluster_found:
            currentMaxCluster += 1
            dicSubClusters[currentMaxCluster] = Autocluster.updateFeatureCohesionRatios(None, featuresNewDocument,newDocument["file_name"], None, level='subcluster')
            dicDocuments[newDocument["file_name"]]['subcluster'] = currentMaxCluster
            dicDocuments[newDocument["file_name"]]['subcluster_admission_distance']= 1
            dicDocuments[newDocument["file_name"]]['subcluster_affinity_ratio_delta']= 0
            dicDocuments[newDocument["file_name"]]['subcluster_total_features']= len( dicSubClusters[currentMaxCluster]['features'])
            dicDocuments[newDocument["file_name"]]['subcluster_features']= featuresNewDocument
            dicSubClusters[currentMaxCluster]["documents"] = [newDocument['file_name']]    


        return dicSubClusters, dicDocuments

    @staticmethod
    def extractFeatures(document):
        features = {}
        financial_statement_filter = {
            'Les notes figurant en annexe font partie intégrante des comptes annuels':0,
            'Die Anhänge sind integraler Bestandteil der Jahresabschlüsse':0,
            'The notes in the annex form an integral part of the annual accounts':0
        } 
        annexes = []
        current_annex = 0
        for page_number in document:
            page = document[page_number]
            financial_statements_check = [1 for line in page if financial_statement_filter.get(line['text']) is not None]
            if len(financial_statements_check) > 0:
                continue #skip financial statements for working only with annexes
            else:
                annexes.append(page_number)

            current_annex += 1 
            feature_template =  {'language': page[0]['language'],'contribution': 1 ,'type':'', 'level':None, 'page':[page_number], 'frequency':1,'documents':[]}
            #A. Page-level features
            #A.1. Language
            feature_name = 'language_'+ str(page_number) + '_' +  page[0]['language']
            features[feature_name] = copy.deepcopy(feature_template)
            features[feature_name]['type'] = 'page' 
            features[feature_name]['level'] = 'cluster' 

            #A.2. Page orientation
            feature_name = 'orientation_'+ str(page_number) + '_' + page[0]['orientation']
            features[feature_name] = copy.deepcopy(feature_template)
            features[feature_name]['type'] = 'page'
            features[feature_name]['level'] = 'cluster'

            if current_annex <3 or page_number == len(document)-1:
                #A.3. X Position with more than one repetition 
                #Get lines
                line_set = {item['y'] for item in page}
                for y_posline in line_set:
                    line_xPos = min([item['x'] for item in page if item['y'] == y_posline])
                    if line_xPos < 500:
                        feature_name = 'xPos_'+ str(line_xPos) #str(page_number) +
                        if features.get(feature_name) is None: 
                            features[feature_name] = copy.deepcopy(feature_template)
                            features[feature_name]['type'] = 'position' 
                            features[feature_name]['level'] = 'subcluster' 
                            features[feature_name]['value'] = line_xPos
                            features[feature_name]['frequency'] = 1
                            features[feature_name]['values'] = [line_xPos] 
                        else:
                            features[feature_name]['frequency'] += 1
                            features[feature_name]['page'] = features[feature_name]['page'] + [page_number]
                            features[feature_name]['values'] += [line_xPos]

                #A.4. Text width size with more than one repetition
                wSize_list = Counter([item['w'] for item in page] ).most_common()
                wSize_list = [item for item in wSize_list if item[1]>1]
                for wSize in wSize_list:
                    feature_name = 'wSize_' + str(wSize[0]) #+ str(page_number)
                    if features.get(feature_name) is None: 
                        features[feature_name] = copy.deepcopy(feature_template)
                        features[feature_name]['type'] = 'size'
                        features[feature_name]['level'] = 'subcluster'
                        features[feature_name]['value'] = wSize[0]
                        features[feature_name]['frequency'] = wSize[1]
                        features[feature_name]['values'] = [wSize[1]]
                    else:
                        features[feature_name]['frequency'] = features[feature_name]['frequency'] + wSize[1]
                        features[feature_name]['page'] = features[feature_name]['page'] + [page_number]
                        features[feature_name]['values'] += [wSize[1]]
                        

                #A.5. Text height size
                hSize_list = Counter([item['h'] for item in page] ).most_common() 
                for hSize in hSize_list:
                    feature_name = 'hSize_'+ str(hSize[0]) # str(page_number) +
                    if features.get(feature_name) is None: 
                        features[feature_name] = copy.deepcopy(feature_template)
                        features[feature_name]['type'] = 'size'
                        features[feature_name]['level'] = 'subcluster'
                        features[feature_name]['value'] = hSize[0]
                        features[feature_name]['frequency'] = hSize[1]
                        features[feature_name]['values'] = [hSize[1]]
                    else:
                        features[feature_name]['frequency'] = features[feature_name]['frequency'] + hSize[1]
                        features[feature_name]['page'] = features[feature_name]['page'] + [page_number]
                        features[feature_name]['values'] += [hSize[1]]

            #B. Line-level features
            #B.1. Look for title and subtitle candidates.
            enumeratorList = {}
            for i_line, line in enumerate(page):
                #for subtitles if it has enumerator la, le, el, lo, il, 
                text =  NLP.replaceSpecialVowels(line['text'].lower().strip())
                coincidences =NLP.checkIfStartsWithEnumerator(text, lang=line['language'], includePrefixWord='DefaultNotes')
                if len(coincidences)>0 and ' ' + coincidences[0].strip() + ' ' not in [' la ', ' le ', ' el ', ' lo ', ' il ']: 
                    text = text.replace(coincidences[0], '').strip()
                    if NLP.hasText(text):
                        enumeratorList.update({item.strip():1 for item in coincidences}) 
                        
                        feature_name = 'enum_' +  NLP.getTypeEnumeratorPattern(coincidences[0])
                        features[feature_name] = copy.deepcopy(feature_template)
                        features[feature_name]['type'] = 'enumerator'
                        features[feature_name]['level'] = 'cluster'

                        text2 = NLP.removeDatesInSentence([text])
                        text2 = NLP.removeNumbersInText(text2) 
                        if len(text2.strip()) <= 5:
                            continue

                        if not (text2[-1] !='.' and line['text'][0] == line['text'][0].upper()):
                            continue                                

                        feature_name = 'subtitle_' +  text2
                        features[feature_name] = copy.deepcopy(feature_template)
                        features[feature_name]['type'] = 'subtitle'
                        features[feature_name]['level'] = 'cluster'

                        
                        if i_line < len(page)-1:
                            text = page[i_line+1]['text'].lower().strip()
                            text_coincidences = NLP.checkIfStartsWithEnumerator(text, lang=line['language'], includePrefixWord='DefaultNotes')
                            if len(text_coincidences) ==0:
                                text2 = NLP.removeDatesInSentence([text])
                                text2 = NLP.removeNumbersInText(text2)
                                text2 = ' '.join(text2.split(' ')[:10])
                                if len(text2)>4 and NLP.hasText(text2):
                                    if ':' in text2:
                                        text2 = text2[:text2.index(':')].strip()
                                    feature_name = 'text_' +  text2
                                    features[feature_name] = copy.deepcopy(feature_template)
                                    features[feature_name]['type'] = 'text'
                                    features[feature_name]['level'] = 'subcluster'

                        

        if len({item for item in features if features[item]['level']=='cluster' and features[item]['type'] in ['subtitle', 'title']}) <3:
            for page_number in annexes:
                page = document[page_number]
                for i_line, line in enumerate(page):
                    w_text = line['w']
                    x_pos = line['x']
                    if x_pos < page[0]['page_width']/3 and w_text < page[0]['page_width']/3:
                        text =  NLP.replaceSpecialVowels(line['text'].lower().strip())
                        text2 = NLP.removeDatesInSentence([text])
                        text2 = NLP.removeNumbersInText(text2)
                        if len(text2)>4 and NLP.hasText(text2) and text2[-1] !='.' and line['text'][0] == line['text'][0].upper():
                            if ':' in text2:
                                text2 = text2[:text2.index(':')]
                            feature_name = 'title_' +  text2
                            features[feature_name] = copy.deepcopy(feature_template)
                            features[feature_name]['type'] = 'title'
                            features[feature_name]['level'] = 'cluster'
        return features

    @staticmethod
    def updateFeatureCohesionRatios(cluster, featuresNewDocument, newDocumentId, sharedFeatures, level): 
        if level is not None:
            doc_level_features = {item:featuresNewDocument[item] for item in featuresNewDocument if featuresNewDocument[item]['level']==level}
        else:
            doc_level_features = featuresNewDocument

        if cluster is None:
            for feature in doc_level_features:
                doc_level_features[feature]['documents'] = [newDocumentId]

            return {'features': doc_level_features, 'affinity_ratio': 1, 'numberDocuments': 1, 'average_features':len(doc_level_features)}
        else:
            avgConfidence = 0
            if level is not None: 
                totalListFeatures = {item:cluster['features'][item] for item in cluster['features'] if cluster['features'][item]['level']==level}
            else:
                totalListFeatures = {item:cluster['features'][item] for item in cluster['features']} 

            #Append features of the new document to cluster's
            for feature in doc_level_features:
                if totalListFeatures.get(feature) is None:
                    totalListFeatures[feature] = doc_level_features[feature]
                    cluster['features'][feature] = doc_level_features[feature]
                    cluster['features'][feature]['contribution'] = 0
                    

            cluster['numberDocuments'] += 1
            totalDocs = cluster['numberDocuments']
            for feature in totalListFeatures:
                newContribution = 0
                if sharedFeatures.get(feature) is not None or cluster['features'][feature]['contribution'] == 0:
                    newContribution = 1/totalDocs
                    cluster['features'][feature]['documents'].append(newDocumentId)

                if cluster['features'].get(feature) is not None:
                    cluster['features'][feature]['contribution'] = cluster['features'][feature]['contribution'] *(totalDocs -1) / totalDocs + newContribution
                else:
                    featureContribution = totalListFeatures[feature]
                    featureContribution['contribution'] = newContribution
                    cluster['features'][feature] = featureContribution
            avgFeatures = 0
            for feature in cluster['features']:
                avgConfidence += cluster['features'][feature]['contribution']
                avgFeatures +=1

            if avgFeatures >0:
                avgConfidence = avgConfidence / avgFeatures
            cluster['affinity_ratio'] = avgConfidence
            cluster['average_features'] = avgFeatures
            
        return cluster



    def save_model(self):
        assert os.path.isdir(self.output_dir)
        pickle.dump(self.cluster_set, open(self.clusters_model_filename, 'wb'))
        pickle.dump(self.documents_set, open(self.documents_set_filename, 'wb'))

        training_args = {}
        training_args['features_weights'] = self.features_weights
        training_args['data_directory'] = self.data_directory
        training_args['thr_similarity'] = self.thr_similarity
        training_args['thr_clustering_merging'] = self.thr_clustering_merging
        training_args['thr_subclustering_merging'] = self.thr_subclustering_merging
        training_args['full_computation_ph1'] = self.full_computation_ph1
        training_args['number_workers'] = self.number_workers
        training_args['batch_size'] = self.batch_size
        training_args['file_limit'] = self.file_limit
        training_args['short_path_ph_merging'] = self.short_path_ph_merging
        training_args['use_case_name'] = self.use_case_name
        training_args['only_last_presented_doc'] = self.only_last_presented_doc
        training_args['filter_type_doc'] = self.filter_type_doc
        training_args['output_dir'] = self.output_dir
        pickle.dump(training_args, open(self.training_args_filename, 'wb'))




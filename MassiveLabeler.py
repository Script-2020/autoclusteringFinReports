import os
import numpy as np
from Autocluster import Autocluster
from util.FileManager import FileManager

def generateCandidates(output_dir, model_dir, use_case_name, min_docs_threshold=2, min_feature_coverage_ratio=0.1, max_number_iterations=10):
    oAutocluster = Autocluster("load",  output_dir, model_dir, use_case_name) 
    document_set = oAutocluster.documents_set
    selected_documents = {}
    feature_scope_set ={}
    
    number_documents_dataset = sum([oAutocluster.cluster_set[cluster_id]['numberDocuments'] for cluster_id in oAutocluster.cluster_set if oAutocluster.cluster_set[cluster_id]['numberDocuments'] >=2])
    for cluster_id in oAutocluster.cluster_set:
        if cluster_id >0:
            document_list = oAutocluster.cluster_set[cluster_id]['documents']
            feature_set = {}
            if len(document_list) >= min_docs_threshold: 
                subtitle_list = {}
                lowconf_subtitle_list = {}
                selected_documents[cluster_id] = [] 
                for document_id in document_list: 
                    oDocument = document_set[document_id]
                    doc_subtitle_list = {item:oAutocluster.cluster_set[cluster_id]['features'][item]['contribution'] for item in oDocument['features'] if oDocument['features'][item]['type'] in ['title','subtitle'] 
                                            and oAutocluster.cluster_set[cluster_id]['features'].get(item) is not None and 
                                            #oAutocluster.cluster_set[cluster_id]['features'][item]['contribution'] >= min_confidence_score and 
                                            oAutocluster.cluster_set[cluster_id]['features'][item]['contribution'] > 1/oAutocluster.cluster_set[cluster_id]['numberDocuments']}
                    lowconfident_subtitles = {item:oAutocluster.cluster_set[cluster_id]['features'][item]['contribution'] for item in oDocument['features'] if oDocument['features'][item]['type'] in ['title' ,'subtitle'] 
                                            and oAutocluster.cluster_set[cluster_id]['features'].get(item) is not None and 
                                            (#oAutocluster.cluster_set[cluster_id]['features'][item]['contribution'] < min_confidence_score or 
                                            oAutocluster.cluster_set[cluster_id]['features'][item]['contribution'] > 1/oAutocluster.cluster_set[cluster_id]['numberDocuments'])}
                    feature_set[document_id] = doc_subtitle_list
                    subtitle_list.update({item:np.zeros(len(document_list)) for item in doc_subtitle_list})
                    lowconf_subtitle_list.update(lowconfident_subtitles)
                
                document_indexes = {item:i for i, item in enumerate(document_list)}
                indexed_documents = {i:item for i, item in enumerate(document_list)}
                for document_id in document_list: 
                    for subtitle in feature_set[document_id]:
                        if subtitle_list.get(subtitle) is not None:
                            subtitle_list[subtitle][document_indexes[document_id]] = 1

                total_features= len(subtitle_list)
                if total_features ==0: 
                    continue

                number_iterations = 0
                features_cluster_labeled = {}
                while number_iterations < max_number_iterations:
                    number_iterations+=1
                    totals = np.zeros(len(document_list))
                    for subtitle in subtitle_list:
                        for i, _ in enumerate(totals):
                            totals[i] += subtitle_list[subtitle][i]
                    
                    max_index = np.argmax(totals)
                    selected_documents[cluster_id].append(indexed_documents[max_index]) 
                    
                    for feature in feature_set[indexed_documents[max_index]]:
                        if subtitle_list.get(feature) is not None:
                            subtitle_list.pop(feature)
                            features_cluster_labeled[feature] = 1

                    if len(subtitle_list) ==0 or len(subtitle_list) /total_features <= min_feature_coverage_ratio:
                        break
                
                feature_scope_set[cluster_id] ={'total_features':total_features,
                                                'low_confident_features':len(lowconf_subtitle_list), 
                                                'selected_features':total_features-len(subtitle_list), 
                                                'number_documents':oAutocluster.cluster_set[cluster_id]['numberDocuments'],
                                                'selected_documents':len(selected_documents[cluster_id]),
                                                'features': features_cluster_labeled}

    number_selected_docs = sum([feature_scope_set[item]['selected_documents'] for item in feature_scope_set])
    number_docs = sum([feature_scope_set[item]['number_documents'] for item in feature_scope_set])
    ratio_docs = number_selected_docs / number_docs

    number_selected_features = sum([feature_scope_set[item]['selected_features'] for item in feature_scope_set])
    number_features_highconf= sum([feature_scope_set[item]['total_features'] for item in feature_scope_set])
    ratio_features = number_selected_features/number_features_highconf
    
    number_features_total= sum([feature_scope_set[item]['total_features'] + feature_scope_set[item]['low_confident_features']  for item in feature_scope_set])
    ratio_features = number_selected_features/number_features_total
    ratio_features_highconf = number_selected_features/number_features_highconf

    report_docs = []
    rep_perc = 0
    for cluster_id in selected_documents: 
        if feature_scope_set.get(cluster_id) is not None:
            for document_id in oAutocluster.cluster_set[cluster_id]['documents']:  
                oAutocluster.documents_set[document_id]['low_confident_features'] = feature_scope_set[cluster_id]['low_confident_features']
                if document_id in selected_documents[cluster_id]:
                    report_docs.append(oAutocluster.documents_set[document_id])
                
                labeled_features_doc = 0
                subtitle_features_doc = [item for item in oAutocluster.documents_set[document_id]['features'] if oAutocluster.documents_set[document_id]['features'][item]['type'] in ['title','subtitle']]
                features_cluster_labeled = feature_scope_set[cluster_id]['features']
                for feature_doc in subtitle_features_doc:
                    if features_cluster_labeled.get(feature_doc) is not None:
                        labeled_features_doc +=1
                rep_doc =  labeled_features_doc / len(subtitle_features_doc)
                oAutocluster.documents_set[document_id]['feature_rep'] = rep_doc 
                rep_perc += rep_doc
    
    rep_perc2 = rep_perc /number_documents_dataset
    print('Rep perc total', rep_perc2)
    
    rep_perc3 = rep_perc /number_docs
    print('Rep perc ', rep_perc3)

    ratio_clusters = 0
    for cluster_id in feature_scope_set:
        ratio_clusters += feature_scope_set[cluster_id]['selected_documents'] / feature_scope_set[cluster_id]['number_documents']
    
    ratio_clusters = ratio_clusters/len(feature_scope_set)
    print('Ratio clusters', ratio_clusters)

    reportFile = os.path.join(output_dir,use_case_name,'documents_to_label')
    FileManager.saveDictToExcel(reportFile, report_docs)
    print('Document saved in ', reportFile)
    print('Number of documents in dataset ', number_documents_dataset)

    return selected_documents, ratio_docs, ratio_features_highconf, ratio_features, number_selected_docs, number_docs
                        
                    
                    

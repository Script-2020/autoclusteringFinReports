from Autocluster import Autocluster 
import MassiveLabeler
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--action",
        default=None,
        type=str,
        required=True, )

    parser.add_argument(
        "--data_directory",
        default=None,
        type=str,
        required=False, )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True, )
        
    parser.add_argument(
        "--model_dir",
        default=None,
        type=str,
        required=True, )
        
    parser.add_argument(
        "--thr_similarity",
        default=0.83,
        type=float,
        required=False, )

    parser.add_argument(
        "--thr_clustering_merging",
        default=0.8,
        type=float,
        required=False, )
    
    parser.add_argument(
        "--thr_subclustering_merging",
        default=0.70,
        type=float,
        required=False, )
        
    parser.add_argument(
        "--only_last_presented_doc",
        default=False,
        type=bool,
        required=False, )

    parser.add_argument(
        "--number_workers",
        default=10,
        type=int,
        required=False, )

    parser.add_argument(
        "--batch_size",
        default=None,
        type=int,
        required=False, )

    parser.add_argument(
        "--file_limit",
        default=100,
        type=int,
        required=False, )

    parser.add_argument(
        "--short_path_ph_merging",
        default=False,
        type=bool,
        required=False, )

    parser.add_argument(
        "--full_computation_ph1",
        default=False,
        type=bool,
        required=False, )

    parser.add_argument(
        "--use_case_name",
        default=None,
        type=str,
        required=True, )

    parser.add_argument(
        "--min_year",
        default=2014,
        type=int,
        required=False, )

    parser.add_argument(
        "--dataset_perc",
        default=1,
        type=float,
        required=False, )

    parser.add_argument(
        "--using_mp",
        default=False,
        type=bool,
        required=False, ) 
        
    parser.add_argument(
        "--filter_type_doc",
        default="eCDF",
        type=str,
        required=False, ) 
    
     

    args = parser.parse_args()
    if args.action == 'train':
        oAutocluster = Autocluster(args.action, args.output_dir, args.model_dir, args.use_case_name, args.data_directory,
                args.thr_similarity, args.thr_clustering_merging, args.thr_subclustering_merging, args.full_computation_ph1, args.short_path_ph_merging,
                args.using_mp, args.number_workers, args.batch_size, args.file_limit, 
                args.dataset_perc, args.filter_type_doc, args.only_last_presented_doc, args.min_year)
    
        oAutocluster.clusterAndSaveModel()
    elif args.action == 'append':
        oAutocluster = Autocluster(args.action, args.output_dir, args.model_dir, args.use_case_name, args.data_directory)
        oAutocluster.appendDocumentsToModel()
    elif args.action == 'generate_candidates':
        min_feature_coverage_ratio = [0, 0.1, 0.2, 0.25]
        for min_coverage in min_feature_coverage_ratio:  
            selected_documents, ratio_docs, ratio_highconf_feat, ratio_feat,number_selected_docs, number_docs = MassiveLabeler.generateCandidates(args.output_dir, args.model_dir, args.use_case_name, min_feature_coverage_ratio=min_coverage,min_docs_threshold=3)
            print("-----------------------------------------------")
            print('Threshold min feature coverage ratio: ' , min_coverage)
            print('Ratio Docs:' , int(ratio_docs*10000)/100)
            print('Number selected Docs:' , number_selected_docs)
            print('Total Docs:' , number_docs)
            print('Ratio Features:' , int(ratio_feat*10000)/100)
            print('Ratio High confident Features:' , int(ratio_highconf_feat*10000)/100)
            print("-----------------------------------------------")

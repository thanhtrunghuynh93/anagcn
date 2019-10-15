from input.dataset import Dataset, SynDataset
from time import time
from algorithms import *
from evaluation.metrics import get_statistics
import utils.graph_utils as graph_utils
import random
import numpy as np
import torch
import argparse
import os
import pdb


def parse_args():
    parser = argparse.ArgumentParser(description="Network alignment")
    parser.add_argument('--source_dataset', default="data/allmv_tmdb/allmv/graphsage/")
    parser.add_argument('--target_dataset', default="data/allmv_tmdb/tmdb/graphsage/")
    parser.add_argument('--groundtruth',    default="data/allmv_tmdb/dictionaries/groundtruth")
    parser.add_argument('--seed',           default=123,    type=int)
    subparsers = parser.add_subparsers(dest="algorithm", help='Choose 1 of the algorithm from: FINAL, REGAL, ANAGCN')
    

    # FINAL
    parser_final = subparsers.add_parser('FINAL', help='FINAL algorithm')
    parser_final.add_argument('--H',                   default=None, help="Priority matrix")
    parser_final.add_argument('--max_iter',            default=30, type=int, help="Max iteration")
    parser_final.add_argument('--alpha',               default=0.6, type=float)
    parser_final.add_argument('--tol',                 default=1e-2, type=float)
    parser_final.add_argument('--train_dict', default='data/allmv_tmdb/dictionaries/node,split=01.train.dict', type=str)


    # REGAL
    parser_regal = subparsers.add_parser('REGAL', help='REGAL algorithm')
    parser_regal.add_argument('--attrvals', type=int, default=2,
                        help='Number of attribute values. Only used if synthetic attributes are generated')
    parser_regal.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser_regal.add_argument('--k', type=int, default=10,
                        help='Controls of landmarks to sample. Default is 10.')
    parser_regal.add_argument('--max_layer', type=int, default=2,
                        help='Calculation until the layer for xNetMF.')
    parser_regal.add_argument('--alpha', type=float, default=0.01, help="Discount factor for further layers")
    parser_regal.add_argument('--gammastruc', type=float, default=1, help="Weight on structural similarity")
    parser_regal.add_argument('--gammaattr', type=float, default=1, help="Weight on attribute similarity")
    parser_regal.add_argument('--num_top', type=int, default=10,
                        help="Number of top similarities to compute with kd-tree.  If 0, computes all pairwise similarities.")
    parser_regal.add_argument('--buckets', default=2, type=float, help="base of log for degree (node feature) binning")


    
    # ANAGCN
    parser_ANAGCN = subparsers.add_parser("ANAGCN", help="ANAGCN algorithm")
    parser_ANAGCN.add_argument('--cuda',                action="store_true")
    
    parser_ANAGCN.add_argument('--embedding_dim',       default=200,         type=int)
    parser_ANAGCN.add_argument('--emb_epochs',    default=20,        type=int)
    parser_ANAGCN.add_argument('--lr', default=0.01, type=float)
    parser_ANAGCN.add_argument('--num_GCN_blocks', type=int, default=2)
    parser_ANAGCN.add_argument('--act', type=str, default='tanh')
    parser_ANAGCN.add_argument('--log', action="store_true", help="Just to print loss")
    parser_ANAGCN.add_argument('--invest', action="store_true", help="To do some statistics")
    parser_ANAGCN.add_argument('--alpha0', type=float, default=1)
    parser_ANAGCN.add_argument('--alpha1', type=float, default=1)
    parser_ANAGCN.add_argument('--alpha2', type=float, default=1)
    # refinement
    parser_ANAGCN.add_argument('--refinement_epochs', default=10, type=int)
    parser_ANAGCN.add_argument('--refine', action="store_true", help="wheather to use refinement step")
    parser_ANAGCN.add_argument('--threshold_refine', type=float, default=0.94, help="The threshold value to get stable candidates")
    # augmentation, let noise_level = 0 if dont want to use it
    parser_ANAGCN.add_argument('--noise_level', default=0.001, type=float, help="noise to add to augment graph")
    parser_ANAGCN.add_argument('--beta', default=0.2, type=float, 
                    help="weight balance between graph structure and augment graph structure")
    parser_ANAGCN.add_argument('--coe_consistency', default=0.2, type=float, help="consistency weight")
    parser_ANAGCN.add_argument('--threshold', default=0.01, type=float, 
                    help="Threshold of for sharpenning")


    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    start_time = time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    source_dataset = Dataset(args.source_dataset)
    target_dataset = Dataset(args.target_dataset)
    groundtruth = graph_utils.load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')

    algorithm = args.algorithm

    if algorithm == "FINAL":
        train_dict = None
        if args.train_dict != "":
            train_dict = graph_utils.load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        model = FINAL(source_dataset, target_dataset, H=args.H, alpha=args.alpha, maxiter=args.max_iter, tol=args.tol, train_dict=train_dict)
    elif algorithm == "REGAL":
        model = REGAL(source_dataset, target_dataset, max_layer=args.max_layer, alpha=args.alpha, k=args.k, num_buckets=args.buckets,
                      gammastruc = args.gammastruc, gammaattr = args.gammaattr, normalize=True, num_top=args.num_top)
    elif algorithm == "ANAGCN":
        model = ANAGCN(source_dataset, target_dataset, args)
    else:
        raise Exception("Unsupported algorithm")

    S = model.align()

    
    acc, MAP, Hit, AUC, top5, top10, top20, top30, top50, top100 = get_statistics(S, groundtruth, get_all_metric=True)
    print("MAP: {:.4f}".format(MAP))
    print("AUC: {:.4f}".format(AUC))
    print("Top_1: {:.4f}".format(acc))
    print("Top_5: {:.4f}".format(top5))
    print("Top_10: {:.4f}".format(top10))
    print("Top_20: {:.4f}".format(top20))
    print("Top_30: {:.4f}".format(top30))
    print("Top_50: {:.4f}".format(top50))
    print("Top_100: {:.4f}".format(top100))
    print("Full_time: {:.4f}".format(time() - start_time))


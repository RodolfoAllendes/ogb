# -*- coding: utf-8 -*-

import os
import logging
import time

from dataloader import EvalDataset, TrainDataset, NewBidirectionalOneShotIterator
# from .dataloader import get_dataset
from dataloader import RandomPartition

from ogb.lsc import WikiKG90MDataset#, WikiKG90MEvaluator
# from ogb.linkproppred import DglLinkPropPredDataset

from utils import get_compatible_batch_size, save_model, CommonArgParser

backend = os.environ.get('DGLBACKEND', 'pytorch')
assert backend.lower() == 'pytorch'
import torch
import numpy as np
import scipy as sp
import torch.multiprocessing as mp
# from .train_pytorch import load_model
# from .train_pytorch import train, train_mp
# from .train_pytorch import test, test_mp

import dgl.backend as F
import dgl
from dgl.base import NID, EID

def CustomPartition(edges, n, p, has_importance=False):
    """Based on BalancedPartition and RandomPartition (from sampler), we will 
    extract only part of the original dataset to each partition.
    
    Algorithm:
    For r in relations:
      Find partition with fewest edges
      if r.size() > num_of empty_slot
         put edges of r into this partition to fill the partition,
         find next partition with fewest edges to put r in.
      else
         put edges of r into this partition.

    Returns
    -------
    List of np.array
        Edges of each partition
    """
    # split the triplets into heads, relation and tails
    if has_importance:
        heads, rels, tails, e_impts = edges
    else:
        heads, rels, tails = edges

    print('Custom partition - Total Edges: {}, Using (aprox): {} into {} parts'.format(len(heads),np.ceil(len(heads)*p),n))
    # name and count of each type of edge,
    # sorted from largest to smallest 
    uniq, cnts = np.unique(rels, return_counts=True)
    idx = np.flip(np.argsort(cnts))
    cnts = cnts[idx]
    uniq = uniq[idx]

    # adjusted count for the percentage of extraction
    p_cnts = np.ceil(cnts*p).astype(int)
    
    # array of indices for each part in the division of the triplets
    parts = []
    for i in range(n):
        parts.append([])
    
    #for each type of relation (edge)
    for i, type_rel in enumerate(uniq):
        # total number of triplets of the current type to be added to the different 
        # parts 
        cnt = p_cnts[i]
        x = cnt//n if cnt >= n else 1 # number of elements for each part
        
        # a mask for the triplets whose relation type matches type_rel
        t_idx = rels == type_rel
        a_idx = np.arange(start=0, stop=len(rels))
        # now t_idx is a list of indices of triplets whose relation matches type_rel
        # (we used the mask to filter out the indices)
        t_idx = a_idx[t_idx]
        # in order to make a random selection of the indices, we permutate them
        t_idx = np.random.permutation(t_idx)
        print('for type {} we have {} items, will use {}'.format(i,len(t_idx),cnt), end='\r', flush=True)
        t_idx = t_idx[:cnt]
        # for each part 
        off = 0
        for j in range(n):
            
            # add indices for each part (only if there are indices left to add)
            if off+x < cnt: 
                parts[j].append(t_idx[off:off+x])
                off += x 
            # if not enough (but still some) add it
            elif off < cnt:
                parts[j].append(t_idx[off:])
                off += x
            else:
                pass
    # at this point, all parts should already have the indices they need, but 
    # they are ordered sequentially from largest to smallest number of edge_type
    # so, shuffle the indices
    # first convert the parts to numpy ndarrays
    for i, part in enumerate(parts):
        parts[i] = np.array(part, dtype=np.int64)
        print('Part[{}] has {} elements: {}'.format(i,len(parts[i]), parts[i]))
    # put them all together
    shuffle_idx = np.concatenate(parts)
    # extract only the parts that were randomly selected
    heads[:] = heads[shuffle_idx]
    rels[:] = rels[shuffle_idx]
    tails[:] = tails[shuffle_idx]
    if has_importance:
        e_impts[:] = e_impts[shuffle_idx]

    # and return the results
    return parts

class SimpleRunner(object):
    
    def __init__(self):

        # Step 1 of the process LOAD THE DATASET

        # contests dataset
        self.dataset = WikiKG90MDataset(root = '/home/nibiohnproj9/public/dataset/')
        self.n_entities = self.dataset.num_entities
        self.n_relations = self.dataset.num_relations

        # The prediction task is to predict:
        # Tail given Head and Relation; or h,r -> t
        # head: the source node 
        # relation: the type of edge
        # tail: the destination node
        self.train = self.dataset.train_hrt.T # the transpose of all the triplets
               
        # even when the argument has_edge_importance seems to be set by default 
        # to True, the edges appear to have no additional information, so we will
        # keep this as False
        self.has_edge_importance = False # conditionals involving this variable will be removed for simplicity

        # neg_sample_size is the number of negative samples we use for each 
        # positive sample in training; and _eval is the number of neg samples used
        # to evaluate a positive sample.
        self.neg_sample_size = 100
        self.neg_sample_size_eval = 1000
     
        # batch sizes for training and evaluation are both of type int and are set
        # by default to 400 and 50 respectively if not specified as an argument in
        # setup.sh 
        self.batch_size = 400
        self.batch_size_eval = 50
        # we correct the values based on neg_sample sizes
        self.batch_size = get_compatible_batch_size(self.batch_size, self.neg_sample_size)
        self.batch_size_eval = get_compatible_batch_size(self.batch_size_eval, self.neg_sample_size_eval)

        # We should turn on mix CPU-GPU training for multi-GPU training.
        # mix_cpu_gpu: Training a knowledge graph embedding model with both CPUs and GPUs.
        self.mix_cpu_gpu = True
        # num_proc: The number of processes to train the model in parallel.
        # In multi-GPU training, the number of processes by default is set to match the number of GPUs.
        self.gpu = [0, 1, 2, 3]
        self.num_proc = len(self.gpu)
        # We force a synchronization between processes every x steps for 
        # multiprocessing training. This potentially stablizes the training process
        # to get a better performance. 
        self.force_sync_interval = 1000

        # Disable filter positive edges from randomly constructed negative edges 
        # for evaluation
        self.no_eval_filter = False# if included in args, set True
        self.eval_filter = not self.no_eval_filter

        # Construct negative samples proportional to vertex degree in the evaluation.
        # True if included as argument
        self.neg_deg_sample_eval = False
        if self.neg_deg_sample_eval:
            assert not self.eval_filter, "if negative sampling based on degree, we can't filter positive edges."

# args.soft_rel_part = args.mix_cpu_gpu and args.rel_part

# # if there is no cross partition relaiton, we fall back to strict_rel_part
# args.strict_rel_part = args.mix_cpu_gpu and (train_data.cross_part == False)
        self.num_workers = 8 # fix num_worker to 8

    def build_training_dataset(self):

        # construct the training Graph
        print ("To build training dataset")       
        t1 = time.time()
        # extract source, edge typy and target from the triplets
        src, etype_id, dst = self.train
        coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)), shape=[self.n_entities, self.n_entities])
        self.train_data = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)
        self.train_data.edata['tid'] = F.tensor(etype_id, F.int64)
        self.train_data.edge_parts = CustomPartition(edges=self.train, n=self.num_proc, p=0.1, has_importance=self.has_edge_importance)
        # self.train_data.edge_parts = RandomPartition(edges=self.train, n=self.num_proc, has_importance=self.has_edge_importance)
        print(self.train_data.edge_parts[0])
        print(len(self.train_data.edge_parts[0]))
        self.train_data.cross_part = True
        print ("Training dataset built, it takes %d seconds"%(time.time()-t1))



# set_logger(args)
# with open(os.path.join(args.save_path, args.encoder_model_name), 'w') as f:
#     f.write(args.encoder_model_name)

    def generate_samplers(self):

        # Generate train samplers
        train_samplers = []
        for i in range(self.num_proc):
            print ("Building training sampler for proc %d"%i)
            t1 = time.time()
            # for each GPU, allocate num_proc // num_GPU processes
            train_sampler_head = dgl.contrib.sampling.EdgeSampler(
                self.train_data,
                seed_edges = F.tensor(self.train_data.edge_parts[i]),
                batch_size = self.batch_size,
                neg_sample_size = self.neg_sample_size, 
                chunk_size = self.neg_sample_size,
                negative_mode = 'head',
                num_workers = self.num_workers,
                shuffle = True,
                exclude_positive = False,
                return_false_neg = False,
            )

            train_sampler_tail = dgl.contrib.sampling.EdgeSampler(
                self.train_data,
                seed_edges = F.tensor(self.train_data.edge_parts[i]),   
                batch_size = self.batch_size,
                neg_sample_size = self.neg_sample_size,
                chunk_size = self.neg_sample_size,
                negative_mode = 'tail',
                num_workers = self.num_workers,
                shuffle = True,
                exclude_positive = False,
                return_false_neg = False,
            )

            print(train_sampler_head)
            print(train_sampler_tail)

            train_samplers.append(NewBidirectionalOneShotIterator(
                dataloader_head = train_sampler_head, 
                dataloader_tail = train_sampler_tail,
                neg_chunk_size = self.neg_sample_size,
                neg_sample_size = self.neg_sample_size,
                is_chunked = True,
                num_nodes = self.n_entities,
                has_edge_importance = self.has_edge_importance,
            ))
            print("Training sampler for proc {} created, it takes {} seconds".format(i, time.time()-t1))

    def build_evaluation_dataset(self):

        # Create the Evaluation Datset
        self.num_test_proc = self.num_proc
        print("To create eval_dataset")
        t1 = time.time()
        eval_dataset = EvalDataset(dataset, args)
        print("eval_dataset created, it takes %d seconds" % (time.time() - t1))

# if args.valid:
#     if args.num_proc > 1:
#         # valid_sampler_heads = []
#         valid_sampler_tails = []
#         for i in range(args.num_proc):
#             print("creating valid sampler for proc %d"%i)
#             t1 = time.time()
#             valid_sampler_tail = eval_dataset.create_sampler('valid', args.batch_size_eval,
#                                                                 args.neg_sample_size_eval,
#                                                                 args.neg_sample_size_eval,
#                                                                 args.eval_filter,
#                                                                 mode='tail',
#                                                                 num_workers=args.num_workers,
#                                                                 rank=i, ranks=args.num_proc)
#             # valid_sampler_heads.append(valid_sampler_head)
#             valid_sampler_tails.append(valid_sampler_tail)
#             print("Valid sampler for proc %d created, it takes %s seconds"%(i, time.time()-t1))
#     else: # This is used for debug
#         valid_sampler_tail = eval_dataset.create_sampler('valid', args.batch_size_eval,
#                                                             args.neg_sample_size_eval,
#                                                             1,
#                                                             args.eval_filter,
#                                                             mode='tail',
#                                                             num_workers=args.num_workers,
#                                                             rank=0, ranks=1)
# if args.test:
#     if args.num_test_proc > 1:
#         test_sampler_tails = []
#         # test_sampler_heads = []
#         for i in range(args.num_test_proc):
#             print("creating test sampler for proc %d"%i)
#             t1 = time.time()
#             # test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
#             test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
#                                                                 args.neg_sample_size_eval,
#                                                                 args.neg_sample_size_eval,
#                                                                 args.eval_filter,
#                                                                 mode='tail',
#                                                                 num_workers=args.num_workers,
#                                                                 rank=i, ranks=args.num_test_proc)
#             # test_sampler_heads.append(test_sampler_head)
#             test_sampler_tails.append(test_sampler_tail)
#             print("Test sampler for proc %d created, it takes %s seconds"%(i, time.time()-t1))
#     else:
#         # test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
#         test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
#                                                         args.neg_sample_size_eval,
#                                                         1,
#                                                         args.eval_filter,
#                                                         mode='tail',
#                                                         num_workers=args.num_workers,
#                                                         rank=0, ranks=1)
# # pdb.set_trace()
# # load model
# print("To create model")
# t1 = time.time()
# model = load_model(args, dataset.n_entities, dataset.n_relations, dataset.entity_feat.shape[1], dataset.relation_feat.shape[1])
# if args.encoder_model_name in ['roberta', 'concat']:
#     model.entity_feat.emb = dataset.entity_feat
#     model.relation_feat.emb = dataset.relation_feat
# print("Model created, it takes %s seconds" % (time.time()-t1))
# model.evaluator = WikiKG90MEvaluator()

# if args.num_proc > 1 or args.async_update:
#     model.share_memory()

# emap_file = dataset.emap_fname
# rmap_file = dataset.rmap_fname
# # We need to free all memory referenced by dataset.
# eval_dataset = None
# dataset = None

# print('Total initialize time {:.3f} seconds'.format(time.time() - init_time_start))

# # train
# start = time.time()
# rel_parts = train_data.rel_parts if args.strict_rel_part or args.soft_rel_part else None
# cross_rels = train_data.cross_rels if args.soft_rel_part else None

# if args.num_proc > 1:
#     procs = []
#     barrier = mp.Barrier(args.num_proc)
#     for i in range(args.num_proc):
#         # valid_sampler = [valid_sampler_heads[i], valid_sampler_tails[i]] if args.valid else None
#         # test_sampler = [test_sampler_heads[i], test_sampler_tails[i]] if args.test else None
#         valid_sampler = [valid_sampler_tails[i]] if args.valid else None
#         test_sampler = [test_sampler_tails[i]] if args.test else None
#     pen failed
            # proc = mp.Process(target=train_mp, args=(args,
#                                                     model,
#                                                     train_samplers[i],
#                                                     valid_sampler,
#                                                     test_sampler,
#                                                     i,
#                                                     rel_parts,
#                                                     cross_rels,
#                                                     barrier,
#                                                     ))
#         procs.append(proc)
#         proc.start()
#     for proc in procs:
#         proc.join()
# else:
#     valid_samplers = [valid_sampler_tail] if args.valid else None
#     test_samplers = [test_sampler_tail] if args.test else None
#     # valid_samplers = [valid_sampler_head, valid_sampler_tail] if args.valid else None
#     # test_samplers = [test_sampler_head, test_sampler_tail] if args.test else None
#     train(args, model, train_sampler, valid_samplers, test_samplers, rel_parts=rel_parts)

# print('training takes {} seconds'.format(time.time() - start))
   

if __name__ == '__main__':
    
    sr = SimpleRunner()

    sr.build_training_dataset()
    # sr.generate_samplers()

    # sr.build_evaluation_dataset()
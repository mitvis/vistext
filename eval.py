"""Evaluation for VisText models.

Example usage:
python eval.py \
"""

import argparse
from utils_gen import *
from utils_eval import *
import importlib
import pandas as pd
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# import tarfile
from tqdm import tqdm
import time
# from timeit import default_timer as timer
import math
import json

from nltk.corpus import stopwords
from nltk import download
download('stopwords', download_dir='./nltkdata/')
stop_words = stopwords.words('english')
import gensim.downloader as api
api.BASE_DIR = './nltkdata/'
w2v_model = api.load('word2vec-google-news-300')

import torch
import torch.nn as nn
import torch.utils.data as torch_data
# from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
import pickle
import shutil
import re

import evaluate
from PIL import Image

import gc

def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to test set')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model folder')
    parser.add_argument('--results_path', type=str, required=True,
                        help='Path to results file')
    parser.add_argument('--seed', type=int, default=2022,
                        help='Seed for L1 caption generation')
    parser.add_argument('--split_eval', type=bool, default=True,
                        help='Boolean to evaluate L1 and L2L3 captions in addition to the joint L1L2L3 captions (default=True)')
    parser.add_argument('--prefixtuning', type=bool, required=True,
                        help='Boolean to specify whether the trained model uses prefix tuning (default=True)')
    parser.add_argument('--metric_bleu', type=bool, default=True,
                        help='Run BLEU score evaluation (default=True)')
    parser.add_argument('--metric_bleurt', type=bool, default=True,
                        help='Run BLEURT score evaluation (default=True)')
    parser.add_argument('--metric_perp', type=bool, default=True,
                        help='Run Perplexity score evaluation (default=True)')
    parser.add_argument('--metric_rg', type=bool, default=True,
                        help='Run Relation Generation score evaluation (default=True)')
    parser.add_argument('--metric_rouge', type=bool, default=True,
                        help='Run ROUGE score evaluation (default=True)')
    parser.add_argument('--metric_ter', type=bool, default=True,
                        help='Run Translation Edit Rate score evaluation (default=True)')
    parser.add_argument('--metric_wmd', type=bool, default=True,
                        help='Run Word Mover\'s Distance score evaluation (default=True)')
    
    args = parser.parse_args()
    seed = args.seed
    path_test = args.test_file
    prefix = args.prefixtuning
    split_eval = args.split_eval
    
    # Set seed value
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Initialize evaluators
    if args.metric_bleu:
        bleu = evaluate.load("bleu", module_type="metric")
    if args.metric_bleurt:
        bleurt = evaluate.load("bleurt", module_type="metric")
    if args.metric_perp:
        perplexity = evaluate.load("perplexity", module_type="metric")
    if args.metric_rouge:
        rouge = evaluate.load("rouge")
    if args.metric_ter:
        ter = evaluate.load("ter")
        
    # Load test set
    testdata = testdata_L1 = testdata_L2L3 = testmetadata = []
    
    with open(path_test) as f:
        data_vistext = json.load(f)

    if prefix==True:
        testdata_L1 = []
        testdata_L2L3 = []
        for i in data_vistext:
            testdata.append(i['caption_L1'] + " " + i['caption_L2L3'])
            
            testdata_L1.append(i['caption_L1'])
            testdata_L2L3.append(i['caption_L2L3'])
            
            testmetadata.append(i["L1_properties"])
        logger.info(f"L1 test set lines loaded: {len(testdata_L1)}")
        logger.info(f"L2L3 test set lines loaded: {len(testdata_L2L3)}")
        logger.info(f"Total test set lines loaded: {len(testdata)}")
    else:
        for i in data_vistext:
            testdata.append(i['caption_L1'] + " " + i['caption_L2L3'])
            testmetadata.append(i["L1_properties"])
        logger.info(f"Test set lines loaded: {len(testdata)}")
        
    # Load predictions
    path_model = args.model_path
    file = open(path_model, "r")
    predictions = file.read()
    file.close()
    predictions = predictions.split("\n")
    logger.info(f"Predictions loaded: {len(predictions)}")
    
    if len(testdata) != len(predictions):
        raise ValueError(f"Your predictions have {len(predictions)} lines while your test set has {len(testdata)} lines!")
        
    if prefix==True:
        if (len(predictions) % 2 != 0) or (len(testdata) % 2 != 0):
            raise ValueError(f"One of either your predictions or test set has an odd number of lines. \
                               Should be an even number using prefix tuning (one each of L1 and L2L3 per caption).")
            
        L1predictions = L2L3predictions = []
        for idx, i in enumerate(predictions):
            if (idx % 2 == 0):
                L1predictions.append(i)
            else:
                L2L3predictions.append(i)
        logger.info(f"L1 predictions loaded: {len(L1predictions)}")
        logger.info(f"L2L3 predictions loaded: {len(L2L3predictions)}")
        predictions_cat =  [i + " " + j for i, j in zip(L1predictions, L2L3predictions)]
        
        length = len(list1)
        if any(len(lst) != predictions_cat for lst in [L1predictions, L2predictions, predictions_cat]):
            raise ValueError(f"Mismatch in prediction list lengths!")
    else:
        predictions_cat = predictions

    scores = {}
    if split_eval:
        scores_pairwise = {}

    # Run evaluation metrics
    logger.info("Scoring...")
    # Run BLEU
    if args.metric_bleu:
        timer_metric_start = timer()
        
        logger.info("Scoring: BLEU...")
        bleu_result = bleu.compute(predictions=predictions_cat,
                                   references=[[x] for x in testdata])
        scores["BLEU"] = round(bleu_result['bleu'], 4)
        
        if split_eval:
            scores_pairwise["BLEU"] = {}
            bleu_result = bleu.compute(predictions=L1predictions,
                           references=[[x] for x in testdata_L1])
            scores_pairwise["BLEU"]["L1"] = round(bleu_result['bleu'], 4)
            
            bleu_result = bleu.compute(predictions=L2L3predictions,
                           references=[[x] for x in testdata_L2L3])
            scores_pairwise["BLEU"]["L2L3"] = round(bleu_result['bleu'], 4)
        
        logger.info('Metric evaluation time:{0:.2f} minutes.'.format((timer()-timer_metric_start)/60))

    if args.metric_bleurt:
        timer_metric_start = timer()
        
        logger.info("Scoring: BLEURT...")
        bleurt_results = bleurt.compute(predictions=predictions_cat,
                                        references=testdata)
        scores["BLEURT"] = round(np.mean(bleurt_results['scores']), 4)
        
        if split_eval:
            scores_pairwise["BLEURT"] = {}
            bleurt_results = bleurt.compute(predictions=L1predictions,
                                          references=testdata_L1)
            scores_pairwise["BLEURT"]["L1"] = round(np.mean(bleurt_results['scores']), 4)
            
            bleurt_results = bleurt.compute(predictions=L2L3predictions,
                                            references=testdata_L2L3)
            scores_pairwise["BLEURT"]["L2L3"] = round(np.mean(bleurt_results['scores']), 4)
        
        logger.info('Metric evaluation time:{0:.2f} minutes.'.format((timer()-timer_metric_start)/60))
    
    if args.metric_perp:
        timer_metric_start = timer()
        
        logger.info("Scoring: Perplexity...")
        ppl_results = perplexity.compute(predictions=predictions_cat,
                                         model_id='gpt2')
        scores["Perplexity"] = round(ppl_results['mean_perplexity'], 4)
        
        if split_eval:
            scores_pairwise["Perplexity"] = {}
            ppl_results = perplexity.compute(predictions=L1predictions,
                                             model_id='gpt2')
            scores_pairwise["Perplexity"]["L1"] = round(ppl_results['mean_perplexity'], 4)
            
            ppl_results = perplexity.compute(predictions=L2L3predictions,
                                             model_id='gpt2')
            scores_pairwise["Perplexity"]["L2L3"] = round(ppl_results['mean_perplexity'], 4)
        
        logger.info('Metric evaluation time:{0:.2f} minutes.'.format((timer()-timer_metric_start)/60))
    
    if args.metric_rg:
        timer_metric_start = timer()
        
        logger.info("Scoring: Relation Generation...")
        rg_result = evalRG(predictions_cat, testmetadata)
        scores["Relation Generation"] = round(np.mean(rg_result), 4)
        
        logger.info('Metric evaluation time:{0:.2f} minutes.'.format((timer()-timer_metric_start)/60))
        
    if args.metric_rouge:
        timer_metric_start = timer()
        
        logger.info("Scoring: ROUGE...")
        rouge_results = rouge.compute(predictions=predictions_cat,
                                      references=testdata)
        scores["ROUGE"] = rouge_results
        
        if split_eval:
            scores_pairwise["ROUGE"] = {}
            rouge_results = rouge.compute(predictions=L1predictions,
                                          references=testdata_L1)
            scores_pairwise["ROUGE"]["L1"] = rouge_results
            
            rouge_results = rouge.compute(predictions=L2L3predictions,
                                          references=testdata_L2L3)
            scores_pairwise["ROUGE"]["L2L3"] = rouge_results
        
        logger.info('Metric evaluation time:{0:.2f} minutes.'.format((timer()-timer_metric_start)/60))

    if args.metric_ter:
        timer_metric_start = timer()
        
        logger.info("Scoring: Translation Edit Rate...")
        ter_results = ter.compute(predictions=predictions_cat,
                                  references=testdata)
        scores["TER"] = ter_results["score"]
        
        if split_eval:
            scores_pairwise["TER"] = {}
            ter_results = ter.compute(predictions=L1predictions,
                                      references=testdata_L1)
            scores_pairwise["TER"]["L1"] = ter_results
            
            ter_results = ter.compute(predictions=L2L3predictions,
                                      references=testdata_L2L3)
            scores_pairwise["TER"]["L2L3"] = ter_results
        
        logger.info('Metric evaluation time:{0:.2f} minutes.'.format((timer()-timer_metric_start)/60))
        
    if args.metric_wmd:
        timer_metric_start = timer()
        
        logger.info("Scoring: Word Mover's Distance...")
        wmd_result = evalWMD(predictions_cat, testdata)
        scores["WMD"] = round(np.mean(wmd_result), 4)
        
        if split_eval:
            scores_pairwise["WMD"] = {}
            wmd_result = evalWMD(L1predictions, testdata_L1)
            scores_pairwise["WMD"]["L1"] = round(np.mean(wmd_result), 4)
            
            wmd_result = evalWMD(L2L3predictions, testdata_L2L3)
            scores_pairwise["WMD"]["L2L3"] = round(np.mean(wmd_result), 4)
        
        logger.info('Metric evaluation time:{0:.2f} minutes.'.format((timer()-timer_metric_start)/60))
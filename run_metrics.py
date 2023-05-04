"""Evaluation for VisText models.

Example usage:
python run_metrics.py \
    --test_file data/data_test.json \
    --predictions_path models/BART-base_run2/generated_predictions.txt \
    --save_results \
    --results_path metrics/BART-base_run2_scores.txt 
"""

import logging
logger = logging.getLogger(__name__)
import sys
import argparse
from utils_gen import *
from utils_metrics import *
import importlib
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import time
from timeit import default_timer as timer
import math
import json

from nltk.corpus import stopwords
from nltk import download
import gensim.downloader as api
api.BASE_DIR = './nltkdata/'

from pathlib import Path
import random
import shutil
import re
import evaluate

def main():
#     logging.basicConfig(level=logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to test set')
    parser.add_argument('--predictions_path', type=str, required=True,
                        help='Path to predictions file')
    parser.add_argument('--save_results', default=False, action='store_true',
                        help='Save results or not (default=False)')
    parser.add_argument('--results_path', type=str, default=None,
                        help='Path to results file, if none, will save in the same directory to results.txt; if existing, will overwrite (default=None)')
    parser.add_argument('--seed', type=int, default=2022,
                        help='Seed for L1 caption generation (default=2022)')
    parser.add_argument('--split_eval', default=False, action='store_true',
                        help='Whether to evaluate L1 and L2L3 captions in addition to the joint L1L2L3 captions (default=False)')
    parser.add_argument('--prefixtuning', default=False, action='store_true',
                        help='Whether the trained model uses prefix tuning (default=False)')
    parser.add_argument('--no_bleu', default=False, action='store_true',
                        help='Remove BLEU score evaluation (default=False)')
    parser.add_argument('--no_bleurt', default=False, action='store_true',
                        help='Remove BLEURT score evaluation (default=False)')
    parser.add_argument('--no_perp', default=False, action='store_true',
                        help='Remove Perplexity score evaluation (default=False)')
    parser.add_argument('--no_rg', default=False, action='store_true',
                        help='Remove Relation Generation score evaluation (default=False)')
    parser.add_argument('--no_rouge', default=False, action='store_true',
                        help='Remove ROUGE score evaluation (default=False)')
    parser.add_argument('--no_ter', default=False, action='store_true',
                        help='Remove Translation Edit Rate score evaluation (default=False)')
    parser.add_argument('--no_wmd', default=False, action='store_true',
                        help='Remove Word Mover\'s Distance score evaluation (default=False)')
    
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    path_test = args.test_file
    path_predictions = args.predictions_path
    save_results = args.save_results
    path_results = args.results_path
    seed = args.seed
    split_eval = args.split_eval
    prefix = args.prefixtuning
    logger.info(f"args: {args}")
    
    # Set seed value
    np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)

    if (split_eval) and (prefix==False):
        raise ValueError(f"Prefix tuning has to be enabled to do evaluate L1 and L2L3 captions separately.")
    
    # Initialize evaluators
    logger.info(f"Initializing evaluators...")
    if args.no_bleu==False:
        bleu = evaluate.load("bleu", module_type="metric")
    if args.no_bleurt==False:
        bleurt = evaluate.load("bleurt", module_type="metric")
    if args.no_perp==False:
        perplexity = evaluate.load("perplexity", module_type="metric")
    if args.no_rouge==False:
        rouge = evaluate.load("rouge")
    if args.no_ter==False:
        ter = evaluate.load("ter")
    if args.no_wmd==False:
        download('stopwords', download_dir='./nltkdata/')
        stop_words = stopwords.words('english')
        w2v_model = api.load('word2vec-google-news-300')
        
    # Load test set
    logger.info(f"Loading test set...")
    testdata = []
    testdata_L1 = []
    testdata_L2L3 = []
    testmetadata = []
    
    with open(path_test) as f:
        data_vistext = json.load(f)
    logger.info(f"Test set JSON items: {len(data_vistext)}")

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
    logger.info(f"Loading predictions...")
    file = open(path_predictions, "r")
    predictions = file.read()
    file.close()
    predictions = predictions.split("\n")
    logger.info(f"Predictions loaded: {len(predictions)}")
        
    if prefix==True:
        if (len(predictions) % 2 != 0) or (len(testdata) % 2 != 0):
            raise ValueError(f"One of either your predictions or test set has an odd number of lines. \
                               Should be an even number using prefix tuning (one each of L1 and L2L3 per caption).")
            
        L1predictions = []
        L2L3predictions = []
        for idx, i in enumerate(predictions):
            if (idx % 2 == 0):
                L1predictions.append(i)
            else:
                L2L3predictions.append(i)
        logger.info(f"L1 predictions loaded: {len(L1predictions)}")
        logger.info(f"L2L3 predictions loaded: {len(L2L3predictions)}")
        predictions_cat =  [i + " " + j for i, j in zip(L1predictions, L2L3predictions)]
        
        if any(len(lst) != len(predictions_cat) for lst in [L1predictions, L2L3predictions, predictions_cat]):
            raise ValueError(f"Mismatch in prediction list lengths!")
    else:
        predictions_cat = predictions
    
    if len(testdata) != len(predictions_cat):
        raise ValueError(f"Your predictions have {len(predictions_cat)} lines while your test set has {len(testdata)} lines!")

    scores = {}
    if split_eval:
        scores_pairwise = {}

    # Run evaluation metrics
    logger.info("Scoring...")
    # Run BLEU
    if args.no_bleu==False:
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
        
        logger.info('BLEU evaluation time:{0:.2f} minutes.'.format((timer()-timer_metric_start)/60))

    if args.no_bleurt==False:
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
        
        logger.info('BLEURT evaluation time:{0:.2f} minutes.'.format((timer()-timer_metric_start)/60))
    
    if args.no_perp==False:
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
        
        logger.info('Perplexity evaluation time:{0:.2f} minutes.'.format((timer()-timer_metric_start)/60))
    
    if args.no_rg==False:
        timer_metric_start = timer()
        
        logger.info("Scoring: Relation Generation...")
        rg_result = evalRG(predictions_cat, testmetadata)
        scores["Relation Generation"] = round(np.mean(rg_result), 4)
        
        logger.info('Relation Generation evaluation time:{0:.2f} minutes.'.format((timer()-timer_metric_start)/60))
        
    if args.no_rouge==False:
        timer_metric_start = timer()
        
        logger.info("Scoring: ROUGE...")
        rouge_results = rouge.compute(predictions=predictions_cat,
                                      references=testdata)
        scores["ROUGE"] = {key:round(rouge_results[key], 4) for key in rouge_results}
        
        if split_eval:
            scores_pairwise["ROUGE"] = {}
            rouge_results = rouge.compute(predictions=L1predictions,
                                          references=testdata_L1)
            scores_pairwise["ROUGE"]["L1"] = {key:round(rouge_results[key], 4) for key in rouge_results}
            
            rouge_results = rouge.compute(predictions=L2L3predictions,
                                          references=testdata_L2L3)
            scores_pairwise["ROUGE"]["L2L3"] = {key:round(rouge_results[key], 4) for key in rouge_results}
        
        logger.info('ROUGE evaluation time:{0:.2f} minutes.'.format((timer()-timer_metric_start)/60))

    if args.no_ter==False:
        timer_metric_start = timer()
        
        logger.info("Scoring: Translation Edit Rate...")
        ter_results = ter.compute(predictions=predictions_cat,
                                  references=testdata)
        scores["TER"] = round(ter_results["score"], 4)
        
        if split_eval:
            scores_pairwise["Translation Edit Rate"] = {}
            ter_results = ter.compute(predictions=L1predictions,
                                      references=testdata_L1)
            scores_pairwise["Translation Edit Rate"]["L1"] = round(ter_results["score"], 4)
            
            ter_results = ter.compute(predictions=L2L3predictions,
                                      references=testdata_L2L3)
            scores_pairwise["Translation Edit Rate"]["L2L3"] = round(ter_results["score"], 4)
        
        logger.info('Translation Edit Rate evaluation time:{0:.2f} minutes.'.format((timer()-timer_metric_start)/60))
        
    if args.no_wmd==False:
        timer_metric_start = timer()
        
        logger.info("Scoring: Word Mover's Distance...")
        wmd_result = evalWMD(predictions_cat, testdata, w2v_model, stop_words)
        scores["Word Mover's Distance"] = round(np.mean(wmd_result), 4)
        
        if split_eval:
            scores_pairwise["Word Mover's Distance"] = {}
            wmd_result = evalWMD(L1predictions, testdata_L1, w2v_model, stop_words)
            scores_pairwise["Word Mover's Distance"]["L1"] = round(np.mean(wmd_result), 4)
            
            wmd_result = evalWMD(L2L3predictions, testdata_L2L3, w2v_model, stop_words)
            scores_pairwise["Word Mover's Distance"]["L2L3"] = round(np.mean(wmd_result), 4)
        
        logger.info('Word Mover\'s Distance evaluation time:{0:.2f} minutes.'.format((timer()-timer_metric_start)/60))

    if save_results:
        if path_results is not None:
            savepath = path_results
        else:
            savepath = Path(path_results).parent.absolute().joinpath("results.txt")
            
        with open(savepath, 'w') as f:
            f.write(f"VisText Model Evaluation Results\n")
            f.write(f"---------------------------------------------\n")
            f.write(f"Model Results Path: {path_predictions}\n")
            f.write(f"Seed: {seed}\n\n")
            
            f.write(f"L1L2L3 Results\n")
            f.write(f"---------------------------------------------\n")
            for k,v in scores.items():
                if isinstance(v, dict):
                    for k2,v2 in v.items():
                        f.write(f"{k2}:\t{v2}\n")
                else:
                    f.write(f"{k}:\t{v}\n")
                    
            if split_eval:
                f.write(f"\nL1 Results\n")
                f.write(f"---------------------------------------------\n")
                for k,v in scores_pairwise.items():
                    if isinstance(v["L1"], dict):
                        for k2,v2 in v["L1"].items():
                            f.write(f"{k2}:\t{v2}\n")
                    else:
                        printresult = v["L1"]
                        f.write(f"{k}:\t{printresult}\n")
                        
                f.write(f"\nL2L3 Results\n")
                f.write(f"---------------------------------------------\n")
                for k,v in scores_pairwise.items():
                    if isinstance(v["L2L3"], dict):
                        for k2,v2 in v["L2L3"].items():
                            f.write(f"{k2}:\t{v2}\n")
                    else:
                        printresult = v["L2L3"]
                        f.write(f"{k}:\t{printresult}\n")

        
    sys.stdout.write(f"L1L2L3 Results\n")
    sys.stdout.write(f"---------------------------------------------\n")
    for k,v in scores.items():
        if isinstance(v, dict):
            for k2,v2 in v.items():
                sys.stdout.write(f"{k2}:\t{v2}\n")
        else:
            sys.stdout.write(f"{k}:\t{v}\n")
            
    if split_eval:
        sys.stdout.write(f"\nL1 Results\n")
        sys.stdout.write(f"---------------------------------------------\n")
        for k,v in scores_pairwise.items():
            if isinstance(v["L1"], dict):
                for k2,v2 in v["L1"].items():
                    sys.stdout.write(f"{k2}:\t{v2}\n")
            else:
                printresult = v["L1"]
                sys.stdout.write(f"{k}:\t{printresult}\n")

        sys.stdout.write(f"\nL2L3 Results\n")
        sys.stdout.write(f"---------------------------------------------\n")
        for k,v in scores_pairwise.items():
            if isinstance(v["L2L3"], dict):
                for k2,v2 in v["L2L3"].items():
                    sys.stdout.write(f"{k2}:\t{v2}\n")
            else:
                printresult = v["L2L3"]
                sys.stdout.write(f"{k}:\t{printresult}\n")
                
if __name__ == '__main__':
    main()
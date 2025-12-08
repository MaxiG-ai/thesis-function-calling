# -*- coding: utf-8 -*- 
import json
import random
import argparse
import os
import logging
from collections import defaultdict
import multiprocessing
from multiprocessing import Pool, Manager
from functools import partial

from benchmarks.complex_func_bench.utils.logger import Logger
from benchmarks.complex_func_bench.utils.utils import load_json

from benchmarks.complex_func_bench.runner.sap_gpt_runner import SAPGPTRunner
from benchmarks.complex_func_bench.runner.response_runner import RespEvalRunner

from src.utils.config import load_configs


cfg = load_configs()

def process_example(data, args):
    log_dir = f"{args.log_dir}/{data['id']}.log"
    logger = Logger(f"evaluation_logger_{data['id']}", log_dir, logging.DEBUG)

    model = SAPGPTRunner(args=args, logger=logger)
    resp_eval_model = RespEvalRunner(args=args, logger=logger)

    logger.info(f"Test Example {data['id']}")
    logger.info(f"Query: {data['conversations'][0]['content']}")
    
    turn_count, call_count = 0, 0
    for turn in data['conversations']:
        if turn['role'] == "assistant" and "function_call" in turn:
            turn_count += 1
            call_count += len(turn["function_call"])

    convs, message, turn_id, correct_count = model.run(data)

    # API Error
    if isinstance(message, dict) and message["error_type"] == "unknown_error":
        return None
    
    real_turn_count = 0
    for turn in convs:
        if turn['role'] == "assistant" and "function_call" in turn:
            real_turn_count += 1
    
    if convs[-1]['role'] == "assistant" and "content" in convs[-1]:
        gen_response = convs[-1]['content']
        resp_eval_result = resp_eval_model.run(data, gen_response)
    else:
        resp_eval_result = None

    logger.info(f"Message: {message}")
    logger.info(f"Success turn num = {turn_id}")
    logger.info("-" * 100)

    result = {
        "id": data['id'],
        "gen_convs": convs,
        "message": message,
        "count_dict": {
            "success_turn_num": turn_id,
            "total_turn_num": turn_count,
            "correct_call_num": correct_count,
            "total_call_num": call_count,
            "real_turn_num": real_turn_count
        },
        "resp_eval": resp_eval_result
    }

    with open(args.output_dir, 'a+') as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()

    return result


def main(cfg):
    test_data = load_json(cfg.input_file)
    if cfg.benchmark_sample_size is not None:
        test_data = random.sample(test_data, cfg.benchmark_sample_size)
    
    # if os.path.exists(cfg.results_dir):
    #     finished_data = load_json(cfg.results_dir)
    #     finished_ids = [d["id"] for d in finished_data]
    # else:
    #     finished_ids = []
    # test_data = [d for d in test_data if d['id'] not in finished_ids]
            
    with Manager():
        pool = Pool(processes=cfg.proc_num)
        process_example_partial = partial(process_example)
        _ = pool.starmap(process_example_partial, [(data, cfg) for data in test_data])
        
    pool.close()
    pool.join()


### RESULTS SCRIPT

def basic_metric(result_dir):
    results = load_json(result_dir)
    domain_success = defaultdict(int)
    domain_turn_count = defaultdict(lambda: [0, 0])
    domain_call_count = defaultdict(lambda: [0, 0])
    complete_score_count = defaultdict(lambda: [0, 0])
    correct_score_count = defaultdict(lambda: [0, 0])
    for result in results:
        domain = result['id'].rsplit("-", 1)[0]
        if result['message'] == "Success.":
            domain_success[domain] += 1
        domain_turn_count[domain][0] += result['count_dict']['success_turn_num']
        domain_turn_count[domain][1] += result['count_dict']['total_turn_num']

        domain_call_count[domain][0] += result['count_dict']['correct_call_num']
        domain_call_count[domain][1] += result['count_dict']['total_call_num']

        if result["resp_eval"] is None:
            continue

        if result["resp_eval"]['complete']['score'] in {0, 1, 2}:
            complete_score_count[domain][0] += result["resp_eval"]['complete']['score']
            complete_score_count[domain][1] += 1
        
        if result["resp_eval"]['correct']['score'] in {0, 1, 2}:
            correct_score_count[domain][0] += result["resp_eval"]['correct']['score']
            correct_score_count[domain][1] += 1

    domain_success_rate = {k: v / 150 * 100 if k != "Cross" else v / 400 * 100 for k, v in domain_success.items()}
    domain_turn_acc = {k: v[0] / v[1] * 100 if v[1] != 0 else 0 for k, v in domain_turn_count.items()}
    domain_call_acc = {k: v[0] / v[1] * 100 if v[1] != 0 else 0 for k, v in domain_call_count.items()}

    overall_success = sum(domain_success.values()) / 1000 * 100
    overall_call_acc = sum([v[0] for v in domain_call_count.values()]) / sum([v[1] for v in domain_call_count.values()]) * 100

    complete_score, complete_total = 0, 0
    for k, v in complete_score_count.items():
        complete_score += v[0]
        complete_total += v[1]
    complete_score_avg = complete_score / complete_total if complete_total != 0 else 0

    correct_score, correct_total = 0, 0
    for k, v in correct_score_count.items():
        correct_score += v[0]
        correct_total += v[1]  
    correct_score_avg = correct_score / correct_total if correct_total != 0 else 0

    
    print(f"Domain Success Rate: {domain_success_rate}")
    print(f"Domain Turn Accuracy: {domain_turn_acc}")
    print(f"Domain Call Accuracy: {domain_call_acc}")
    print(f"Overall Success Rate: {overall_success}")
    print(f"Overall Call Accuracy: {overall_call_acc}")
    print(f"Complete Score: {complete_score_avg}")
    print(f"Correct Score: {correct_score_avg}")

if __name__ == "__main__":
    cfg = load_configs()
    multiprocessing.set_start_method('spawn')
    main(cfg)
    basic_metric(result_dir=cfg.results_dir)
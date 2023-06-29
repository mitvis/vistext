#!/bin/bash
# Script to run image-guided chart captioning experiments.

function usage {
  echo "usage: $0 [-p predictions_path] [-f results_filename] [--split_eval] [--prefix_tuning]"
  echo " "
  echo "  -p predictions_path Path to test set predictions"
  echo "  -f results_filename Filename to save results under in metrics/"
  echo "  --split_eval        Also separately run evaluations for L1 and L2/L3 captions"
  echo "  --prefix_tuning     Apply semantic prefix-tuning"
  exit 1
}

# Default parameter values.
predictions_path=""            # Path to test set predictions.
results_filename="results.txt" # Filename of results.
split_eval=false               # True for separately evaluating L1 and L2/L3 captions; false otherwise.
prefix_tuning=false            # True 

# Update parameters based on arguments passed to the script.
while [[ $1 != "" ]]; do
    case $1 in
    -p | --predictions_path)
        shift
        predictions_path=$1
        ;;
    -f | --results_filename)
        shift
        results_filename=$1
        ;;
    --split_eval)
        shift
        split_eval=true
        ;;
    --prefix_tuning)
        prefix_tuning=true
    esac
    shift
done

if [[ $predictions_path = "" ]]; then
    echo "Invalid predictions file path: {$predictions_path}"
    usage
fi

# Join results path
results_path="$(pwd)/metrics/$results_filename"

# Run test
if [[ $prefix_tuning = false ]]; then
    if [[ $split_eval = true ]]; then
        echo "Invalid argument: Split evaluation requires can only be used with prefix tuning."
        usage
    else
        PYTHONPATH=$PYTHONPATH:./src \
        python code/run_metrics.py \
            --test_file data/data_test.json \
            --predictions_path $predictions_path \
            --save_results \
            --results_path $results_path \
            --no_bleurt
    fi
else
    if [[ $split_eval = true ]]; then
        PYTHONPATH=$PYTHONPATH:./src \
        python code/run_metrics.py \
            --test_file data/data_test.json \
            --predictions_path $predictions_path \
            --save_results \
            --results_path $results_path \
            --split_eval \
            --prefixtuning \
            --no_bleurt
    else
        PYTHONPATH=$PYTHONPATH:./src \
        python code/run_metrics.py \
            --test_file data/data_test.json \
            --predictions_path $predictions_path \
            --save_results \
            --results_path $results_path \
            --prefixtuning \
            --no_bleurt
    fi
fi


# If the `--save_results` flag is not specified, metrics will only be printed to stdout.
# `--results_path` can be used to specify where to save the metrics to, otherwise it will be saved to the same folder as the predictions file under `results.txt`, overwriting any existing.
# To evaluate the L1 and L2L3 captions separately, use the `--split_eval` flag. This will not disable the combined L1L2L3 evaluation, and all three sets will be output (and saved, if specified).
# Metrics can be disabled from running using the `--no_X` flag, where `X` is the metric desired to be disabled. This can be helpful in the case of `--no_bleurt`, which disables BLEURT evaluation in the event that you do not available GPU resources.
# Specify the `--prefixtuning` flag for the prefix tuning case.
# VisText: A Benchmark for Semantically Rich Chart Captioning
![Saliency card teaser.](teaser.png)

VisText is a benchmark dataset of over 12,000 charts and semantically rich captions! In the VisText dataset, each chart is represented as its rasterized image, scene graph specification, and underlying datatable. Each chart is paired with a synthetic L1 caption that describes the chart's elemental and ecoded properties and a human-generated L2/L3 caption that describes trends and statistics about the chart.

In the VisText paper, we train text-based models (i.e, models that use the scene graph and data table chart representations) as well as image-guided models that include the chart image. We also include semantic prefix-tuning, allowing our models to customize the level of semantic content in the chat. Our models output verbose chart captions that contain varying levels of semantic content.

This repository contains code for training and evaluating the VisText models. For more info, see: [VisText: A Benchmark for Semantically Rich Chart Captioning (ACL 2023)](http://vis.csail.mit.edu/pubs/vistext)

## Repo Contents

`data/` - Data files

`models/` - Pretrained models and checkpoints

## Using VisText
### Step 0: Clone the VisText repo

### Step 1: Download the raw data
Download the raw data from the [dataset site](http://vis.csail.mit.edu/) and unzip to `data/`.
Ensure that you have three folders, `data/images`, `data/scenegraphs`, and `data/features`.

### Step 2: Generate the VisText dataset from raw data
Run the `dataset_generation.ipynb` notebook from start to finish, which will generate the three split dataset files `data/data_train.json`, `data/data_test.json`, and `data/data_validation.json`.

### Step 3: Training and evaluating the VisText model
Model training and evaluation can be run from `run_train_eval_predict.py`.
As an example, our finetuned ByT5 scenegraph model was trained and evaluated with the following command:
```
python run_train_eval_predict.py     \
    --model_name_or_path google/byt5-small   \
    --do_train --do_eval     \
    --train_file data/data_train.json \
    --validation_file data/data_validation.json \
    --test_file data/data_test.json     \
    --output_dir models/ByT5-small-scenegraph-prefix   \
    --max_target_length 512 \
    --per_device_train_batch_size=3     \
    --per_device_eval_batch_size=6 \
    --text_column scenegraph    \
    --summary_column caption \
    --evaluation_strategy epoch \
    --eval_accumulation_steps 1000 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --load_best_model_at_end=True \
    --num_train_epochs 50 \
    --prefixtuning=True \
    --seed 10
```
The `--text_caption` flag specifies which textual data source to use, either `scenegraph` or `datatable`. It is also important to correctly specify the `--prefixtuning` flag if the intent is to train a model with prefix tuning for L1 and L2L3 captions. Other arguments specified are documented in Hugging Face Transformers [TrainingArguments](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments).

### Step 4: Generating predictions
Generating predictions can be run from the same `run_train_eval_predict.py` script.
Our predictions were generated as follows:
```
python run_train_eval_predict.py     \
    --model_name_or_path models/ByT5-small-scenegraph-prefix/checkpoint-XXXXX   \
    --do_predict \
    --train_file data/data_train.json \
    --validation_file data/data_validation.json \
    --test_file data/data_test.json     \
    --output_dir models/ByT5-small-scenegraph-prefix   --overwrite_output_dir     \
    --max_predict_samples 2540     \
    --max_target_length 512 \
    --per_device_eval_batch_size=10 \
    --predict_with_generate=True \
    --auto_find_batch_size=True \
    --text_column scenegraph    \
    --summary_column caption \
    --prefixtuning=True \
    --gradient_checkpointing=True
```
We choose the checkpoint with the lowest evaluation loss for generating our predictions from, which we specify in `--model_name_or_path`. As before, correctly check that `--prefixtuning` is specified for the prefix tuning case.

### Step 5: Evaluating metrics for predictions
Once our predictions are generated, we can evaluate them on our metrics.
Our metrics were evaluated as follows:
```
python run_metrics.py
    --test_file data/data_test.json
    --predictions_path models/ByT5-small-scenegraph-prefix/generated_predictions.txt
    --save_results
    --results_path metrics/ByT5-small-scenegraph-prefix_scores.txt
    --split_eval
    --prefixtuning
```
If the `--save_results` flag is not specified, metrics will only be printed to stdout.
`--results_path` can be used to specify where to save the metrics to, otherwise it will be saved to the same folder as the predictions file under `results.txt`, overwriting any existing.
To evaluate the L1 and L2L3 captions separately, use the `--split_eval` flag. This will not disable the combined L1L2L3 evaluation, and all three sets will be output (and saved, if specified).
Metrics can be disabled from running using the `--no_X` flag, where `X` is the metric desired to be disabled. This can be helpful in the case of `--no_bleurt`, which disables BLEURT evaluation in the event that you do not available GPU resources.
As in previous steps, specify the `--prefixtuning` flag for the prefix tuning case.

## Citation
For more information about VisText, check out [VisText: A Benchmark for Semantically Rich Chart Captioning](http://vis.csail.mit.edu/pubs/vistext/)
```
@inproceedings{saliencycards,
  title={{VisText: A Benchmark for Semantically Rich Chart Captioning}},
  author={Tang, Benny J. and Boggust, Angie and Satyanarayan, Arvind},
  booktitle = {Annual Meeting of the Association for Computational Linguistics ({ACL})},
  year={2023}
}
```

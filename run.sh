#!/bin/bash
# Script to run image-guided chart captioning experiments.

function usage {
  echo "usage: $0 [-c model_class] [-b batch_size] [-e num_epochs] [-g num_gpus] [-i input_type]"
  echo "       [-m model_backbone] [-s seed] [--prefix_tuning]"
  echo " "
  echo "  -c model_class      Class of models to use."
  echo "                      Options: 'text_only' or 'image_guided'"
  echo "  -b batch_size       Batch size for training, validation, and testing."
  echo "                      Default batch_size = 4."
  echo "  -e num_epochs       Number of epochs to train for."
  echo "                      Default num_epochs = 50."
  echo "  -g num_gpus         Number of gpus to parallelize across."
  echo "                      Default num_gpus = 1"
  echo "  -i input_type       Chart text representation."
  echo "                      Options: 'scenegraph', 'datatable', or 'imageonly'"
  echo "                      Default input_type: 'scenegraph'"
  echo "  -m model_backbone   Model architecture to finetune."
  echo "                      Options: 'byt5', 't5', or 'bart'"
  echo "                      Default model_backbone = 'byt5'"
  echo "  -s seed             Seed number. Default seed = random integer."
  echo "  --prefix_tuning     Apply semantic prefix tuning"
  exit 1
}

# Default parameter values.
model_class="text_only" # Class of models to use.
batch_size=4            # Train, val, and test batch size.
num_epochs=50           # Number of epochs to train for.
num_gpus=1              # Number of GPUs to parallelize across.
seed=$RANDOM            # Set seed to sepcific number for reproducible results.
prefix_tuning=false     # true for semantic prefix tuning; false for full captions.
input_type="scenegraph" # Text chart representation ("scenegraph", "datatable", or "imageonly").
model_backbone="byt5"     # Model backbone to use ("byt5", "t5", or "bart").

# Update parameters based on arguments passed to the script.
while [[ $1 != "" ]]; do
    case $1 in
    -c | --model_class)
        shift
        input_type=$1
        ;;
    -b | --batch_size)
        shift
        batch_size=$1
        ;;
    -e | --num_epochs)
        shift
        num_epochs=$1
        ;;
    -g | --num_gpus)
        shift
        num_gpus=$1
        ;;
    -i | --input_type)
        shift
        input_type=$1
        ;;
    -m | --model_backbone)
        shift
        model_backbone=$1
        ;;
    -s | --seed)
        shift
        seed=$1
        ;;
    --prefix_tuning)
        prefix_tuning=true
    esac
    shift
done

# Create experiment directory based on modeling parameters.

experiment_name="vistext_${input_type}_${model_backbone}_prefixtuning${prefix_tuning}_seed${seed}"
experiment_directory="$(pwd)/models/${experiment_name}"
if [[ ! -d $experiment_directory ]]; then
    echo "Making output directory at ${experiment_directory}"
    mkdir -p $experiment_directory
fi

if [[ $model_class == "textonly" ]]; then
    
    if [[ $input_type == "image_only" ]]; then
        echo "Invalid argument: ${image_only} cannot be used with ${model_class}."
        usage
    else
    
        if [[ $model_backbone == "t5" ]]; then
            backbone_name="google/byt5-small"
        elif [[ $model_backbone == "t5" ]]; then
            backbone_name="t5-small"
        elif [[ $model_backbone == "bart" ]]; then
            backbone_name="facebook/bart-base"
        else
            echo "Invalid argument: model_backbone is ${model_backbone}."
            usage
        fi

        PYTHONPATH=$PYTHONPATH:./src \
        python run_train_eval_predict.py     \
            --model_name_or_path $model_backbone   \
            --do_train --do_eval     \
            --train_file data/data_train.json \
            --validation_file data/data_validation.json \
            --test_file data/data_test.json     \
            --output_dir $experiment_directory   \
            --max_target_length 512 \
            --per_device_train_batch_size $batch_size     \
            --per_device_eval_batch_size $batch_size \
            --text_column $input_type    \
            --summary_column caption \
            --evaluation_strategy epoch \
            --eval_accumulation_steps 1000 \
            --save_strategy epoch \
            --save_total_limit 2 \
            --load_best_model_at_end=True \
            --num_train_epochs $num_epochs \
            --prefixtuning $prefix_tuning \
            --seed $seed
    fi
    
elif [[ $model_class == "image_guided" ]]; then
    # Convert model backbone name to backbone name used by VLT5
    if [[ $model_backbone == "t5" ]]; then
        vl_backbone_name="t5-small"
    elif [[ $model_backbone == "bart" ]]; then
        vl_backbone_name="facebook/bart-base"
    else
        echo "Invalid argument: model_backbone is ${model_backbone}."
        usage
    fi

    data_directory="$(pwd)/data/data" # data is in vistext/data/
    pretrained_model_path="$(pwd)/models/pretrain/VL${model_backbone^}/Epoch30.pth"

    # Run VisText model.
    PYTHONPATH=$PYTHONPATH:./src \
    python -m torch.distributed.launch \
        --nproc_per_node=$num_gpus \
        src/chart_caption.py \
        --distributed \
        --multiGPU \
        --train \
        --predict \
        --seed  $seed \
        --output $experiment_directory \
        --data_directory $data_directory \
        --load $pretrained_model_path \
        --backbone $backbone_name \
        --batch_size $batch_size \
        --epochs $num_epochs \
        --input_type $input_type \
        --prefix_tuning $prefix_tuning \
else
    echo "Invalid argument: model_class is ${model_class}."
    usage
fi
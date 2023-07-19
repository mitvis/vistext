#!/bin/bash
# Script to download and unzip data files.

function usage {
  echo "usage: $0 [--images] [--scenegraphs] [--image_guided]"
  echo "  --images          Download rasterized chart images."
  echo "  --scenegraphs     Download full scenegraphs."
  echo "  --vl_spec         Download Vega-Lite specs."
  echo "  --image_guided    Download multimodal features and weights."
  exit 1
}

# Default parameter values.
images=false        # true to download images; false otherwise.
scenegraphs=false   # true to download scenegraphs; false otherwise.
vl_spec=false       # true to download vega-lite specs; false otherwise.
mm=false            # true to download multimodal features and weights; false otherwise.


# Update parameters based on arguments passed to the script.
while [[ $1 != "" ]]; do
    case $1 in
    --images)
        images=true
        ;;
    --scenegraphs)
        scenegraphs=true
        ;;
    --vl_spec)
        vl_spec=true
        ;;
    --image_guided)
        mm=true
    esac
    shift
done

# Download tabular data
echo "Downloading tabular data zip"
wget https://vis.csail.mit.edu/vistext/tabular.zip -P ./data/

# Download images
if [[ $images = true ]]; then
    echo "Downloading rasterized chart images zip."
    wget https://vis.csail.mit.edu/vistext/images.zip -P ./data/
fi
# Download scenegraphs
if [[ $scenegraphs = true ]]; then
    echo "Downloading full scenegraphs zip."
    wget https://vis.csail.mit.edu/vistext/scenegraphs.zip -P ./data/
fi
# Download vl_specs
if [[ $vl_spec = true ]]; then
    echo "Downloading full vega-lite specs zip."
    wget https://vis.csail.mit.edu/vistext/vl_spec.zip -P ./data/
fi
# Download multimodal features
if [[ $mm = true ]]; then
    echo "Downloading multimodal features and weights zips."
    wget https://vis.csail.mit.edu/vistext/visual_features.zip -P ./data/
    # mkdir -p ./models/pretrain/VLT5/
    # wget --load-cookies ./tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=100qajGncE_vc4bfjVxxICwz3dwiAxbIZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=100qajGncE_vc4bfjVxxICwz3dwiAxbIZ" -O ./models/pretrain/VLT5/Epoch30.pth && rm -rf ./tmp/cookies.txt
    # mkdir -p ./models/pretrain/VLBart/
    # wget --load-cookies ./tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fTKGCBfMe2eJ_rivPQu0YkNJTNdv_0aq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fTKGCBfMe2eJ_rivPQu0YkNJTNdv_0aq" -O ./models/pretrain/VLBart/Epoch30.pth && rm -rf ./tmp/cookies.txt
    wget https://vis.csail.mit.edu/vistext/vl_weights.zip -P ./models/
fi

echo "Downloading complete. Unzipping archives."

# Unzip tabular data
unzip ./data/tabular.zip -d ./data/
# Unzip images
if [[ $images = true ]]; then
    unzip ./data/images.zip -d ./data/
fi
# Unzip scenegraphs
if [[ $scenegraphs = true ]]; then
    unzip ./data/scenegraphs.zip -d ./data/
fi
# Unzip vl_spec
if [[ $vl_spec = true ]]; then
    unzip ./data/vl_spec.zip -d ./data/
fi
# Unzip multimodal features
if [[ $mm = true ]]; then
    unzip ./data/visual_features.zip -d ./data/
    unzip ./models/vl_weights.zip -d ./models/
fi
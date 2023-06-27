#!/bin/bash
# Script to download and unzip data files.

function usage {
  echo "usage: $0 [--images] [--scenegraphs] [--mm_features]"
  echo "  --images          Download rasterized chart images."
  echo "  --scenegraphs     Download full scenegraphs."
  echo "  --mm_features     Download multimodal features."
  exit 1
}

# Default parameter values.
images=false        # true to download images; false otherwise.
scenegraphs=false   # true to download scenegraphs; false otherwise.
features=false      # true to download multimodal features; false otherwise.

# Update parameters based on arguments passed to the script.
while [[ $1 != "" ]]; do
    case $1 in
    --images)
        images=true
        ;;
    --scenegraphs)
        scenegraphs=true
        ;;
    --mm_features)
        features=true
    esac
    shift
done

# Download tabular data
echo "Downloading tabular data zip"
wget https://vis.csail.mit.edu/vis-text/tabular.zip -P ./data/

# Download images
if [[ $images = true ]]; then
    echo "Downloading rasterized chart images zip."
    wget https://vis.csail.mit.edu/vis-text/images.zip -P ./data/
fi
# Download scenegraphs
if [[ $scenegraphs = true ]]; then
    echo "Downloading full scenegraphs zip."
    wget https://vis.csail.mit.edu/vis-text/scenegraphs.zip -P ./data/
fi
# Download multimodal features
if [[ $features = true ]]; then
    echo "Downloading multimodal scenegraphs zip."
    wget https://vis.csail.mit.edu/vis-text/visual_features.zip -P ./data/
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
# Unzip multimodal features
if [[ $features = true ]]; then
    unzip ./data/visual_features.zip -d ./data/
fi
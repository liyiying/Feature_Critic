#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR

echo "Downloading the trained model ..."
mega-get 'https://mega.nz/#F!rRkgzawL!qoGX4bT3sif88Ho1Ke8j1Q' $DIR
echo "Done. Please verify the integrality of files"

function readfile ()
{
    for file in `ls $1` 
    do
        if [ -d $1"/"$file ]
	then
	    readfile $1"/"$file
	else
	    echo $1"/"$file
    fi
    done
}

readfile $DIR/model_output


echo "Downloading the dataset of PACS ..."
mega-get 'https://mega.nz/#F!jBllFAaI!gOXRx97YHx-zorH5wvS6uw' $DIR/data

echo "Unzipping..."

unzip $DIR/data/PACS/pacs_data.zip
echo "finishing pacs data."
unzip $DIR/data/PACS/pacs_label.zip
echo "finishing pacs label."


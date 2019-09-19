#!/bin/bash

# A script to submit to this competition

if [ "$1" == ""]; then
    echo "Provide model number to submit when calling this script"
fi

$MODEL= $1

kaggle competitions submit ieee-fraud-detection -f $MODEL*.csv -m "$MODEL"

#!/bin/bash
# Script to evaluate single model performance

for id in 14 16 17 18 29 30 37; 
do 
    echo "on run $id"
    ./scripts/run_new.sh $id
done
#!/bin/bash

while getopts k:d:r: flag
do
    case "${flag}" in
        k) key=${OPTARG};;
        d) dataset=${OPTARG};;
        r) remote=${OPTARG};;
    esac
done

scp -i $key $dataset $remote:efs/datasets

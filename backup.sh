#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# This script is used to backup models and data stored on tmp scratch for the unsupervised control project.


# Path to the unsupervised control project
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BACKUP_DIR="/mnt/storage/clwork/users/kew/unsup_ctrl/"

# Backup data
# note: use the trailing slash to copy the contents of the directory
DATA_DIR="$SCRIPT_DIR/resources/data/"
echo "Backing up contents of $DATA_DIR to $BACKUP_DIR/data ..."
# rsync -azn --progress --partial --exclude='.git' resources/data/ kew@midgard.ifi.uzh.ch:/mnt/storage/clwork/users/kew/unsup_ctrl/data
rsync -avz --progress --partial --exclude='.git' $DATA_DIR kew@midgard.ifi.uzh.ch:$BACKUP_DIR/data

# Backup models 
# note: use the trailing slash to copy the contents of the directory
MODEL_DIR="$SCRIPT_DIR/resources/models/"
echo "Backing up contents of $MODEL_DIR to $BACKUP_DIR/models ..."
# rsync -azn --progress --partial --exclude='.git' resources/models/ kew@midgard.ifi.uzh.ch:/mnt/storage/clwork/users/kew/unsup_ctrl/models
rsync -avz --progress --partial --exclude='.git' $MODEL_DIR kew@midgard.ifi.uzh.ch:$BACKUP_DIR/models



# 259512
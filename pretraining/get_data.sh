#!/usr/bin/env bash
# -*- coding: utf-8 -*-

script_dir=$( dirname -- "$( readlink -f -- "$0"; )"; )
base="$script_dir/.."
cd "$base" && echo $(pwd) || exit 1

data_dir="$base/resources/data"
mkdir -p "$data_dir"

echo ""
echo "Downloading data"
echo ""
wget https://battle.shawwn.com/sdb/books1/books1.tar.gz -P "$data_dir"

echo ""
echo "Unpacking data"
echo ""
tar -xvf "$data_dir/books1.tar.gz" -C "$data_dir"
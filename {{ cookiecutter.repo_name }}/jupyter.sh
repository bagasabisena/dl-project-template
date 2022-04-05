#!/usr/bin/env bash
nohup jupyter lab --ip=0.0.0.0 --no-browser > jupyter.log &
echo $!> jupyter.pid

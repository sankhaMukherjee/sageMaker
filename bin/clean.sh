#!/bin/bash

find . -name __pycache__ | xargs rm -rf
find . -name .ipynb_checkpoints | xargs rm -rf

# ------ Clean all results ----------------
cd results
ls | grep $(date +'%Y') | xargs rm -rf  
cd ..

#!/bin/bash

export JUPYTER_TOKEN='veg4ADPkFV5TMf6XbwH3mGB2ScjtKEaWu8pZr7RNhLsy9ndq'
jupyter notebook --ip=0.0.0.0 --port=8887 --no-browser --NotebookApp.token="$JUPYTER_TOKEN" --allow-root


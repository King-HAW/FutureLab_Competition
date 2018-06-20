#!/bin/bash
echo "getting data from King-HAW's repository, make sure you are connected to internet"
wget https://github.com/King-HAW/FutureLab_Competition/releases/download/v1.0/weights-best-inception-resnet-v2-ft-futurelab.hdf5
echo "inception-resnet-v2 download complete!"
wget https://github.com/King-HAW/FutureLab_Competition/releases/download/v1.0/weights-best-inception-v3-ft-futurelab-150.hdf5
echo "inception-v3 download complete!"


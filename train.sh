export raw_data_base=$HOME/workspace/SegUnc_project/Pytorch-UNet/data/
echo "data directory is: "$raw_data_base
nohup python -u train.py -e 100 -b 2 > nohup_train.out 2>&1 &

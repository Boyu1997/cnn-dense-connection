''' training '''

# replication for the original densenet configuration
python3 main.py --cross_block_rate=0 --end_block_reduction_rate=0.5 --save_folder="densenet"

# experiments to add cross-block connection base on densenet
python3 main.py --cross_block_rate=0.2 --end_block_reduction_rate=0.5 --save_folder="cbr=0.2-rdc=0.5"
python3 main.py --cross_block_rate=0.4 --end_block_reduction_rate=0.5 --save_folder="cbr=0.4-rdc=0.5"
python3 main.py --cross_block_rate=0.6 --end_block_reduction_rate=0.5 --save_folder="cbr=0.6-rdc=0.5"


# replication for original condensenet without learned group convolution
python3 main.py --cross_block_rate=1 --end_block_reduction_rate=0.1 --save_folder="condensenet"

# experiments to reduce cross-block connection base on condensenet
python3 main.py --cross_block_rate=0.8 --end_block_reduction_rate=0.1 --save_folder="cbr=0.8-rdc=0.1"
python3 main.py --cross_block_rate=0.6 --end_block_reduction_rate=0.1 --save_folder="cbr=0.6-rdc=0.1"
python3 main.py --cross_block_rate=0.4 --end_block_reduction_rate=0.1 --save_folder="cbr=0.4-rdc=0.1"


''' testing '''

cd evaluation

# densenet
python3 model_eval.py --directory='../save/densenet'
python3 model_eval.py --directory='../save/cbr=0.2-rdc=0.5'
python3 model_eval.py --directory='../save/cbr=0.4-rdc=0.5'
python3 model_eval.py --directory='../save/cbr=0.6-rdc=0.5'

# condensenet
python3 model_eval.py --directory='../save/condensenet'
python3 model_eval.py --directory='../save/cbr=0.8-rdc=0.1'
python3 model_eval.py --directory='../save/cbr=0.6-rdc=0.1'
python3 model_eval.py --directory='../save/cbr=0.4-rdc=0.1'

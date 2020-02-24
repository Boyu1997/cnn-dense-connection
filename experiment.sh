# replication for the original densenet configuration
python3 main.py --cross_block_rate=0 --reduction=0.5 --save_folder="densenet-original"

# experiments to add cross-block connection base on densenet
python3 main.py --cross_block_rate=0.2 --reduction=0.5 --save_folder="cbr=0.2-reduction=0.5"
python3 main.py --cross_block_rate=0.4 --reduction=0.5 --save_folder="cbr=0.4-reduction=0.5"
python3 main.py --cross_block_rate=0.6 --reduction=0.5 --save_folder="cbr=0.6-reduction=0.5"


# replication for original condensenet without learned group convolution
python3 main.py --cross_block_rate=1 --reduction=0.1 --save_folder="condensenet-original"

# experiments to reduce cross-block connection base on condensenet
python3 main.py --cross_block_rate=0.8 --reduction=0.1 --save_folder="cbr=0.8-reduction=0.1"
python3 main.py --cross_block_rate=0.6 --reduction=0.1 --save_folder="cbr=0.6-reduction=0.1"
python3 main.py --cross_block_rate=0.4 --reduction=0.1 --save_folder="cbr=0.4-reduction=0.1"

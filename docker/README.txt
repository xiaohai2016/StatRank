Functionality in this folder only works with LinkedIn internal infrastructure

Commands to generate result files:

# dataset - MQ2007, Ranking function - Single layer, Divergence - listed below
python -u main.py --listnet --listmle --alpha-divergence --weighted-kl-divergence | tee ../mq2007-onelayer-multi.txt

# dataset - MQ2008, Ranking function - Single layer, Divergence - listed below
python -u main.py --listnet --listmle --alpha-divergence --weighted-kl-divergence --data-set mq2008 | tee ../mq2008-onelayer-multi.txt

# dataset - MQ2007, Ranking function - Single layer, Divergence - alpha divergence with entropy regularization
python -u main.py --alpha-and-entropy | tee ../mq2007-onelayer-alpha-entropy.txt

# dataset - MQ2008, Ranking function - Single layer, Divergence - alpha divergence with entropy regularization
python -u main.py --alpha-and-entropy --data-set mq2008 | tee ../mq2008-onelayer-alpha-entropy.txt

# dataset - MQ2007, Ranking function - Three layer, Divergence - listed below
python -u main.py --listnet --listmle --alpha-divergence --weighted-kl-divergence --model three_layer | tee ../mq2007-threelayer-multi.txt

# dataset - MQ2008, Ranking function - Three layer, Divergence - listed below
python -u main.py --listnet --listmle --alpha-divergence --weighted-kl-divergence --data-set mq2008 --model three_layer | tee ../mq2008-threelayer-multi.txt

# dataset - MQ2007, Ranking function - Three layer, Divergence - alpha divergence with entropy regularization
python -u main.py --alpha-and-entropy --model three_layer | tee ../mq2007-threelayer-alpha-entropy.txt

# dataset - MQ2008, Ranking function - Three layer, Divergence - alpha divergence with entropy regularization
python -u main.py --alpha-and-entropy --data-set mq2008 --model three_layer | tee ../mq2008-threelayer-alpha-entropy.txt
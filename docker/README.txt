Functionality in this folder only works with LinkedIn internal infrastructure

Commands to generate result files:

# dataset - MQ2007, Ranking function - Single layer, Divergence - listed below
nohup python -u main.py --listnet --listmle --alpha-divergence --weighted-kl-divergence > ../mq2007-onelayer-multi.txt &

# dataset - MQ2008, Ranking function - Single layer, Divergence - listed below
nohup python -u main.py --listnet --listmle --alpha-divergence --weighted-kl-divergence --data-set mq2008 > ../mq2008-onelayer-multi.txt &

# dataset - MQ2007, Ranking function - Single layer, Divergence - alpha divergence with entropy regularization
nohup python -u main.py --alpha-and-entropy > ../mq2007-onelayer-alpha-entropy.txt &

# dataset - MQ2008, Ranking function - Single layer, Divergence - alpha divergence with entropy regularization
nohup python -u main.py --alpha-and-entropy --data-set mq2008 > ../mq2008-onelayer-alpha-entropy.txt &

# dataset - MQ2007, Ranking function - Three layer, Divergence - listed below
nohup python -u main.py --listnet --listmle --alpha-divergence --weighted-kl-divergence --model three_layer > ../mq2007-threelayer-multi.txt &

# dataset - MQ2008, Ranking function - Three layer, Divergence - listed below
nohup python -u main.py --listnet --listmle --alpha-divergence --weighted-kl-divergence --data-set mq2008 --model three_layer > ../mq2008-threelayer-multi.txt &

# dataset - MQ2007, Ranking function - Three layer, Divergence - alpha divergence with entropy regularization
nohup python -u main.py --alpha-and-entropy --model three_layer > ../mq2007-threelayer-alpha-entropy.txt &

# dataset - MQ2008, Ranking function - Three layer, Divergence - alpha divergence with entropy regularization
nohup python -u main.py --alpha-and-entropy --data-set mq2008 --model three_layer > ../mq2008-threelayer-alpha-entropy.txt &

# rerun for MQ2007 three_layer
nohup python -u main.py --alphas -0.9 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --lambdas -0.9 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --alpha-divergence --weighted-kl-divergence --model three_layer > ../mq2007-threelayer-alpha-and-weighted-kl.txt &
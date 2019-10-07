# Learning to Rank with Alpha-divergence and Entropy Regularization

## Steps to reproduce results in the paper

1. Create a folder named *resources* under the root of the cloned code tree;
2. Download MQ2007 and MQ2008 datasets and decompress them under the *resources* folder.
3. Prepare your execution environment with PyTorch 1.1.0 and Scikit.
4. Run the following commands to generate the test results:

```bash
# dataset - MQ2007, Ranking function - Single layer, Divergence - listed below
nohup python -u main.py --listnet --listmle --alpha-divergence --weighted-kl-divergence > ../mq2007-onelayer-multi.txt &

# dataset - MQ2008, Ranking function - Single layer, Divergence - listed below
nohup python -u main.py --listnet --listmle --alpha-divergence --weighted-kl-divergence --data-set mq2008 > ../mq2008-onelayer-multi.txt &

# dataset - MQ2007, Ranking function - Single layer, Divergence - alpha divergence with entropy regularization
nohup python -u main.py --alpha-and-entropy > ../mq2007-onelayer-alpha-entropy.txt &

# dataset - MQ2008, Ranking function - Single layer, Divergence - alpha divergence with entropy regularization
nohup python -u main.py --alpha-and-entropy --data-set mq2008 > ../mq2008-onelayer-alpha-entropy.txt &

# dataset - MQ2007, Ranking function - Three layer, Divergence - listed below
nohup python -u main.py --listnet --listmle > ../mq2007-threelayer-multi.txt &

# dataset - MQ2008, Ranking function - Three layer, Divergence - listed below
nohup python -u main.py --listnet --listmle --alpha-divergence --weighted-kl-divergence --data-set mq2008 --model three_layer > ../mq2008-threelayer-multi.txt &

# dataset - MQ2007, Ranking function - Three layer, Divergence - alpha divergence with entropy regularization
#nohup python -u main.py --alpha-and-entropy --model three_layer > ../mq2007-threelayer-alpha-entropy.txt &
# I once did run the above command and later abandoned it. It takes many days to complete and the hyper parameters are not good.

# dataset - MQ2008, Ranking function - Three layer, Divergence - alpha divergence with entropy regularization
nohup python -u main.py --alpha-and-entropy --data-set mq2008 --model three_layer > ../mq2008-threelayer-alpha-entropy.txt &

# rerun for MQ2007 three_layer
nohup python -u main.py --alphas -0.9 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --lambdas -0.9 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --alpha-divergence --model three_layer > ../mq2007-threelayer-alpha.txt &
nohup python -u main.py --alphas -0.9 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --lambdas -0.9 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --weighted-kl-divergence --model three_layer > ../mq2007-threelayer-weighted-kl.txt &

nohup python -u main.py --alphas -0.9 --lambdas -0.9 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --alpha-and-entropy --model three_layer > ../mq2007-alpha-and-entropy-neg0.9.txt &
nohup python -u main.py --alphas -0.6 --lambdas -0.9 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --alpha-and-entropy --model three_layer > ../mq2007-alpha-and-entropy-neg0.6.txt &
nohup python -u main.py --alphas -0.4 --lambdas -0.9 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --alpha-and-entropy --model three_layer > ../mq2007-alpha-and-entropy-neg0.4.txt &
nohup python -u main.py --alphas -0.2 --lambdas -0.9 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --alpha-and-entropy --model three_layer > ../mq2007-alpha-and-entropy-neg0.2.txt &
nohup python -u main.py --alphas -0.1 --lambdas -0.9 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --alpha-and-entropy --model three_layer > ../mq2007-alpha-and-entropy-neg0.1.txt &
nohup python -u main.py --alphas 0.1 --lambdas -0.9 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --alpha-and-entropy --model three_layer > ../mq2007-alpha-and-entropy-0.1.txt &
nohup python -u main.py --alphas 0.2 --lambdas -0.9 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --alpha-and-entropy --model three_layer > ../mq2007-alpha-and-entropy-0.2.txt &
nohup python -u main.py --alphas 0.3 --lambdas -0.9 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --alpha-and-entropy --model three_layer > ../mq2007-alpha-and-entropy-0.3.txt &
nohup python -u main.py --alphas 0.4 --lambdas -0.9 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --alpha-and-entropy --model three_layer > ../mq2007-alpha-and-entropy-0.4.txt &
nohup python -u main.py --alphas 0.5 --lambdas -0.9 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --alpha-and-entropy --model three_layer > ../mq2007-alpha-and-entropy-0.5.txt &
nohup python -u main.py --alphas 0.6 --lambdas -0.9 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --alpha-and-entropy --model three_layer > ../mq2007-alpha-and-entropy-0.6.txt &
nohup python -u main.py --alphas 0.7 --lambdas -0.9 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --alpha-and-entropy --model three_layer > ../mq2007-alpha-and-entropy-0.7.txt &
```
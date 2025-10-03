#from trainer import *
import subprocess
import multiprocessing

def hidden_dim_task(i):
    sh = "python train.py --hidden_dim {:d}".format(i)
    # print(sh)
    subprocess.run(sh, shell=True)

def dropout_task(i):
    sh = "python train.py --hidden_dim 256 --dropout {:f}".format(i)
    subprocess.run(sh, shell=True)

def attention_dim_task(i):
    sh = "python train.py --hidden_dim 128 --dropout 0.2 --n_batch 50 --n_tbatch 32 --attn_dim {:d}".format(i)
    subprocess.run(sh, shell=True)

def layer_num_task(i):
    sh = "python train.py --hidden_dim 128 --dropout 0.3 --attn_dim 5 --n_batch 50 --n_tbatch 32 --n_layer {:d}".format(i)
    subprocess.run(sh, shell=True)

def activation_task(i):
    sh = "python train.py --hidden_dim 128 --dropout 0.3 --attn_dim 5 --n_batch 50 --n_tbatch 32 --n_layer 3 --act {:s}".format(i)
    subprocess.run(sh, shell=True)

def different_split_task(i):
    sh = "python train.py --hidden_dim 128  --n_batch 32 --n_tbatch 32  --n_epoch 50  --data_path {:s}".format(i)
    subprocess.run(sh, shell=True)


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=1)

    hidden_dim = [8,16,32,64,128,256,512]
    dropout = [0.1,0.3,0.4,0.5]
    attn_dim = [5]
    n_layer = [5]
    data_path = ['data/onlydrds_' + str(i) for i in range(1,10)]


    # results = pool.map(hidden_dim_task, hidden_dim)
    # results = pool.map(attention_dim_task, attn_dim)
    # results = pool.map(layer_num_task, n_layer)
    # results = pool.map(activation_task, act)
    # results = pool.map(dropout_task, dropout)
    results = pool.map(different_split_task,data_path)

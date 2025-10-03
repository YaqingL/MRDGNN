import os
import argparse
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel
from utils import seed_everything
import logging


def logger_config():
    if not os.path.exists('logs_'):
        os.mkdir('logs_')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler('logs_/{}.txt'.format(args.prefix_file), mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


def Options():
    parser = argparse.ArgumentParser(description="Parser for MRDGNN")
    parser.add_argument('--data_path', type=str, default='data/onlydrds/')
    parser.add_argument('--seed', type=str, default=1234)

    # hyper-parameters for optimizer
    parser.add_argument('--lr', type=float, default=0.0036)
    parser.add_argument('--decay_rate', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--warm_up_step', type=int, default=1000)
    parser.add_argument("--max_batches", default=300000, type=int)
    parser.add_argument('--lamb', type=float, default=0.000017)

    # hyper-parameters for model
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--attn_dim', type=int, default=5)
    parser.add_argument('--n_layer', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--n_epoch', type=int, default=50)

    # hyper-parameters for training
    parser.add_argument('--n_batch', type=int, default=64)
    parser.add_argument('--n_tbatch', type=int, default=50)

    # for ablation study
    parser.add_argument('--use_pretrain', type=bool, default=True)
    parser.add_argument('--layer_attention', type=bool, default=True)

    args = parser.parse_args()

    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]

    args.prefix_file = (args.hidden_dim.__str__() + '_' +
                   args.attn_dim.__str__() + '_' +
                   args.n_layer.__str__() + '_' +
                   args.dropout.__str__() + '_'+
                   dataset + '_' )


    return args

if __name__ == '__main__':
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    args = Options()
    logger_config()

    seed_everything(seed=args.seed)

    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]
   
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


    # gpu = select_gpu()
    torch.cuda.set_device("cuda:0")
    # print('gpu:', gpu)

    if args.use_pretrain:
        loader = DataLoader(args.data_path, embedding_file='transe_drds_embeddings.txt')  # use pretrain embedding
    else:
        loader = DataLoader(args.data_path, embedding_file=None)

    args.n_ent = loader.n_ent
    args.n_rel = loader.n_rel

    # print(args.use_pretrain, args.layer_attention)

    config_str = '%s, %s, %.4f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (str(args.use_pretrain), str(args.layer_attention), args.lr, args.decay_rate, args.lamb, args.hidden_dim, args.attn_dim, args.n_layer, args.n_batch, args.dropout, args.act)
    logging.info(config_str)

    # with open(args.perf_file, 'a+') as f:
    #     f.write(config_str)

    model = BaseModel(args, loader)

    start_epoch = 0
    # checkpoint_path = f"checkpoint/layer{args.n_layer}_{dataset}_indication.pth"
    # if os.path.exists(checkpoint_path):
    #     print(f"Checkpoint detected. Loading from {checkpoint_path}...")
    #     model.load_model(checkpoint_path)
    #     start_epoch = model.current_epoch + 1
    #     print(f"Resuming training from epoch {start_epoch}")
    # else:
    #     print("No checkpoint found. Starting training from scratch.")

    # best_mrr = 0
    best_mrr_per_relation = {'indication': 0, 'contraindication': 0}
    best_str_per_relation = {'indication': '', 'contraindication': ''}

    for epoch in range(start_epoch, args.n_epoch):
        logging.info(f'Epoch:{epoch}')
        model.current_epoch = epoch
        mrr_per_relation, out_str = model.train_batch()

        logging.info(out_str)


        for rel, mrr in mrr_per_relation.items():
            if mrr > best_mrr_per_relation[rel]:
                best_mrr_per_relation[rel] = mrr
                best_str_per_relation[rel] = out_str
                logging.info(f'{epoch}\t[Best {rel}] ' + best_str_per_relation[rel])  # best metrics for each relation

                # save best model
                best_model_path = f"checkpoint/{dataset}_{rel}.pth"
                model.save_model(best_model_path)
                logging.info(f"Model saved at {best_model_path} (Best MRR for {rel}: {mrr:.4f})")

    logging.info(f"Final Best Results in {dataset}_{args.n_layer}layer\n")


    for rel, best_mrr in best_mrr_per_relation.items():
        # line1 = f'Best MRR for {rel}: {best_mrr:.4f}'
        # line2 = f'Best metrics for {rel}: {best_str_per_relation[rel]}'
        #
        # print(line1)
        # print(line2)

        # f.write(line1 + '\n')
        # f.write(line2 + '\n')
        logging.info(f'Best MRR for {rel}: {best_mrr:.4f}')
        logging.info(f'Best metrics for {rel}: {best_str_per_relation[rel]}')

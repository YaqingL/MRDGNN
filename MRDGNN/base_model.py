import torch
import numpy as np
import time

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from models import MRD_GNN
from utils import *

class BaseModel(object):
    def __init__(self, args, loader):
        self.model = MRD_GNN(args, loader, loader.entity_embeddings, use_layer_attention=args.layer_attention)

        self.model.cuda()

        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_rel = loader.n_rel
        self.n_batch = args.n_batch
        self.n_tbatch = args.n_tbatch

        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test  = loader.n_test
        self.n_layer = args.n_layer
        self.args = args


        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.smooth = 1e-5
        self.t_time = 0

        self.current_epoch = 0

    def train_batch(self,):
        epoch_loss = 0
        i = 0

        batch_size = self.n_batch
        n_batch = self.loader.n_train // batch_size + (self.loader.n_train % batch_size > 0)

        # indication_id = self.loader.relation2id['indication']
        # contraindication_id = self.loader.relation2id['contraindication']

        t_time = time.time()
        self.model.train()

        print("n_batch: {}".format(n_batch))
        #for i in range(2):
        for i in range(n_batch):
            if i % 100 == 0:
                print("batch: {}".format(i))
            start = i*batch_size
            end = min(self.loader.n_train, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            triple = self.loader.get_batch(batch_idx)

            # indication_mask = (triple[:, 1] == indication_id)
            # contraindication_mask = (triple[:, 1] == contraindication_id)
            #
            # indication_triples = triple[indication_mask]
            # contraindication_triples = triple[contraindication_mask]

            #self.model.zero_grad()

            scores = self.model(triple[:,0], triple[:,1])
            pos_scores = scores[[torch.arange(len(scores)).cuda(),torch.LongTensor(triple[:,2]).cuda()]]
            max_n = torch.max(scores, 1, keepdim=True)[0]
            loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1)))

            lr = adjust_learning_rate(optimizer=self.optimizer, epoch=i, lr=self.args.lr,
                                      warm_up_step=self.args.warm_up_step,
                                      max_update_step=self.args.max_batches)


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            logging.info('batch: {}, learning rate: {}'.format(i, lr))

            # # indication
            # if len(indication_triples) > 0:
            #     scores_indication = self.model(indication_triples[:, 0], indication_triples[:, 1])
            #     pos_scores_indication = scores_indication[
            #         torch.arange(len(scores_indication)).cuda(),
            #         torch.LongTensor(indication_triples[:, 2]).cuda()
            #     ]
            #     max_n_indication = torch.max(scores_indication, 1, keepdim=True)[0]
            #     loss_indication = torch.sum(
            #         -pos_scores_indication + max_n_indication + torch.log(
            #             torch.sum(torch.exp(scores_indication - max_n_indication), 1))
            #     )
            # else:
            #     loss_indication = torch.zeros(1, device='cuda', requires_grad=True)
            #
            # #  contraindication
            # if len(contraindication_triples) > 0:
            #     scores_contraindication = self.model(contraindication_triples[:, 0], contraindication_triples[:, 1])
            #     pos_scores_contraindication = scores_contraindication[
            #         torch.arange(len(scores_contraindication)).cuda(),
            #         torch.LongTensor(contraindication_triples[:, 2]).cuda()
            #     ]
            #     max_n_contraindication = torch.max(scores_contraindication, 1, keepdim=True)[0]
            #     loss_contraindication = torch.sum(
            #         -pos_scores_contraindication + max_n_contraindication + torch.log(
            #             torch.sum(torch.exp(scores_contraindication - max_n_contraindication), 1))
            #     )
            # else:
            #     loss_contraindication = torch.zeros(1, device='cuda', requires_grad=True)
            #
            # # total loss
            # loss = loss_indication + loss_contraindication
            # loss.backward()
            # self.optimizer.step()

            # avoid NaN
            time1 = time.time()
            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            time2 = time.time()
            print("time (parameters): {}".format(time2-time1))

            epoch_loss += loss.item()
        self.scheduler.step()
        self.t_time += time.time() - t_time

        valid_per_mrr, out_str = self.evaluate()
        # test_per_mrr, out_str = self.evaluate()
        self.loader.shuffle_train()
        return valid_per_mrr, out_str

    def evaluate(self, ):
        global scores, objs, rels
        batch_size = self.n_tbatch
        relations_to_evaluate = ['contraindication', 'indication']
        metrics_per_relation = {rel: [] for rel in relations_to_evaluate}
        scores_per_relation = {rel: [] for rel in relations_to_evaluate}
        objs_per_relation = {rel: [] for rel in relations_to_evaluate}

        # n_data = self.n_valid
        n_batch = self.n_valid // batch_size + (self.n_valid % batch_size > 0)
        self.model.eval()
        i_time = time.time()

        for i in range(n_batch):
            start = i * batch_size
            end = min(self.n_valid, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='valid')
            scores = self.model(subs, rels, mode='valid').data.cpu().numpy()

            filters = []
            for j in range(len(subs)):
                filt = self.loader.filters[(subs[j], rels[j])]
                filt_1hot = np.zeros((self.n_ent,))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)

            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)

            # Categorize ranking by relation
            for rel, rank in zip(rels, ranks):
                if rel in self.loader.id2relation:
                    rel_name = self.loader.id2relation[rel]
                    if rel_name in metrics_per_relation:
                        metrics_per_relation[rel_name].append(rank)

            # Categorize scores&objs by relation
            for rel, score, obj in zip(rels, scores, objs):
                if rel in self.loader.id2relation:
                    rel_name = self.loader.id2relation[rel]
                    if rel_name in scores_per_relation:
                        scores_per_relation[rel_name].append(score)
                        objs_per_relation[rel_name].append(obj)

        v_mrr_per_relation = {}# Calculate metrics for each relation
        v_auc_per_relation = {}
        v_aupr_per_relation = {}
        out_str = ''

        for rel_name, ranks in metrics_per_relation.items():
            if ranks:
                ranking = np.array(ranks)
                v_mrr, v_h1, v_h3, v_h10 = cal_performance(ranking)
                v_mrr_per_relation[rel_name] = v_mrr
                out_str += f'[VALID - {rel_name}] MRR: {v_mrr:.4f}, H@1: {v_h1:.4f}, H@3: {v_h3:.4f}, H@10: {v_h10:.4f}\n'
                # calculate AUC and AUPR
                rel_scores = np.array(scores_per_relation[rel_name])
                rel_objs = np.array(objs_per_relation[rel_name])

                if rel_scores.size > 0 and rel_objs.size > 0:
                    v_auc, v_aupr = cal_auc_aupr(rel_scores.flatten(), rel_objs.flatten())
                    v_auc_per_relation[rel_name] = v_auc
                    v_aupr_per_relation[rel_name] = v_aupr
                    out_str += f'[VALID - {rel_name}] AUC: {v_auc:.4f}, AUPR: {v_aupr:.4f}\n'

        # n_data = self.n_test
        n_batch = self.n_test // batch_size + (self.n_test % batch_size > 0)
        metrics_per_relation = {rel: [] for rel in relations_to_evaluate}
        scores_per_relation = {rel: [] for rel in relations_to_evaluate}
        objs_per_relation = {rel: [] for rel in relations_to_evaluate}
        self.model.eval()

        for i in range(n_batch):
            start = i * batch_size
            end = min(self.n_test, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='test')
            scores = self.model(subs, rels, mode='test').data.cpu().numpy()

            filters = []
            for j in range(len(subs)):
                filt = self.loader.filters[(subs[j], rels[j])]
                filt_1hot = np.zeros((self.n_ent,))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)

            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)

            for rel, rank in zip(rels, ranks):
                if rel in self.loader.id2relation:
                    rel_name = self.loader.id2relation[rel]
                    if rel_name in metrics_per_relation:
                        metrics_per_relation[rel_name].append(rank)

            for rel, score, obj in zip(rels, scores, objs):
                if rel in self.loader.id2relation:
                    rel_name = self.loader.id2relation[rel]
                    if rel_name in scores_per_relation:
                        scores_per_relation[rel_name].append(score)
                        objs_per_relation[rel_name].append(obj)

        t_mrr_per_relation = {}
        t_auc_per_relation = {}
        t_aupr_per_relation = {}

        for rel_name, ranks in metrics_per_relation.items():
            if ranks:
                ranking = np.array(ranks)
                t_mrr, t_h1, t_h3, t_h10 = cal_performance(ranking)
                t_mrr_per_relation[rel_name] = t_mrr
                out_str += f'[TEST - {rel_name}] MRR: {t_mrr:.4f}, H@1: {t_h1:.4f}, H@3: {t_h3:.4f}, H@10: {t_h10:.4f}\n'

                # calculate AUC and AUPR
                rel_scores = np.array(scores_per_relation[rel_name])
                rel_objs = np.array(objs_per_relation[rel_name])

                if rel_scores.size > 0 and rel_objs.size > 0:
                    t_auc, t_aupr = cal_auc_aupr(rel_scores.flatten(), rel_objs.flatten())
                    t_auc_per_relation[rel_name] = t_auc
                    t_aupr_per_relation[rel_name] = t_aupr
                    out_str += f'[TEST - {rel_name}] AUC: {t_auc:.4f}, AUPR: {t_aupr:.4f}\n'


        i_time = time.time() - i_time
        out_str += f'[TIME] train: {self.t_time:.4f}, inference: {i_time:.4f}\n'

        return v_mrr_per_relation, out_str

    def save_model(self, file_path="checkpoint.pth"):
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(checkpoint, file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path="checkpoint.pth"):
        checkpoint = torch.load(file_path)
        self.current_epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Model loaded from {file_path}, starting from epoch {self.current_epoch}")



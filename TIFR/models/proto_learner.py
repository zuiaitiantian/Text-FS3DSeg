""" ProtoNet with/without attention learner for Few-shot 3D Point Cloud Semantic Segmentation


"""
import torch
from torch import optim
from torch.nn import functional as F

from models.protonet_TIFR import ProtoNet
from utils.checkpoint_util import load_pretrain_checkpoint, load_model_checkpoint


class ProtoLearner(object):

    def __init__(self, args, mode='train'):

        # init model and optimizer
        self.model = ProtoNet(args)
        print(self.model)
        
        if torch.cuda.is_available():
            self.model.cuda()

        if mode == 'train':
            
            self.optimizer = torch.optim.AdamW(
            [{'params': self.model.encoder.parameters(), 'lr': args.lr * 0.1},
             {'params': self.model.base_learner.parameters()},
             {'params': self.model.att_learner.parameters()},

             {'params': self.model.MLP_feat1.parameters()},
             {'params': self.model.CAFP.parameters()},
             
             {'params': self.model.text_MLP.parameters()},
             
             {'params': self.model.TAMA.parameters()},
             {'params': self.model.self_att.parameters()},
             # {'params': self.model.cross_att.parameters()},
             

             {'params': self.model.transformer.parameters(), 'lr': args.trans_lr},
             # ], lr=args.lr, weight_decay=1e-3)
             ], lr=args.lr, weight_decay=5e-4, betas=(0.9, 0.98), eps=1e-8)
            

            # set learning rate scheduler
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=args.n_iters // 3, 
                T_mult=1,             
                eta_min=1e-6
            )
            
            
            # load pretrained model for point cloud encoding
            self.model = load_pretrain_checkpoint(self.model, args.pretrain_checkpoint_path)
        elif mode == 'test':
            # Load model checkpoint
            self.model = load_model_checkpoint(self.model, args.model_checkpoint_path, mode='test')
        else:
            raise ValueError('Wrong GMMLearner mode (%s)! Option:train/test' %mode)

    def train(self, data, sampled_classes):
        """
        Args:
            data: a list of torch tensors wit the following entries.
            - support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            - support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
            - query_x: query point clouds with shape (n_queries, in_channels, num_points)
            - query_y: query labels with shape (n_queries, num_points)
        """

        [support_x, support_y, query_x, query_y] = data
        self.model.train()

        query_logits, loss = self.model(support_x, support_y, query_x, query_y, sampled_classes)

        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
        
        self.optimizer.step()
        self.lr_scheduler.step()

        query_pred = F.softmax(query_logits, dim=1).argmax(dim=1)
        correct = torch.eq(query_pred, query_y).sum().item()  # including background class
        accuracy = correct / (query_y.shape[0]*query_y.shape[1])

        return loss, accuracy

    def test(self, data, sampled_classes):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points), each point \in {0,1}.
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
        """
        [support_x, support_y, query_x, query_y] = data
        self.model.eval()

        with torch.no_grad():
            logits, loss = self.model(support_x, support_y, query_x, query_y, sampled_classes)
            pred = F.softmax(logits, dim=1).argmax(dim=1)
            correct = torch.eq(pred, query_y).sum().item()
            accuracy = correct / (query_y.shape[0]*query_y.shape[1])

        return pred, loss, accuracy
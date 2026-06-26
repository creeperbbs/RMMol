# -*- coding: utf-8 -*-
# @Author: Hongzhi Yao
# @Date:   2025-06-21 09:15:55
# @Last Modified by:   Hongzhi Yao
# @Last Modified time: 2025-7-5 18:33:14
import os
import shutil
import sys
from functools import partial
import argparse

import pandas as pd
import torch
import random
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, load_from_disk
import yaml
import getpass
import numpy as np
from datetime import datetime
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch.distributed as dist
import pytorch_lightning as pl
import os
from ..model.rmmol_gnn_model import GNNDecoder, GNN
from ..utils.loss import NTXentLoss, sce_loss
from torch.cuda.amp import autocast, GradScaler
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only, seed
from pytorch_lightning.callbacks import LearningRateMonitor
apex_support = False
# from apex import optimizers
from ..loader.loader import MoleculeProcessor,molcae_embed
from torch.distributed import broadcast_object_list
import subprocess

import glob
SEED = 2025
seed.seed_everything(SEED)
# try:
#     sys.path.append('./apex')
#     # from apex import amp
#
#     apex_support = True
# except:
#     print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
#     apex_support = False

class CheckpointEveryNSteps(pl.Callback):
    """
        Save a checkpoint every N steps, instead of Lightning's default that checkpoints
        based on validation loss.
    """

    def __init__(self, save_step_frequency=-1,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
        ):
        """
        Args:
        save_step_frequency: how often to save in steps
        prefix: add a prefix to the name, only used if
        use_modelcheckpoint_filename=False
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if global_step % self.save_step_frequency == 0 and self.save_step_frequency > 10:

            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
class ModelCheckpointAtEpochEnd(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics['epoch'] = trainer.current_epoch
        if trainer.disable_validation:
            trainer.checkpoint_callback.on_validation_end(trainer, pl_module)


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch


def load_qm9_data(csv_path, smiles_col='smiles', target_cols=None):
    df = pd.read_csv(csv_path)
    smiles = df[smiles_col].tolist()
    if target_cols is None:
        target_cols = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
    targets = df[target_cols].values.astype(np.float32)
    return smiles, targets

train_csv = r'\smiles\qm9\qm9_small_train.csv'
valid_csv = r':\smiles\qm9\qm9_small_valid.csv'
test_csv = r'E:\smiles\qm9\qm9_small_test.csv'

smiles_train, y_train_raw = load_qm9_data(train_csv)
smiles_valid, y_valid_raw = load_qm9_data(valid_csv)
smiles_test, y_test_raw = load_qm9_data(test_csv)

scaler = StandardScaler()
y_train = scaler.fit_transform(y_train_raw)
y_valid = scaler.transform(y_valid_raw)
y_test = scaler.transform(y_test_raw)


def evaluate_linear_probe(encoder, smiles_train, y_train, smiles_test, y_test, device='cuda'):

    Z_train = molcae_embed(encoder, smiles_train, device=device).cpu().numpy()
    Z_test = molcae_embed(encoder, smiles_test, device=device).cpu().numpy()
    print(Z_train.shape)
    maes = []
    for i in range(y_train.shape[1]):
        reg = Ridge(alpha=1.0)
        reg.fit(Z_train, y_train[:, i])
        pred = reg.predict(Z_test)
        mae = mean_absolute_error(y_test[:, i], pred)
        maes.append(mae)
    return np.mean(maes)
@rank_zero_only
def remove_tree(cachefiles):
    if type(cachefiles) == type([]):
        #if cachefiles are identical remove all but one file path
        cachefiles = list(set(cachefiles))
        for cache in cachefiles:
            shutil.rmtree(cache)
    else:
        shutil.rmtree(cachefiles)

def random_remask(dec_mask_token, rep, x, device, remask_rate=0.5):
    num_nodes = x.num_nodes
    perm = torch.randperm(num_nodes,device = device)
    num_remask_nodes = int(remask_rate * num_nodes)
    remask_nodes = perm[:num_remask_nodes]
    
    rep_new = torch.zeros_like(rep)
    rep_new.copy_(rep)
    
    mask = torch.zeros_like(rep_new, dtype=torch.bool)
    mask[remask_nodes] = True
    
    rep_new = torch.where(
        mask,
        dec_mask_token.expand_as(rep_new),  
        rep_new
    )
    
    return rep_new, remask_nodes, None

class MoleculeModule(pl.LightningDataModule):
    def __init__(self,  max_len, data_path, train_args):
        super().__init__()
        self.data_path = ['pubchem']
        self.train_args = train_args  # dict with keys {'batch_size', 'shuffle', 'num_workers', 'pin_memory'}
        print(train_args)
        self.data_collector = MoleculeProcessor()
        
        
    def prepare_data(self):
        pass

    def get_cache(self):
        return self.cache_files
    def setup(self, stage=None):
        zinc_path = r'\Data\benchmark_smi'
        zinc_files = [f for f in glob.glob(os.path.join(zinc_path, '*.smi'))]
        for zfile in zinc_files:
            print(zfile)
        data_path = {'train': zinc_files}
        dataset_dict = load_dataset(
            './zinc_script.py',
            data_files=data_path,
            cache_dir=os.path.join('E:/生信/scMCP/DrugGCL/tmp', getpass.getuser(), 'zinc'),
            split='train'
        )

        # 记录缓存路径（原始实现）
        self.cache_files = []
        for cache in dataset_dict.cache_files:
            tmp = '/'.join(cache['filename'].split('/'))
            self.cache_files.append(tmp)

        train_test_split = dataset_dict.train_test_split(test_size=0.05, seed=42)
        self.pubchem_train = train_test_split['train']
        self.pubchem_val = train_test_split['test']


    def train_dataloader(self):
        return DataLoader(
            self.pubchem_train,
            collate_fn=self.data_collector.process, drop_last=True,
            shuffle=True,         
            **self.train_args
        )

    def val_dataloader(self):
        return DataLoader(
            self.pubchem_val,
            collate_fn=self.data_collector.process, drop_last=True,
            shuffle=False,        
            **self.train_args
        )
    def test_dataloader(self):
        return []
class MolGATMAE(pl.LightningModule):
    
    def __init__(self,  config):
        super().__init__() 
        self.config = config
        self.lambda_divergence = config.get('lambda_divergence', 0.0)
        self._num_remasking =2
        self._remask_rate=0.5
        self.loss_fn="sce"
        NUM_NODE_ATTR = 119+4+7+12+10+12+8
        NUM_BOND_ATTR = 5 + 3 + 3 + 3
        self.cur_device = 'cuda' 
        self.encoder = GNN(num_layer=self.config['num_layer'], emb_dim=self.config['emb_dim'],
                    JK=self.config['JK'],feat_dim=self.config['feat_dim'], drop_ratio=self.config['dropout_ratio'],
                    gnn_type='gin',degree_list=[0, 1, 2, 3, 4, 5 ,6 ,7, 8, 9, 10],
                    batch_size=self.config['batch_size'],device=self.cur_device).double()
        # self.encoder = self.GNN(x, edge_index, edge_attr)
        # self.encoder = GNN(num_layer=self.config['num_layer'], emb_dim=self.config['emb_dim'],
        #             JK=self.config['JK'],feat_dim=self.config['feat_dim'], drop_ratio=self.config['dropout_ratio'],
        #             gnn_type='degree',degree_list=[0, 1, 2, 3, 4, 5 ,6 ,7, 8, 9, 10],
        #             batch_size=self.config['batch_size'],device=self.cur_device).double()
        if self.config['input_model_file'] is not None and self.config['input_model_file'] != "":
            # model.load_state_dict(torch.load(self.config['input_model_file']))
            # print("Resume training from:", self.config['input_model_file'])
            self.resume = True
        else:
            self.resume = False
        self.dec_pred_atoms = GNNDecoder(self.config['emb_dim'], NUM_NODE_ATTR, JK=self.config['JK'], gnn_type=self.config['gnn_type']).double()
        self.dec_pred_bonds = GNNDecoder(self.config['emb_dim'], NUM_BOND_ATTR, JK=self.config['JK'], gnn_type='linear').double()
            
        self.nt_xent_criterion = NTXentLoss(self.cur_device, config['batch_size'], config['temperature'], config['use_cosine_similarity']).double()
        alpha_l=1.0
        self.criterion = partial(sce_loss, alpha=alpha_l)
    def on_save_checkpoint(self, checkpoint):
        #save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict['torch_state']=torch.get_rng_state()
        out_dict['cuda_state']=torch.cuda.get_rng_state()
        if np:
            out_dict['numpy_state']=np.random.get_state()
        if random:
            out_dict['python_state']=random.getstate()
        checkpoint['rng'] = out_dict

    def on_load_checkpoint(self, checkpoint):
        #load RNG states each time the model and states are loaded from checkpoint
        rng = checkpoint['rng']
        for key, value in rng.items():
            if key =='torch_state':
                torch.set_rng_state(value)
            elif key =='cuda_state':
                torch.cuda.set_rng_state(value)
            elif key =='numpy_state':
                np.random.set_state(value)
            elif key =='python_state':
                random.setstate(value)
            else:
                print('unrecognized state')
    # def on_validation_epoch_end(self, outputs):
    #
    #     avg_loss = torch.tensor([output['loss'] for output in outputs]).mean()
    #     loss = {'loss': avg_loss.item()}
    #     self.log('validation_loss', loss['loss'])
    def validation_step(self, batch, batch_idx):
        xis, xjs = batch

        node_rep, zis = self.encoder(xis)
        node_rep_j, zjs = self.encoder(xjs)

        node_rep = node_rep.clone()
        node_rep_j = node_rep_j.clone()

        loss_rec_all = 0
        masked_node_indices_i = xis.masked_atom_indices
        masked_node_indices_j = xjs.masked_atom_indices

        for i in range(self._num_remasking):
            rep = node_rep.clone().detach().requires_grad_(True)
            rep_j = node_rep_j.clone().detach().requires_grad_(True)

            with torch.no_grad():
                rep_masked, remask_nodes, _ = random_remask(
                    self.encoder.dec_mask_token, rep, xis,
                    self.cur_device, self._remask_rate
                )
                rep_j_masked, remask_nodes_j, _ = random_remask(
                    self.encoder.dec_mask_token, rep_j, xjs,
                    self.cur_device, self._remask_rate
                )

            rep_masked = rep + (rep_masked - rep).detach()
            rep_j_masked = rep_j + (rep_j_masked - rep_j).detach()

            recon = self.dec_pred_atoms(rep_masked,
                                        xjs.edge_index, 
                                        xjs.edge_attr,
                                        masked_node_indices_j)
            recon_j = self.dec_pred_atoms(rep_j_masked,
                                          xis.edge_index,
                                          xis.edge_attr,
                                          masked_node_indices_i)

            loss_rec_all += self.criterion(xjs.node_attr_label[masked_node_indices_j],
                                           recon[masked_node_indices_j])
            loss_rec_all += self.criterion(xis.node_attr_label[masked_node_indices_i],
                                           recon_j[masked_node_indices_i])

        pred_node = self.dec_pred_atoms(node_rep,
                                        xjs.edge_index,
                                        xjs.edge_attr,
                                        masked_node_indices_j)
        pred_node_j = self.dec_pred_atoms(node_rep_j,
                                          xis.edge_index,
                                          xis.edge_attr,
                                          masked_node_indices_i)

        if self.loss_fn == "sce":
            latent_loss = self.criterion(xjs.node_attr_label, pred_node[masked_node_indices_j])
            latent_loss += self.criterion(xis.node_attr_label, pred_node_j[masked_node_indices_i])
        else:
            latent_loss = self.criterion(pred_node.double()[masked_node_indices_i],
                                         xis.mask_node_label[:, 0])

        edge_loss = 0.0
        if self.config['mask_edge']:
            masked_edge_index_i = xis.edge_index[:, xis.connected_edge_indices]
            masked_edge_index_j = xjs.edge_index[:, xjs.connected_edge_indices]

            edge_rep = node_rep[masked_edge_index_j[0]] + node_rep[masked_edge_index_j[1]]
            pred_edge = self.dec_pred_bonds(edge_rep,
                                            xjs.edge_index,
                                            xjs.edge_attr,
                                            masked_node_indices_j)
            edge_loss = self.criterion(pred_edge.double(), xjs.edge_attr_label)

            edge_rep_j = node_rep_j[masked_edge_index_i[0]] + node_rep_j[masked_edge_index_i[1]]
            pred_edge_j = self.dec_pred_bonds(edge_rep_j,
                                              xis.edge_index,
                                              xis.edge_attr,
                                              masked_node_indices_i)
            edge_loss += self.criterion(pred_edge_j.double(), xis.edge_attr_label)
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        if zis.shape[0] != self.nt_xent_criterion.batch_size:
            self.nt_xent_criterion.batch_size = zis.shape[0]
            self.nt_xent_criterion.mask_samples_from_same_repr = \
                self.nt_xent_criterion._get_correlated_mask().type(torch.bool)

        total_loss = latent_loss + loss_rec_all
        # if hasattr(self, 'lambda_divergence') and self.lambda_divergence > 0.0:
        #     div_loss = self.lambda_divergence * torch.norm(zis - zjs, p=2, dim=1).mean()
        #     total_loss = total_loss + div_loss
        return {'loss': total_loss}

    def training_step(self, batch, batch_idx):
        xis, xjs = batch

        node_rep, zis = self.encoder(xis)
        node_rep_j, zjs = self.encoder(xjs)
        node_rep = node_rep.clone()
        node_rep_j = node_rep_j.clone()

        loss_rec_all = 0
        masked_node_indices_i = xis.masked_atom_indices  
        masked_node_indices_j = xjs.masked_atom_indices

        for i in range(self._num_remasking):
            rep = node_rep.clone().detach().requires_grad_(True)
            rep_j = node_rep_j.clone().detach().requires_grad_(True)

            with torch.no_grad():
                rep_masked, remask_nodes, _ = random_remask(
                    self.encoder.dec_mask_token, rep, xis,
                    self.cur_device, self._remask_rate
                )
                rep_j_masked, remask_nodes_j, _ = random_remask(
                    self.encoder.dec_mask_token, rep_j, xjs,
                    self.cur_device, self._remask_rate
                )

            rep_masked = rep + (rep_masked - rep).detach()
            rep_j_masked = rep_j + (rep_j_masked - rep_j).detach()

            recon = self.dec_pred_atoms(rep_masked,
                                        xis.edge_index, 
                                        xis.edge_attr, 
                                        masked_node_indices_i)  
            recon_j = self.dec_pred_atoms(rep_j_masked,
                                          xjs.edge_index,
                                          xjs.edge_attr,  
                                          masked_node_indices_j)  

            loss_rec_all += self.criterion(xis.node_attr_label[masked_node_indices_i],
                                           recon[masked_node_indices_i])
            loss_rec_all += self.criterion(xjs.node_attr_label[masked_node_indices_j],
                                           recon_j[masked_node_indices_j])
        pred_node = self.dec_pred_atoms(node_rep,
                                        xjs.edge_index,
                                        xjs.edge_attr,
                                        masked_node_indices_j)
        pred_node_j = self.dec_pred_atoms(node_rep_j,
                                          xis.edge_index,
                                          xis.edge_attr,
                                          masked_node_indices_i)

        latent_loss = self.criterion(xjs.node_attr_label[masked_node_indices_j],
                                     pred_node[masked_node_indices_j])
        latent_loss += self.criterion(xis.node_attr_label[masked_node_indices_i],
                                      pred_node_j[masked_node_indices_i])

        edge_loss = 0.0
        if self.config['mask_edge']:
            if xjs.connected_edge_indices.numel() > 0:
                masked_edge_index_j = xjs.edge_index[:, xjs.connected_edge_indices]
                edge_rep = node_rep[masked_edge_index_j[0]] + node_rep[masked_edge_index_j[1]]
                pred_edge = self.dec_pred_bonds(edge_rep, xjs.edge_index, xjs.edge_attr, masked_node_indices_j)
                edge_loss += self.criterion(pred_edge, xjs.edge_attr_label[xjs.connected_edge_indices])
       
            if xis.connected_edge_indices.numel() > 0:
                masked_edge_index_i = xis.edge_index[:, xis.connected_edge_indices]
                edge_rep_j = node_rep_j[masked_edge_index_i[0]] + node_rep_j[masked_edge_index_i[1]]
                pred_edge_j = self.dec_pred_bonds(edge_rep_j, xis.edge_index, xis.edge_attr, masked_node_indices_i)
                edge_loss += self.criterion(pred_edge_j, xis.edge_attr_label[xis.connected_edge_indices])


        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        contrast_loss = self.nt_xent_criterion(zis, zjs)  
        total_loss =edge_loss +latent_loss+ loss_rec_all + contrast_loss

        return {'loss': total_loss}
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def configure_optimizers(self):
        model_params = []
        dec_pred_atoms_params = []
        dec_pred_bonds_params = []
        # print(self.named_parameters())
        for name, param in self.named_parameters():
            if 'dec_pred_atoms' in name: 
                dec_pred_atoms_params.append(param)
            elif 'dec_pred_bonds' in name: 
                dec_pred_bonds_params.append(param)
            else: 
                model_params.append(param)

     
        param_groups = [
            {
                'params': model_params,
                'lr': self.config['init_lr'],  
                'weight_decay': self.config['weight_decay']  
            },
            {
                'params': dec_pred_atoms_params,
                'lr': self.config['init_lr'],  
                'weight_decay': self.config['weight_decay'] 
            },
            {
                'params': dec_pred_bonds_params,
                'lr': self.config['init_lr'],  
                'weight_decay': self.config['weight_decay']
            }
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.99) 
        )
        return optimizer


from pytorch_lightning.callbacks import ModelCheckpoint
import torch
torch.serialization.add_safe_globals([ModelCheckpoint])

def load_smiles_from_files(filepaths):
    smiles_list = []
    for filepath in filepaths:
        with open(filepath, 'r') as file:
            for line in file:
                smiles = line.strip().split()[0] 
                if smiles=='smiles':
                    continue
                smiles_list.append(smiles)
    return smiles_list

def get_nccl_socket_ifname():
    return "" 

def fix_infiniband():
    pass  
def main():
    for var in ["MASTER_ADDR", "MASTER_PORT", "NODE_RANK", "LOCAL_RANK"]:
        os.environ.pop(var, None)
    fix_infiniband()
    config={
        'batch_size': 128,  # batch size
        'warm_up': 1,  # warm-up epochs
        'epochs': 1, # total number of epochs
'num_nodes':1,
        'load_model': None,  # resume training
        'eval_every_n_epochs': 1,  # validation frequency
        'save_every_n_epochs': 1 , # automatic model saving frequecy
        'log_every_n_steps': 1,  # print training log frequency

        'fp16_precision': False,  # float precision 16 (i.e. True/False)
        'init_lr': 3e-4*1 ,  # initial learning rate for Adam
        'weight_decay': 1e-5,  # weight decay for Adam
'world_size':1,
    'model_type': 'gin' , # GNN backbone (i.e., gin/gcn)

    'num_layer': 6,  # number of graph conv layers
    'emb_dim': 1024,  # embedding dimension in graph conv layers
    'feat_dim': 768,  # output feature dimention
    'drop_ratio': 0,  # dropout ratio
    'pool': 'mean',  # readout pooling (i.e., mean/max/add)

    'aug': 'node',  # molecule graph augmentation strategy (i.e., node/subgraph/mix)

    'num_workers':0,  # dataloader number of workers
    'valid_size': 0.05,  # ratio of validation data
    
    'temperature': 0.1,  # temperature of NT-Xent loss
    'use_cosine_similarity': True,  # whet
        'JK': 'last',
    'dropout_ratio': 0.0,
    'gnn_type': 'gin',
    'mask_rate':0.15,
    'mask_edge':True,
        'input_model_file':'',
        'use_scheduler':True,
        'alpha_l':1.0,
        'output_model_file':'gin_pretrain_remask_20250529_ZINC_dist',
        'loss_fn':'sce',
        'restart_path':'',
        'max_len':202,
        'lambda_divergence': 0.0,  
    }
    import scanpy as sc
    import getpass
    if config['num_nodes'] > 1:
        # print("Using " + str(config.num_nodes) + " Nodes----------------------------------------------------------------------")
        LSB_MCPU_HOSTS = os.environ["LSB_MCPU_HOSTS"].split(' ') # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
        HOST_LIST = LSB_MCPU_HOSTS[::2] # Strips the cores per node items in the list
        os.environ["MASTER_ADDR"] = HOST_LIST[0] # Sets the MasterNode to thefirst node on the list of hosts
        os.environ["MASTER_PORT"] = "54966"
        os.environ["NODE_RANK"] = str(HOST_LIST.index(os.environ["HOSTNAME"])) #Uses the list index for node rank, master node rank must be 0
        #os.environ["NCCL_SOCKET_IFNAME"] = 'ib,bond'  # avoids using docker of loopback interface
        os.environ["NCCL_DEBUG"] = "INFO" #sets NCCL debug to info, during distributed training, bugs in code show up as nccl errors
        #os.environ["NCCL_IB_CUDA_SUPPORT"] = '1' #Force use of infiniband
        #os.environ["NCCL_TOPO_DUMP_FILE"] = 'NCCL_TOP.%h.xml'
        #os.environ["NCCL_DEBUG_FILE"] = 'NCCL_DEBUG.%h.%p.txt'
        print(os.environ["HOSTNAME"] + " MASTER_ADDR: " + os.environ["MASTER_ADDR"])
        print(os.environ["HOSTNAME"] + " MASTER_PORT: " + os.environ["MASTER_PORT"])
        print(os.environ["HOSTNAME"] + " NODE_RANK " + os.environ["NODE_RANK"])
        print(os.environ["HOSTNAME"] + " NCCL_SOCKET_IFNAME: " + os.environ["NCCL_SOCKET_IFNAME"])
        print(os.environ["HOSTNAME"] + " NCCL_DEBUG: " + os.environ["NCCL_DEBUG"])
        print(os.environ["HOSTNAME"] + " NCCL_IB_CUDA_SUPPORT: " + os.environ["NCCL_IB_CUDA_SUPPORT"])
        print("Using " + str(config['num_nodes']) + " Nodes---------------------------------------------------------------------")
        print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    else:
        print("Using " + str(config['num_nodes']) + " Node----------------------------------------------------------------------")
        print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    train_config = {'batch_size':config['batch_size'], 'num_workers':config['num_workers'], 'pin_memory':True}
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
  
    lambda_values = [0]

    results = []

    base_config = config.copy() 

    for lam in lambda_values:
        print(f"\n{'='*40}\nTraining with λ = {lam}\n{'='*40}")
        current_config = base_config.copy()
        current_config['lambda_divergence'] = lam

        data_loader = MoleculeModule(current_config['max_len'], None, train_config)
        data_loader.setup()
        cachefiles = data_loader.get_cache()
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            # Save checkpoint every 1 epoch
            save_top_k=-1,
            # Save all checkpoints (you can change this to a positive integer to save top-k checkpoints based on a monitored metric)
            verbose=True  # Print information about checkpoint saving
        )
        try:
            checkpoint_path = '\version_160\checkpoints\epoch=0-step=4688.ckpt'

            molgatmae = MolGATMAE(current_config)
            # molgatmae.load_state_dict(model_state, strict=True)
            molgatmae.to('cuda')
            # molgatmae.load_state_dict(model_state)
            # molgatmae = MolGATMAE.load_from_checkpoint('E:\生信\scMCP\DrugGCL\\version_160\checkpoints\epoch=0-step=4688.ckpt', config=current_config)
            #
        except RuntimeError as e:
            if "inline_container.cc" in str(e):
                print(f"Skipping λ={lam} due to corrupted checkpoint: {e}")
                continue
            else:
                raise
        trainer = pl.Trainer(
            default_root_dir=f'./lambda_{lam:.4f}/',
            max_epochs=current_config['epochs'],
            precision=16,
            gpus=1,
            num_nodes=1,
            callbacks=[checkpoint_callback, ModelCheckpointAtEpochEnd(), CheckpointEveryNSteps(1000)],
            accumulate_grad_batches=1,
            num_sanity_val_steps=10,
            val_check_interval=30,
        )
        try:
            train_loader = data_loader.train_dataloader()
            print(f"Training batches: {len(train_loader)}")
            trainer.fit(model=molgatmae, train_dataloader=train_loader)
        except Exception as exp:
            print(f"Training failed for λ={lam}: {exp}")
            rank_zero_warn('Error caught, cleaning up')
            remove_tree(cachefiles)
            continue  

        molgatmae.eval()
        molgatmae.to('cuda')
        val_loader = data_loader.val_dataloader()
        all_D = []
        with torch.no_grad():
            # train_loader_for_D = train_loader.train_dataloader()
            for i, batch in enumerate(val_loader):
                if i >= 30:
                    break
                xis, xjs = batch
                xis, xjs = xis.to('cuda'), xjs.to('cuda')
                _, zis = molgatmae.encoder(xis)  # 假设 encoder 返回 (_, z)
                _, zjs = molgatmae.encoder(xjs)
                D_batch = torch.norm(zis - zjs, p=2, dim=1).mean()
                all_D.append(D_batch.item())
        mean_D = sum(all_D) / len(all_D) if all_D else float('nan')

        total_val_loss = 0.0
        num_val_batches = 0
        test_mae = evaluate_linear_probe(
            molgatmae.encoder,
            smiles_train, y_train,
            smiles_test, y_test,
            device='cuda'
        )
        results.append({'lambda': lam, 'mean_D': mean_D,  'test_mae': test_mae})
        print(f"λ={lam:.4f}, mean D={mean_D:.4f},  test_MAE={test_mae:.4f}")

        remove_tree(cachefiles)

        del train_loader, val_loader
        torch.cuda.empty_cache()
        import gc;
        gc.collect()

    print("\n===== Scan Results =====")
    for r in results:
        print(f"λ={r['lambda']:.4f}, D={r['mean_D']:.4f}")

    import matplotlib.pyplot as plt
    lambs = [r['lambda'] for r in results]
    Ds = [r['mean_D'] for r in results]
    plt.figure()
    plt.plot(lambs, Ds, 'o-')
    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('Mean representation divergence D')
    plt.title('Effect of λ on inter-view divergence')
    plt.grid(True)
    plt.savefig('lambda_D_curve.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()

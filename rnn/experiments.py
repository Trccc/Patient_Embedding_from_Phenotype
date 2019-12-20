import os
import nltk
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn
from EHR_utils import *
from model import RecurrentNetwork,DenseNetwork
from sklearn.metrics import roc_auc_score
from config import cfg



def single_train():
    pre_embed = torch.tensor(weights)
    pre_embed = pre_embed.clone().detach().requires_grad_(True).cuda()
    USE_CUDA = cfg.CUDA
    loss_fn = nn.CrossEntropyLoss()
    model = RecurrentNetwork(vecs=pre_embed, embed_dim=EMBEDDING_DIM, dropout=0.1, hidden_dim=64, num_layers=2, num_labels=2)
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4)
    if USE_CUDA:
        loss_fn = nn.CrossEntropyLoss().cuda()
        model.cuda()
    dmodel,_ = train_model(model, loss_fn,optimizer, train_generator, dev_generator)
    sen,spe,auc = test_model(dmodel.eval(), loss_fn, test_generator)
    return sen,spe,auc

if __name__ == "__main__":
    print(cfg)
    EMBEDDING_DIM = cfg.EMBEDDING_DIM
    batch_size = cfg.BATCH_SIZE
    weights_path = cfg.PATH.WEIGHTS
    weights_idx_path = cfg.PATH.WEIGHTS_IDX
#     data_path = cfg.PATH.DATA
    base = cfg.PATH.BASE
    


############# diag

    weights = np.load(weights_path)
    word2idx = np.load(weights_idx_path,allow_pickle=True).item()


#     dict_idx = np.load('dict_idx.npy',allow_pickle=True).item()
#     word2idx = dict_idx['word2idx']
#     idx2word = dict_idx['idx2word']

#     weights = embidx2fullidx(weight1,word2int,idx2word,word2idx)
    
    

############ pred diag lab

#     wt_diag = np.load('./ckd_data/pre_training/diag_embedding/weights60.npy')
#     word2idx_diag = np.load('./ckd_data/pre_training/diag_embedding/word2int60.npy',allow_pickle=True).item()

#     wt_lab = np.load('./ckd_data/pre_training/lab_emb/weights60.npy')
#     word2idx_lab = np.load('./ckd_data/pre_training/lab_emb/word2int60.npy',allow_pickle=True).item()

#     wt_pres = np.load('./ckd_data/pre_training/pres_emb/weights60.npy')
#     word2idx_pres = np.load('./ckd_data/pre_training/pres_emb/word2int60.npy',allow_pickle=True).item()
#     # idx2word = {v: k for k, v in word2idx.items()}

#     weights = np.vstack((wt_diag,wt_lab,wt_pres))

#     word2idx = {}
#     for k,v in word2idx_diag.items():
#         word2idx[str(k)] = len(word2idx)
#     for k,v in word2idx_lab.items():
#         word2idx[str(k)] = len(word2idx)   
#     n = len(word2idx)
#     for k,v in word2idx_pres.items():
#         word2idx[str(k)] = n+v
#     word2idx['nan'] = 0
    
############ diag lab

############ pred diag lab

#     wt_diag = np.load('./ckd_data/pre_training/diag_embedding/weights60.npy')
#     word2idx_diag = np.load('./ckd_data/pre_training/diag_embedding/word2int60.npy',allow_pickle=True).item()

#     wt_lab = np.load('./ckd_data/pre_training/lab_emb/weights60.npy')
#     word2idx_lab = np.load('./ckd_data/pre_training/lab_emb/word2int60.npy',allow_pickle=True).item()

#     weights = np.vstack((wt_diag,wt_lab))

#     word2idx = {}
#     for k,v in word2idx_diag.items():
#         word2idx[str(k)] = len(word2idx)
#     for k,v in word2idx_lab.items():
#         word2idx[str(k)] = len(word2idx)   

#     word2idx['nan'] = 0

###################
    
    idx2word = {v: k for k, v in word2idx.items()}
    for file in os.listdir(base):
        if file.endswith("pres.csv"):
            fn = os.path.join(base, file)
            print(fn)
            
            pre_embed = torch.tensor(weights)
            pre_embed = pre_embed.clone().detach().requires_grad_(True).cuda()

            train_generator,dev_generator,test_generator = create_dataloader(fn,word2idx,batch_size)


            sen = []
            spe = []
            auc = []
            filename = '../experiment_shuffle_result/res'+file
            for i in range(cfg.REPEAT):
                sen_0,spe_0,auc_0 =  single_train()
                auc.append(auc_0)
                sen.append(sen_0)
                spe.append(spe_0)
                if i%10==0 and i>0:
                    res = {'Sensitivity':sen, 'Specificity':spe,'AUC':auc} 
                    df = pd.DataFrame(res) 
                    df.to_csv(filename)
            res = {'Sensitivity':sen, 'Specificity':spe,'AUC':auc} 
            df = pd.DataFrame(res) 
            df.to_csv(filename)
        
        
    
  
    


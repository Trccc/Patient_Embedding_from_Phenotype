import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
import sys
import matplotlib.pyplot as plt 
import data
import scipy.io

from torch import nn, optim
from torch.nn import functional as F

from etm import ETM
from utils import nearest_neighbors, get_topic_coherence, get_topic_diversity

from easydict import EasyDict as edict

__C = edict()
args = __C

if __name__ =="__main__": 

    ### data and file related arguments
    args.dataset = 'diag'
    args.data_path = './scripts/diag'
    args.emb_path = './diag_embedding.txt'
    args.save_path = './results'
    args.batch_size = 1000

    ### model-related arguments

    args.num_topics = 10
    args.rho_size = 60
    args.emb_size = 60
    args.t_hidden_size = 800
    args.theta_act = 'relu'
    args.train_embeddings = 1

    ### optimization-related arguments

    args.lr = 0.005
    args.lr_factor = 4.0
    args.epochs = 2
    args.mode = 'train'  #train or eval
    args.optimizer = 'adam'
    args.seed = 2019
    args.enc_drop = 0.0
    args.clip = 0.0
    args.nonmono = 10
    args.wdecay = 1.2e-6
    args.anneal_lr = 0
    args.bow_norm = 1


    ### evaluation, visualization, and logging-related arguments

    args.num_words = 10  # number of words for topic viz
    args.log_interval = 40
    args.visualize_every = 10
    args.eval_batch_size = 1000
    # args.load_from = './results/etm_diag_K_40_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.005_Bsz_1000_RhoSize_60_trainEmbeddings_0'
    args.load_from = ''
    args.tc = 1
    args.td = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('\n')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    ## get data
    # 1. vocabulary
    vocab, train, valid, test = data.get_data(os.path.join(args.data_path))
    vocab_size = len(vocab)
    args.vocab_size = vocab_size

    # 1. training data
    train_tokens = train['tokens']
    train_counts = train['counts']
    args.num_docs_train = len(train_tokens)

    # 2. dev set
    valid_tokens = valid['tokens']
    valid_counts = valid['counts']
    args.num_docs_valid = len(valid_tokens)

    # 3. test data
    test_tokens = test['tokens']
    test_counts = test['counts']
    args.num_docs_test = len(test_tokens)
    test_1_tokens = test['tokens_1']
    test_1_counts = test['counts_1']
    args.num_docs_test_1 = len(test_1_tokens)
    test_2_tokens = test['tokens_2']
    test_2_counts = test['counts_2']
    args.num_docs_test_2 = len(test_2_tokens)


    embeddings = None
    if not args.train_embeddings:
        emb_path = args.emb_path
        vect_path = os.path.join(args.data_path.split('/')[0], 'embeddings.pkl')   
        vectors = {}
        with open(emb_path, 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                if word in vocab:
                    vect = np.array(line[1:]).astype(np.float)
                    vectors[word] = vect
        embeddings = np.zeros((vocab_size, args.emb_size))
        words_found = 0
        for i, word in enumerate(vocab):
            try: 
                embeddings[i] = vectors[word]
                words_found += 1
            except KeyError:
                embeddings[i] = np.random.normal(scale=0.6, size=(args.emb_size, ))
        embeddings = torch.from_numpy(embeddings).to(device)
        args.embeddings_dim = embeddings.size()

    print('=*'*100)
    print('Training an Embedded Topic Model on {} with the following settings: {}'.format(args.dataset.upper(), args))
    print('=*'*100)

    ## define checkpoint
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.mode == 'eval':
        ckpt = args.load_from
    else:
        ckpt = os.path.join(args.save_path, 
    'Dec17_etm_{}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}'.format(
            args.dataset, args.num_topics, args.t_hidden_size, args.optimizer, args.clip, args.theta_act, 
                args.lr, args.batch_size, args.rho_size, args.train_embeddings))


    for num_topics in [10,15,20,25,30,35,40,45,50]:
        args.num_topics = num_topics
    ## define model and optimizer
        model = ETM(args.num_topics, vocab_size, args.t_hidden_size, args.rho_size, args.emb_size, 
                        args.theta_act, embeddings, args.train_embeddings, args.enc_drop).to(device)
        # print('model: {}'.format(model))
        if args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)


        def train(epoch):
            model.train()
            acc_loss = 0
            acc_kl_theta_loss = 0
            cnt = 0
            indices = torch.randperm(args.num_docs_train)
            indices = torch.split(indices, args.batch_size)
            for idx, ind in enumerate(indices):
                optimizer.zero_grad()
                model.zero_grad()
                data_batch = data.get_batch(train_tokens, train_counts, ind, args.vocab_size, device)
                sums = data_batch.sum(1).unsqueeze(1)
                if args.bow_norm:
                    normalized_data_batch = data_batch / sums
                else:
                    normalized_data_batch = data_batch
                recon_loss, kld_theta = model(data_batch, normalized_data_batch)
                total_loss = recon_loss + kld_theta
                total_loss.backward()

                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

                acc_loss += torch.sum(recon_loss).item()
                acc_kl_theta_loss += torch.sum(kld_theta).item()
                cnt += 1

                if idx % args.log_interval == 0 and idx > 0:
                    cur_loss = round(acc_loss / cnt, 2) 
                    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
                    cur_real_loss = round(cur_loss + cur_kl_theta, 2)

                    print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                        epoch, idx, len(indices), optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))

            cur_loss = round(acc_loss / cnt, 2) 
            cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
            cur_real_loss = round(cur_loss + cur_kl_theta, 2)
            print('*'*100)
            print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                    epoch, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
            print('*'*100)

        def visualize(m, show_emb=True):
            if not os.path.exists('./results'):
                os.makedirs('./results')

            m.eval()
        #     queries = ['andrew', 'computer', 'sports', 'religion', 'man', 'love', 
        #                 'intelligence', 'money', 'politics', 'health', 'people', 'family']
            queries = ['7429']
            ## visualize topics using monte carlo
            with torch.no_grad():
                print('#'*100)
                print('Visualize topics...')
                topics_words = []
                gammas = m.get_beta()
                for k in range(args.num_topics):
                    print(gammas.shape)
                    gamma = gammas[k]
                    top_words = list(gamma.cpu().numpy().argsort()[-args.num_words:][::-1])
                    topic_words = [vocab[a] for a in top_words]
                    topics_words.append(' '.join(topic_words))
                    print('Topic {}: {}'.format(k, topic_words))

                if show_emb:
                    ## visualize word embeddings by using V to get nearest neighbors
                    print('#'*100)
                    print('Visualize word embeddings by using output embedding matrix')
                    try:
                        embeddings = m.rho.weight  # Vocab_size x E
                    except:
                        embeddings = m.rho         # Vocab_size x E
                    neighbors = []
                    for word in queries:
                        print('word: {} .. neighbors: {}'.format(
                            word, nearest_neighbors(word, embeddings, vocab)))
                    print('#'*100)

        def evaluate(m, source, tc=False, td=False):
            """Compute perplexity on document completion.
            """
            m.eval()
            with torch.no_grad():
                if source == 'val':
                    indices = torch.split(torch.tensor(range(args.num_docs_valid)), args.eval_batch_size)
                    tokens = valid_tokens
                    counts = valid_counts
                else: 
                    indices = torch.split(torch.tensor(range(args.num_docs_test)), args.eval_batch_size)
                    tokens = test_tokens
                    counts = test_counts

                ## get \beta here
                beta = m.get_beta()

                ### do dc and tc here
                acc_loss = 0
                cnt = 0
                indices_1 = torch.split(torch.tensor(range(args.num_docs_test_1)), args.eval_batch_size)
                for idx, ind in enumerate(indices_1):
                    ## get theta from first half of docs
                    data_batch_1 = data.get_batch(test_1_tokens, test_1_counts, ind, args.vocab_size, device)
                    sums_1 = data_batch_1.sum(1).unsqueeze(1)
                    if args.bow_norm:
                        normalized_data_batch_1 = data_batch_1 / sums_1
                    else:
                        normalized_data_batch_1 = data_batch_1
                    theta, _ = m.get_theta(normalized_data_batch_1)

                    ## get prediction loss using second half
                    data_batch_2 = data.get_batch(test_2_tokens, test_2_counts, ind, args.vocab_size, device)
                    sums_2 = data_batch_2.sum(1).unsqueeze(1)
                    res = torch.mm(theta, beta)
                    preds = torch.log(res)
                    recon_loss = -(preds * data_batch_2).sum(1)

                    loss = recon_loss / sums_2.squeeze()
                    loss = loss.mean().item()
                    acc_loss += loss
                    cnt += 1
                cur_loss = acc_loss / cnt
                ppl_dc = round(math.exp(cur_loss), 1)
                print('*'*100)
                print('{} Doc Completion PPL: {}'.format(source.upper(), ppl_dc))
                print('*'*100)
                if tc or td:
                    beta = beta.data.cpu().numpy()
                    if tc:
                        print('Computing topic coherence...')
                        get_topic_coherence(beta, train_tokens, vocab)
                    if td:
                        print('Computing topic diversity...')
                        get_topic_diversity(beta, 25)
                return ppl_dc


            ## train model on data 
            best_epoch = 0
            best_val_ppl = 1e9
            all_val_ppls = []
            print('\n')
            print('Visualizing model quality before training...')
            visualize(model)
            print('\n')
            for epoch in range(1, args.epochs):
                train(epoch)
                val_ppl = evaluate(model, 'val')
                if val_ppl < best_val_ppl:
                    with open(ckpt, 'wb') as f:
                        torch.save(model, f)
                    best_epoch = epoch
                    best_val_ppl = val_ppl
                else:
                    ## check whether to anneal lr
                    lr = optimizer.param_groups[0]['lr']
                    if args.anneal_lr and (len(all_val_ppls) > args.nonmono and val_ppl > min(all_val_ppls[:-args.nonmono]) and lr > 1e-5):
                        optimizer.param_groups[0]['lr'] /= args.lr_factor
                if epoch % args.visualize_every == 0:
                    visualize(model)
                all_val_ppls.append(val_ppl)
            with open(ckpt, 'rb') as f:
                model = torch.load(f)
            model = model.to(device)
            val_ppl = evaluate(model, 'val')
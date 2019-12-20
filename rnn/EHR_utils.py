from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch.nn as nn
import time
from sklearn.metrics import f1_score, confusion_matrix,roc_auc_score,classification_report
import torch
from sklearn.preprocessing import LabelEncoder
from config import cfg
import time 
def get_data(data_fn):
    """
    Load the data from file and split it randomly into train, dev, and test sets.
    :param data_fn: the name of the .csv file in which the data is stored
    :return: train, dev, test; splits of the pandas dataframe
    """

    # Load the data
    df = pd.read_csv(data_fn,index_col=0)

    # Split the data into train, dev, and test
    # .sample() shuffles the data - set the random seed so we always grab the same data split(!)
    train, dev, test = np.split(df.sample(frac=1, random_state=4705), [int(.8 * len(df)), int(.9 * len(df))])

    # Grab some samples from the train set to view
    print("Random samples from train set")
    print(train.sample(10))
    print("\n\n")

    return train, dev, test

def make_vectors(data, word2idx, label_enc=None):
    """
    Helper function: given text data and labels, transform them into vectors, where text becomes lists of word indices.
    Words not in the provided vocabulary get mapped to an "unknown" token, <UNK>.
    :param data: a pandas DataFrame including 'content' and 'sentiment' columns
    :param word2idx: a {word: index} dictionary defining the vocabulary for this data.
    :param label_enc: a OneHotEncoder that turns labels into one-hot vectors. Pass None to fit a new one from the data.
    :return: X (a list of lists of word indices), y (a numpy matrix of class indices), label_enc (as in parameters)
    """
    colname = cfg.COLNAME
    X_pre = data[colname].apply(lambda x: x.strip('][').replace("'",'').split(', '))
#     for datapoint in X_pre:
#         if len(datapoint)>0:
#             torch.tensor([word2idx[word] if word in word2idx else word2idx['<unk>'] for word in
#                                                  datapoint])
            
    X = nn.utils.rnn.pad_sequence([torch.tensor([word2idx[word] if word in word2idx else word2idx['<unk>'] for word in
                                                 datapoint]) for datapoint in X_pre])
    y = data['ckd'].to_numpy()
    if label_enc is None:
        label_enc = LabelEncoder()
        y = label_enc.fit_transform(y)
    else:
        y = label_enc.transform(y)
    return X, y, label_enc

class DiagDataset(Dataset):
#     def __init__(self, data, word2idx=None, encoder=None, glove=None):
    def __init__(self, data, word2idx=None, encoder=None):
        """
        Dataset class to help load and handle the data, specific to our format.
        Provides X, y pairs.
        Create the dataset by providing the data as well as possible predefined preprocessing
        (e.g., to prevent data leakage between train and test).

        Args:
            :param data (pandas dataframe): data you want to represent
            :param word2idx: either the prepared {word: index} dictionary or None (to create one from the data)
            :param encoder: either a prepared label encoder or None (to create one from the data)
            :param glove: a prepared {word: embedding} dictionary, if creating the vocabulary
        """
        if word2idx is None and glove is None:
            raise ValueError("Must pass either a predefined vocabulary or a GloVe dictionary to make a dataset.")

        self.label_enc = encoder  # encoder may be None, in which case make_vectors() will handle it
        self.diag_frame = data
        self.word2idx = word2idx
        self.X, self.y, self.label_enc = make_vectors(self.diag_frame, self.word2idx, self.label_enc)

    def __len__(self):
        return len(self.diag_frame)

    def __getitem__(self, idx):
        X = self.X[:, idx]
        y = self.y[idx]

        return X, y


def train_model(model, loss_fn, optimizer, train_generator, dev_generator):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: one of your models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """
    epochs = cfg.MAX_EPOCH
    valloss = []
    valacc = []
    best_loss = 100
    best_acc = 0
    best_model = model
    time0 = time.time()
    for epoch in range(epochs):
        start = time.time()
        epoch_loss = 0
        train_acc = 0
        j = 0
        for X_b, y_b in train_generator:
            j+=1
            num_b = len(y_b)
            X_b = X_b.long().cuda()           
            y_b = y_b.cuda()
            optimizer.zero_grad()
            y_pred = model(X_b)
            train_loss = loss_fn(y_pred, y_b.long())
            _,indices = y_pred.max(1)
            correct = (indices == y_b).float() #convert into float for division 
            train_acc += correct.sum() / len(correct)
            epoch_loss += train_loss
            train_loss.backward()
            optimizer.step()
        epoch_loss = epoch_loss/len(train_generator)
        train_acc = train_acc/len(train_generator)
    # on validation set
        with torch.no_grad():
            valid_acc = 0
            valid_loss = 0
            for X_b, y_b in dev_generator:
                # Predict
                X_b = X_b.long().cuda()
                y_b = y_b.cuda()
                y_pred = model(X_b)
                valid_loss += loss_fn(y_pred, y_b.long()).item()
                _,indices = y_pred.max(1)
                correct = (indices == y_b).float()
                valid_acc += correct.sum() / len(correct)
            valid_loss = valid_loss/len(dev_generator)
            valid_acc = valid_acc/len(dev_generator)
            end = time.time()
            if cfg.VERBOSE:
                print(f'Epoch: {epoch+1:02}')
                print(f'\tTrain Loss: {epoch_loss:.3f}| Train Acc: {train_acc*100:.2f}% | Time: {end-start:.2f}s')
                print(f'\t Val. Loss: {valid_loss:.3f}|  Val. Acc: {valid_acc*100:.2f}%')
            valacc.append(valid_acc)
            valloss.append(valid_loss)
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = model
            if (epoch>=cfg.MIN_EPOCH) and (valloss[-1]> valloss[-2]) and (valloss[-1]> valloss[-3]) and (valloss[-1]> valloss[-4]):
                if cfg.VERBOSE:
                    print(f'\tBest Loss: {best_loss:.3f}')
                    print("Early Stop")
                break
    print('Total time = %.2fs'%(end - time0))
    return best_model,valloss
  
def test_model(model, loss_fn, test_generator):
    """
    Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
    :param model: a model that performs 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param test_generator: a DataLoader that provides batches of the testing set
    """
    gold = []
    predicted = []

    # Keep track of the loss
    loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
    if cfg.CUDA:
        loss = loss.cuda()

    model.eval()
    # Iterate over batches in the test dataset
    with torch.no_grad():
        for X_b, y_b in test_generator:
            # Predict
            if cfg.CUDA:
                y_pred = model(X_b.cuda())
            # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
                gold.extend(y_b.cpu().detach().numpy())
                predicted.extend(y_pred.argmax(1).cpu().detach().numpy())
                loss += loss_fn(y_pred.double(), y_b.long().cuda()).data
                
                
            else:
                y_pred = model(X_b)
                gold.extend(y_b.cpu().detach().numpy())
                predicted.extend(y_pred.argmax(1).cpu().detach().numpy())
                loss += loss_fn(y_pred.double(), y_b.long())
                

    # Print total loss and AUC score
    print("Test loss: %.3f"%loss)
    report = classification_report(gold, predicted,output_dict=True)
    auc = roc_auc_score(gold, predicted)
    sensitivity = report['1']['recall']
    specificity = report['0']['recall']
    print("AUC-score: %.3f"%auc)
    return sensitivity,specificity,auc

    
def embidx2fullidx(weights,word2int,idx2word,word2idx):
    V = len(word2idx)
    d = weights.shape[1]
    weight2 = np.zeros((V,d))
    c = 0
    for i in range(1,V):
        word = idx2word[i]
        if word2int.get(word):
            weight2[i] = weights[word2int.get(word)]
            c+=1
        else:
            weight2[i] = np.random.normal(0,1,d)
        return weight2

def create_dataloader(path,word2idx,batch_size):
    train, dev, test = get_data(path)
    LABEL_NAMES = [1,0]
    label_enc = LabelEncoder()
    label_enc.fit(np.array(LABEL_NAMES))
    print("Labels are encoded in the following class order:")
    print(label_enc.inverse_transform(np.arange(2)))
    print("\n\n")

    # Create the three Datasets
    train_data = DiagDataset(train, word2idx,encoder=label_enc)
    dev_data = DiagDataset(dev, word2idx, encoder=label_enc)
    test_data = DiagDataset(test, word2idx, encoder=label_enc)
#     print('num of ckd = ',sum(test['ckd']==1),'num of not ckd = ',sum(test['ckd']==0))
    train_generator = DataLoader(dataset=train_data, batch_size=batch_size,drop_last=True)
    dev_generator = DataLoader(dataset=dev_data, batch_size=batch_size,drop_last=True)
    test_generator = DataLoader(dataset=test_data, batch_size=batch_size,drop_last=False)

    return train_generator,dev_generator,test_generator
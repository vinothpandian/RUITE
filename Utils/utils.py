from sklearn.metrics import *
from matplotlib import pyplot
import pickle
import json
from config.uiconfig import *


def save_params(best_params, name):
    with open(name, 'wb') as handle:
        pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(name):
    with open(name, 'rb') as handle:
        best_params_dict =  pickle.load(handle)
    return best_params_dict

def save_json(params, path):
    with open(path, 'w') as fp:
        json.dump(params, fp)

def load_param(path):
    with open(path, 'r') as fp:
        params = json.load(fp)
    return params

def f1score_without_group(y_true, y_pred, average = 'macro'):
    #acc = accuracy_score(y_true, y_pred)
    tru_align=[]
    pred_align=[]
    print('pref', y_pred[:100])
    for i in range(len(y_pred)):
        if ((i+1)%6 == 0):
            if(i < len(y_pred)-(len(y_pred)/6)):
                tru_align.append(y_true[i])

        else:
            pred_align.append(y_pred[i])
            if (i < len(y_pred) - (len(y_pred) / 6)):
                tru_align.append(y_true[i])
    #pred_align = pred_align[:len(y_true)]
    print('truu', tru_align[:100])
    print('pred', pred_align[:100])
    align_acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    return f1, align_acc

def f1score_with_group(y_true, y_pred, average = 'macro'):

    f1 = f1_score(y_true, y_pred, average=average)
    acc = accuracy_score(y_true, y_pred)
    tru_group=y_true[5::6]
    pred_group=y_pred[5::6]
    group_acc = accuracy_score(tru_group,pred_group)
    tru_align=[]
    pred_align=[]
    for i in range(len(y_true)):
        if ((i+1)%6 == 0):
            continue
        else:
            tru_align.append(y_true[i])
            pred_align.append(y_pred[i])
    align_acc = accuracy_score(tru_align, pred_align)
    return f1, acc, group_acc, align_acc

def gestalt_alignment(pred_df):
    import pandas as pd

    # def gest_alig_fun(group_i):
    #     df.groupby(by='domain', as_index=False).agg({'ID': pd.Series.nunique})


    pred_x1 = pred_df.groupby('filename', as_index=False).agg({'x1': pd.Series.nunique})
    x1_align = len(pred_x1[pred_x1['x1']!=0])/len(pred_x1)            #).apply(gest_alig_fun)

    pred_x1 = pred_df.groupby('filename', as_index=False).agg({'y1': pd.Series.nunique})
    y1_align = len(pred_x1[pred_x1['y1']!=0])/len(pred_x1)

    return x1_align+y1_align



def classificationreport(y_true, y_pred):
    print(classification_report(y_true, y_pred))

class score_cal:
    def __init__(self):
        self.train_f1 = []
        self.valid_f1 = []
        self.train_acc = []
        self.valid_acc = []
        self.train_loss = []
        self.valid_loss = []

def print_result(score:score_cal, epochs):
    epoch_vals = [i + 1 for i in range(epochs)]
    pyplot.subplot(311)
    pyplot.title("Loss")
    pyplot.plot(epoch_vals, score.train_loss, label='train')
    pyplot.plot(epoch_vals, score.valid_loss, label='valid')
    pyplot.legend()
    pyplot.xticks(epoch_vals)

    if(score.train_f1 != []):
        pyplot.subplot(312)
        pyplot.title("F1")
        pyplot.plot(epoch_vals, score.train_f1, label='train')
        pyplot.plot(epoch_vals, score.valid_f1, label='valid')
        pyplot.legend()
        pyplot.xticks(epoch_vals)

        pyplot.subplot(313)
        pyplot.title("acc")
        pyplot.plot(epoch_vals, score.train_acc, label='train')
        pyplot.plot(epoch_vals, score.valid_acc, label='valid')
        pyplot.legend()
        pyplot.xticks(epoch_vals)

    pyplot.show()


def prune_preds(y_true, y_pred):
    true = []
    preds = []
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    for i in range(len(y_true)):
        if(y_true[i] != UIconfig.ignore_index): # ignore padding index
            true.append(y_true[i])
            preds.append(y_pred[i])
    return true, preds

def create_masks(batch):
    input_seq = batch.seq_x.transpose(0,1)
    input_pad = 1
    # creates mask with 1s wherever there is padding in the input
    input_msk = (input_seq == input_pad)
    return input_msk

def plot_save(ytrue, ypred):
    print(ytrue[0])
    print(ypred[0])
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import model as matchmakerCrossNetEmb
import performance_metrics
import argparse
import matplotlib.pyplot as plt
from tensorflow import keras

import datetime
from sklearn.model_selection import PredefinedSplit



# --------------- Parse MolCross arguments --------------- #

parser = argparse.ArgumentParser(description='REQUEST REQUIRED PARAMETERS OF MolCross')

parser.add_argument('--comb-data-name', default='data/DrugCombinationData.tsv',
                    help="Name of the drug combination data")

parser.add_argument('--cell_line-gex', default='data/cell_line_gex.csv',
                    help="Name of the cell line gene expression data")

parser.add_argument('--drug1-chemicals', default='data/drug1_chem.csv',
                    help="Name of the chemical features data for drug 1")

parser.add_argument('--drug2-chemicals', default='data/drug2_chem.csv',
                    help="Name of the chemical features data for drug 2")

parser.add_argument('--gpu-devices', default='3', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')

parser.add_argument('--train-test-mode', default=1, type = int,
                    help="Test of train mode (0: test, 1: train)")

parser.add_argument('--train-ind', default='data/train_inds.txt',
                    help="Data indices that will be used for training")

parser.add_argument('--val-ind', default='data/val_inds.txt',
                    help="Data indices that will be used for validation")

parser.add_argument('--test-ind', default='data/test_inds.txt',
                    help="Data indices that will be used for test")

parser.add_argument('--arch', default='data/architecture.txt',
                    help="Architecute file to construct MolCross layers")

parser.add_argument('--gpu-support', default=True,
                    help='Use GPU support or not')

parser.add_argument('--saved-model-name', default="MolCross.h5",
                    help='Model name to save weights')
args = parser.parse_args()
# ---------------------------------------------------------- #
num_cores = 8
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
GPU = True
if args.gpu_support:
    num_GPU = 1
    num_CPU = 1
else:
    num_CPU = 2
    num_GPU = 0

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores,
                        allow_soft_placement=True,
                        device_count = {'CPU' : num_CPU,
                                        'GPU' : num_GPU}
                       )

tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))



def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


#load data
A_exp=pd.read_csv('../data/drugA2000_mm.csv',header=None)
B_exp=pd.read_csv('../data/drugB2000_mm.csv',header=None)
cell=pd.read_csv('../data/exp2000_mm.csv',header=None)
labels=pd.read_csv('../data/lables_ABBA.csv')
emb=pd.read_csv('../data/embed.csv',header=None)

A = np.array(A_exp.values)
B = np.array(B_exp.values)
cell = np.array(cell.values)
emb = np.array(emb.values)

synergies=np.array(labels['synergy'].values).reshape(-1)

test_data = {}
#fold=4 作为最后的测试集
test_ind = list(np.loadtxt('../data/onlyForTest_idx.txt', dtype=np.int, delimiter=','))
test_data['drug1']=np.concatenate((A[test_ind, :], cell[test_ind, :]), axis=1)
test_data['drug2']=np.concatenate((B[test_ind, :], cell[test_ind, :]), axis=1)
test_data['full']=np.concatenate((test_data['drug1'], B[test_ind, :]), axis=1)
test_data['emb'] =emb[test_ind,:]
test_data['y'] =synergies[test_ind]
print(test_data['drug1'].shape)
print(test_data['drug2'].shape)
print(test_data['full'].shape)
print(test_data['y'].shape)
print(test_data['emb'].shape)

tr_ind = list(np.loadtxt('../data/onlyForTrain_idx.txt', dtype=np.int, delimiter=','))
A_tr = A[tr_ind,:]
B_tr = B[tr_ind,:]
cell_tr = cell[tr_ind,:]
emb_tr=emb[tr_ind,:]
synergies_tr = synergies[tr_ind]

X1=np.concatenate((A_tr, cell_tr), axis=1)
X2=np.concatenate((B_tr, cell_tr), axis=1)
X3=np.concatenate((X1, B_tr), axis=1)
y =synergies_tr
X4=emb_tr

test_fold=labels['fold'].values.tolist()
test_fold=[5 if i ==1 else i for i in test_fold]
test_fold=[1 if i ==4 else i for i in test_fold]
test_fold=list(filter(lambda x : x != 5, test_fold))

print("load data is ok========================")
ps = PredefinedSplit(test_fold=test_fold)
ps.get_n_splits()

# calculate weights for weighted MSE loss
#min_s = np.amin(train_data['y'])
#loss_weight = np.log(train_data['y'] - min_s + np.e)

# load architecture file
architecture = pd.read_csv('architectureCrossNet.txt')

# prepare layers of the model and the model name
layers = {}
layers['DSN_1'] = architecture['DSN_1'][0] # layers of Drug Synergy Network 1
layers['DSN_2'] = architecture['DSN_2'][0] # layers of Drug Synergy Network 2
layers['SPN'] = architecture['SPN'][0] # layers of Synergy Prediction Network
modelName = args.saved_model_name # name of the model to save the weights

# define constant parameters of MatchMaker
l_rate = 0.0001
inDrop = 0.2
drop = 0.5
max_epoch = 1000
batch_size = 128
earlyStop_patience = 100


k=0
mse=[]
for train_index, test_index in ps.split():
    log_dir = "../logs/fit/SubtypeCrossNet/K"+str(k)+'__' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    modelName ='Kfold/SubtypeCrossNetK'+str(k)+'.h5'
    train_data = {}
    val_data = {}
    train_data['drug1'] = X1[train_index]
    train_data['drug2'] = X2[train_index]
    train_data['full'] = X3[train_index]
    train_data['emb'] = X4[train_index]
    train_data['y'] = y[train_index]
    # calculate weights for weighted MSE loss
    min_s = np.amin(train_data['y'])
    loss_weight = np.log(train_data['y'] - min_s + np.e)

    val_data['drug1'] = X1[test_index]
    val_data['drug2'] = X2[test_index]
    val_data['full'] = X3[test_index]
    val_data['emb'] = X4[test_index]
    val_data['y'] = y[test_index]
    model = matchmakerCrossNetEmb.generate_network(train_data, layers, inDrop, drop)


    if (args.train_test_mode == 1):
    # if we are in training mode
        model,val_loss = matchmakerCrossNetEmb.trainer(model, l_rate, train_data, val_data, max_epoch, batch_size,
                                earlyStop_patience, modelName,loss_weight,log_dir)

# load the best model
    model.load_weights(modelName)

# predict in Drug1, Drug2 order
    pred1 = matchmakerCrossNetEmb.predict(model, [test_data["emb"],test_data["full"],test_data['drug1'],test_data['drug2']])
    mse_value = performance_metrics.mse(test_data['y'], pred1)
    spearman_value = performance_metrics.spearman(test_data['y'], pred1)
    pearson_value = performance_metrics.pearson(test_data['y'], pred1)

# predict in Drug2, Drug1 order
    pred2 = matchmakerCrossNetEmb.predict(model, [test_data["emb"],test_data["full"],test_data['drug2'],test_data['drug1']])

# take the mean for final prediction
    pred = (pred1 + pred2) / 2

    np.savetxt("Kfold/pred_final-" + str(k) + ".txt", np.asarray(pred), delimiter=",")
    np.savetxt("Kfold/y_test_final-" + str(k) + ".txt", np.asarray(test_data['y']), delimiter=",")

    mse_value = performance_metrics.mse(test_data['y'], pred)
    spearman_value = performance_metrics.spearman(test_data['y'], pred)
    pearson_value = performance_metrics.pearson(test_data['y'], pred)
    mse.append(mse_value)

    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(val_loss, label=('validation loss'),color=('b'))
    ax.legend()

    plt.savefig('Kfold/' + str(k) + '.png')
    k = k + 1

print("%.2f%% (+/- %.2f%%)" % (np.mean(mse), np.std(mse)))



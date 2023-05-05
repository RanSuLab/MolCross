import tensorflow as tf
import pandas as pd
import numpy as np
import json
import performance_metrics
from tensorflow.keras import backend as K
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate, BatchNormalization, Activation, Conv1D, MaxPooling1D, Flatten, Lambda, Add, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from helper_funcs import normalize




from tensorflow.keras.layers import Layer
class ComputeLayer(Layer):
    def __init__(self, W, b,**kwargs):
        self.W=W
        self.b=b
        super(ComputeLayer, self).__init__( **kwargs)
    def call(self,dim, inputs,input):
        return Add()([K.sum(self.W * K.batch_dot(K.reshape(inputs, (-1, 1, dim)), input), 1, keepdims = True), self.b, inputs])




class CrossLayer(Layer):
    def __init__(self, output_dim,num_layer, **kwargs):
        self.output_dim = output_dim
        self.num_layer = num_layer
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[1]
        self.W = []
        self.bias = []
        for i in range(self.num_layer):
            self.W.append(self.add_weight(shape = [1, self.input_dim], initializer = 'glorot_uniform',regularizer='l2', name = 'w_' + str(i), trainable = True))
            self.bias.append(self.add_weight(shape = [1, self.input_dim], initializer = 'zeros',regularizer='l2', name = 'b_' + str(i), trainable = True))

        self.built = True

    def call(self, input):

        for i in range(self.num_layer):
            compute=ComputeLayer( self.W[i],self.bias[i],name = "Compute_layer")

            if i == 0:
                cross = compute(self.input_dim,input,input)
            else:
                cross = compute(self.input_dim,cross,input)

        return Flatten()(cross)

    def compute_output_shape(self, input_shape):
        return (None, self.output_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "arg1": self.output_dim,
            "arg2": self.num_layer,
        })
        return config

class EmbLayer(Layer):
    def __init__(self, num_layer,**kwargs):
        self.num_layer = num_layer
        self.embed_layers = {
            'embed_0' : Embedding(6, 64),
            'embed_1': Embedding(36, 64),
            'embed_2': Embedding(4, 64)
        }
        super(EmbLayer, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](inputs[:, i]) if i==2
                                  else self.embed_layers['embed_{}'.format(i)](inputs[:, i]-1)
                                  for i in range(inputs.shape[1])], axis=1)
        embedding = tf.reshape(sparse_embed, [-1, 64*3])

        return embedding

    def get_config(self):
        config = super().get_config()
        config.update({
            "arg1": self.num_layer,
        })
        return config


def data_loader(drug1_chemicals,drug2_chemicals,cell_line_gex,comb_data_name):
    print("File reading ...")
    comb_data = pd.read_csv(comb_data_name, sep="\t")
    cell_line = pd.read_csv(cell_line_gex,header=None)
    chem1 = pd.read_csv(drug1_chemicals,header=None)
    chem2 = pd.read_csv(drug2_chemicals,header=None)
    synergies = np.array(comb_data["synergy_loewe"])

    cell_line = np.array(cell_line.values)
    chem1 = np.array(chem1.values)
    chem2 = np.array(chem2.values)
    return chem1, chem2, cell_line, synergies


def prepare_data(chem1, chem2, cell_line, synergies, norm, train_ind_fname, val_ind_fname, test_ind_fname):
    print("Data normalization and preparation of train/validation/test data")
    test_ind = list(np.loadtxt(test_ind_fname,dtype=np.int,delimiter=','))
    val_ind = list(np.loadtxt(val_ind_fname,dtype=np.int,delimiter=','))
    train_ind = list(np.loadtxt(train_ind_fname,dtype=np.int,delimiter=','))

    train_data = {}
    val_data = {}
    test_data = {}


    train1 = np.concatenate((chem1[train_ind,:],chem2[train_ind,:]),axis=0)

    train_data['drug1'], mean1, std1, mean2, std2, feat_filt = normalize(train1, norm=norm)
    val_data['drug1'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(chem1[val_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
    test_data['drug1'], mean1, std1, mean2, std2, feat_filt = normalize(chem1[test_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)


    train2 = np.concatenate((chem2[train_ind,:],chem1[train_ind,:]),axis=0)
    train_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(train2, norm=norm)
    val_data['drug2'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(chem2[val_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
    test_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(chem2[test_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)

    train3 = np.concatenate((cell_line[train_ind,:],cell_line[train_ind,:]),axis=0)
    train_cell_line, mean1, std1, mean2, std2, feat_filt = normalize(train3, norm=norm)
    val_cell_line, mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(cell_line[val_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
    test_cell_line, mean1, std1, mean2, std2, feat_filt = normalize(cell_line[test_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)

    train_data['drug1'] = np.concatenate((train_data['drug1'],train_cell_line),axis=1)
    train_data['full'] = np.concatenate((train_data['drug1'], train_data['drug2']), axis=1)
    train_data['drug2'] = np.concatenate((train_data['drug2'],train_cell_line),axis=1)
    train_data['cell'] = train_cell_line


    val_data['drug1'] = np.concatenate((val_data['drug1'],val_cell_line),axis=1)
    val_data['full'] = np.concatenate((val_data['drug1'], val_data['drug2']), axis=1)
    val_data['drug2'] = np.concatenate((val_data['drug2'],val_cell_line),axis=1)
    val_data['cell'] = val_cell_line


    test_data['drug1'] = np.concatenate((test_data['drug1'],test_cell_line),axis=1)
    test_data['full'] = np.concatenate((test_data['drug1'], test_data['drug2']), axis=1)
    test_data['drug2'] = np.concatenate((test_data['drug2'],test_cell_line),axis=1)
    test_data['cell'] = test_cell_line


    train_data['y'] = np.concatenate((synergies[train_ind],synergies[train_ind]),axis=0)
    val_data['y'] = synergies[val_ind]
    test_data['y'] = synergies[test_ind]
    print(test_data['drug1'].shape)
    print(test_data['drug2'].shape)
    print(test_data['cell'].shape)
    return train_data, val_data, test_data

def generate_network(train, layers, inDrop, drop):
    # fill the architecture params from dict
    dsn1_layers = layers["DSN_1"].split("-")
    dsn2_layers = layers["DSN_2"].split("-")
    snp_layers = layers["SPN"].split("-")

    input_embed = Input(shape=(train["emb"].shape[1],))
    emb_layer=EmbLayer(num_layer = 3,name='emb_layer')
    concatModel_emb=emb_layer(input_embed)


    input_cross = Input(shape=(train["full"].shape[1],))
    emb_cross=concatenate([input_cross, concatModel_emb])

    cross_layers=CrossLayer(output_dim=(emb_cross.shape[1]),num_layer = 6, name = "cross_layer")
    cross_output=cross_layers(emb_cross)
    cross_output = BatchNormalization()(cross_output)


    # contruct two parallel networks
    for l in range(len(dsn1_layers)):
        if l == 0:
            input_drug1    = Input(shape=(train["drug1"].shape[1],))
            emb_drug1 = concatenate([input_drug1, concatModel_emb])
            middle_layer = Dense(int(dsn1_layers[l]), activation='relu', kernel_initializer='he_normal')(emb_drug1)
            middle_layer = Dropout(float(inDrop))(middle_layer)

        elif l == (len(dsn1_layers)-1):
            dsn1_output = Dense(int(dsn1_layers[l]), activation='linear')(middle_layer)
        else:
            middle_layer = Dense(int(dsn1_layers[l]), activation='relu')(middle_layer)
            middle_layer = Dropout(float(drop))(middle_layer)

    for l in range(len(dsn2_layers)):
        if l == 0:
            input_drug2    = Input(shape=(train["drug2"].shape[1],))
            emb_drug2 = concatenate([input_drug2, concatModel_emb])
            middle_layer = Dense(int(dsn2_layers[l]), activation='relu', kernel_initializer='he_normal')(emb_drug2)
            middle_layer = Dropout(float(inDrop))(middle_layer)

        elif l == (len(dsn2_layers)-1):
            dsn2_output = Dense(int(dsn2_layers[l]), activation='linear')(middle_layer)
        else:
            middle_layer = Dense(int(dsn2_layers[l]), activation='relu')(middle_layer)
            middle_layer = Dropout(float(drop))(middle_layer)


    concatModel = concatenate([dsn1_output, dsn2_output])

    for snp_layer in range(len(snp_layers)):
        if len(snp_layers) == 1:
            snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(concatModel)

            snp_output = Dense(1, activation='linear')(snpFC)
        else:
            # more than one FC layer at concat
            if snp_layer == 0:
                snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(concatModel)
                snpFC = Dropout(float(drop))(snpFC)

            elif snp_layer == (len(snp_layers)-1):
                snp_output  = Dense(int(snp_layers[snp_layer]), activation='linear')(snpFC)

            else:
                snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(snpFC)
                snpFC = Dropout(float(drop))(snpFC)


    concatModel2 = concatenate([cross_output, snp_output])
    output=Dense(1, activation='linear')(concatModel2)
    model = Model([input_embed,input_cross, input_drug1, input_drug2], output)


    model.summary()
    return model


def trainer(model, l_rate, train, val, epo, batch_size, earlyStop, modelName,weights,log_d):
    cb_check = ModelCheckpoint((modelName), verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_d, histogram_freq=1)
    model.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(learning_rate=float(l_rate), beta_1=0.9, beta_2=0.99, epsilon=1e-08,amsgrad=False))

    hist = model.fit([train["emb"],train["full"],train["drug1"], train["drug2"]], train["y"], epochs=epo, shuffle=True, batch_size=batch_size,
                     verbose=1,
                     validation_data=([val["emb"],val["full"],val["drug1"], val["drug2"]], val["y"]), sample_weight=weights,
                     callbacks=[EarlyStopping(monitor='val_loss', mode='auto', patience=earlyStop), cb_check,tensorboard_callback])

    val_loss = hist.history['val_loss']

    return model,val_loss

def predict(model, data):
    pred = model.predict(data)
    return pred.flatten()

def random_batch(y, batch_size):
    idx = np.random.randint(0, len(y), size = batch_size)
    return idx

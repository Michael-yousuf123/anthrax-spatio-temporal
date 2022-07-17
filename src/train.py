import tensorflow as tf
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from kerashypetune import KerasGridSearch
from keras.layers import Dense, LSTM, TimeDistributed, RepeatVector
import random 
from feat import *

def set_seed(seed):
    
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_model(param):
    
    set_seed(33)

    model = Sequential()
    model.add(LSTM(param['unit'], activation=param['act']))
    model.add(RepeatVector(end_id))
    model.add(LSTM(param['unit'], activation=param['act'], return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    
    opt_choices = {'adam': Adam(),
                   'rms': RMSprop()}
    
    opt = opt_choices[param['opt']]
    opt.lr = param['lr'] 
    
    model.compile(opt, 'mae')
    
    return model

def model_tuning(X_train, y_train, X_test, y_test, param_grid):
    hypermodel = get_model
    es = EarlyStopping(patience=10, verbose=0, min_delta=0.001, monitor='val_loss', mode='auto', restore_best_weights=True)
    kgs = KerasGridSearch(hypermodel, param_grid, monitor='val_loss', greater_is_better=False, tuner_verbose=1)
    save_weights_at = os.path.join('keras_models', 'best_model.{epoch:02d}-{val_loss:.4f}.hdf5')
    save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)
    kgs.search(X_train, y_train, validation_data=(X_test, y_test), callbacks=[save_best])
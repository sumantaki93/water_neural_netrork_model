from pyscf import gto,dft,lib,scf
from pyscf import grad
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import lammps
from lammps import LMP_STYLE_GLOBAL,LMP_TYPE_ARRAY
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers

fname="Input_test_data_lammps_thermal_sample"

mol_hf = gto.M(atom='H 0 0 0; H 0 0 0')  # Just geeting the mf_hf object from pyscf
mf_hf = dft.RKS(mol_hf)


if __name__ == "__main__":


    n=100
    training_data_shape=100
    orbitals= np.zeros((n,24,24),dtype=float)
    density_matrix= np.zeros((n,24,24),dtype=float)
    upper_traingle_dm=np.zeros((n,24,24),dtype=float)
    
    x_1=np.zeros(n,dtype=float) 
    x_2=np.zeros(n,dtype=float) 
    y_2=np.zeros(n,dtype=float)
    
    x_1_train=np.zeros(training_data_shape,dtype=float) 
    x_2_train=np.zeros(training_data_shape,dtype=float) 
    y_2_train=np.zeros(training_data_shape,dtype=float)
    
    a =np.zeros(shape=(training_data_shape,3))
    train_dm=np.zeros(training_data_shape,dtype=float)
    
    
    
    
    for i in range (n):
        mo_coeff = lib.chkfile.load("{}/chkfile/input_data_h20{}.chk".format(fname,i) , 'scf/mo_coeff')
        orbitals[i,:,:]=mo_coeff
        mo_occ = lib.chkfile.load("{}/chkfile/input_data_h20{}.chk".format(fname,i) , 'scf/mo_occ')
        dm = mf_hf.make_rdm1(mo_coeff, mo_occ)
        density_matrix[i,:,:]=dm
        mask = np.less_equal(np.absolute(density_matrix[i,:,:]),1e-03)
        density_matrix[i,mask]=0.00
        upper_traingle_dm[i,:,:]=np.triu(density_matrix[i,:])
        file=open('{}/geometry/X_coo_for_H1_{}.dat'.format(fname,i),'r')
        lines=file.read()
        file.close()
        x_1[i]=float(lines)
        file=open('{}/geometry/X_coo_for_H2_{}.dat'.format(fname,i),'r')
        lines=file.read()
        file.close()
        x_2[i]=float(lines)
        file=open('{}/geometry/Y_coo_for_H2_{}.dat'.format(fname,i),'r')
        lines=file.read()
        file.close()
        y_2[i]=float(lines)
        






    x_1_train[:]=x_1[0:training_data_shape]
    x_2_train[:]=x_2[0:training_data_shape]
    y_2_train[:]=y_2[0:training_data_shape]
    train_dm=np.array([d.flatten() for d in density_matrix[0:training_data_shape]])
    a[:,:] =(np.column_stack([x_1_train,x_2_train,y_2_train]))
    
   
    #--------------------------------------------------------------------------------
    inputs=keras.Input(shape=(3))
    dense=layers.Dense(18,kernel_regularizer=regularizers.l2(0.0000),activation='relu')
    x=dense(inputs)
    x=layers.Dense(32,kernel_regularizer=regularizers.l2(0.0000),activation='relu')(x)    
    outputs=layers.Dense(24*5)(x)
    r1=layers.Reshape((24,5))(outputs)
    t1=layers.Permute((2,1))(r1)
    d1=layers.Dot(axes=(2,1))([r1,t1])
    f1=layers.Flatten()(d1)
    #--------------------------------------------------------------------------------
    
    water_ml_model= keras.Model(inputs=inputs, outputs=f1, name="water_ml_model")
    
    
    water_ml_model.compile(loss =tf.losses.MeanSquaredError(),
                            optimizer = tf.optimizers.Adam())
    
    
    water_ml_model.fit(a,train_dm,shuffle=True, epochs=150, validation_split=0.05)

    
    dmpredicted=water_ml_model.predict(a)


    def R_squared(y, y_pred):
        residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
        total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
        r2 = tf.subtract(1.0, tf.divide(residual, total))
        return r2

    r_square_train=R_squared(train_dm,dmpredicted)
    error_1_train=np.mean(np.square(train_dm-dmpredicted))
    RMSE_train=np.sqrt(error_1_train)
    MAE_train=np.mean(np.absolute(train_dm-dmpredicted))
    MAX_train=np.max(np.absolute(train_dm-dmpredicted))



    print()
    print('------------------------------------------')
    print('Number of Training  structures:', training_data_shape, )
    print('---------------------------------------')
    print()
    print ('           |        R2       |      RMSE       |       MAE       |      MaxAE       ')
    print(f'Training   |  {r_square_train:8.6e}   |  {RMSE_train: 8.6e} |  {MAE_train: 8.6e} |  {MAX_train: 8.6e}')





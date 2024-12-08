import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io
import math
import time
import sys
import os
from Compute_Jacobian import jacobian
tf.disable_v2_behavior()


def tf_session():
    # tf session
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    config.gpu_options.force_gpu_compatible = True
    sess = tf.Session(config=config)

    # init
    init = tf.global_variables_initializer()
    sess.run(init)

    return sess


def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return tf.reduce_mean(tf.square(pred - exact))


def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y)
    G = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    Y_x = tf.gradients(G, dummy, colocate_gradients_with_ops=True)[0]
    return Y_x


class neural_net(object):

    def __init__(self, *inputs, layers):

        self.layers = layers
        self.num_layers = len(self.layers)

        X = np.concatenate(inputs, 1)

        self.X_mean = X.mean(0, keepdims=True)
        self.X_std  = X.std(0, keepdims=True)

        self.weights = []
        self.biases = []
        self.gammas = []

        for l in range(0,self.num_layers-1):
            in_dim = self.layers[l]
            out_dim = self.layers[l+1]
            W = self.xavier_init(size=[in_dim, out_dim])
            b = np.zeros([1, out_dim])
            g = np.ones([1, out_dim])
            # weights, biases
            self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True))
            self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
            self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))
            
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def __call__(self, *inputs):

        H = (tf.concat(inputs, 1) - self.X_mean) / self.X_std
    
        for l in range(0, self.num_layers-1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            V = W/tf.norm(W, axis = 0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            # add bias
            H = g * H + b
            # activation
            if l < self.num_layers-2:
                H = H * tf.sigmoid(H) 
                
        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)
    
        return Y, self.weights, self.biases, self.gammas
    
    
def Navier_Stokes_2D(vr, vth, p, r, th, t, Re):

    Y = tf.concat([vr, vth, p], 1)

    Y_r  = fwd_gradients(Y, r)
    Y_th = fwd_gradients(Y, th)
    Y_t  = fwd_gradients(Y, t)
    
    Y_rr   = fwd_gradients(Y_r, r)
    Y_thth = fwd_gradients(Y_th, th)

    vr  = Y[:,0:1]
    vth = Y[:,1:2]
    p   = Y[:,2:3]
        
    vr_r  = Y_r[:,0:1]
    vr_th = Y_th[:,0:1]
    vr_t  = Y_t[:,0:1]
    
    vth_r  = Y_r[:,1:2]
    vth_th = Y_th[:,1:2]
    vth_t  = Y_t[:,1:2]
    
    p_r  = Y_r[:,2:3]
    p_th = Y_th[:,2:3]
    
    vr_rr    = Y_rr[:,0:1]
    vr_thth  = Y_thth[:,0:1]
    vth_rr   = Y_rr[:,1:2]
    vth_thth = Y_thth[:,1:2]
    
    j1 = vr_t  + vr*vr_r  + (vth/r)*vr_th - vth*vth/r + p_r    - (1/Re)*(vr_r/r  - vr/(r*r)  + vr_rr  + vr_thth/(r*r)  - 2*vth_th/(r*r))
    j2 = vth_t + vr*vth_r + vth*vth_th/r  + vr*vth/r  + p_th/r - (1/Re)*(vth_r/r - vth/(r*r) + vth_rr + vth_thth/(r*r) + 2*vr_th/(r*r))
    j3 = (vr   + r*vr_r   + vth_th)/r
    
    return j1, j2, j3


class AIVFM(object):

    def __init__(self, r_data, th_data, t_data, vr_data,

                       r_cons, th_cons, t_cons,

                       r_bc,   th_bc,   t_bc,   vth_bc,

                       layers, batch_size, Re, v_en, kernel_size):

        # 
        self.layers      = layers
        self.batch_size  = batch_size
        self.Re          = Re
        self.v_en        = v_en
        self.kernel_size = kernel_size

        # Initialize the weights of each loss term as ones
        self.lam_vr_val = np.array(1.0)
        self.lam_bc_val = np.array(1.0)
        self.lam_j1_val = np.array(1.0)
        self.lam_j2_val = np.array(1.0)
        self.lam_j3_val = np.array(1.0)

        # 
        [self.r_data, self.th_data, self.t_data, self.vr_data] = [r_data, th_data, t_data, vr_data]
        [self.r_cons, self.th_cons, self.t_cons]               = [r_cons, th_cons, t_cons]
        [self.r_bc,   self.th_bc,   self.t_bc,   self.vth_bc]  = [r_bc,   th_bc,   t_bc,   vth_bc]

        # Placeholders
        [self.r_data_tf, self.th_data_tf, self.t_data_tf, self.vr_data_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
        [self.r_cons_tf, self.th_cons_tf, self.t_cons_tf]                  = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
        [self.r_bc_tf,   self.th_bc_tf,   self.t_bc_tf,   self.vth_bc_tf]  = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]

        self.lam_vr_tf = tf.placeholder(tf.float32, shape=self.lam_vr_val.shape)
        self.lam_bc_tf = tf.placeholder(tf.float32, shape=self.lam_bc_val.shape)
        self.lam_j1_tf = tf.placeholder(tf.float32, shape=self.lam_j1_val.shape)
        self.lam_j2_tf = tf.placeholder(tf.float32, shape=self.lam_j2_val.shape)
        self.lam_j3_tf = tf.placeholder(tf.float32, shape=self.lam_j3_val.shape)

        # Define placeholders for NTK computation
        D = self.kernel_size 

        [self.r_vr_ntk_tf, self.th_vr_ntk_tf, self.t_vr_ntk_tf] = [tf.placeholder(tf.float32, shape=[D, 1]) for _ in range(3)]
        [self.r_bc_ntk_tf, self.th_bc_ntk_tf, self.t_bc_ntk_tf] = [tf.placeholder(tf.float32, shape=[D, 1]) for _ in range(3)]
        [self.r_js_ntk_tf, self.th_js_ntk_tf, self.t_js_ntk_tf] = [tf.placeholder(tf.float32, shape=[D, 1]) for _ in range(3)]

        # Feed forward fully-connected neural network
        self.FCN_net = neural_net(self.r_data, self.th_data, self.t_data, layers=self.layers)

        [self.vr_data_pred,
         self.vth_data_pred,
         self.p_data_pred], self.weights, self.biases, self.gammas  =  self.FCN_net(self.r_data_tf,
                                                                                    self.th_data_tf,
                                                                                    self.t_data_tf)

        [self.vr_cons_pred,
         self.vth_cons_pred,
         self.p_cons_pred], _, _, _   =  self.FCN_net(self.r_cons_tf,
                                                      self.th_cons_tf,
                                                      self.t_cons_tf)

        [self.vr_bc_pred,
         self.vth_bc_pred,
         _], _, _, _         =  self.FCN_net(self.r_bc_tf,
                                             self.th_bc_tf,
                                             self.t_bc_tf)

        [self.j1_cons_pred,
         self.j2_cons_pred,
         self.j3_cons_pred] = Navier_Stokes_2D(self.vr_cons_pred,
                                               self.vth_cons_pred,
                                               self.p_cons_pred,
                                               self.r_cons_tf,
                                               self.th_cons_tf,
                                               self.t_cons_tf,
                                               self.Re)

        [self.vr_ntk_pred,
         _,
         _], _, _, _   =  self.FCN_net(self.r_vr_ntk_tf,
                                       self.th_vr_ntk_tf,
                                       self.t_vr_ntk_tf)

        [_,
         self.vth_bc_ntk_pred,
         _], _, _, _            =  self.FCN_net(self.r_bc_ntk_tf,
                                                self.th_bc_ntk_tf,
                                                self.t_bc_ntk_tf)

        [self.vr_js_ntk_pred,
         self.vth_js_ntk_pred,
         self.p_js_ntk_pred], _, _, _   =  self.FCN_net(self.r_js_ntk_tf,
                                                        self.th_js_ntk_tf,
                                                        self.t_js_ntk_tf)

        [self.j1_ntk_pred,
         self.j2_ntk_pred,
         self.j3_ntk_pred] = Navier_Stokes_2D(self.vr_js_ntk_pred,
                                              self.vth_js_ntk_pred,
                                              self.p_js_ntk_pred,
                                              self.r_js_ntk_tf,
                                              self.th_js_ntk_tf,
                                              self.t_js_ntk_tf,
                                              self.Re)

        # Doppler training data loss
        self.loss_vr = mean_squared_error(tf.math.cos((tf.constant(math.pi)*self.vr_data_pred/self.v_en)), tf.math.cos((tf.constant(math.pi)*self.vr_data_tf/self.v_en))) + \
                       mean_squared_error(tf.math.sin((tf.constant(math.pi)*self.vr_data_pred/self.v_en)), tf.math.sin((tf.constant(math.pi)*self.vr_data_tf/self.v_en)))

        # No-slip boundary condition loss
        self.loss_bc = mean_squared_error(self.vth_bc_pred,  self.vth_bc_tf)

        # Navier-Stokes residual equation losses
        self.loss_j1 = mean_squared_error(self.j1_cons_pred, 0.0)
        self.loss_j2 = mean_squared_error(self.j2_cons_pred, 0.0)
        self.loss_j3 = mean_squared_error(self.j3_cons_pred, 0.0)

        # Total loss
        self.loss_total = self.lam_vr_tf * self.loss_vr + self.lam_bc_tf * self.loss_bc + self.lam_j1_tf * self.loss_j1 + self.lam_j2_tf * self.loss_j2 + self.lam_j3_tf * self.loss_j3

        # Compute the Jacobian for weights and biases in each hidden layer
        self.J_vr = self.compute_jacobian(self.vr_ntk_pred)
        self.J_bc = self.compute_jacobian(self.vth_bc_ntk_pred)
        self.J_j1 = self.compute_jacobian(self.j1_ntk_pred)
        self.J_j2 = self.compute_jacobian(self.j2_ntk_pred)
        self.J_j3 = self.compute_jacobian(self.j3_ntk_pred)

        self.K_vr = self.compute_ntk(self.J_vr, D, self.J_vr, D)
        self.K_bc = self.compute_ntk(self.J_bc, D, self.J_bc, D)
        self.K_j1 = self.compute_ntk(self.J_j1, D, self.J_j1, D)
        self.K_j2 = self.compute_ntk(self.J_j2, D, self.J_j2, D)
        self.K_j3 = self.compute_ntk(self.J_j3, D, self.J_j3, D)

        # Optimizer specifications
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_total)
        
        self.sess = tf_session()


    # Compute Jacobian for each weights and biases in each layer and retrun a list 
    def compute_jacobian(self, f):

        J_list =[]
        L = len(self.weights)
        for i in range(L):
            J_w = jacobian(f, self.weights[i])
            J_list.append(J_w)

        for i in range(L):
            J_b = jacobian(f, self.biases[i])
            J_list.append(J_b)

        for i in range(L):
            J_g = jacobian(f, self.gammas[i])
            J_list.append(J_g)

        return J_list

    
    def compute_ntk(self, J1_list, D1, J2_list, D2):

        N = len(J1_list)

        Ker = tf.zeros((D1,D2))

        for k in range(N):

            J1 = tf.reshape(J1_list[k], shape=(D1,-1))
            J2 = tf.reshape(J2_list[k], shape=(D2,-1))

            K = tf.matmul(J1, tf.transpose(J2))

            Ker = Ker + K

        return Ker


    def train(self, maxIte, starter_lr,

              output_weight_path, output_loss_path, output_filename):

        self.loss_vr_save = []
        self.loss_bc_save = []
        self.loss_j1_save = []
        self.loss_j2_save = []
        self.loss_j3_save = []

        self.lam_log_vr = []
        self.lam_log_bc = []
        self.lam_log_j1 = []
        self.lam_log_j2 = []
        self.lam_log_j3 = []

        self.K_log_vr = []
        self.K_log_bc = []
        self.K_log_j1 = []
        self.K_log_j2 = []
        self.K_log_j3 = []

        self.time_save = []
        start_time     = time.time()
        running_time   = 0
        it             = 0
            
        N_data = self.t_data.shape[0]
        N_cons = self.t_cons.shape[0]

        while it < maxIte:

            idx_data = np.random.choice(N_data, self.batch_size)
            idx_cons = np.random.choice(N_cons, self.batch_size)

            (r_data_batch,
             th_data_batch,
             t_data_batch,
             vr_data_batch) = (self.r_data[idx_data,  :],
                               self.th_data[idx_data, :],
                               self.t_data[idx_data,  :],
                               self.vr_data[idx_data, :])

            (r_cons_batch,
             th_cons_batch,
             t_cons_batch) = (self.r_cons[idx_cons,  :],
                              self.th_cons[idx_cons, :],
                              self.t_cons[idx_cons,  :])

            # Learning rate scheduler
            decay_ite  = 10000
            decay_rate = 0.9

            learning_rate = starter_lr * decay_rate ** (it / decay_ite)

            tf_dict = {self.r_data_tf: r_data_batch,
                       self.th_data_tf: th_data_batch,
                       self.t_data_tf: t_data_batch,
                       self.vr_data_tf: vr_data_batch,
                       self.r_cons_tf: r_cons_batch,
                       self.th_cons_tf: th_cons_batch,
                       self.t_cons_tf: t_cons_batch,
                       self.r_bc_tf: self.r_bc,
                       self.th_bc_tf: self.th_bc,
                       self.t_bc_tf: self.t_bc,
                       self.vth_bc_tf: self.vth_bc,
                       self.lam_vr_tf: self.lam_vr_val,
                       self.lam_bc_tf: self.lam_bc_val,
                       self.lam_j1_tf: self.lam_j1_val,
                       self.lam_j2_tf: self.lam_j2_val,
                       self.lam_j3_tf: self.lam_j3_val,
                       self.learning_rate: learning_rate}

            self.sess.run([self.train_op], tf_dict)

            if it % 10 == 0:
                
                loss_vr_value = self.sess.run(self.loss_vr, tf_dict)
                loss_j1_value = self.sess.run(self.loss_j1, tf_dict)
                loss_j2_value = self.sess.run(self.loss_j2, tf_dict)
                loss_j3_value = self.sess.run(self.loss_j3, tf_dict)
                loss_bc_value = self.sess.run(self.loss_bc, tf_dict)

                elapsed = time.time() - start_time
                running_time += elapsed/3600.0
                [loss_value,
                 learning_rate_value,
                 lam_vr_value,
                 lam_bc_value,
                 lam_j1_value,
                 lam_j2_value,
                 lam_j3_value] = self.sess.run([self.loss_total,
                                                self.learning_rate,
                                                self.lam_vr_tf,
                                                self.lam_bc_tf,
                                                self.lam_j1_tf,
                                                self.lam_j2_tf,
                                                self.lam_j3_tf], tf_dict)

                print(output_filename, ' ', 'It: %d/%d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh, Learning Rate: %.1e'
                      %(it, maxIte, loss_value, elapsed, running_time, learning_rate_value))
                print('lam_vr: %.3f, lam_bc: %.3f, lam_j1: %.3f, lam_j2: %.3f, lam_j3: %.3f' % (lam_vr_value, lam_bc_value, lam_j1_value, lam_j2_value, lam_j3_value))
                print("\n")
                
                self.loss_vr_save.append(loss_vr_value)
                self.loss_bc_save.append(loss_bc_value)
                self.loss_j1_save.append(loss_j1_value)
                self.loss_j2_save.append(loss_j2_value)
                self.loss_j3_save.append(loss_j3_value)

                self.lam_log_vr.append(self.lam_vr_val)
                self.lam_log_bc.append(self.lam_bc_val)
                self.lam_log_j1.append(self.lam_j1_val)
                self.lam_log_j2.append(self.lam_j2_val)
                self.lam_log_j3.append(self.lam_j3_val)

                self.time_save.append(running_time)

                if it % 100 == 0 and it !=0 :
                    print("Compute NTK...")

                    idx_data_ntk = np.random.choice(N_data, self.kernel_size)
                    idx_cons_ntk = np.random.choice(N_cons, self.kernel_size)
                    idx_bc_ntk   = np.random.choice(self.t_bc.shape[0], self.kernel_size)

                    (r_data_ntk_batch,
                     th_data_ntk_batch,
                     t_data_ntk_batch)  =  (self.r_data[idx_data_ntk,:],
                                            self.th_data[idx_data_ntk,:],
                                            self.t_data[idx_data_ntk,:])

                    (r_cons_ntk_batch,
                     th_cons_ntk_batch,
                     t_cons_ntk_batch)  =  (self.r_cons[idx_cons_ntk,:],
                                            self.th_cons[idx_cons_ntk,:],
                                            self.t_cons[idx_cons_ntk,:])

                    (r_bc_ntk_batch,
                     th_bc_ntk_batch,
                     t_bc_ntk_batch)  =  (self.r_bc[idx_bc_ntk,:],
                                          self.th_bc[idx_bc_ntk,:],
                                          self.t_bc[idx_bc_ntk,:])
                    
                    tf_dict = {self.r_vr_ntk_tf: r_data_ntk_batch,
                               self.th_vr_ntk_tf: th_data_ntk_batch,
                               self.t_vr_ntk_tf: t_data_ntk_batch,
                               self.r_bc_ntk_tf: r_bc_ntk_batch,
                               self.th_bc_ntk_tf: th_bc_ntk_batch,
                               self.t_bc_ntk_tf: t_bc_ntk_batch,
                               self.r_js_ntk_tf: r_cons_ntk_batch,
                               self.th_js_ntk_tf: th_cons_ntk_batch,
                               self.t_js_ntk_tf: t_cons_ntk_batch}

                    K_vr_value, K_bc_value, K_j1_value, K_j2_value, K_j3_value =  self.sess.run([self.K_vr, self.K_bc, self.K_j1, self.K_j2, self.K_j3], tf_dict)

                    trace_K = np.trace(K_vr_value) + np.trace(K_bc_value) + np.trace(K_j1_value) + np.trace(K_j2_value) + np.trace(K_j3_value)

                    self.lam_vr_val = trace_K / np.trace(K_vr_value)
                    self.lam_bc_val = trace_K / np.trace(K_bc_value)
                    self.lam_j1_val = trace_K / np.trace(K_j1_value)
                    self.lam_j2_val = trace_K / np.trace(K_j2_value)
                    self.lam_j3_val = trace_K / np.trace(K_j3_value)

                sys.stdout.flush()
                start_time = time.time()

            it += 1

            if it == maxIte:

                print('Saving weights and losses')

                # Store
                self.K_log_vr.append(K_vr_value)
                self.K_log_bc.append(K_bc_value)
                self.K_log_j1.append(K_j1_value)
                self.K_log_j2.append(K_j2_value)
                self.K_log_j3.append(K_j3_value)

                saver = tf.train.Saver()
                saver.save(self.sess, os.path.join(output_weight_path,output_filename,'LV')) 

                scipy.io.savemat(os.path.join(output_loss_path,output_filename + '_loss.mat'), 
                        {'runtime':self.time_save,
                         'loss_vr':self.loss_vr_save,
                         'loss_bc':self.loss_bc_save,
                         'loss_j1':self.loss_j1_save,
                         'loss_j2':self.loss_j2_save,
                         'loss_j3':self.loss_j3_save,
                         'lam_log_vr':self.lam_log_vr,
                         'lam_log_bc':self.lam_log_bc,
                         'lam_log_j1':self.lam_log_j1,
                         'lam_log_j2':self.lam_log_j2,
                         'lam_log_j3':self.lam_log_j3,
                         'K_log_vr':self.K_log_vr,
                         'K_log_bc':self.K_log_bc,
                         'K_log_j1':self.K_log_j1,
                         'K_log_j2':self.K_log_j2,
                         'K_log_j3':self.K_log_j3})
                

    def predict(self, r_star, th_star, t_star):

        tf_dict = {self.r_data_tf: r_star, self.th_data_tf: th_star, self.t_data_tf: t_star}

        vr_star  = self.sess.run(self.vr_data_pred, tf_dict)
        vth_star = self.sess.run(self.vth_data_pred, tf_dict)
        p_star   = self.sess.run(self.p_data_pred, tf_dict)

        return vr_star, vth_star, p_star


if __name__ == "__main__": 

    # Define layers
    layers = [3] + 4 * [150] + [3]

    # Load data
    data = scipy.io.loadmat('../AIVFM_LV_data.mat')

    # Characteristic length scale i.e., MV diameter [cm]
    D = 2.731

    # Characteristic velocity scale i.e., Nyquist velocity [cm/s]
    U = 50

    # Kinematic viscosity [cm2/s]
    nu = 0.04

    # Reynolds number
    Re = D * U / nu
    
    VR  = data['VR_aliased']
    VTh = data['VTh']
    P   = data['P']
    
    masks   = np.ma.make_mask(data['masks'])
    masksbc = np.ma.make_mask(data['masksbc'])
    
    R  = data['R']
    Th = data['Th']
    T  = data['T']
    
    # Reshape and repeats grids into the same shape as VR, VTh, P i.e., [50,220,200]
    R  = np.repeat(R[:, :, np.newaxis],  VR.shape[2], axis=2)
    Th = np.repeat(Th[:, :, np.newaxis], VR.shape[2], axis=2)
    T  = np.tile(T.reshape(1, 1, VR.shape[2]), (VR.shape[0], VR.shape[1], 1))

    # Number of Doppler frames used for training
    N_dop  = 15

    # Number of frames used to impose boundary conditions
    N_bc   = 100

    # Number of frames used to impose physical constraints
    N_cons = 200

    train_data_list = np.arange(start=0, stop=200, step=math.ceil(200/N_dop))
    train_bc_list   = np.arange(start=0, stop=200, step=math.ceil(200/N_bc)) 
    train_cons_list = np.arange(start=0, stop=200, step=math.ceil(200/N_cons))
    
    r_data  = R[:,:, train_data_list][masks[:,:,train_data_list]][:,None]
    th_data = Th[:,:,train_data_list][masks[:,:,train_data_list]][:,None]
    t_data  = T[:,:, train_data_list][masks[:,:,train_data_list]][:,None]
    vr_data = VR[:,:,train_data_list][masks[:,:,train_data_list]][:,None]
    
    r_bc    = R[:,:,   train_bc_list][masksbc[:,:,train_bc_list]][:,None]
    th_bc   = Th[:,:,  train_bc_list][masksbc[:,:,train_bc_list]][:,None]
    t_bc    = T[:,:,   train_bc_list][masksbc[:,:,train_bc_list]][:,None]
    vth_bc  = VTh[:,:, train_bc_list][masksbc[:,:,train_bc_list]][:,None]
    
    r_cons  = R[:,:, train_cons_list][masks[:,:,train_cons_list]][:,None]
    th_cons = Th[:,:,train_cons_list][masks[:,:,train_cons_list]][:,None]
    t_cons  = T[:,:, train_cons_list][masks[:,:,train_cons_list]][:,None]

    # Non-dimensionalization using characteristic scales
    r_data  = r_data / D
    th_data = th_data
    t_data  = t_data / (D/U)
    vr_data = vr_data / U

    r_cons  = r_cons / D
    th_cons = th_cons
    t_cons  = t_cons / (D/U)

    r_bc    = r_bc / D
    th_bc   = th_bc
    t_bc    = t_bc / (D/U)
    vth_bc  = vth_bc / U

    # Nyquist velocity 50 cm/s
    v_en = 50 / U
    
    # Sample 10000 number of points each iteration
    batch_size  = 10000

    # Define kernel size used to create kernel matrix and compute NTK
    kernel_size = 300

    # Maximum training iterations
    maxIte = 300000

    # Initial learning rate
    starter_lr = 1e-3

    output_weight_path = '../weights/'
    output_loss_path   = '../loss/'

    output_filename = 'AIVFM_results'
    
    model = AIVFM(r_data, th_data, t_data, vr_data,
                  
                  r_cons, th_cons, t_cons,
                  
                  r_bc,   th_bc,   t_bc,   vth_bc,
                  
                  layers, batch_size, Re, v_en, kernel_size)


    model.train(maxIte = maxIte, starter_lr = starter_lr,
    
                output_weight_path = output_weight_path, output_loss_path = output_loss_path, output_filename = output_filename) 
    
    
    # Prediction lists
    VR_pred  = []
    VTh_pred = []
    P_pred   = []
    
    for n in range(0,VR.shape[2]):
        
        r_test = R[:,:,n].flatten(order='F')[masks[:,:,n].flatten(order='F')][:,None]
        r_test = r_test / D
        
        th_test = Th[:,:,n].flatten(order='F')[masks[:,:,n].flatten(order='F')][:,None]
        
        t_test = T[:,:,n].flatten(order='F')[masks[:,:,n].flatten(order='F')][:,None]
        t_test = t_test / (D/U)
        
        vr_pred, vth_pred, p_pred = model.predict(r_test, th_test, t_test)
        
        vr_pred  = vr_pred * U
        vth_pred = vth_pred * U
        p_pred   = p_pred * U**2
    
        VR_pred.append(vr_pred)
        VTh_pred.append(vth_pred)
        P_pred.append(p_pred)
    
        scipy.io.savemat(os.path.join('../results/', output_filename + '_%s.mat' % (time.strftime('%m_%d'))),  
                             {'VR_pred':VR_pred, 'VTh_pred':VTh_pred, 'P_pred':P_pred})


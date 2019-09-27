import numpy as np

class IHDP_loader:

    def __init__(self, file_name, split_type, in_sample=False, batch_size=32):
        """dataloader for IHDP dataset

        Arguments:
            file_name {str} -- datapath for IHDP dataset
            split_type {str} -- split type of dataloader. train/valid/test

        Keyword Arguments:
            in_sample {bool} -- for test dataloader we distinguish in/out-sample error (default: {False})
            batch_size {int} -- mini-batch size (default: {32})
        """
        assert (split_type == 'train') or (split_type == 'valid') or (split_type == 'test')
        if in_sample:
            assert split_type == 'test'

        self.split_type = split_type
        self.batch_size = batch_size

        # set data_path
        if self.split_type=='train' or self.split_type=='valid':
            file_name = file_name + '.train.npz'
        elif self.split_type=='test':
            if in_sample:
                file_name = file_name + '.train.npz'
            else:
                file_name = file_name + '.test.npz'

        # get data
        self.data = np.load(file_name)
        self.x, self.t, self.yf, self.ycf, self.mu0, self.mu1 = None, None, None, None, None, None

    def set_id(self, i):
        """modify repetition number for dataloader

        Arguments:
            i {int} -- repetition id
        """
        self.x = self.data['x'][:,:,i]
        self.t = self.data['t'][:,i]
        # concat x and t
        self.feat = np.concatenate((self.x, np.expand_dims(self.t, axis=1)) axis=1)

        self.yf = self.data['yf'][:,i]
        self.ycf = self.data['ycf'][:,i]
        self.mu0 = self.data['mu0'][:,i]
        self.mu1 = self.data['mu1'][:,i]

        self.train_size = int(0.7 * len(self.feat))

        if self.split_type=='train':
            self.feat = self.feat[:self.train_size]
            self.t = self.t[:self.train_size]
            self.yf = self.yf[:self.train_size]
            self.ycf = self.ycf[:self.train_size]
            self.mu0 = self.mu0[:self.train_size]
            self.mu1 = self.mu1[:self.train_size]

        elif self.split_type=='valid':
            self.feat = self.feat[self.train_size:]
            self.t = self.t[self.train_size:]
            self.yf = self.yf[self.train_size:]
            self.ycf = self.ycf[self.train_size:]
            self.mu0 = self.mu0[self.train_size:]
            self.mu1 = self.mu1[self.train_size:]

        # self.x_mean = np.mean(self.feat, axis=0)
        self.yf_0_mean = np.mean(self.yf[self.t==0], axis=0)
        self.yf_1_mean = np.mean(self.yf[self.t==1], axis=0)
        return None

    def __next__(self):
        # work as generator for time efficiency
        if False: #self.split_type=='train':
            mb_arr = np.random.choice(list(range(self.train_size)), self.batch_size)
            mb_x = self.x[mb_arr]
            mb_t = self.t[mb_arr]
            mb_yf = self.yf[mb_arr]
            mb_ycf = self.ycf[mb_arr]
            mb_mu0 = self.mu0[mb_arr]
            mb_mu1 = self.mu1[mb_arr]
            return mb_x, mb_t, mb_yf, mb_ycf, mb_mu0, mb_mu1

        else:
            return self.feat, self.t, self.yf, self.ycf, self.mu0, self.mu1

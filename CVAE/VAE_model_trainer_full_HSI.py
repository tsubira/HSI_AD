from HSI_utils import *
from losses import *
from timeit import default_timer as timer
# import conv_models
import conv_models_AD_full_HSI
# import dense_models
# import VAE_model_trainer

from datetime import datetime, timedelta
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
import numpy as np
# from torchsummary import summary


class HSI_convolutional_anomaly_detector:
    def __init__(self, 
                 dir_results, 
                 zdim=256, 
                 input_shape=(207, 100, 100), 
                 loss_kl = 'sum', 
                 load_weights = False, 
                 loss_reconstruction='bce',
                 level_cams=-4, 
                 verbose = 0):

        # Init input variables
        self.dir_results = dir_results
        self.zdim = zdim
        self.input_shape = input_shape
        self.loss_reconstruction = loss_reconstruction
        self.level_cams = level_cams
        self.verbose = verbose
        self.loss_kl = loss_kl
        self.load_weights = load_weights

        # Init encoder and decoder
        self.conv_encoder = conv_models_AD_full_HSI.Encoder(n_channels=self.input_shape[0], zdim=self.zdim, spatial_dim=self.input_shape[-1], verbose=self.verbose)
        self.conv_decoder = conv_models_AD_full_HSI.Decoder(n_channels=self.input_shape[0], zdim=self.zdim, spatial_dim=self.input_shape[-1], verbose=self.verbose)

        # Prepare the models to run on GPU
        if torch.cuda.is_available():
            self.conv_encoder.cuda()
            self.conv_decoder.cuda()

        # Load model weights from files
        if self.load_weights:
            self.conv_encoder.load_state_dict(torch.load(os.path.join(self.dir_results, 'encoder_weights.pth')))
            self.conv_decoder.load_state_dict(torch.load(os.path.join(self.dir_results, 'decoder_weights.pth')))

        # Set parameters
        self.params = list(self.conv_encoder.parameters()) + list(self.conv_decoder.parameters())

        # Set reconstruction losses
        if self.loss_reconstruction == 'l2':
            self.Lr = torch.nn.MSELoss(reduction='sum')
        elif self.loss_reconstruction == 'bce_logits':
            self.Lr = torch.nn.BCEWithLogitsLoss(reduction='sum')
        elif self.loss_reconstruction == 'bce':
            self.Lr = torch.nn.BCELoss(reduction='sum')
        elif self.loss_reconstruction == 'weighted_l2':
            self.Lr = weighted_MSE

        # Set distribution losses (KL divergence)
        if self.loss_kl == 'sum':
            self.Lkl = kl_loss
        elif self.loss_kl == 'mean':
            self.Lkl = kl_loss_mean

        # Spectral angle losses, for HSI
        self.Lsa = SAD

        # Init additional variables and objects
        self.epochs = 0.
        self.iterations = 0.
        self.init_time = 0.
        self.lr_iteration = 0.
        self.lsa_iteration = 0.
        self.lr_epoch = 0.
        self.kl_iteration = 0.
        self.kl_epoch = 0.
        self.H_iteration = 0.
        self.H_epoch = 0.
        self.i_epoch = 0.
        self.train_generator = []
        self.dataset_test = []
        self.metrics = {}
        self.aucroc_lc = []
        self.auprc_lc = []
        self.auroc_det = []
        self.lr_lc = []
        self.lkl_lc = []
        self.lae_lc = []
        self.H_lc = []
        self.lsa_lc = []
        self.auroc_det_lc = []
        self.refCam = 0.

    #########################
    def train(self, 
              training_data_index, 
              data_path, 
              epochs, 
              update_weights = 100, 
              lr=1*1e-5, 
              alpha_kl=1, 
              alpha_lr = 1, 
              alpha_entropy=1, 
              alpha_spectral_angle=1, 
              pre_training_epochs=5, 
              train_verbose = 0):
        
        
        self.epochs = epochs
        self.init_time = time.time()
        self.training_data_index = training_data_index
        self.pre_training_epochs = pre_training_epochs
        self.lr = lr
        self.alpha_spectral_angle = alpha_spectral_angle
        self.alpha_entropy = alpha_entropy
        self.alpha_kl = alpha_kl
        self.alpha_lr = alpha_lr
        self.update_weights = update_weights

        # torch.autograd.set_detect_anomaly(True)

        # Set optimizers
        self.opt = torch.optim.Adam(self.params, lr=self.lr)

        self.conv_encoder.train()
        self.conv_decoder.train()

        ## 1) Loading and managing training data

        # Load training data
        if train_verbose: print(f'Loading training data into memory...')
        df_files = create_df_from_files_in_path(data_path, verbose = False)
        training_maps, training_data = load_HSI_from_idx(self.training_data_index, df_files, verbose = train_verbose)
        training_data = (training_data/training_data.max())

        # Convert ndarray to Tensor
        x_n = torch.from_numpy(np.float32(training_data))
        x_n = torch.permute(x_n, (2, 0, 1))
        x_n = torch.unsqueeze(x_n, dim=0)
        # Move tensor to gpu
        x_n = x_n.to('cuda')

        # Initial weights matrix, when using weighted l2
        weights = torch.ones_like(x_n)

        if train_verbose: print(f'Data Tensor prepared with shape {torch.Tensor.size(x_n)}.')

        ## 2) Loop over epochs
        for self.i_epoch in np.arange(self.epochs):

            # init epoch losses
            self.lr_epoch = 0   # Reconstruction
            self.kl_epoch = 0.  # KL divergence
            self.H_epoch = 0.   # Entropy
            self.lsa_epoch = 0  # Spectral angle

            # Obtain latent space from HSI via encoder
            z, z_mu, z_logvar, allF = self.conv_encoder(x_n)
            if train_verbose == 1: print(f'  Dimmensions of latent vector sample z = {np.shape(z)}')

            if torch.any(torch.isinf(z)):
                print(f'>>> ERROR! There are infinite values in the latent vector.')
                return
            if torch.any(torch.isnan(z)):
                print(f'>>> ERROR! There are NaN values in the latent vector.')
                return

            # Obtain reconstructed images through decoder
            x_hat, _ = self.conv_decoder(z)
            if train_verbose == 1: print(f'  Dimmensions of reconstructed image x_hat = {np.shape(x_hat)}')
            # Apply sigmoid to the output when using MSE losses
            if self.loss_reconstruction == 'l2' or self.loss_reconstruction == 'weighted_l2':
                x_hat = torch.sigmoid(x_hat)
    
            # Calculate losses
            if self.loss_reconstruction == 'weighted_l2':
                # Update weights for weighted loss function
                if self.i_epoch % self.update_weights == 0:
                    # print('Updating weights...')
                    weights = weight_map(x_n.detach(), x_hat.detach())
                self.lr_iteration = self.Lr(x_n, x_hat, weights) # Weighted reconstruction loss
            else:
                self.lr_iteration = self.Lr(x_hat, x_n) # Reconstruction loss

            self.kl_iteration = self.Lkl(mu=z_mu, logvar=z_logvar) # kl loss
            self.lsa_iteration = self.Lsa(x_hat, x_n)   # Spectral angle losses

            if self.alpha_entropy > 0:
                # Compute Attention Homogeneization loss via Entropy
                am = torch.mean(allF[self.level_cams], 1)
                # Restore original shape
                am = torch.nn.functional.interpolate(am.unsqueeze(1),
                                                    size=(self.input_shape[-1], self.input_shape[-1]),
                                                    mode='bilinear',
                                                    align_corners=True)
                am = am.view((am.shape[0], -1))

                am = am.view((am.shape[0], -1))                         # Attention vector
                p = torch.nn.functional.softmax(am, dim=-1)             # Probabilities
                sum = -torch.sum(p * torch.log(p + 1e-12), dim=(-1))    # Entropy
                self.H_iteration = torch.mean(sum)                      # Mean entropy of the batch
            else:
                self.H_iteration = 0

            # Init overall losses. Reconstruction and spectral angle
            L = alpha_lr * self.lr_iteration + self.alpha_spectral_angle * self.lsa_iteration
            # Add the KL and Entropy terms to the losses only after the pre-training epochs
            if self.i_epoch > self.pre_training_epochs:
                if self.alpha_entropy > 0:
                    # Entropy Maximization
                    L += - self.alpha_entropy * self.H_iteration
                if self.alpha_kl > 0:
                    # Gaussian distribution Maximization
                    L += self.alpha_kl * self.kl_iteration

            # Update weights
            if train_verbose == 1: print(f'Backpropagating error...')
            # L.backward(retain_graph=True)  # Backward
            L.backward()
            self.opt.step()  # Update weights
            self.opt.zero_grad()  # Clear gradients

            """
            ON ITERATION/EPOCH END PROCESS
            """

            # Update epoch's losses
            self.lr_epoch += self.lr_iteration
            self.kl_epoch += self.kl_iteration
            # self.H_epoch += self.H_iteration
            self.lsa_epoch += self.lsa_iteration
            
            # Display losses per iteration
            # self.display_losses(on_epoch_end=False)

            # Epoch-end processes
            self.on_epoch_end()

        # Save the model at the end of the training
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        torch.save(self.conv_encoder.state_dict(), os.path.join(self.dir_results, f'encoder_weights_{df_files.Filename[training_data_index]}.pth'))
        torch.save(self.conv_decoder.state_dict(), os.path.join(self.dir_results, f'decoder_weights_{df_files.Filename[training_data_index]}.pth'))

        # move losses history to cpu
        reconstruction_losses = torch.stack(self.lr_lc).cpu().detach().numpy()
        KL_div_losses = torch.stack(self.lkl_lc).cpu().detach().numpy()
        # entropy_losses = torch.stack(self.H_lc).cpu().detach().numpy()
        spectral_angle_losses = torch.stack(self.lsa_lc).cpu().detach().numpy()
        # Save losses history to dataframe
        history = pd.DataFrame(list(zip(reconstruction_losses, KL_div_losses, spectral_angle_losses)),
                                       columns=['reconstruction_losses', 'KL_div_losses', 'spectral_angle_losses'])
        # history = pd.DataFrame(list(zip(reconstruction_losses, KL_div_losses, entropy_losses, spectral_angle_losses)),
        #                                columns=['reconstruction_losses', 'KL_div_losses', 'entropy_losses', 'spectral_angle_losses'])
        history.to_csv(self.dir_results + 'training_losses.csv')


    def on_epoch_end(self):

        # Display losses
        self.display_losses(on_epoch_end=True)

        # Update learning curves
        self.lr_lc.append(self.lr_epoch)
        self.lkl_lc.append(self.kl_epoch)
        self.H_lc.append(self.H_epoch)
        self.lsa_lc.append(self.lsa_epoch)

    # Evaluate the predictions
    def predict_score(self, x):
        # Turn on the "evaluation mode" of the model
        self.conv_encoder.eval()
        self.conv_decoder.eval()

        # Get reconstruction error map
        # z, z_mu, z_logvar, f = self.conv_encoder(torch.tensor(x).cuda().float().unsqueeze(0))
        z, z_mu, z_logvar, f = self.conv_encoder(torch.tensor(x).cuda().float())
        x_hat, f_dec = self.conv_decoder(z)
        am = torch.mean(f[self.level_cams], 1)
        # am = torch.sum(f[self.level_cams], 1)
   
        if self.verbose == 1: print(f'Attention mask extracted, with shape {np.shape(am)}.')

        # Restore original shape
        m_hat = torch.nn.functional.interpolate(am.unsqueeze(0), size=(self.input_shape[-1], self.input_shape[-1]),
                                               mode='bilinear', align_corners=True).squeeze().detach().cpu().numpy()
        

        if self.verbose == 1: print(f'Attention mask restored, with shape {np.shape(m_hat)}.')

        # Repeat the process with all the layers of the encoder
        am_list_enc = []
        for attention_map in (self.conv_encoder.act_maps):
            # am = torch.sum(attention_map, 1)
            am = torch.mean(attention_map, 1)
            m_hat_enc = torch.nn.functional.interpolate(am.unsqueeze(0), size=(self.input_shape[-1], self.input_shape[-1]),
                                                    mode='bilinear', align_corners=True).squeeze().detach().cpu().numpy()
            am_list_enc.append(m_hat_enc)

        # Repeat the process with all the layers of the decoder
        am_list_dec = []
        for attention_map in (self.conv_decoder.act_maps):
            # am = torch.sum(attention_map, 1)
            am = torch.mean(attention_map, 1)
            m_hat_enc = torch.nn.functional.interpolate(am.unsqueeze(0), size=(self.input_shape[-1], self.input_shape[-1]),
                                                    mode='bilinear', align_corners=True).squeeze().detach().cpu().numpy()
            am_list_dec.append(m_hat_enc)


        # Get outputs
        score = np.std(m_hat)
        if self.verbose == 1: print(f'Score calculated, with value {score}.')

        # Turn back to the "training mode" of the model
        self.conv_encoder.train()
        self.conv_decoder.train()

        return score, m_hat, x_hat, am_list_enc, am_list_dec

    # Write the value of the losses on every epoch or step of the training process
    def display_losses(self, on_epoch_end=False):
        # Init info display
        info =(f"[INFO] Epoch {self.i_epoch + 1}/{self.epochs}: ")
        # Prepare values to show
        if on_epoch_end:
            lr = self.lr_epoch
            lkl = self.kl_epoch
            # lH = self.H_epoch
            lsa = self.lsa_epoch
        else:
            lr = self.lr_iteration
            lkl = self.kl_iteration
            # lH = self.H_iteration
            lsa = self.lsa_iteration
        # Init losses display
        info += f"RL={lr:.4f} || SAD={lsa:.4f} || KL={lkl:.4f}"
        # info += f"RL={lr:.4f} || SAD={lsa:.4f} || KL={lkl:.4f} || H={lH:.4f}"
        # Print losses
        et = time.time() - self.init_time
        print(f'{info}, ET= {et} \r')


    def plot_learning_curves(self):
        def plot_subplot(axes, x, y, y_axis):
            axes.grid()
            axes.plot(x, y, 'o-')
            axes.set_ylabel(y_axis)

        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        plot_subplot(axes[0, 0], np.arange(self.i_epoch + 1) + 1, np.array(self.lr_lc), "Reconstruc loss")
        plot_subplot(axes[0, 1], np.arange(self.i_epoch + 1) + 1, np.array(self.lkl_lc), "KL loss")
        plot_subplot(axes[1, 0], np.arange(self.i_epoch + 1) + 1, np.array(self.H_lc), "H")
        plt.savefig(self.dir_results + 'learning_curve.png')
        plt.close()
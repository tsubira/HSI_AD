import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
import time
from losses import *

from vit import Transformer

class MAE(nn.Module):

    ############################
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64,
        lr = 1e-4,
        verbose = 0
    ):
        super().__init__()

        self.loss_history = []
        self.verbose = verbose
        self.lr = lr

        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.num_patches = num_patches

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    ############################
    def forward(self, img):
        device = img.device

        # get patches
        self.patches = self.to_patch(img)
        if self.verbose > 1: print(f'Size of patches = {self.patches.size()}')

        batch, num_patches, *_ = self.patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(self.patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # calculate number of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        self.masked_patches = self.patches[batch_range, masked_indices]

        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = torch.sigmoid(self.decoder(decoder_tokens))
        if self.verbose > 1: print(f'Size of decoded tokens = {decoded_tokens.size()}')

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)
        recons_image = self.to_pixels(decoded_tokens)


        if self.verbose > 1: print(f'Size of reconstructed images = {recons_image.size()}')

        return recons_image, pred_pixel_values, masked_indices


    ############################
    def encode_image(self, img):
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        encoded_tokens = self.encoder.transformer(tokens)
        encoded_tokens = self.enc_to_dec(encoded_tokens)

        # encode the tokens
        return self.to_pixels(encoded_tokens)
        # return (encoded_tokens)

    ############################
    def reconstruct_image(self, img):
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # encode the tokens
        encoded_tokens = self.encoder.transformer(tokens)

        # convert to decoder dim
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # We use non-random indices, since we want the output image organized
        all_indices = torch.rand(batch, num_patches, device = img.device).argsort(dim = -1)
        all_indices[:,:] = torch.arange(0, num_patches, 1)

        if self.verbose > 1: print(f'>> Indices = {all_indices}')
        decoder_tokens = decoder_tokens + self.decoder_pos_emb(all_indices)

        # decode the tokens
        decoded_tokens = torch.sigmoid(self.decoder(decoder_tokens))
        self.decoded_tokens = decoded_tokens

        # convert back to pixels
        return (self.to_pixels(decoded_tokens))

    ############################
    def training_loop(self, images, epochs, autosave = 5, loss_fcn = 'mae', update_weights = 100):
        # Set the model in training mode
        self.train()

        # Define the loss function
        if loss_fcn == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_fcn == 'weighted_l2':
            self.criterion = weighted_MSE
        else:
            self.criterion = nn.L1Loss()
        
        self.init_time = time.time()
        if self.verbose != 0:
            print(f'\n-- TRAINING START -- ')
            print(f'    Number of epochs = {epochs}')
            print(f'    Number of images = {images.size(0)}')
            print(f'    Number of patches per image = {self.num_patches}')

        # Set the optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(epochs):
            # Forward the batch
            recons_image, pred_pixel_values, masked_indices = self.forward(images)

            # Calculate the reconstruction loss
            # loss = self.criterion(self.masked_patches, pred_pixel_values)     # loss from the masked pixels, according to MAE paper
            if loss_fcn == 'weighted_l2':
                if epoch % update_weights == 0:
                    if self.verbose > 0: print(f'>> Updating RL weights...')
                    weights = weight_map(self.patches.detach(), recons_image.detach())
                loss = self.criterion(self.patches, recons_image, weights)
            else:
                loss = self.criterion(self.patches, recons_image)                   # loss from the whole image

            # Add the losses to the history log
            self.loss_history.append(loss.detach().numpy())
            if self.verbose != 0:
                print(f'>> EPOCH {epoch+1} | RL = {loss:.4f} | ET = {(time.time() - self.init_time):.4f}')
            
            loss.backward()
            self.opt.step()         # Update weights
            self.opt.zero_grad()    # Clear gradients

            if autosave > 0 and epoch%autosave == 0:
                torch.save(self.state_dict(), f'checkpoint_mae_epoch{epoch}.pt')


        
        if self.verbose != 0:
            print(f'\n-- TRAINING END -- ')
        
        self.eval()
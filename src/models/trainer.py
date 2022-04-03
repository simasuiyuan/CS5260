import torch.utils.data as data
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import sys
import os

def train(epoch, 
          encoder,
          decoder,
          encoder_optimizer,
          decoder_optimizer,
          criterion, 
          total_step, 
          num_epochs, 
          data_loader, 
          write_file,
          vocab_size,
          batch_size,
          device,
          print_every=1,
          save_every = 1):
    """ Train function for a single epoch. 
    Arguments: 
    ----------
    - epoch - number of current epoch
    - encoder - model's Encoder
    - decoder - model's Decoder
    - optimizer - model's optimizer (Adam in our case)
    - criterion - loss function to optimize
    - num_epochs - total number of epochs
    - data_loader - specified data loader (for training, validation or test)
    - write_file - file to write the training logs
    
    """
    epoch_loss = 0.0
    epoch_perplex = 0.0
    
    for i_step in range(1, total_step+1):
        # training mode on
        encoder.train() # no fine-tuning for Encoder
        # mlp.train()
        decoder.train()
        
        # Randomly sample a caption length, and sample indices with that length.
        indices = data_loader.dataset.get_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler
        
        # Obtain the batch.
        images, onehot_cat, onehot_seq, captions = next(iter(data_loader))
        # target captions, excluding the first word
        captions_target = captions[:, 1:].to(device) 
        # captions for training without the last word
        captions_train = captions[:, :-1].to(device)

        # Move batch of images and captions to GPU if CUDA is available.
        images = images.to(device)

        onehot_cat = torch.cat((onehot_cat.view(batch_size,-1), 
                                onehot_seq.view(batch_size,-1)), 1).type('torch.FloatTensor').view(batch_size,-1).to(device)
        # Zero the gradients.
        decoder.zero_grad()
        # mlp.zero_grad()
        encoder.zero_grad()
        
        # Pass the inputs through the CNN-RNN model.
        features = encoder(images)

        outputs, atten_weights = decoder(captions= captions_train,
                                         features = features,
                                         category_enc = onehot_cat)
        
        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, vocab_size).to(device), captions_target.reshape(-1))
        
        # Backward pass.
        loss.backward()
        
        # Update the parameters in the optimizer.
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        perplex = np.exp(loss.item())
        epoch_loss += loss.item()
        epoch_perplex += perplex
        
        stats = 'Epoch train: [%d/%d], Step train: [%d/%d], Loss train: %.4f, Perplexity train: %5.4f' % (epoch, num_epochs, i_step, total_step, loss.item(), perplex)
        
        
        # Print training statistics (on same line).
        print('\r' + stats, end="")
        sys.stdout.flush()
        
        # Print training statistics to file.
        write_file.write(stats + '\n')
        write_file.flush()
        
        # Print training statistics (on different line).
        if i_step % print_every == 0:
            print('\r' + stats)
        
    epoch_loss_avg = epoch_loss / total_step
    epoch_perp_avg = epoch_perplex / total_step
    
    print('\r')
    print('Epoch train:', epoch)
    print('\r' + 'Avg. Loss train: %.4f, Avg. Perplexity train: %5.4f' % (epoch_loss_avg, epoch_perp_avg), end="")
    print('\r')
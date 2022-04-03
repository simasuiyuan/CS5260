import torch.utils.data as data
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import sys
import os
from src.utils.utils_trainer import get_batch_caps, get_hypothesis, adjust_learning_rate
from nltk.translate.bleu_score import corpus_bleu

def validate(epoch, 
             encoder, 
             decoder, 
             encoder_optimizer, 
             decoder_optimizer, 
             criterion, 
             vocab_size,
             batch_size,
             device,
             print_every,
             total_step, 
             num_epochs, 
             data_loader, 
             write_file, 
             bleu_score_file):
    """ Validation function for a single epoch. 
    Arguments: 
    ----------
    - epoch - number of current epoch
    - encoder - model's Encoder (evaluation)
    - decoder - model's Decoder (evaluation)
    - optimizer - model's optimizer (Adam in our case)
    - criterion - optimized loss function
    - num_epochs - total number of epochs
    - data_loader - specified data loader (for training, validation or test)
    - write_file - file to write the validation logs
    """
    epoch_loss = 0.0
    epoch_perplex = 0.0
    references = []
    hypothesis = []
      
    for i_step in range(1, total_step+1):
        # evaluation of encoder and decoder
        encoder.eval()
        decoder.eval()
        val_images, val_onehot_cat, val_onehot_seq, val_captions, caps_all = next(iter(data_loader))
        
        val_captions_target = val_captions[:, 1:].to(device) 
        val_captions = val_captions[:, :-1].to(device)
        val_images = val_images.to(device)
       
        val_onehot_cat = torch.cat((val_onehot_cat.view(batch_size,-1), 
                                    val_onehot_seq.view(batch_size,-1)), 1).type('torch.FloatTensor').view(batch_size,-1).to(device)
        
        features_val = encoder(val_images)
        outputs_val, atten_weights_val = decoder(captions= val_captions,
                                                 features = features_val,
                                                 category_enc=val_onehot_cat)
        loss_val = criterion(outputs_val.view(-1, vocab_size), 
                             val_captions_target.reshape(-1))
        
        # preprocess captions and add them to the list
        caps_processed = get_batch_caps(caps_all, batch_size=batch_size)
        references.append(caps_processed)
        # get corresponding indicies from predictions
        # and form hypothesis from output
        terms_idx = torch.max(outputs_val, dim=2)[1]
        hyp_list = get_hypothesis(terms_idx, data_loader=data_loader)
        hypothesis.append(hyp_list)
        
        perplex = np.exp(loss_val.item())
        epoch_loss += loss_val.item()
        epoch_perplex += perplex
        
        stats = 'Epoch valid: [%d/%d], Step valid: [%d/%d], Loss valid: %.4f, Perplexity valid: %5.4f' % (epoch, num_epochs, i_step, total_step, loss_val.item(), perplex)
        
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
            
    # prepare the proper shape for computing BLEU scores
    references = np.array(references).reshape(total_step*batch_size, -1)
    #hyps = np.array(hypothesis).reshape(total_step*batch_size, -1)
    hyps = np.concatenate(np.array(hypothesis))
        
    bleu_1 = corpus_bleu(references, hyps, weights = (1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hyps, weights = (0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(references, hyps, weights = (1.0/3.0, 1.0/3.0, 1.0/3.0, 0))
    bleu_4 = corpus_bleu(references, hyps, weights = (0.25, 0.25, 0.25, 0.25))
    # append individual n_gram scores
    #bleu_score_list.append((bleu_1, bleu_2, bleu_3, bleu_4))
    
    print('\r')
    print('Epoch valid:', epoch)
    epoch_stat = 'Avg. Loss valid: %.4f, Avg. Perplexity valid: %5.4f, \
    BLEU-1: %.2f, BLEU-2: %.2f, BLEU-3: %.2f, BLEU-4: %.2f' % (epoch_loss_avg, epoch_perp_avg, bleu_1, bleu_2, bleu_3, bleu_4)
    
    print('\r' + epoch_stat, end="")
    print('\r')
    
    bleu_score_file.write(epoch_stat + '\n')
    bleu_score_file.flush()
    return bleu_1, bleu_2, bleu_3, bleu_4
__author__ = 'ma_yuan'
__version__ = '1.0'
import nltk
import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from typing import Union
import skimage.io as io
from PIL import Image
import torch.utils.data as data
from sklearn.preprocessing import OneHotEncoder
import time

from src.utils.vocabulary import Vocabulary

def get_loader(
  source:Union[dict, pd.DataFrame],
  transform,
  mode='train',
  # default batch size
  image_type="imageURL",
  caption_type="description",
  sentence_index = "sentence_index",
  batch_size=1,
  vocab_threshold=None,
  vocab_file='./assets/vocab.pkl',
  start_word="<start>",
  end_word="<end>",
  unk_word="<unk>",
  vocab_from_file=True,
  num_workers=0):

  """Returns the data loader.
  Args:
    transform: Image transform.
    mode: One of 'train', 'valid or 'test'.
    batch_size: Batch size (if in testing mode, must have batch_size=1).
    vocab_threshold: Minimum word count threshold.
    vocab_file: File containing the vocabulary. 
    start_word: Special word denoting sentence start.
    end_word: Special word denoting sentence end.
    unk_word: Special word denoting unknown words.
    vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                      If True, load vocab from from existing vocab_file, if it exists.
    num_workers: Number of subprocesses to use for data loading 
  """
  assert mode in ['train', 'valid', 'test'], "mode must be one of 'train', 'valid' or 'test'."
  if vocab_from_file==False: assert mode=='train', "To generate vocab from captions file, must be in training mode (mode='train')."

  # Based on mode (train, val, test), obtain img_folder and annotations_file.
  if mode == 'train':
    if vocab_from_file==True: assert os.path.exists(vocab_file), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."

  elif mode == 'valid':
    assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
    assert vocab_from_file==True, "Change vocab_from_file to True."
      
  elif mode == 'test':
    assert batch_size==1, "Please change batch_size to 1 for testing your model."
    assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
    assert vocab_from_file==True, "Change vocab_from_file to True."
  
  image_dict = source[image_type].to_dict()
  caption_dict = source[caption_type].to_dict()
  category_dict = source["perCategory"].to_dict()
  if sentence_index in source.columns:
        sentence_id_dict = source["sentence_index"].to_dict()
  else:
        sentence_id_dict = None

  dataset= myDataset(
    image_dict, 
    caption_dict,
    category_dict,
    transform, 
    mode, 
    batch_size, 
    vocab_threshold, 
    vocab_file, 
    start_word, 
    end_word, 
    unk_word, 
    vocab_from_file,
    sentence_id_dict=sentence_id_dict)

  if mode == 'train' or mode == 'valid':
      # Randomly sample a caption length and indices of that length
      indices = dataset.get_indices()
      # Create and assign a batch sampler to retrieve a batch with the sampled indices
      # functionality from torch.utils
      initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
      data_loader = data.DataLoader(dataset=dataset, 
                                    num_workers=num_workers,
                                    batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                            batch_size=dataset.batch_size,
                                                                            drop_last=False))
  elif mode == 'test':
      data_loader = data.DataLoader(dataset=dataset,
                                    batch_size=dataset.batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)

  return data_loader

class myDataset(data.Dataset):
  def __init__(self, image_dict, caption_dict, category_dict, transform, 
  mode, batch_size, vocab_threshold, vocab_file, start_word, 
  end_word, unk_word, vocab_from_file,sentence_id_dict=None):
    self.image_dict = image_dict
    self.caption_dict = caption_dict
    self.category_dict = category_dict
    self.sentence_id_dict = sentence_id_dict
    #onehot encode the category data
    self.onehot_enc_cat = OneHotEncoder(handle_unknown='ignore').fit(np.array([*category_dict.values()], dtype=object).reshape(-1, 1))
    if sentence_id_dict is not None:
          self.onehot_enc_seq = OneHotEncoder(handle_unknown='ignore').fit(np.array([*sentence_id_dict.values()], dtype=object).reshape(-1, 1))
          
    self.transform = transform
    self.mode = mode
    self.batch_size = batch_size
    self.vocab = Vocabulary(caption_dict, vocab_threshold, vocab_file, start_word,
                            end_word, unk_word, vocab_from_file)
    # self.img_folder = img_folder
    # if training and validation
    if self.mode == 'train' or self.mode == 'valid':
      self.ids = list(caption_dict.keys())
      print('Obtaining caption lengths...')
      # get all_tokens - a big list of lists. Each is a list of tokens for specific caption
      all_tokens = [nltk.tokenize.word_tokenize(str(caption_dict[index]).lower()) for index in tqdm(np.arange(len(self.ids)))]
      # list of token lengths (number of words for each caption)
      self.caption_lengths = [len(token) for token in all_tokens]
      

  def __getitem__(self, index):
    # obtain image and caption if in training mode
    if self.mode == 'train':
      # if we are in training mode
      # we retrieve an id of specified annotation
      # get caption for annotation based on its id
      caption = self.caption_dict[index]
      # get image id
      img_url = self.image_dict[index]
      # get an image name, like 'https://images-na.ssl-images-amazon.com/images/I/41WGCY65ENL._US40_.jpg'
      # Convert image to tensor and pre-process using transform
      # we open specified image and convert it to RGB
      time.sleep(0.01)
      image = Image.fromarray(io.imread(img_url)).convert('RGB')
      # specified image transformer - the way we want to augment/modify image
      image = self.transform(image)

      # image pre-processed with tranformer applied
      onehot_cat = torch.Tensor(self.onehot_enc_cat.transform(
        np.array(self.category_dict[index], 
        dtype=object).reshape(-1, 1)).toarray()).long()
      
      if self.sentence_id_dict is not None:
            onehot_seq = torch.Tensor(self.onehot_enc_seq.transform(
              np.array(self.sentence_id_dict[index], 
                       dtype=object).reshape(-1, 1)).toarray()).long()

      # Convert caption to tensor of word ids.
      tokens = nltk.tokenize.word_tokenize(str(caption).lower())
      caption = []
      caption.append(self.vocab(self.vocab.start_word))
      caption.extend([self.vocab(token) for token in tokens])
      caption.append(self.vocab(self.vocab.end_word))
      caption = torch.Tensor(caption).long()
      # return pre-processed image and caption tensors
      return image, onehot_cat, onehot_seq, caption

    # obtain image if in test mode
    elif self.mode == 'valid':
      # Convert image to tensor and pre-process using transform           
      caption = self.caption_dict[index]
      img_url = self.image_dict[index]
      time.sleep(0.01)
      image = Image.fromarray(io.imread(img_url)).convert('RGB')
      image = self.transform(image)
      
      onehot_cat = torch.Tensor(self.onehot_enc_cat.transform(
        np.array(self.category_dict[index], 
        dtype=object).reshape(-1, 1)).toarray()).long()
      
      if self.sentence_id_dict is not None:
            onehot_seq = torch.Tensor(self.onehot_enc_seq.transform(
              np.array(self.sentence_id_dict[index], 
                       dtype=object).reshape(-1, 1)).toarray()).long()

      # Convert caption to tensor of word ids.
      tokens = nltk.tokenize.word_tokenize(str(caption).lower())
      caption = []
      caption.append(self.vocab(self.vocab.start_word))
      caption.extend([self.vocab(token) for token in tokens])
      caption.append(self.vocab(self.vocab.end_word))
      caption = torch.Tensor(caption).long()
      # retrun all captions for image (will be required for calculating BLEU score)
      caps_all = list(self.caption_dict.values())
      # return original image and pre-processed image tensor
      return image, onehot_cat, onehot_seq, caption, caps_all

    elif self.mode == 'test':
      img_url = self.image_dict[index]
      caption = self.caption_dict[index]
      # Convert image to tensor and pre-process using transform
      PIL_image = Image.fromarray(io.imread(img_url)).convert('RGB')
      orig_image = np.array(PIL_image)
      image = self.transform(PIL_image)
      
      onehot_cat = torch.Tensor(self.onehot_enc_cat.transform(
        np.array(self.category_dict[index], 
        dtype=object).reshape(-1, 1)).toarray()).long()
      
      if self.sentence_id_dict is not None:
            onehot_seq = torch.Tensor(self.onehot_enc_seq.transform(
              np.array(self.sentence_id_dict[index], 
                       dtype=object).reshape(-1, 1)).toarray()).long()
      # return original image and pre-processed image tensor
      return orig_image, image, onehot_cat, onehot_seq, caption

  def get_indices(self):
    # randomly select the caption length from the list of lengths
    sel_length = np.random.choice(self.caption_lengths)
    all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
    # select m = batch_size captions from list above
    indices = list(np.random.choice(all_indices, size=self.batch_size))
    # return the caption indices of specified batch
    return indices

  def __len__(self):
    if self.mode == 'train' or self.mode == 'valid':
        return len(self.ids)
    else:
        return len(self.image_dict.values())



# class myDataset:
#   def __init__(self, source:Union[dict, pd.DataFrame]):
#     """
#     Constructor of dataset helper class for reading and visualizing annotations.
#     :param source (str): location of annotation file
#     :return:
#     """
#     # load dataset
#     self.dataset,self.cats,self.titles,self.imgs = dict(),dict(),dict(),dict()
#     self.titleToImgs, self.catToImgs = defaultdict(list),defaultdict(list)
#     if not source == None:
#       print('loading dataset into memory...')
#       tic = time.time()
#       if type(source)==pd.DataFrame:
#         source = source.to_dict()
#       assert type(source)==dict, 'source file format {} not supported'.format(type(source))
#       print('Done (t={:0.2f}s)'.format(time.time()- tic))
#       self.dataset = source
#       self.createIndex()
  
#   def createIndex(self):
#     # create index
#     print('creating index...')
#     titleToImgs, catToImgs = defaultdict(list), defaultdict(list)

#     if 'title' in self.dataset:
#         for image_id, title in self.dataset['title'].items():
#             titleToImgs[title].append(image_id)

#     if 'perCategory' in self.dataset:
#         for image_id, cat in self.dataset['title'].items():
#             titleToImgs[cat].append(image_id)

#     print('index created!')

#     # create class members
#     self.titles = self.dataset["title"]
#     self.titleToImgs = titleToImgs
#     self.catToImgs = catToImgs
#     self.imgs = self.dataset["imageURL"]
#     self.cats = self.dataset["perCategory"]
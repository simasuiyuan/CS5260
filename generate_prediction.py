#%%
from traceback import print_tb
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
import json
import nltk
from tqdm import tqdm
from pathlib import Path
tqdm.pandas()
import warnings
warnings.filterwarnings('ignore')
# nltk.download('stopwords')
# nltk.download('punkt')
%matplotlib inline
# IMAGE_PATH = Path("/etlstage/PEE_joint/mine/image_data")
# %%
nltk.data.path.append("/home/hdfsf10n/nltk_data")
nltk.data.path
# %%
%load_ext autoreload
%autoreload
from src.utils.data_loader import get_loader
# %%
with open('./data/vaild_dataset.json', 'r') as outfile:
  product_info = pd.read_json(json.load(outfile), orient="records")
print(product_info.shape)
# product_info["imageURLHighRes"] = product_info["file_name"].apply(lambda file_name : IMAGE_PATH/file_name)

import re
def get_space_len(s):
  return len(max(re.findall(' +', s), key=len, default=[0]))

# print(product_info["perCategory"].unique())
product_info = product_info[product_info.valid == True]
product_info = product_info[product_info["perCategory"].isin(["AMAZON_FASHION", 'All_Beauty',
                                                               'Toys_and_Games','Office_Products',
                                                              'Home_and_Kitchen','Electronics',
                                                              'Clothing_Shoes_and_Jewelry'])]


# array(['AMAZON_FASHION', 'All_Beauty', 'Appliances',
#        'Arts_Crafts_and_Sewing', 'Automotive', 'Books', 'CDs_and_Vinyl',
#        'Cell_Phones_and_Accessories', 'Clothing_Shoes_and_Jewelry',
#        'Digital_Music', 'Electronics', 'Gift_Cards',
#        'Grocery_and_Gourmet_Food', 'Home_and_Kitchen',
#        'Industrial_and_Scientific', 'Luxury_Beauty',
#        'Magazine_Subscriptions', 'Movies_and_TV', 'Musical_Instruments',
#        'Office_Products', 'Patio_Lawn_and_Garden', 'Pet_Supplies',
#        'Prime_Pantry', 'Software', 'Sports_and_Outdoors',
#        'Tools_and_Home_Improvement', 'Toys_and_Games', 'Video_Games'],
#       dtype=object)

# product_info = product_info[product_info["perCategory"].isin(["AMAZON_FASHION", 'All_Beauty',
#                                                                 'Cell_Phones_and_Accessories',
#                                                                 'Toys_and_Games','Luxury_Beauty',
#                                                                 'Home_and_Kitchen','Electronics',
#                                                                 'Grocery_and_Gourmet_Food',
#                                                                 'Automotive','Office_Products',
#                                                                 'Clothing_Shoes_and_Jewelry'])]

print(product_info["perCategory"].unique())

product_info["file_name"] = product_info["imageURLHighRes"].str.split("/").str[-1]

product_info = product_info.explode("description")
product_info["description"] = product_info["description"].str.split(".")
product_info = product_info.explode("description")
product_info = product_info.where(
    (product_info["description"].str.len()>5) & 
    (product_info["description"].astype(str).apply(get_space_len)<4)).dropna().reset_index(drop=True)
product_info['sentence_index'] = product_info.groupby('file_name').cumcount()  # Enumerate Groups
product_info = product_info[product_info['sentence_index'] <= product_info["description"].str.len().min()].reset_index(drop=True)
print(product_info.shape)
product_info.head(5)
# %%
for descript in product_info.loc[:100,"description"]:
    print("="*100)
    print(descript)
# %%
import random
idex = random.randint(0, product_info.shape[0])
# img_url = product_info.loc[idex,"imageURLHighRes"]
img_loc = product_info.loc[idex,"imageURLHighRes"]
caption = product_info.loc[idex,"description"]
Category = product_info.loc[idex,["perCategory", "sentence_index"]]
print(caption)
print(Category)
print(img_loc)
I = io.imread(img_loc)
plt.axis('off')
plt.imshow(I)
# %%
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder

image_dict = product_info["imageURLHighRes"].to_dict()
caption_dict = product_info["description"].to_dict()
category_dict = product_info["perCategory"].to_dict()
onehot_cat = OneHotEncoder().fit_transform(np.array([*category_dict.values()], dtype=object).reshape(-1, 1)).toarray()
sentence_id_dict = product_info["sentence_index"].to_dict()
onehot_seq = OneHotEncoder().fit_transform(np.array([*sentence_id_dict.values()], dtype=object).reshape(-1, 1)).toarray()
one_hot = np.concatenate((onehot_cat, onehot_seq), axis=1)
print(one_hot.shape)

# Define a transform to pre-process the training images.
# transform_train = transforms.Compose([ 
#     transforms.Resize(500),                          # smaller edge of image resized to 256
#     transforms.Resize((320,320)),                      # get 224x224 crop from random location
#     transforms.ToTensor(),                           # convert the PIL Image to a tensor
#     transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
#                          (0.229, 0.224, 0.225))])
transform_train = transforms.Compose([ 
    transforms.Resize(500),                          # smaller edge of image resized to 256
    transforms.Resize((320,320)),                      # get 224x224 crop from random location
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
])

# Set the minimum word count threshold.
vocab_threshold = 5

mode = "train"

# Specify the batch size.
# we will pass 10 images at a time. So, m = 10
batch_size = 10
# %%
%load_ext autoreload
%autoreload
from src.utils.data_loader import get_loader
# %%
data_loader = get_loader(product_info,
                        transform=transform_train,
                        mode='train',
                        image_type="imageURLHighRes",
                        caption_type="description",
                        batch_size=batch_size,
                        vocab_threshold=vocab_threshold,
                        vocab_from_file=True)

print('The shape of first image:', data_loader.dataset[0][0].shape)
print('Total number of tokens in vocabulary:', len(data_loader.dataset.vocab))

# Randomly sample a caption length, and sample indices with that length.
indices = data_loader.dataset.get_indices()
print('sampled indices:', indices)

# Create and assign a batch sampler to retrieve a batch with the sampled indices.
new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
data_loader.batch_sampler.sampler = new_sampler
    
# Obtain the batch.
images, onehot_cat, onehot_enc_seq, captions = next(iter(data_loader))
    
print('images.shape:', images.shape)
print('onehot_cat.shape:', onehot_cat.shape)
print('captions.shape:', captions.shape)
# %%
plt.axis('off')
plt.imshow(images[0].detach().T)
plt.show()

# %%
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import sys
import os
import math
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import nltk
from nltk.translate.bleu_score import corpus_bleu

%load_ext autoreload
%autoreload
from src.utils.data_loader import get_loader
from src.models.CNN_Encoder import EncoderCNN
from src.models.RNN_decoder_catAttention import DecoderRNN
from src.models.MLP_Encoder import MlpEncoder
from src.utils.utils_trainer import get_batch_caps, get_hypothesis, adjust_learning_rate

# %%
train_df, val_test_df = train_test_split(product_info, test_size=0.4)
train_df.reset_index(drop=True,inplace=True)
valid_df, test_df = train_test_split(val_test_df, test_size=0.5)
valid_df.reset_index(drop=True,inplace=True)
test_df.reset_index(drop=True,inplace=True)

transform_test = transforms.Compose([ 
    transforms.Resize(500),                          # smaller edge of image resized to 256
    transforms.Resize((320,320)),                      # get 224x224 crop from random location
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
])

test_data_loader = get_loader(test_df,
                              image_type="imageURLHighRes",
                              caption_type="description",
                              transform=transform_test,
                              batch_size=1,
                              mode='test')
# %%
len(set(category_dict.values())) + len(set(sentence_id_dict.values()))
# %%
# current_time = "latest_model"
# encoder_file = 'encoder-57.pkl' 
# decoder_file = 'decoder-57.pkl'

current_time = "2022-04-02-02-25-40"
encoder_file = 'encoder-61.pkl' 
decoder_file = 'decoder-61.pkl'

embed_size = 125           # dimensionality of image and word embeddings
hidden_size = 512          # number of features in hidden state of the RNN decoder
num_features = 2048        # number of feature maps, produced by Encoder

# The size of the vocabulary.
vocab_size = 2435#len(test_data_loader.dataset.vocab)

# Initialize the encoder and decoder, and set each to inference mode.
encoder = EncoderCNN()
encoder.eval()
decoder = DecoderRNN(num_features = num_features, 
                     embedding_dim = embed_size,
                     category_dim = len(set(category_dict.values())) + len(set(sentence_id_dict.values())),
                     hidden_dim = hidden_size, 
                     vocab_size = vocab_size,
                     cat_attention=True)
decoder.eval()

# Load the trained weights.
encoder.load_state_dict(torch.load(os.path.join(f'../models_new/{current_time}', encoder_file), map_location='cpu'))
decoder.load_state_dict(torch.load(os.path.join(f'../models_new/{current_time}', decoder_file), map_location='cpu'))

device = "cuda"

# Move models to GPU if CUDA is available.
encoder.to(device)
decoder.to(device)
# %%
def clean_sentence(output, data_loader):
    vocab = data_loader.dataset.vocab.idx2word
    words = [vocab.get(idx) for idx in output]
    words = [word for word in words if word not in (',', '.', '<end>')]
    sentence = " ".join(words)
    
    return sentence

def get_prediction(data_loader, df):
    print("="*20)
    orig_image, image, cat_onehot, seq_onehot, caption= next(iter(data_loader))
    plt.imshow(orig_image.squeeze())
    plt.title(caption)
    plt.show()
    file_name = df[df["description"] == caption[0]]["file_name"].values[0]
    image = image.to(device)
    #display(test_df[(test_df["file_name"]==file_name)])
    for sentence_id in range(6):
        seq_onehot = torch.Tensor(data_loader.dataset.onehot_enc_seq.transform(np.array([sentence_id], 
                                                                            dtype=object).reshape(-1, 1)).toarray()).long()
        seq_onehot.to(device)
        print(sentence_id)
        print(seq_onehot)
        onehot = torch.cat((cat_onehot.view(1,-1), 
                            seq_onehot.view(1,-1)), 1).type('torch.FloatTensor').view(1,-1).to(device)
        
        features = encoder(image)
        output, atten_weights = decoder.greedy_search(features, onehot)    
        sentence = clean_sentence(output,data_loader)
        print(df[(df["file_name"]==file_name) & (df["sentence_index"]==sentence_id)]["description"].values)
        print(sentence)
        #print(sentence.replace("<unk>", ""))

# %%
for i in range(4):
    get_prediction(test_data_loader, test_df)

 # %%

# %%

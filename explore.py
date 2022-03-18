#%%
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt
%matplotlib inline
# %%
%load_ext autoreload
%autoreload
from src.utils.data_loader import get_loader
# %%
import pandas as pd
product_info = pd.read_json("data/output/sample_data.json")
display(product_info.head())
# %%
product_info = product_info.loc[:,["title", "imageURL", "description","perCategory"]].copy()
product_info.dropna(inplace=True)
product_info.reset_index(drop=True, inplace=True)
product_info = product_info.explode('imageURL').reset_index(drop=True)
product_info = product_info[product_info.imageURL != "https://images-na.ssl-images-amazon.com/images/I/31CoVdns-FL._US40_.jpg"].reset_index(drop=True)
product_info.head()
# %%
idex = 3
img_url = product_info.loc[idex,"imageURL"]
caption = product_info.loc[idex,"title"]
print(caption)
print(img_url)
I = io.imread(img_url)
plt.axis('off')
plt.imshow(I)
plt.show()

#%%
from torchvision import transforms

image_dict = product_info["imageURL"].to_dict()
caption_dict = product_info["title"].to_dict()

# Define a transform to pre-process the training images.
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    # for each color channel we pass 2 values: 
    # mean and std. deviation
    # since out image has 3 color channels, we pass 3 values for mean and  values for std. deviation
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Set the minimum word count threshold.
vocab_threshold = 5

mode = "train"

# Specify the batch size.
# we will pass 10 images at a time. So, m = 10
batch_size = 10

# %%

data_loader = get_loader(product_info,
                        transform=transform_train,mode='train',
                        batch_size=batch_size,
                        vocab_threshold=vocab_threshold,
                        vocab_from_file=False)
# %%
print('The shape of first image:', data_loader.dataset[0][0].shape)
# %%
# Preview the word2idx dictionary.
dict(list(data_loader.dataset.vocab.word2idx.items())[:10])
# %%
print('The length of entire vocabulary is:', len(data_loader.dataset.vocab.word2idx.items()))
# %%
dict(list(data_loader.dataset.vocab.idx2word.items())[:10])
# %%
# Print the total number of keys in the word2idx dictionary.
print('Total number of tokens in vocabulary:', len(data_loader.dataset.vocab))
# %%
import numpy as np
import torch.utils.data as data

# Randomly sample a caption length, and sample indices with that length.
indices = data_loader.dataset.get_indices()
print('sampled indices:', indices)

# Create and assign a batch sampler to retrieve a batch with the sampled indices.
new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
data_loader.batch_sampler.sampler = new_sampler
    
# Obtain the batch.
images, captions = next(iter(data_loader))
    
print('images.shape:', images.shape)
print('captions.shape:', captions.shape)
# %%

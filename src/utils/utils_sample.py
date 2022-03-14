# %%
import os
import json
import gzip
import pandas as pd
from random import seed, sample
seed(29)

# %%
CATEGORIES = [
	"AMAZON_FASHION", "All_Beauty", "Appliances", "Arts_Crafts_and_Sewing",
	"Automotive", "Books", "CDs_and_Vinyl", "Cell_Phones_and_Accessories",
	"Clothing_Shoes_and_Jewelry", "Digital_Music", "Electronics", "Gift_Cards",
	"Grocery_and_Gourmet_Food", "Home_and_Kitchen", "Industrial_and_Scientific",
	"Kindle_Store", "Luxury_Beauty", "Magazine_Subscriptions", "Movies_and_TV",
	"Musical_Instruments", "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies",
	"Prime_Pantry", "Software", "Sports_and_Outdoors", "Tools_and_Home_Improvement",
	"Toys_and_Games", "Video_Games"
]

INDEXES = sample(range(1200), 100)

# http://deepyeti.ucsd.edu/jianmo/amazon/index.html
URL = "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/meta_{}.json.gz" 

IN_PATH = "data/input"
OUT_PATH = "data/output"
FILEPATH = "{}/meta_{}.json.gz"


if __name__ == "__main__":
    for cat in CATEGORIES:
        if os.path.exists(FILEPATH.format(IN_PATH, cat)):
            print("{} already downloaded".format(FILEPATH.format(IN_PATH, cat)))
            continue

        print("Downloading {}".format(FILEPATH.format(IN_PATH, cat)))
        os.system("wget {} -P {}".format(URL.format(cat), IN_PATH))

    # %%
    sample_data = []
    for cat in CATEGORIES:
        data = []
        with gzip.open(FILEPATH.format(IN_PATH, cat)) as f:
            for l in f:
                row = json.loads(l.strip())
                title = row.get("title", "")
                if ("getTime" in title) or (title == ""):
                    pass
                row['perCategory'] = cat
                data.append(row)
                if len(data) > 2000:
                    break
            for ix, r in enumerate(data):
                if ix in INDEXES:
                    sample_data.append(r)
        
    with open('{}/sample_data.json'.format(OUT_PATH), 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=4)
    json.dump(INDEXES, open("{}/sample_indexes.json".format(OUT_PATH), "w"))

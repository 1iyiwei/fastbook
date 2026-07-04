import pandas as pd
import numpy as np
from fastai.collab import *
from fastai.tabular import *
from fastai.vision.all import *

# Create synthetic data for demonstration
np.random.seed(42)
n_users = 100
n_items = 50

# Generate synthetic user-item ratings (1-5 stars)
users = np.random.randint(1, n_users + 1, 1000)
items = np.random.randint(1, n_items + 1, 1000)
ratings = np.random.randint(1, 6, 1000)  # Ratings from 1 to 5

# Create DataFrame
data = pd.DataFrame({
    'user': users,
    'item': items,
    'rating': ratings
})

print("Sample of synthetic data:")
print(data.head(10))

# Binning ratings into discrete classes (1-5 stars)
# We'll create 5 classes for the 5-star rating system
num_rating_bins = 5
data['rating_binned'] = pd.cut(data['rating'], bins=num_rating_bins, labels=False, include_lowest=True)

print("\nBinned ratings:")
print(data[['rating', 'rating_binned']].head(10))

# Create DataLoaders for collaborative filtering with binned ratings
dls = CollabDataLoaders.from_df(data,
                                user_name='user', item_name='item',
                                rating_name='rating_binned',  y_block=CategoryBlock,
                                valid_pct=0.2, bs=64)

print("\nDataLoaders created:")
print(dls)

# Create a collaborative filtering model with cross-entropy loss
# This is a classification problem now (5 classes)
learn = collab_learner(dls, n_factors=10, y_range=(0,num_rating_bins-1), use_nn=True, layers=[10], loss_func=CrossEntropyLossFlat())

print("\nModel architecture:")
print(learn.model)

# Train the model for a few epochs to see if it works
print("\nStarting training...")
learn.fit_one_cycle(5, 5e-3, wd=0.1)
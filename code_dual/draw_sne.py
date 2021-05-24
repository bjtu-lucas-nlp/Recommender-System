#Import numpy
import numpy as np

#Import scikitlearn for machine learning functionalities
import sklearn
from sklearn.manifold import TSNE 
from sklearn.datasets import load_digits # For the UCI ML handwritten digits dataset

# Import matplotlib for plotting graphs ans seaborn for attractive graphics.
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sb

iteration = 5
perplexity = 10

data_path = './latent_feature/latent_features_1.npz'
with np.load(data_path) as data:
    source_user_1 = data['source_users']
    source_item_1 = data['source_items']
    target_user_1 = data['target_users']
    target_item_1 = data['target_items']


data_path = './latent_feature/latent_features_'+str(iteration)+'.npz'
with np.load(data_path) as data:
    source_user_2 = data['source_users']
    source_item_2 = data['source_items']
    target_user_2 = data['target_users']
    target_item_2 = data['target_items']
    

# X = np.vstack([digits.data[digits.target==i] for i in range(10)]) # Place the arrays of data of each digit on top of each other and store in X
# Y = np.hstack([digits.target[digits.target==i] for i in range(10)]) # Place the arrays of data of each target digit by the side of each other continuosly and store in Y

#Implementing the TSNE Function - ah Scikit learn makes it so easy!

x1 = TSNE(perplexity=perplexity).fit_transform(source_user_1[1:8938,:])
y1 = TSNE(perplexity=perplexity).fit_transform(target_user_1[1:,:])

x2 = TSNE(perplexity=perplexity).fit_transform(source_user_2[1:8938,:])
y2 = TSNE(perplexity=perplexity).fit_transform(target_user_2[1:,:])


x3 = TSNE(perplexity=perplexity).fit_transform(source_item_1[1:,:])
y3 = TSNE(perplexity=perplexity).fit_transform(target_item_1[1:7404,:])

x4 = TSNE(perplexity=perplexity).fit_transform(source_item_2[1:,:])
y4 = TSNE(perplexity=perplexity).fit_transform(target_item_2[1:7404,:])

np.savez('./latent_feature/latent_features_t_sne_'+ str(perplexity) +'_iter_'+str(iteration)+'.npz', \
            x1 = x1, y1 = y1, x2 = x2, y2 = y2, x3 = x3, y3 = y3, x4 = x4, y4 = y4)



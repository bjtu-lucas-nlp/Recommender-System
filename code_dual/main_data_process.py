# import scipy.io as sio
# import torch
# import numpy as np

# data_path = './data/movie_ml_amazon_vali.mat'

# mat_contents = sio.loadmat(data_path)
# target_train_m = np.float32(mat_contents['movie_2_Xt'].toarray())
# source_train_m = np.float32(mat_contents['movie_1_Xt'].toarray())

# num_user_target, num_item_target = np.shape(target_train_m)
# num_user_source, num_item_source = np.shape(source_train_m)

# target_train_m = np.concatenate([np.zeros((1, num_item_target)),target_train_m])
# target_train_m = np.hstack([np.zeros((num_user_target+1, 1)),target_train_m])

# source_train_m = np.concatenate([np.zeros((1, num_item_source)),source_train_m])
# source_train_m = np.hstack([np.zeros((num_user_source+1, 1)),source_train_m])

# target_test = np.float32(mat_contents['movie_2_test_vec'])
# source_test = np.float32(mat_contents['movie_1_test_vec'])
# target_train = np.float32(mat_contents['movie_2_Xt_vec'])
# source_train = np.float32(mat_contents['movie_1_Xt_vec'])
# target_vali = np.float32(mat_contents['movie_2_vali_vec'])
# source_vali = np.float32(mat_contents['movie_1_vali_vec'])

# np.savez('./data/movie_ml_amazon.npz', \
#     target_train_m = target_train_m, target_train = target_train, target_vali = target_vali, target_test = target_test, \
#         source_train_m = source_train_m, source_train = source_train, source_vali = source_vali, source_test = source_test)



# import scipy.io as sio
# import torch
# import numpy as np

# data_path = './data/book_douban_amazon_vali.mat'

# mat_contents = sio.loadmat(data_path)
# target_train_m = np.float32(mat_contents['am_Xt'].toarray())
# source_train_m = np.float32(mat_contents['db_Xt'].toarray())

# num_user_target, num_item_target = np.shape(target_train_m)
# num_user_source, num_item_source = np.shape(source_train_m)

# target_train_m = np.concatenate([np.zeros((1, num_item_target)),target_train_m])
# target_train_m = np.hstack([np.zeros((num_user_target+1, 1)),target_train_m])

# source_train_m = np.concatenate([np.zeros((1, num_item_source)),source_train_m])
# source_train_m = np.hstack([np.zeros((num_user_source+1, 1)),source_train_m])

# target_test = np.float32(mat_contents['am_test_vec'])
# source_test = np.float32(mat_contents['db_test_vec'])
# target_train = np.float32(mat_contents['am_Xt_vec'])
# source_train = np.float32(mat_contents['db_Xt_vec'])
# target_vali = np.float32(mat_contents['am_vali_vec'])
# source_vali = np.float32(mat_contents['db_vali_vec'])

# np.savez('./data/book_douban_amazon.npz', \
#     target_train_m = target_train_m, target_train = target_train, target_vali = target_vali, target_test = target_test, \
#         source_train_m = source_train_m, source_train = source_train, source_vali = source_vali, source_test = source_test)



import scipy.io as sio
import torch
import numpy as np

data_path = './data/amazon_same_user_vali.mat'

mat_contents = sio.loadmat(data_path)
target_train_m = np.float32(mat_contents['movie_1_Xt'].toarray())
source_train_m = np.float32(mat_contents['book_1_Xt'].toarray())

num_user_target, num_item_target = np.shape(target_train_m)
num_user_source, num_item_source = np.shape(source_train_m)

target_train_m = np.concatenate([np.zeros((1, num_item_target)),target_train_m])
target_train_m = np.hstack([np.zeros((num_user_target+1, 1)),target_train_m])

source_train_m = np.concatenate([np.zeros((1, num_item_source)),source_train_m])
source_train_m = np.hstack([np.zeros((num_user_source+1, 1)),source_train_m])

target_test = np.float32(mat_contents['movie_1_test_vec'])
source_test = np.float32(mat_contents['book_1_test_vec'])
target_train = np.float32(mat_contents['movie_1_Xt_vec'])
source_train = np.float32(mat_contents['book_1_Xt_vec'])
target_vali = np.float32(mat_contents['movie_1_vali_vec'])
source_vali = np.float32(mat_contents['book_1_vali_vec'])

np.savez('./data/amazon_book2movie_same.npz', \
    target_train_m = target_train_m, target_train = target_train, target_vali = target_vali, target_test = target_test, \
        source_train_m = source_train_m, source_train = source_train, source_vali = source_vali, source_test = source_test)


 
# source_train_m = np.float32(mat_contents['book_2_Xt'].toarray())

# num_user_source, num_item_source = np.shape(source_train_m)
# source_train_m = np.concatenate([np.zeros((1, num_item_source)), source_train_m])
# source_train_m = np.hstack([np.zeros((num_user_source+1, 1)), source_train_m])

# source_test = np.float32(mat_contents['book_2_test_vec'])
# source_train = np.float32(mat_contents['book_2_Xt_vec'])
# source_vali = np.float32(mat_contents['book_2_vali_vec'])

# np.savez('./data/amazon_book2movie_diff.npz', \
#     target_train_m = target_train_m, target_train = target_train, target_vali = target_vali, target_test = target_test, \
#         source_train_m = source_train_m, source_train = source_train, source_vali = source_vali, source_test = source_test)





# import scipy.io as sio
# import torch
# import numpy as np

# data_path = './NewData/douban_same_user_vali.mat'

# mat_contents = sio.loadmat(data_path)
# target_train_m = np.float32(mat_contents['book_1_Xt'])
# source_train_m = np.float32(mat_contents['movie_1_Xt'])

# num_user_target, num_item_target = np.shape(target_train_m)
# num_user_source, num_item_source = np.shape(source_train_m)

# target_train_m = np.concatenate([np.zeros((1, num_item_target)),target_train_m])
# target_train_m = np.hstack([np.zeros((num_user_target+1, 1)),target_train_m])

# source_train_m = np.concatenate([np.zeros((1, num_item_source)),source_train_m])
# source_train_m = np.hstack([np.zeros((num_user_source+1, 1)),source_train_m])

# target_test = np.float32(mat_contents['book_1_test_vec'])
# source_test = np.float32(mat_contents['movie_1_test_vec'])
# target_train = np.float32(mat_contents['book_1_Xt_vec'])
# source_train = np.float32(mat_contents['movie_1_Xt_vec'])
# target_vali = np.float32(mat_contents['book_1_vali_vec'])
# source_vali = np.float32(mat_contents['movie_1_vali_vec'])

# np.savez('./NewData/douban_movie2book_same.npz', \
#     target_train_m = target_train_m, target_train = target_train, target_vali = target_vali, target_test = target_test, \
#         source_train_m = source_train_m, source_train = source_train, source_vali = source_vali, source_test = source_test)


 

# source_train_m = np.float32(mat_contents['movie_2_Xt'])

# num_user_source, num_item_source = np.shape(source_train_m)
# source_train_m = np.concatenate([np.zeros((1, num_item_source)), source_train_m])
# source_train_m = np.hstack([np.zeros((num_user_source+1, 1)), source_train_m])

# source_test = np.float32(mat_contents['movie_2_test_vec'])
# source_train = np.float32(mat_contents['movie_2_Xt_vec'])
# source_vali = np.float32(mat_contents['movie_2_vali_vec'])

# np.savez('./NewData/douban_movie2book_diff.npz', \
#     target_train_m = target_train_m, target_train = target_train, target_vali = target_vali, target_test = target_test, \
#         source_train_m = source_train_m, source_train = source_train, source_vali = source_vali, source_test = source_test)

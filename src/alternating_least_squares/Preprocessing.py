__author__ = 'Marcelo d\'Almeida'

import pickle

import os.path
import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k, train_test_split
from scipy.sparse import csr_matrix


def preprocess():

    if os.path.isfile('dataset/modified/simple_order_product_data.csv'):
        print("Reading simple_order_product_data.csv")
        order_and_product_data = pd.read_csv('dataset/modified/simple_order_product_data.csv')
    else:

        print("Creating simple_order_product_data.csv")

        order_product_prior_data = pd.read_csv('dataset/order_products__prior.csv')
        order_product_train_data = pd.read_csv('dataset/order_products__train.csv')

        order_and_product_data = pd.concat([order_product_prior_data, order_product_train_data])

        order_and_product_data.to_csv('dataset/modified/simple_order_product_data.csv', index=False);

        print("Done saving simple_order_product_data.csv")

    if os.path.isfile('dataset/modified/user_product_data.csv'):
        print("Reading user_product_data.csv")
        user_product_data = pd.read_csv('dataset/modified/user_product_data.csv')
    else:

        print("Creating user product sparse matrix")

        order_and_product_data = pd.read_csv('dataset/modified/simple_order_product_data.csv')
        order_data = pd.read_csv('dataset/orders.csv')

        product_data = pd.read_csv('dataset/products.csv')

        order_and_user_data = order_data.loc[:,('order_id', 'user_id')];

        user_product_data = pd.merge(order_and_product_data, order_and_user_data,
                       on="order_id",
                       how='left')

        user_product_data = user_product_data.loc[:, ('user_id', 'product_id')]
        user_product_data = user_product_data.groupby(['user_id', 'product_id']).size()
        user_product_data = user_product_data.to_frame().reset_index()
        user_product_data.columns = ['user_id', 'product_id', 'total']

        user_product_data.to_csv('dataset/modified/user_product_data.csv', index=False);

    if not os.path.isfile('dataset/modified/alternating_least_squares/product_user_sparse_matrix.pickle'):

        row_ind = user_product_data['product_id']
        col_ind = user_product_data['user_id']
        data = user_product_data['total']

        product_user_sparse_matrix = csr_matrix((data, (row_ind, col_ind)))

        with open('dataset/modified/alternating_least_squares/product_user_sparse_matrix.pickle', 'wb') as handle:
            pickle.dump(product_user_sparse_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Done pickling user product sparse matrix")

    if not os.path.isfile('dataset/modified/alternating_least_squares/model.pickle'):

        model = AlternatingLeastSquares()
        model.fit(product_user_sparse_matrix)

        with open('dataset/modified/alternating_least_squares/model.pickle', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done fiting model")

    print("Preprocessing Done")



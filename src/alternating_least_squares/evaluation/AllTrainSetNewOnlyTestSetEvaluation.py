__author__ = 'Marcelo d\'Almeida'

from tqdm import tqdm
import os.path
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


def evaluate():

    print("Creating simple_order_product_data.csv")

    order_product_prior_data = pd.read_csv('dataset/order_products__prior.csv')
    order_product_train_data = pd.read_csv('dataset/order_products__train.csv')

    order_and_product_train_data = order_product_prior_data

    print("Done saving simple_order_product_data.csv")

    order_data = pd.read_csv('dataset/orders.csv')


    order_and_user_data = order_data.loc[:,('order_id', 'user_id')];

    user_product_train_data = pd.merge(order_and_product_train_data, order_and_user_data,
                   on="order_id",
                   how='left')

    user_product_train_data = user_product_train_data.loc[:, ('user_id', 'product_id')]
    user_product_train_data = user_product_train_data.groupby(['user_id', 'product_id']).size()
    user_product_train_data = user_product_train_data.to_frame().reset_index()
    user_product_train_data.columns = ['user_id', 'product_id', 'total']

    user_product_data_train_set = user_product_train_data

    row_ind = user_product_data_train_set['product_id']
    col_ind = user_product_data_train_set['user_id']
    data = user_product_data_train_set['total']

    product_user_sparse_matrix = csr_matrix((data, (row_ind, col_ind)))

    print("Done pickling user product sparse matrix")

    model = AlternatingLeastSquares()
    model.fit(product_user_sparse_matrix)

    print("Done fiting model")

    print("Preprocessing Done")


    order_and_product_test_data = order_product_train_data

    user_product_data_test = pd.merge(order_and_product_test_data, order_and_user_data,
                   on="order_id",
                   how='left')

    user_product_data_test_set = user_product_data_test.loc[user_product_data_test['reordered'] == 0]

    sample = 10000
    seed = 145

    users = user_product_data_test_set['user_id'].drop_duplicates()
    users = list(users.sample(sample, random_state=seed))

    correct = 0
    total = 0

    with tqdm(total=len(users)) as progress:
        for user in users:

            user_products_test_set = list(user_product_data_test_set.loc[user_product_data_test_set['user_id'] == user]['product_id'])
            recommendations = model.recommend(user, product_user_sparse_matrix.T.tocsr(), N=len(user_products_test_set))

            for recommendation in recommendations:

                product = recommendation[0]
                if product in user_products_test_set:
                    correct += 1

                total += 1
            progress.update(1)

    print(correct)
    print(total)
    print("% correct:", correct/total)

    data = {'correct': [correct], 'total': [total], 'percentage': [correct/total]}
    evaluation = pd.DataFrame(data=data)

    evaluation.to_csv("dataset/modified/alternating_least_squares/evaluation/all_train-set__new-only_test-set_evaluation--sample-" + str(sample) + "-seed-" + str(seed) + ".csv", index=False)

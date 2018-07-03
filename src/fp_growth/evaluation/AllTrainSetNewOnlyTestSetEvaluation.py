__author__ = 'Marcelo d\'Almeida'

import time

import orangecontrib.associate.fpgrowth as fp
import os.path
import pandas as pd
from tqdm import tqdm

from fp_growth import Main as Fp_Growth_Main


def evaluate():

    support_threshold = 1000
    confidence_threshold = 0.1

    print("Creating simple_order_product_data.csv")

    order_product_prior_data = pd.read_csv('dataset/order_products__prior.csv')
    order_product_train_data = pd.read_csv('dataset/order_products__train.csv')

    order_and_product_data = order_product_prior_data

    print("Done saving simple_order_product_data.csv")

    print("Creating transactions")

    order_and_product_data = pd.read_csv('dataset/modified/simple_order_product_data.csv')
    order_data = pd.read_csv('dataset/orders.csv')

    order_and_user_data = order_data.loc[:, ('order_id', 'user_id')];

    raw_user_product_data = pd.merge(order_and_product_data, order_and_user_data,
                                 on="order_id",
                                 how='left')

    order_and_product_data = order_and_product_data.loc[:, ('order_id', 'product_id')];

    order_and_product_grouped_by_data_dict = order_and_product_data.groupby('order_id')['product_id'].apply(list).to_dict()

    transactions = list(order_and_product_grouped_by_data_dict.values())

    print("Generating itemset")

    t0 = time.time()
    patterns = dict(fp.frequent_itemsets(transactions, support_threshold))
    t1 = time.time()
    total = t1 - t0
    print("patterns time:", total)

    print("Generating rules")

    t0 = time.time()
    rules = list(fp.association_rules(patterns, confidence_threshold))
    t1 = time.time()
    total = t1 - t0
    print("rules generation time:", total)

    order_and_product_test_data = order_product_train_data

    user_product_data_test = pd.merge(order_and_product_test_data, order_and_user_data,
                   on="order_id",
                   how='left')

    user_product_data_test_set = user_product_data_test.loc[user_product_data_test['reordered'] == 0]


    if os.path.isfile('dataset/modified/user_product_data.csv'):
        print("Reading user_product_data.csv")
        user_product_data = pd.read_csv('dataset/modified/user_product_data.csv')

    sample = 10000
    seed = 145

    users = user_product_data_test_set['user_id'].drop_duplicates()
    users = list(users.sample(sample, random_state=seed))

    correct = 0
    total = 0

    with tqdm(total=len(users)) as progress:
        for user in users:

            user_products = list(user_product_data.loc[user_product_data['user_id'] == user]['product_id'])

            user_products_test_set = list(user_product_data_test_set.loc[user_product_data_test_set['user_id'] == user]['product_id'])
            recommendations = Fp_Growth_Main.recommend(rules, user_products, N=len(user_products_test_set))

            for recommendation in recommendations:

                consequent = recommendation[1]

                for product in consequent:
                    if product in user_products_test_set:
                        correct += 1

                total += 1
            progress.update(1)

    print(correct)
    print(total)

    if total != 0:
        percent_correct = correct / total
    else:
        percent_correct = 0

    print("% correct:", percent_correct)

    data = {'correct': [correct], 'total': [total], 'percentage': [percent_correct]}
    evaluation = pd.DataFrame(data=data)

    evaluation.to_csv("dataset/modified/fp_growth/evaluation/all_train-set__new-only_test-set_evaluation--sample-" + str(sample) + "-seed-" + str(seed) + ".csv", index=False)

__author__ = 'Marcelo d\'Almeida'

import pickle
import time

import orangecontrib.associate.fpgrowth as fp
import os.path
import pandas as pd


def preprocess(support_threshold=70000, confidence_threshold=0.01):

    if not os.path.isfile('dataset/modified/fp_growth/transactions.pickle'):

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

        print("Creating transactions")

        order_and_product_data = order_and_product_data.loc[:, ('order_id', 'product_id')];

        order_and_product_grouped_by_data_dict = order_and_product_data.groupby('order_id')['product_id'].apply(list).to_dict()

        transactions = list(order_and_product_grouped_by_data_dict.values())

        with open('dataset/modified/fp_growth/transactions.pickle', 'wb') as handle:
            pickle.dump(transactions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Done pickling transactions")

    if os.path.isfile('dataset/modified/fp_growth/pattern' + str(support_threshold) + '.pickle'):
        with open('dataset/modified/fp_growth/pattern' + str(support_threshold) + '.pickle', 'rb') as handle:
            patterns = pickle.load(handle)
    else:
        with open('dataset/modified/fp_growth/transactions.pickle', 'rb') as handle:
            transactions = pickle.load(handle)

        t0 = time.time()

        patterns = dict(fp.frequent_itemsets(transactions, support_threshold))

        t1 = time.time()
        total = t1 - t0
        print("patterns time:", total)

        with open('dataset/modified/fp_growth/pattern' + str(support_threshold) + '.pickle', 'wb') as handle:
            pickle.dump(patterns, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile('dataset/modified/fp_growth/rules' + str(support_threshold) + '_' + str(confidence_threshold) + '.pickle'):
        with open('dataset/modified/fp_growth/rules' + str(support_threshold) + '_' + str(confidence_threshold) + '.pickle', 'rb') as handle:
            rules = pickle.load(handle)
    else:
        t0 = time.time()
        rules = list(fp.association_rules(patterns, confidence_threshold))
        t1 = time.time()
        total = t1 - t0
        print("rules generation time:", total)

        calculate_lift(rules, patterns)

        with open('dataset/modified/fp_growth/rules' + str(support_threshold) + '_' + str(confidence_threshold) + '.pickle', 'wb') as handle:
            pickle.dump(rules, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Preprocessing Done")

def calculate_lift(rules, patterns):

    for rule in rules:
        consequent = rule[1]
        confidence = rule[3]
        lift = confidence / patterns[consequent]
        rule += (lift,)
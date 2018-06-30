__author__ = 'Marcelo d\'Almeida'

import pickle
import time

import os.path
import pandas as pd

import orangecontrib.associate.fpgrowth as fp


def preprocess(support_threshold=70000, confidence_threshold=0.0):

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
        rules = fp.association_rules(patterns, confidence_threshold)
        t1 = time.time()
        total = t1 - t0
        print("rules generation time:", total)

        #calculate_lift(rules, transactions)

        with open('dataset/modified/fp_growth/rules' + str(support_threshold) + '_' + str(confidence_threshold) + '.pickle', 'wb') as handle:
            pickle.dump(dict(rules), handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Preprocessing Done")

def calculate_lift(rules, transactions):

    support_dict = {}

    tuple_list = []
    for key, value in rules.items():
        tuple = (key, value[0], value[1])
        tuple_list.append(tuple)

    #lift = confidence / support(consequent)

    consequent_count = {}

    valid_transaction_count = 0
    length = len(transactions)
    for transaction in transactions:

        for value in rules.items():
            consequent = value[0]

            valid_transaction = True
            for item in consequent:
                if item not in transaction:
                    valid_transaction = False

            if (valid_transaction):
                valid_transaction_count += 1

    return valid_transaction_count / length
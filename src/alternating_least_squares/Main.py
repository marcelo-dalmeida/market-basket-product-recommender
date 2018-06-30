
import pickle

import os.path
import pandas as pd


def alternating_least_squares_main(user):

    if os.path.isfile('dataset/modified/user_product_data.csv'):
        print("Reading user_product_data.csv")
        user_product_data = pd.read_csv('dataset/modified/user_product_data.csv')

    if os.path.isfile('dataset/products.csv'):
        print("Reading product_data.csv")
        product_data = pd.read_csv('dataset/products.csv')

    if os.path.isfile('dataset/modified/alternating_least_squares/product_user_sparse_matrix.pickle'):
        with open('dataset/modified/alternating_least_squares/product_user_sparse_matrix.pickle', 'rb') as handle:
            product_user_sparse_matrix = pickle.load(handle)

    if os.path.isfile('dataset/modified/alternating_least_squares/model.pickle'):
        with open('dataset/modified/alternating_least_squares/model.pickle', 'rb') as handle:
            model = pickle.load(handle)

    user_products = list(user_product_data.loc[user_product_data['user_id'] == user]['product_id'])

    print("User Current Products")
    for product in user_products:
        product_name = product_data.loc[product_data['product_id'] == product]['product_name'].iat[0]
        print("User:", user, "- Product:", product_name)

    recommendations = model.recommend(user, product_user_sparse_matrix.T.tocsr())

    print("User Recommended Products")
    for recommendation in recommendations:
        product_name = product_data.loc[product_data['product_id'] == recommendation[0]]['product_name'].iat[0]
        print("Recommended Product:", product_name, "- Score:", recommendation[1])

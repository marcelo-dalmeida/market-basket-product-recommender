__author__ = 'Marcelo d\'Almeida'

import pickle
import time

import os.path
import pandas as pd

import src.algorithm.fp_growth.pyfpgrowth.pyfpgrowth as pyfpgrowthpyfpgrowth


def fp_growth_main(user, support_threshold=30000, confidence_threshold=0.0):

    if os.path.isfile('dataset/modified/fp_growth/pattern' + str(support_threshold) + '.pickle'):
        with open('dataset/modified/fp_growth/pattern' + str(support_threshold) + '.pickle', 'rb') as handle:
            patterns = pickle.load(handle)

    if os.path.isfile('dataset/modified/fp_growth/rules' + str(support_threshold) + '_' + str(confidence_threshold) + '.pickle'):
        with open('dataset/modified/fp_growth/rules' + str(support_threshold) + '_' + str(confidence_threshold) + '.pickle', 'rb') as handle:
            rules = pickle.load(handle)

    print(rules)

    #print(recommend(user, rules))

    print(getAllRules(rules))

def getAllRules(rules):

    if os.path.isfile('dataset/products.csv'):
        print("Reading product_data.csv")
        product_data = pd.read_csv('dataset/products.csv')

    tuple_list = []
    for key, value in rules.items():
        tuple = (key, value[0], value[1])
        tuple_list.append(tuple)

    sorted_rules = sorted(tuple_list, key=lambda x: x[2], reverse=True)

    with open('dataset/modified/fp_growth/transactions.pickle', 'rb') as handle:
        transactions = pickle.load(handle)

    sorted_product_rules = []
    for antecedent, consequent, confidence in sorted_rules:

        antecedent_list = []
        for a in antecedent:
            product_name = product_data.loc[product_data['product_id'] == a]['product_name'].iat[0]
            antecedent_list.append(product_name)

        consequent_list = []
        for c in consequent:
            product_name = product_data.loc[product_data['product_id'] == c]['product_name'].iat[0]
            consequent_list.append(product_name)

        sorted_product_rules.append((antecedent_list, consequent_list, confidence))

    for sorted_product_rule in sorted_product_rules:
        print("Antecedent", sorted_product_rule[0], "- Consequent:", sorted_product_rule[1], "- Confidence:", sorted_product_rule[2])



def recommend(user, rules):

    if os.path.isfile('dataset/modified/user_product_data.csv'):
        print("Reading user_product_data.csv")
        user_product_data = pd.read_csv('dataset/modified/user_product_data.csv')

    if os.path.isfile('dataset/products.csv'):
        print("Reading product_data.csv")
        product_data = pd.read_csv('dataset/products.csv')

    user_products = list(user_product_data.loc[user_product_data['user_id'] == user]['product_id'])

    print("User Current Products")
    for product in user_products:
        product_name = product_data.loc[product_data['product_id'] == product]['product_name'].iat[0]
        print("User:", user, "- Product: ", product_name)

    recommendations = {}
    for product in user_products:
        if product in rules:
            recommendations[product] = rules[product]

    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1])[0:10]

    print("User Recommended Products")
    for recommendation in sorted_recommendations:
        product_name = product_data.loc[product_data['product_id'] == recommendation[0]]['product_name'].iat[0]
        print("Recommended Product:", user, "- Product: ", product_name, "- Confidence:", recommendation[1])

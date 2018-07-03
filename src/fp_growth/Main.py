__author__ = 'Marcelo d\'Almeida'

import pickle

import os.path
import pandas as pd


def fp_growth_main(user, support_threshold=30000, confidence_threshold=0.01):

    if os.path.isfile('dataset/modified/fp_growth/pattern' + str(support_threshold) + '.pickle'):
        with open('dataset/modified/fp_growth/pattern' + str(support_threshold) + '.pickle', 'rb') as handle:
            patterns = pickle.load(handle)

    if os.path.isfile('dataset/modified/fp_growth/rules' + str(support_threshold) + '_' + str(confidence_threshold) + '.pickle'):
        with open('dataset/modified/fp_growth/rules' + str(support_threshold) + '_' + str(confidence_threshold) + '.pickle', 'rb') as handle:
            rules = pickle.load(handle)

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

    recommended_rules = recommend(rules, user_products)

    print("User Recommended Products")
    for rule in recommended_rules:

        consequent = rule[1]
        confidence = rule[3]

        for product in consequent:
            product_name = product_data.loc[product_data['product_id'] == product]['product_name'].iat[0]
            print("Recommended Product:", user, "- Product: ", product_name, "- Confidence:", confidence)



def recommend(rules, user_products, N=10):

    applicable_rules = []
    for rule in rules:

        antecedent = rule[0]
        consequent = rule[1]

        is_rule_applicable = True
        for item in antecedent:
            if item not in user_products:
                is_rule_applicable = False

        for item in consequent:
            if item in user_products:
                is_rule_applicable = False

        if (is_rule_applicable):
            applicable_rules.append(rule)


    recommended_rules = sorted(applicable_rules, key=lambda x: x[3], reverse=True)

    i = 0
    products_recommended = []
    rules_used = []

    for rule in recommended_rules:

        if len(products_recommended) >= N:
            break

        consequent = rule[1]

        for product in consequent:

            if product not in products_recommended:
                i += 1
                products_recommended.append(product)
                rules_used.append(rule)

    return rules_used


def getAllRules(rules):

    if os.path.isfile('dataset/products.csv'):
        print("Reading product_data.csv")
        product_data = pd.read_csv('dataset/products.csv')


    sorted_rules = sorted(rules, key=lambda x: x[3], reverse=True)

    with open('dataset/modified/fp_growth/transactions.pickle', 'rb') as handle:
        transactions = pickle.load(handle)

    sorted_product_rules = []
    for rule in sorted_rules:

        antecedent = rule[0]
        consequent = rule[1]
        confidence = rule[2]

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
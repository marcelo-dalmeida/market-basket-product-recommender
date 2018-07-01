__author__ = 'Marcelo d\'Almeida'

from fp_growth import Main as  Fp_Growth_Main
from fp_growth import Preprocessing as FP_Growth_Preprocessing
from alternating_least_squares import Main as Alternating_Least_Squares_Main
from alternating_least_squares import Preprocessing as Alternating_Least_Squares_Preprocessing
from alternating_least_squares.evaluation import AllReorderedTrainSetAllNewTestSetEvaluation as Alternating_Least_Squares_Evaluation

import argparse

def fp_growth_preprocessing(support_threshold, confidence_threshold):
    FP_Growth_Preprocessing.preprocess(support_threshold, confidence_threshold)

def fp_growth_main(user, support_threshold, confidence_threshold):
    Fp_Growth_Main.fp_growth_main(user, support_threshold, confidence_threshold)

def alternating_least_squares_preprocessing():
    Alternating_Least_Squares_Preprocessing.preprocess()

def alternating_least_squares_main(user):
    Alternating_Least_Squares_Main.alternating_least_squares_main(user)

def alternating_least_squares_evaluation():
    Alternating_Least_Squares_Evaluation.evaluate()


def main():

    #alternating_least_squares_evaluation()

    stop_condition = False

    while not stop_condition:
        parser = argparse.ArgumentParser()

        parser.add_argument("-a", "--algorithm")
        parser.add_argument("-s", "--step")

        args = parser.parse_args()

        if not args.algorithm:
            algorithm_input = input('FP-Growth or Alternating Least Square? -> [FP][ALS] - [Q]/[QUIT] ').upper()
        else:
            algorithm_input = args.algorithm

        if (algorithm_input == "Q" or algorithm_input == "QUIT"):
            stop_condition = True
            continue

        if not args.step:
            step_input = input('Preprocess, run or both? -> [PRE][RUN][BOTH] ').upper()
        else:
            step_input = args.step

        if algorithm_input == "FP":
            support_threshold = int(input('Support Threshold: '))
            confidence_threshold = float(input('Confidence Threshold: '))
            if step_input == "PRE" or step_input == "BOTH":
                fp_growth_preprocessing(support_threshold, confidence_threshold)
            if step_input == "RUN" or step_input == "BOTH":
                user_input = input('Recommendations for User: ').upper()
                user = int(user_input)
                fp_growth_main(user, support_threshold, confidence_threshold)

        if algorithm_input == "ALS":
            if step_input == "PRE" or step_input == "BOTH":
                alternating_least_squares_preprocessing()
            if step_input == "RUN" or step_input == "BOTH":
                user = int(input('Recommendations for User: ').upper())
                alternating_least_squares_main(user)

if __name__ == '__main__':
    main()
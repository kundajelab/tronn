# description: post process learning

import os
import h5py

import numpy as np

from scipy.stats import spearmanr
from scipy.stats import pearsonr

from sklearn import linear_model
from sklearn import ensemble
from sklearn import tree
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve


def build_regression_model(X, y, alpha=0.001):
    """Build a regression model and return coefficients
    """
    #clf = linear_model.Lasso(alpha=0.001, positive=True)
    #clf = linear_model.ElasticNet(alpha=0.1)
    #clf = linear_model.HuberRegressor(epsilon=3.0, alpha=0.0001)
    #clf = linear_model.LinearRegression()
    clf = linear_model.ElasticNet(alpha=alpha, positive=True)
    #clf = tree.DecisionTreeRegressor()
    #clf = ensemble.RandomForestRegressor()
    #clf = linear_model.ElasticNet(alpha=alpha, positive=False)
    #clf = linear_model.LogisticRegression()
    clf.fit(X, y)
    #if True:
    #    return clf
    print clf.coef_
    if np.sum(clf.coef_) == 0:
        clf = linear_model.LinearRegression()
        #clf = linear_model.ElasticNet(alpha=alpha, positive=False)
        clf.fit(X, y)
        print clf.coef_

    print spearmanr(clf.predict(X), y)
    #print pearsonr(clf.predict(X), y)
    #print spearmanr(clf.predict_proba(X)[:,1], y)
    
    return clf


def build_polynomial_model(X, y, degree=2, feature_names=None):
    """Build the polynomial features and fit a model, return model
    and feature names
    """
    # build polynomial features
    poly = PolynomialFeatures(
        2, interaction_only=True, include_bias=False)
    X = poly.fit_transform(X)
    poly_names = poly.get_feature_names(feature_names)

    # build model
    clf = build_regression_model(X, y)
    
    return clf, poly_names

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


def build_lasso_regression_models(
        h5_file,
        dataset_keys,
        cluster_key,
        refined_cluster_key,        
        out_key,
        pwm_list,
        pwm_dict):
    """Build regularized regression models
    """
    # for each dataset
    # extract the correct datasets

    
    # first reduce motif redundancy with loose cutoffs to reduce more expansively


    # then run first lasso. 
    # tricky part is that the lasso may find them all correlated and return none of them...
    # so need to do a lasso with negative logits from other examples
    

    # with reduced feature set, run second lasso with pairwise terms.
    # optimize alpha

    # adjust coefficient based on raw scores

    with h5py.File(h5_file, "a") as hf:
        num_examples, num_motifs = hf[dataset_keys[0]].shape

        # set up a new cluster dataset, which will get the results here
        # of grammars after thresholds
        del hf[refined_cluster_key]
        refined_clusters_hf = hf.create_dataset(
            refined_cluster_key, hf[cluster_key].shape, dtype=int)

        # get metaclusters and ignore the non-clustered set (highest ID value)
        metacluster_by_region = hf[cluster_key][:,0]
        metaclusters = list(set(hf[cluster_key][:,0].tolist()))
        max_id = max(metaclusters)
        metaclusters.remove(max_id)

        # set up out matrix {metacluster, task, M} <- save as motif vector
        # TODO also need to save adjacency matrix?
        del hf[out_key]
        metacluster_motifs_hf = hf.create_dataset(
            out_key, (len(metaclusters), len(dataset_keys), num_motifs))
        metacluster_motifs_hf.attrs["pwm_names"] = hf[dataset_keys[0]].attrs["pwm_names"]
        # finish filling out attributes first, then append to dataset
        thresholds = np.zeros((
            len(metaclusters), len(dataset_keys), 1))
        
        # for each metacluster, for each dataset, get matrix
        for i in xrange(len(metaclusters)):
            metacluster_id = metaclusters[i]
            print"metacluster:", metacluster_id
            metacluster_thresholds = np.zeros((len(dataset_keys), 1))
            
            for j in xrange(len(dataset_keys)):
                print "task:", j
                dataset_key = dataset_keys[j]
                
                # get subset
                sub_dataset = hf[dataset_key][:][
                    np.where(metacluster_by_region == metacluster_id)[0],:]
                print "total examples used:", sub_dataset.shape
                
                # reduce pwms by signal similarity - a hierarchical clustering
                pwm_vector = reduce_pwms_by_signal_similarity(
                    sub_dataset, pwm_list, pwm_dict)

                # here, throw in the lasso model - first round
                # how to choose alpha?
                X = np.multiply(pwm_vector, hf[dataset_key][:])
                y = np.multiply(
                    hf["logits"][:, j],
                    #hf["labels"][:, j], # TODO - here need the importance task true indices
                    (metacluster_by_region == metacluster_id).astype(int))

                # row normalize
                #X = X / (np.sum(X, axis=1, keepdims=True) + 0.00000000001)
                
                # balance positives and negatives?
                X_pos = X[y > 0,:]
                y_pos = y[y > 0]
                pos_size = X_pos.shape[0]
                
                X_neg = X[y == 0,:]
                y_neg = y[y == 0]
                neg_size = X_neg.shape[0]
                idx = np.random.randint(neg_size, size=pos_size)
                X_neg = X_neg[idx,:]
                y_neg = y_neg[idx]
    
                X_bal = np.vstack((X_pos, X_neg))
                y_bal = np.hstack((y_pos, y_neg))
    
                print X_bal.shape
                
                print "running linear model"
                clf = linear_model.Lasso(alpha=0.000001)
                #clf = linear_model.LinearRegression()
                #clf = linear_model.LogisticRegression(C=1, penalty="l1", tol=0.0001)
                clf.fit(X, y)
                print clf.coef_
                print clf.score(X, y)
                
                # now filter again on where these coefficients are nonzero

                # add in pairwise features

                # and run again
                
                import ipdb
                ipdb.set_trace()

                
                # then set a cutoff
                pwm_vector = sd_cutoff(sub_dataset, pwm_vector)
                print "final pwm count:", np.sum(pwm_vector)

                # reduce the cluster size to those that have
                # all of the motifs in the pwm vector
                refined_clusters_hf[metacluster_by_region==metacluster_id, 0] = max_id
                masked_data = np.multiply(
                    hf[dataset_key][:],
                    np.expand_dims(pwm_vector, axis=0))
                minimal_motif_mask = np.sum(masked_data > 0, axis=1) >= np.sum(pwm_vector)
                metacluster_mask = metacluster_by_region == metacluster_id
                final_mask = np.multiply(minimal_motif_mask, metacluster_mask)
                refined_clusters_hf[final_mask > 0, 0] = metacluster_id

                # save out the pwm vector
                metacluster_motifs_hf[i, j, :] = pwm_vector

                # determine best threshold and save out to attribute, set at 5% FDR
                scores = np.sum(
                    np.multiply(
                        masked_data,
                        np.expand_dims(minimal_motif_mask, axis=1)),
                    axis=1)
                labels = metacluster_mask
                get_threshold = make_threshold_at_fdr(0.05)
                threshold = get_threshold(labels, scores)
                metacluster_thresholds[j, 0] = threshold
                
            # after finishing all tasks, save out to grammar file
            if grammar_files:
                grammar_file = "{0}.metacluster-{1}.motifset.grammar".format(
                    h5_file.split(".h5")[0], metacluster_id)
                for j in xrange(len(dataset_keys)):
                    pwm_vector = metacluster_motifs_hf[i,j,:]
                    pwm_names = np.array(metacluster_motifs_hf.attrs["pwm_names"])
                    pwm_names = pwm_names[np.where(pwm_vector)]
                    threshold = metacluster_thresholds[j,0]
                    print threshold
                    print pwm_names
                    node_dict = {}
                    for pwm in pwm_names:
                        node_dict[pwm] = 1.0
                    task_grammar = Grammar(
                        pwm_file,
                        node_dict,
                        {},
                        ("taskidx={0};"
                         "type=metacluster;"
                         "directed=no;"
                         "threshold={1}").format(j, threshold),
                        "metacluster-{0}.taskidx-{1}".format(
                            metacluster_id, j)) # TODO consider adjusting the taskidx here
                    task_grammar.to_file(grammar_file)
                    
            if visualize:
                # network plots?
                pass
                    
            # add thresholds    
            thresholds[i,:,:] = metacluster_thresholds

        # append
        metacluster_motifs_hf.attrs["thresholds"] = thresholds                    

    

    return



def threshold_at_recall(labels, scores, recall_thresh=1.0):
    """Get highest threshold at which there is perfect recall
    """
    pr_curve = precision_recall_curve(labels, scores)
    precision, recall, thresholds = pr_curve
    threshold_index = np.searchsorted(1 - recall, 1 - recall_thresh)
    #print "precision at thresh", precision[index]
    #print "recall at thresh", recall[threshold_index]
    try:
        return thresholds[threshold_index]
    except:
        return 0 # TODO figure out what to do here...
    

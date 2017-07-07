"""Contains code to run baseline models as comparisons
"""

def build_estimator(model_dir):
    """Build an estimator."""
    params = tensor_forest.ForestHParams(
        num_classes=10, num_features=784,
        num_trees=FLAGS.num_trees, max_nodes=FLAGS.max_nodes) # fix these
    graph_builder_class = tensor_forest.RandomForestGraphs
    if FLAGS.use_training_loss:
        graph_builder_class = tensor_forest.TrainingLossForest
    # Use the SKCompat wrapper, which gives us a convenient way to split
    # in-memory data like MNIST into batches.
    return estimator.SKCompat(random_forest.TensorForestEstimator(
        params, graph_builder_class=graph_builder_class,
        model_dir=model_dir))


def train_and_eval_tensorforest(
        data_file_list,
        data_loader,
        batch_size,
        tasks,
        out_dir):
    """Runs random forest baseline model
    Note that this is TFLearn's model
    """
    
    est = build_estimator(out_dir)
    features, labels, metadata = data_loader(data_file_list, batch_size, tasks)
    
    #est.fit(x=features, y=labels,
    #        batch_size=batch_size)

    est.fit(input_fn=data_loader, batch_size=batch_size)
    
    # TODO change metrics here
    metric_name = 'accuracy'
    metric = {metric_name:
              metric_spec.MetricSpec(
                  eval_metrics.get_metric(metric_name),
                  prediction_key=eval_metrics.get_prediction_key(metric_name))}

    # get the results
    results = est.score(x=mnist.test.images, y=mnist.test.labels,
                        batch_size=FLAGS.batch_size,
                        metrics=metric)
    
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))

    return None




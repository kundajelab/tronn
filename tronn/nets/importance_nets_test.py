"""description: tests for importance nets
"""
import pytest

import numpy as np
import tensorflow as tf

from tronn.nets.importance_nets import InputxGrad
from tronn.nets.importance_nets import DeltaFeatureImportanceMapper
from tronn.util.utils import DataKeys


class SquareTest(tf.test.TestCase):

    def testSquare(self):
        with self.test_session():
            x = tf.square([2, 3])
            self.assertAllEqual(x.eval(), [4, 9])


@pytest.fixture(scope="class")
def one_hot_sequence(request):
    """create a one hot sequence of shape {1, 1, 1000, 4}
    """
    onehot_vectors = np.eye(4)
    sequence = onehot_vectors[
        np.random.choice(onehot_vectors.shape[0], size=1000)]
    sequence = np.expand_dims(sequence, axis=0)
    sequence = np.expand_dims(sequence, axis=0)
    request.cls.one_hot_sequence = sequence
    
            
@pytest.fixture
def input_x_grad_extractor():
    """fixture that makes an 
    """
    
    pass


@pytest.mark.usefixtures("one_hot_sequence")
class FeatureImportanceExtractorTests(tf.test.TestCase):

    def test_attach_shuffles(self):
        """test attaching dinuc shuffles
        """
        sequence_content = np.sum(self.one_hot_sequence, axis=(0,1,2))
        num_shuffles = 7
        
        with self.test_session():
            # arrange: need class instance, features tensor, no params (minimal inputs)
            extractor = InputxGrad(None, num_shuffles=num_shuffles)
            inputs = {DataKeys.FEATURES: tf.convert_to_tensor(self.one_hot_sequence)}
            params = {}

            # act: run preprocess fn and eval
            outputs, _ = extractor.preprocess(inputs, params)
            results = self.evaluate(outputs)
            
        # assert: original sequence is properly stored
        orig_sequence_maintained = np.array_equal(
            self.one_hot_sequence,
            np.expand_dims(results[DataKeys.ORIG_SEQ][0], axis=0))
        assert orig_sequence_maintained, "orig sequence not maintained"

        # assert: sequence composition is not changed when shuffling
        for i in range(1, num_shuffles+1):
            sequence_content_maintained = np.array_equal(
                sequence_content,
                np.sum(results[DataKeys.FEATURES][i], axis=(0,1)))
            assert sequence_content_maintained, "dinuc shuffle {} don't maintain sequence content".format(i)

        # assert: attached properly
        final_batch_size = self.one_hot_sequence.shape[0] * (num_shuffles + 1)
        assert results[DataKeys.FEATURES].shape[0] == final_batch_size

        # TODO need to do tests where shuffles are interleaved
        

    def test_threshold_fresh(self):
        """test thresholding
        """
        # set up fake weighted sequence with scores 0-999
        weights = np.arange(self.one_hot_sequence.shape[2])
        weights = np.reshape(weights, (1, 1, self.one_hot_sequence.shape[2], 1))
        weighted_seq = np.multiply(self.one_hot_sequence, weights)
        weighted_seq_shuffles = np.stack([weighted_seq]*7, axis=2)

        with self.test_session():
            # arrange: need class instance, features, shuffled features
            extractor = InputxGrad(None)
            inputs = {
                DataKeys.FEATURES: tf.convert_to_tensor(weighted_seq, dtype=tf.float32),
                DataKeys.WEIGHTED_SEQ_SHUF: tf.convert_to_tensor(weighted_seq_shuffles, dtype=tf.float32)}
            params = {"to_threshold": [DataKeys.FEATURES, DataKeys.WEIGHTED_SEQ_SHUF]}

            # act: run preprocess fn and eval
            outputs, _ = extractor.threshold(inputs, params)
            results = self.evaluate(outputs)

        # assert: min score is 991 (for default pval of 0.01)
        nonzero_scores = results[DataKeys.FEATURES][np.where(results[DataKeys.FEATURES] != 0)]
        assert np.min(nonzero_scores) == 991, "correct threshold not found"

        # assert: threshold is saved out correctly
        assert np.all(np.equal(results[DataKeys.WEIGHTED_SEQ_THRESHOLDS], 990)), "threshold not saved out"

        # assert: shuffles were also thresholded
        nonzero_scores = results[DataKeys.FEATURES][np.where(results[DataKeys.FEATURES] != 0)]
        assert np.min(nonzero_scores) == 991, "shuffles not thresholded"

        # TODO - make sure to remove active shuffles?


    # TODO: test thresholding for ensemble, when thresholds are already defined


    def test_denoise(self):
        """test denoising
        """
        # set up features to test denoising
        zero_features = np.zeros_like(self.one_hot_sequence) # {1, 1, 1000, 4}
        test_features = []
        desired_results = []

        # 1) success - 2 positives close enough
        features = np.copy(zero_features)
        features[0,0,420,0] = 1
        features[0,0,423,0] = 1
        test_features.append(features)
        desired_results.append(features)
        
        # 2) failure - 2 positives NOT close enough
        features = np.copy(zero_features)
        features[0,0,420,0] = 1
        features[0,0,424,0] = 1
        test_features.append(features)
        desired_results.append(zero_features)

        # 3) success - 2 negatives close enough
        features = np.copy(zero_features)
        features[0,0,420,0] = -1
        features[0,0,423,0] = -1
        test_features.append(features)
        desired_results.append(features)
        
        # 4) failure - 2 negatives NOT close enough
        features = np.copy(zero_features)
        features[0,0,420,0] = -1
        features[0,0,424,0] = -1
        test_features.append(features)
        desired_results.append(zero_features)
        
        # 5) failure - neg and pos close enough
        features = np.copy(zero_features)
        features[0,0,420,0] = 1
        features[0,0,423,0] = -1
        test_features.append(features)
        desired_results.append(zero_features)

        # concatenate
        features = np.concatenate(test_features, axis=0)
        desired_results = np.concatenate(desired_results, axis=0)
        fake_aux_features = np.expand_dims(features, axis=1)
        
        with self.test_session():
            # arrange: need class instance, features, shuffled features
            extractor = InputxGrad(None)
            inputs = {
                DataKeys.FEATURES: tf.convert_to_tensor(features, dtype=tf.float32),
                DataKeys.WEIGHTED_SEQ_SHUF: tf.convert_to_tensor(fake_aux_features, dtype=tf.float32)}
            params = {"to_denoise_aux": [DataKeys.WEIGHTED_SEQ_SHUF]}

            # act: run preprocess fn and eval
            outputs, _ = extractor.denoise(inputs, params)
            results = self.evaluate(outputs)

        # assert: all denoising happened correctly for features
        matches_desired_results = np.equal(results[DataKeys.FEATURES], desired_results)
        matches_desired_results = np.all(matches_desired_results, axis=(1,2,3))
        assert np.all(matches_desired_results), matches_desired_results

        # assert: denoising correctly done for aux
        matches_desired_results = np.equal(
            np.squeeze(results[DataKeys.WEIGHTED_SEQ_SHUF], axis=1),
            desired_results)
        matches_desired_results = np.all(matches_desired_results, axis=(1,2,3))
        assert np.all(matches_desired_results), matches_desired_results


@pytest.mark.usefixtures("one_hot_sequence")
class DeltaFeatureImportanceMapperTests(tf.test.TestCase):
    
    def test_attach_mutations(self):
        """test attaching mutated sequence
        """
        sequence_content = np.sum(self.one_hot_sequence, axis=(0,1,2))
        num_shuffles = 7

        # make fake sequence to attach
        fake_mutants = np.stack([self.one_hot_sequence]*num_shuffles, axis=1) # {N, shuf, 1, seqlen, 4}
        
        with self.test_session():
            # arrange: need class instance, features tensor, no params (minimal inputs)
            extractor = DeltaFeatureImportanceMapper(None)
            inputs = {
                DataKeys.FEATURES: tf.convert_to_tensor(self.one_hot_sequence),
                DataKeys.MUT_MOTIF_ORIG_SEQ: tf.convert_to_tensor(fake_mutants)}
            params = {}

            # act: run preprocess fn and eval
            outputs, params = extractor.preprocess(inputs, params)
            results = self.evaluate(outputs)
            
        # assert: features properly attached
        final_feature_shape = list(self.one_hot_sequence.shape)
        final_feature_shape[0] += num_shuffles
        assert list(results[DataKeys.FEATURES].shape) == final_feature_shape, "did not attach properly"

        # assert: params contain the relevant details
        assert params["batch_size"] == self.one_hot_sequence.shape[0]

        # TODO need to do tests where shuffles are interleaved

    
    # elsewhere, test attach/detach aux tensors
    # elsewhere, test normalization
    # elsewhere, check confidence intervals
        
if __name__ == '__main__':
    tf.test.main()

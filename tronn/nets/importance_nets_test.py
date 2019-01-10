"""description: tests for importance nets
"""
import pytest

import numpy as np
import tensorflow as tf

from tronn.nets.importance_nets import InputxGrad
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


    # TODO: test denoising....
        
        
if __name__ == '__main__':
    tf.test.main()

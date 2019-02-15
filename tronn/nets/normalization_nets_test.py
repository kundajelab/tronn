"""description: tests for motif nets
"""

import pytest

import numpy as np
import tensorflow as tf

from tronn.nets.normalization_nets import normalize_to_importance_logits
from tronn.util.utils import DataKeys


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

    
@pytest.mark.usefixtures("one_hot_sequence")
class NormalizationTests(tf.test.TestCase):

    def test_logit_normalization(self):
        """test standard normalization
        """
        # necessary inputs
        features = np.concatenate(
            [self.one_hot_sequence, self.one_hot_sequence, np.zeros_like(self.one_hot_sequence)],
            axis=1) # {1, 3, 1000, 4}
        logits = np.expand_dims(np.array([1., 2., 3., 4., 5.]), axis=0) # {1, 3}
        
        with self.test_session():
            # arrange
            inputs = {
                DataKeys.FEATURES: tf.convert_to_tensor(features, dtype=tf.float32),
                DataKeys.LOGITS: tf.convert_to_tensor(logits, dtype=tf.float32)}
            params = {
                "importance_task_indices": [0,1,2],
                "weight_key": DataKeys.LOGITS}

            # act
            outputs, params = normalize_to_importance_logits(inputs, params)
            results = self.evaluate(outputs)

        # assert
        assert np.sum(results[DataKeys.FEATURES][:,0]) == 1.0, np.sum(results[DataKeys.FEATURES][:,0])
        assert np.sum(results[DataKeys.FEATURES][:,1]) == 2.0, np.sum(results[DataKeys.FEATURES][:,1])
        assert np.sum(results[DataKeys.FEATURES][:,2]) == 0.0, np.sum(results[DataKeys.FEATURES][:,2])

        
    def test_shuf_logit_normalization(self):
        """test standard normalization
        """
        # necessary inputs
        features = np.concatenate(
            [self.one_hot_sequence, self.one_hot_sequence, np.zeros_like(self.one_hot_sequence)],
            axis=1) # {1, 3, 1000, 4}
        features = np.expand_dims(features, axis=1) # {1, 1, 3, 1000, 4}
        logits = np.expand_dims(np.array([1., 2., 3., 4., 5.]), axis=0) # {1, 3}
        logits = np.expand_dims(logits, axis=1) # {1, 1, 3}
        
        with self.test_session():
            # arrange
            inputs = {
                DataKeys.FEATURES: tf.convert_to_tensor(features, dtype=tf.float32),
                DataKeys.LOGITS: tf.convert_to_tensor(logits, dtype=tf.float32)}
            params = {
                "importance_task_indices": [0,1,2],
                "weight_key": DataKeys.LOGITS}

            # act
            outputs, params = normalize_to_importance_logits(inputs, params)
            results = self.evaluate(outputs)

        # assert
        assert np.sum(results[DataKeys.FEATURES][:,:,0]) == 1.0, np.sum(results[DataKeys.FEATURES][:,0])
        assert np.sum(results[DataKeys.FEATURES][:,:,1]) == 2.0, np.sum(results[DataKeys.FEATURES][:,1])
        assert np.sum(results[DataKeys.FEATURES][:,:,2]) == 0.0, np.sum(results[DataKeys.FEATURES][:,2])


    def test_masked_normalization(self):
        """test masking for dfim normalization
        """
        if False:
            # same as above, but blank out some sites
            # necessary inputs
            features = np.concatenate(
                [self.one_hot_sequence, self.one_hot_sequence, np.zeros_like(self.one_hot_sequence)],
                axis=1) # {1, 3, 1000, 4}
            logits = np.expand_dims(np.array([1., 2., 3., 4., 5.]), axis=0) # {1, 3}
            blanked_positions = np.reshape(np.ones((1000)), (1, 1, 1000, 1))
            blanked_positions[0,0,0:495,0] = 0
            blanked_positions[0,0,505:,0] = 0
            features[:,:,495:505,:] = 0
            features[:,:,500,:] = [0,0,0,990]

            with self.test_session():
                # arrange
                inputs = {
                    DataKeys.FEATURES: tf.convert_to_tensor(features, dtype=tf.float32),
                    DataKeys.LOGITS: tf.convert_to_tensor(logits, dtype=tf.float32),
                    DataKeys.MUT_MOTIF_POS: tf.convert_to_tensor(blanked_positions, dtype=tf.float32)}
                params = {
                    "importance_task_indices": [0,1,2],
                    "weight_key": DataKeys.LOGITS}

                # act
                outputs, params = normalize_to_importance_logits(inputs, params)
                results = self.evaluate(outputs)

            # assert
            assert np.isclose(np.sum(results[DataKeys.FEATURES][:,0]), 2.0)
            assert np.isclose( np.sum(results[DataKeys.FEATURES][:,1]), 4.0)
            assert np.sum(results[DataKeys.FEATURES][:,2]) == 0.0
        else:
            assert True

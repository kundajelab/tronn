
import pytest

import numpy as np
import tensorflow as tf

from tronn.nets.mutate_nets import Mutagenizer
from tronn.util.utils import DataKeys

# TODO - lots of tests to make sure the mutations are made correctly

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
class MutagenizerTests(tf.test.TestCase):


    def test_preprocess_for_shuffle(self):
        """make sure preprocess correctly pulls out values
        """
        # necessary inputs
        indices = np.zeros((1,12,1))
        vals = np.ones((1,12,1))
        sig_pwms = np.zeros((12))
        sig_pwms[[1,3,5]] = 1
        
        with self.test_session():
            # arrange
            mutagenizer = Mutagenizer(mutation_type="shuffle")
            inputs = {
                DataKeys.FEATURES: tf.convert_to_tensor(self.one_hot_sequence),
                DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX: tf.convert_to_tensor(indices),
                DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL: tf.convert_to_tensor(vals)}
            params = {"sig_pwms": sig_pwms}

            # act: run preprocess fn and eval
            outputs, params = mutagenizer.preprocess(inputs, params)
            results = self.evaluate(outputs)

        #for key in sorted(results.keys()):
        #    print key, results[key].shape
        #print params

        # assert: create new val/idx tensors with correct shapes
        assert results[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL_MUT].shape == (1,3,1)
        assert results[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT].shape == (1,3,1)

        
    def test_preprocess_for_point_mut(self):
        """make sure preprocess correctly pulls out values
        """
        # necessary inputs
        indices = np.zeros((1,12,1))
        vals = np.ones((1,12,1))
        sig_pwms = np.zeros((12))
        sig_pwms[[1,3,5]] = 1
        gradients = -self.one_hot_sequence
        
        with self.test_session():
            # arrange
            mutagenizer = Mutagenizer(mutation_type="point")
            inputs = {
                DataKeys.FEATURES: tf.convert_to_tensor(self.one_hot_sequence),
                DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX: tf.convert_to_tensor(indices),
                DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL: tf.convert_to_tensor(vals),
                DataKeys.IMPORTANCE_GRADIENTS: tf.convert_to_tensor(gradients)}
            params = {"sig_pwms": sig_pwms}

            # act: run preprocess fn and eval
            outputs, params = mutagenizer.preprocess(inputs, params)
            results = self.evaluate(outputs)

        for key in sorted(results.keys()):
            print key, results[key].shape
        print params
        
        # assert: create new val/idx tensors with correct shapes
        assert results[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_VAL_MUT].shape == (1,3,1)
        assert results[DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT].shape == (1,3,1)

        # assert: grad_mins is the min gradient value
        assert np.all(np.equal(results["grad_mins"], self.one_hot_sequence))


    def test_shuffle_mutagenize_single(self):
        """test shuffle mutagenizer
        """
        example = np.squeeze(self.one_hot_sequence, axis=1)
        print example.shape
        motif_present = True
        positions = np.sum(np.zeros_like(example), axis=-1, keepdims=True)
        positions[0,500,0] = 1

        with self.test_session():
            # arrange
            mutagenizer = Mutagenizer()
            tensors = [
                tf.convert_to_tensor(example, dtype=tf.float32),
                tf.convert_to_tensor(motif_present),
                tf.convert_to_tensor(positions, dtype=tf.float32)]

            # act: run preprocess fn and eval
            outputs = mutagenizer.shuffle_mutagenize_example(tensors)
            results = self.evaluate(outputs)

        mut_example = results[0]
        adjusted_positions = results[1]

        # assert: only the shuffled part was changed
        left_unchanged = np.all(np.equal(example[:,:495], mut_example[:,:495]))
        assert left_unchanged
        right_unchanged = np.all(np.equal(example[:,505:], mut_example[:,505:]))
        assert right_unchanged
        shuffle_changed = np.any(np.not_equal(example[:,495:505], mut_example[:,495:505]))
        assert shuffle_changed, "did not change sequence. is random, if fails try run tests again"

        
    def test_point_mutagenize_single(self):
        """test shuffle mutagenizer
        """
        example = np.squeeze(self.one_hot_sequence, axis=1)
        print example.shape
        motif_present = True
        positions = np.sum(np.zeros_like(example), axis=-1, keepdims=True)
        positions[0,500,0] = 1
        gradients = np.copy(example)
        gradients[0,500,3] = 1
        
        with self.test_session():
            # arrange
            mutagenizer = Mutagenizer()
            tensors = [
                tf.convert_to_tensor(example, dtype=tf.float32),
                tf.convert_to_tensor(motif_present),
                tf.convert_to_tensor(positions, dtype=tf.float32),
                tf.convert_to_tensor(gradients, dtype=tf.float32)]

            # act: run preprocess fn and eval
            outputs = mutagenizer.point_mutagenize_example(tensors)
            results = self.evaluate(outputs)

        mut_example = results[0]
        adjusted_positions = results[1]

        # assert: only the shuffled part was changed
        left_unchanged = np.all(np.equal(example[:,:500], mut_example[:,:500]))
        assert left_unchanged
        right_unchanged = np.all(np.equal(example[:,501:], mut_example[:,501:]))
        assert right_unchanged
        shuffle_changed = np.any(np.equal(example[:,500], gradients[:,500]))
        assert shuffle_changed, "did not change sequence. is random, if fails try run tests again"


    def test_mutagenize_multiple(self):
        """make sure properly mutagenizes multiple sites
        """
        # necessary inputs
        indices = np.zeros((1,3,1), dtype=np.int64)
        indices[0,0,0] = 450
        indices[0,1,0] = 475
        indices[0,2,0] = 500

        mut_motif_present = np.expand_dims(np.array([True, True, True]), axis=0)
        
        #vals = np.ones((1,12,1))
        #sig_pwms = np.zeros((12))
        #sig_pwms[[1,3,5]] = 1
        gradients = -self.one_hot_sequence
        
        with self.test_session():
            # arrange
            mutagenizer = Mutagenizer(mutation_type="shuffle")
            inputs = {
                DataKeys.ORIG_SEQ: tf.convert_to_tensor(self.one_hot_sequence),
                DataKeys.WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT: tf.convert_to_tensor(indices),
                DataKeys.MUT_MOTIF_PRESENT: tf.convert_to_tensor(mut_motif_present),
                DataKeys.IMPORTANCE_GRADIENTS: tf.convert_to_tensor(gradients)}
            params = {}
            
            # act: run preprocess fn and eval
            outputs, params = mutagenizer.mutagenize_multiple_motifs(inputs, params)
            results = self.evaluate(outputs)

        for key in sorted(results.keys()):
            print key, results[key].shape
        print params

        # assert

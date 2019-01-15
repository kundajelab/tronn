

import pytest

import numpy as np
import tensorflow as tf

from tronn.nets.motif_nets import filter_for_any_sig_pwms
from tronn.util.utils import DataKeys

@pytest.fixture(scope="class")
def pwm_hits(request):
    """create a motif hits vector {1, 1, 1000, 10}
    """
    hits = np.zeros((1,1,976,10))
    request.cls.pwm_hits = hits

    
@pytest.mark.usefixtures("pwm_hits")
class FilterMotifsTests(tf.test.TestCase):

    def test_filter_sig_pwms(self):
        """test filtering
        """
        # make a batch vector
        pwm_hits_shape = list(self.pwm_hits.shape)
        pwm_hits_shape[0] = 8
        pwm_hits = np.zeros(pwm_hits_shape)
        id_vector = np.arange(pwm_hits_shape[0])

        # make sig pwms vector
        sig_pwms = np.zeros((pwm_hits_shape[3]))
        sig_pwms[1] = 1
        sig_pwms[5] = 1
        
        # 1) has hit in sig
        pwm_hits[0,0,500,1] = 1
        pwm_hits[2,0,500,5] = 1
        pwm_hits[4,0,500,1] = -1
        pwm_hits[6,0,500,5] = -1

        # 2) has hit but not matching sig
        pwm_hits[1,0,500,0] = 1
        pwm_hits[3,0,500,2] = 1
        pwm_hits[5,0,500,3] = 1

        # desired results
        desired_indices = np.array([0,2,4,6])
        
        with self.test_session():
            # arrange: need class instance, features tensor, no params (minimal inputs)
            inputs = {
                DataKeys.ORIG_SEQ_PWM_HITS: tf.convert_to_tensor(pwm_hits),
                "indices": id_vector}
            params = {"sig_pwms": sig_pwms, "use_queue": False}

            # act: run preprocess fn and eval
            outputs, _ = filter_for_any_sig_pwms(inputs, params)
            results = self.evaluate(outputs)
        
        # assert: filtered properly
        for key in results.keys():
            assert results[key].shape[0] == 4
        assert np.all(np.equal(results["indices"], desired_indices))
    

class MotifScannerTests(tf.test.TestCase):

    def test_pwm_initializing(self):
        """make sure pwms get loaded correctly
        """
        assert True


    


# check twotailed pwm scoring works as expected


# shuffles: check reshaping


# check thresholding with shuffles


# check dmim


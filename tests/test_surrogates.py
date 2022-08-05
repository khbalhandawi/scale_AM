import pytest

from src.surrogateLib import KSModel
from src.utilities import check_folder
import numpy as np
import os
import matplotlib

from typing import Tuple, List

matplotlib.use('Agg')

@pytest.fixture
def training_data() -> Tuple[np.ndarray,np.ndarray]:
    x = np.array([
        [0.8699,0.5439,0.3658],
        [0.2648,0.7210,0.7635],
        [0.3181,0.5225,0.6279],
        [0.1192,0.9937,0.7720],
        [0.9398,0.2187,0.9329],
        [0.6456,0.1058,0.9727],
        [0.4795,0.1097,0.1920],
        [0.6393,0.0636,0.1389],
        [0.5447,0.4046,0.6963],
        [0.6473,0.4484,0.0938],
     ])
        
    y = np.array([
        [0.5254,0.5861],
        [0.5303,0.2621],
        [0.8611,0.0445],
        [0.4849,0.7549],
        [0.3935,0.2428],
        [0.6714,0.4424],
        [0.7413,0.6878],
        [0.5201,0.3592],
        [0.3477,0.7363],
        [0.1500,0.3947],
    ])

    return x,y

@pytest.fixture
def test_sites() -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    z = np.array([
        [0.6834,0.4243,0.8878],
        [0.7040,0.2703,0.3912],
        [0.4423,0.1971,0.7691],
        [0.0196,0.8217,0.3968],
        [0.3309,0.4299,0.8085],
    ])

    f_test = np.array([
        [0.4921, 0.4739],
        [0.4945, 0.5012],
        [0.5454, 0.4784],
        [0.5761, 0.4217],
        [0.6025, 0.3561],
    ])

    grad_f_test = np.array([
        [
            [-0.1069,-0.0665, 0.1831],
            [-0.3204, 0.0888,-0.4116],
        ],
        [
            [-0.1610,-0.2811, 0.0795],
            [-0.0125, 0.0352, 0.1159],
        ],
        [
            [-0.2242,-0.2073, 0.1216],
            [ 0.2896, 0.0924,-0.1221],
        ],
        [
            [ 0.0547,-0.2563,-0.0320],
            [-0.2415, 0.6204, 0.1599],
        ],
        [
            [-0.4313, 0.1937,-0.2732],
            [ 0.6713,-0.4640, 0.2957],
        ],
    ])

    return z, f_test, grad_f_test

@pytest.mark.dependency()
def test_against_MATLAB(training_data: Tuple[np.ndarray,np.ndarray],
    test_sites: Tuple[np.ndarray,np.ndarray,np.ndarray]):
    """
    Tests the KS model against known outputs from a MATLAB code

    Parameters
    ----------
    training_data : Tuple[np.ndarray,np.ndarray]
        training data
    test_sites : Tuple[np.ndarray,np.ndarray,np.ndarray]
        test data obtained from MATLAB
    """
    x,y = training_data
    z,f_test,grad_f_test = test_sites

    model = KSModel('test')
    model.train(x,y,5e-1,lb=np.zeros(x.shape[1]),ub=np.ones(x.shape[1]))
    f,grad_f = model.predict(z)

    test_dir = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(test_dir,'..','test_dumps')
    check_folder(folder)
    model.save(os.path.join(folder,'test_model'))

    grad_f_test = np.swapaxes(np.swapaxes(grad_f_test,0,2),0,1)

    assert np.allclose(f, f_test, rtol=1e-1)
    assert np.allclose(grad_f, grad_f_test, rtol=1e-1)

@pytest.mark.dependency(depends=["test_against_MATLAB",])
def test_error_metrics(test_sites: Tuple[np.ndarray,np.ndarray,np.ndarray]):
    """
    Tests the R2 and MSE error metrics

    Parameters
    ----------
    test_sites : Tuple[np.ndarray,np.ndarray,np.ndarray]
        test data obtained from MATLAB
    """
    z,f_test,grad_f_test = test_sites

    model = KSModel('test')

    test_dir = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(test_dir,'..','test_dumps')
    check_folder(folder)
    model.load(os.path.join(folder,'test_model'))

    e = model.error(z,f_test)
    r2 = model.R2(z,f_test)
    
    # since KS is an interpolating model
    assert np.isclose(e,0,rtol=1e-3)
    assert np.isclose(r2,1,rtol=1e-3)

@pytest.mark.dependency(depends=["test_against_MATLAB",])
def test_visualization():
    model = KSModel('test')

    test_dir = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(test_dir,'..','test_dumps')
    check_folder(folder)
    model.load(os.path.join(folder,'test_model'))

    model.view([0,1],0)
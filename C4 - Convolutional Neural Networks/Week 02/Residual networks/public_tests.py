from termcolor import colored
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
import numpy as np

def identity_block_test(target):
    tf.random.set_seed(2)
    np.random.seed(1)
    
    #X = np.random.randn(3, 4, 4, 6).astype(np.float32)
    X1 = np.ones((1, 4, 4, 3)) * -1
    X2 = np.ones((1, 4, 4, 3)) * 1
    X3 = np.ones((1, 4, 4, 3)) * 3

    X = np.concatenate((X1, X2, X3), axis = 0).astype(np.float32)

    A3 = target(X,
                f = 2,
                filters = [4, 4, 3],
                initializer=lambda seed=0:constant(value=1),
                training=False)


    A3np = A3.numpy()
    assert tuple(A3np.shape) == (3, 4, 4, 3), "Shapes does not match. This is really weird"
    assert np.all(A3np >= 0), "The ReLu activation at the last layer is missing"
    resume = A3np[:,(0,-1),:,:].mean(axis = 3)

    assert np.floor(resume[1, 0, 0]) == 2 * np.floor(resume[1, 0, 3]), "Check the padding and strides"
    assert np.floor(resume[1, 0, 3]) == np.floor(resume[1, 1, 0]),     "Check the padding and strides"
    assert np.floor(resume[1, 1, 0]) == 2 * np.floor(resume[1, 1, 3]), "Check the padding and strides"
    assert np.floor(resume[1, 1, 0]) == 2 * np.floor(resume[1, 1, 3]), "Check the padding and strides"

    assert resume[1, 1, 0] - np.floor(resume[1, 1, 0]) > 0.7, "Looks like the BatchNormalization units are not working"

    assert np.allclose(resume, 
                       np.array([[[  0.,        0.,        0.,        0.,     ],
                                  [  0.,        0.,        0.,        0.,     ]],
                                 [[192.6542,  192.6542,  192.6542,   96.8271 ],
                                  [ 96.8271,   96.8271,   96.8271,   48.91355]],
                                 [[578.21246, 578.21246, 578.21246, 290.60623],
                                  [290.60623, 290.60623, 290.60623, 146.80312]]]), atol = 1e-5 ), "Wrong values with training=False"
    
    np.random.seed(1)
    tf.random.set_seed(2)
    A4 = target(X,
                f = 3,
                filters = [3, 3, 3],
                initializer=lambda seed=7:constant(value=1),
                training=True)
    A4np = A4.numpy()
    resume = A4np[:,(0,-1),:,:].mean(axis = 3)
    assert np.allclose(resume, 
                         np.array([[[0.,        0.,        0.,        0.       ],
                                    [0.,        0.,        0.,        0.       ]],
                                   [[0.373974,  0.373974,  0.373974,  0.373974 ],
                                    [0.373974,  0.373974,  0.373974,  0.373974 ]],
                                   [[3.2379792, 4.139072,  4.139072,  3.2379792],
                                    [3.2379792, 4.139072,  4.139072,  3.2379792]]]), atol = 1e-5 ), "Wrong values with training=True"

    print(colored("All tests passed!", "green"))

    
def convolutional_block_test(target):
    np.random.seed(1)
    tf.random.set_seed(2)
    
    convolutional_block_output1 = [[[[0.,         0.,         0.6442667,  0.,         0.13945118, 0.78498244],
                                     [0.01695363, 0.,         0.7052939,  0.,         0.27986753, 0.67453355]],
                                    [[0.,         0.,         0.6702033,  0. ,        0.18277727, 0.7506114 ],
                                     [0.,         0.,         0.68768525, 0. ,        0.25753927, 0.6794529 ]]],
                                   [[[0.,         0.7772112,  0.,        1.4986887,  0.,         0.        ],
                                     [0.,         1.0264266,  0.,        1.274425,   0.,         0.        ]],
                                    [[0.,         1.0375856,  0.,        1.6640364,  0.,         0.        ],
                                     [0.,         1.0398285,  0.,        1.3485202,  0.,         0.        ]]],
                                   [[[0.,         2.3315008,  0.,        4.4961185,  0.,         0.        ],
                                     [0.,         3.0792732,  0.,        3.8233364,  0.,         0.        ]],
                                    [[0.,         3.1125813,  0.,        4.9924607,  0.,         0.        ],
                                     [0.,         3.1193442,  0.,        4.0456157,  0.,         0.        ]]]]
    
    convolutional_block_output2 = [[[[0.0000000e+00, 2.4476275e+00, 1.8830043e+00, 2.1259236e-01, 1.9220030e+00, 0.0000000e+00],
                                     [0.0000000e+00, 2.1546977e+00, 1.6514317e+00, 0.0000000e+00, 1.7889941e+00, 0.0000000e+00]],
                                    [[0.0000000e+00, 1.8540058e+00, 1.3404746e+00, 0.0000000e+00, 1.0688392e+00, 0.0000000e+00],
                                     [0.0000000e+00, 1.6571904e+00, 1.1809819e+00, 0.0000000e+00, 9.4837922e-01, 0.0000000e+00]]],
                                   [[[0.0000000e+00, 5.0503787e-02, 0.0000000e+00, 2.9122047e-03, 8.7130928e-01, 1.0279868e+00],
                                     [0.0000000e+00, 5.0503787e-02, 0.0000000e+00, 2.9122047e-03, 8.7130928e-01, 1.0279868e+00]],
                                    [[0.0000000e+00, 5.0503787e-02, 0.0000000e+00, 2.9122047e-03, 8.7130928e-01, 1.0279868e+00],
                                     [0.0000000e+00, 5.0503787e-02, 0.0000000e+00, 2.9122047e-03, 8.7130928e-01, 1.0279868e+00]]],
                                   [[[1.9959736e+00, 0.0000000e+00, 0.0000000e+00, 2.4793634e+00, 0.0000000e+00, 2.9498351e-01],
                                     [1.4637939e+00, 0.0000000e+00, 0.0000000e+00, 1.3023224e+00, 0.0000000e+00, 1.5583299e+00]],
                                    [[3.1462767e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.1307199e+00],
                                     [1.8378723e+00, 0.0000000e+00, 0.0000000e+00, 1.5683722e-01, 0.0000000e+00, 2.3509054e+00]]]]
    
    #X = np.random.randn(3, 4, 4, 6).astype(np.float32)
    X1 = np.ones((1, 4, 4, 3)) * -1
    X2 = np.ones((1, 4, 4, 3)) * 1
    X3 = np.ones((1, 4, 4, 3)) * 3

    X = np.concatenate((X1, X2, X3), axis = 0).astype(np.float32)

    A = target(X, f = 2, filters = [2, 4, 6], training=False)

    assert type(A) == EagerTensor, "Use only tensorflow and keras functions"
    assert tuple(tf.shape(A).numpy()) == (3, 2, 2, 6), "Wrong shape."
    assert np.allclose(A.numpy(), convolutional_block_output1), "Wrong values when training=False."
    print(A[0])

    B = target(X, f = 2, filters = [2, 4, 6], training=True)
    assert np.allclose(B.numpy(), convolutional_block_output2), "Wrong values when training=True."

    print('\033[92mAll tests passed!')
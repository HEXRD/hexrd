import pickle

import numpy as np
import sys
sys.path.append('..')
from hexrd.extensions import inverse_distortion

RHO_MAX = 204.8
params = [-2.277777438488093e-05, -8.763805995117837e-05, -0.00047451698761967085]

big_test_in = np.array([[ 47.031483 ,  -5.2170362],
                        [ 60.0171   ,  27.218563 ],
                        [ 60.697784 ,  25.48354  ],
                        [ 56.90082  , -35.88738  ],
                        [ 55.631718 , -37.62758  ],
                        [ 41.258152 , -63.237328 ],
                        [ 78.00906  ,  -8.576369 ],
                        [ 77.7207   , -10.315189 ],
                        [ 73.20562  ,  32.426453 ],
                        [ 92.81449  , -13.717096 ],
                        [101.26153  ,  18.790167 ],
                        [ 33.428936 ,  99.08392  ],
                        [101.51741  ,  17.049505 ],
                        [ 85.12195  ,  59.84272  ],
                        [-44.172375 ,  11.69572  ],
                        [-57.850574 , -19.013659 ],
                        [-57.169926 , -20.749386 ],
                        [-52.772987 ,  44.117252 ],
                        [-54.043003 ,  42.37685  ],
                        [-74.88209  ,  16.800816 ],
                        [-70.36986  , -25.959877 ],
                        [-38.385185 ,  69.727234 ],
                        [-75.17097  ,  15.061297 ],
                        [-89.98957  ,  20.207851 ],
                        [-98.708176 , -10.574363 ],
                        [-82.30401  , -53.390797 ],
                        [-98.45244  , -12.316093 ]], dtype=np.float32)

big_test_out = np.array([[ 47.03288205,  -5.21719147],
                         [ 60.02002476,  27.21988891],
                         [ 60.70080301,  25.48480692],
                         [ 56.90341105, -35.88901181],
                         [ 55.63418288, -37.62924612],
                         [ 41.26039938, -63.24077218],
                         [ 78.01561783,  -8.57708985],
                         [ 77.72717551, -10.31604838],
                         [ 73.21095917,  32.42881771],
                         [ 92.82553762, -13.71872887],
                         [101.27584945,  18.79282435],
                         [ 33.43310185,  99.09627097],
                         [101.5318596 ,  17.05193195],
                         [ 85.13101706,  59.84909563],
                         [-44.17351296,  11.69602109],
                         [-57.85318107, -19.01451522],
                         [-57.17243759, -20.75029751],
                         [-52.77530392,  44.11918895],
                         [-54.04540365,  42.37873249],
                         [-74.88783308,  16.80210463],
                         [-70.37458323, -25.96162026],
                         [-38.38762237,  69.73166096],
                         [-75.17679039,  15.06246417],
                         [-89.99957766,  20.21009857],
                         [-98.72150605, -10.57579081],
                         [-82.31199676, -53.39597868],
                         [-98.46565139, -12.31774635]])

def test_known_values():
  xy_in = np.array([[140.40087891, 117.74253845]])
  expected_output = np.array([[140.44540352, 117.77987754]])
  xy_out = inverse_distortion.ge_41rt_inverse_distortion(xy_in, RHO_MAX, params)
  assert np.allclose(xy_out, expected_output)

def test_big_input():
  xy_out = inverse_distortion.ge_41rt_inverse_distortion(big_test_in, RHO_MAX, params)
  assert np.allclose(xy_out, big_test_out)

def test_large_input():
  xy_in = np.array([[1e5, 1e5]])
  xy_out = inverse_distortion.ge_41rt_inverse_distortion(xy_in, RHO_MAX, params)
  # No specific expected output here, just ensure it doesn't fail
  assert xy_out.shape == xy_in.shape

def test_logged_data():
    # Load logged data
    with open('data/inverse_distortion_in_out.pkl', 'rb') as f:
        logged_data = pickle.load(f)

    logged_inputs = logged_data['inputs']
    logged_outputs = logged_data['outputs']
    logged_params = logged_data['params']

    for xy_in, xy_out_expected, params in zip(logged_inputs, logged_outputs, logged_params):
        xy_out = inverse_distortion.ge_41rt_inverse_distortion(xy_in, RHO_MAX, params)
        assert np.allclose(xy_out, xy_out_expected, atol=1e-6)

def test_random_values():
  np.random.seed(42)
  xy_in = np.random.rand(10, 2) * 200
  xy_out = inverse_distortion.ge_41rt_inverse_distortion(xy_in, RHO_MAX, params)
  # Verify function does not raise any exception
  assert xy_out.shape == xy_in.shape

from collections import namedtuple
import copy
import json

import numpy as np

from hexrd.transforms.xfcapi import gvec_to_xy

from common import convert_axis_angle_to_rmat


_flds = [
    "tvec_c", "tvec_d", "tvec_s","rmat_c", "rmat_d", "rmat_s",
    "gvec_c", "beam_vec", "xy"
]
GvecXYData = namedtuple("GvecData", _flds)


def test_gvec_to_xy(test_data_dir):
    with open(test_data_dir / 'gvec_to_xy.json') as rf:
        test_data = json.load(rf)

    for entry in test_data:
        kwargs = entry['input']
        expected = entry['output']

        kwargs = {k: np.asarray(v) for k, v in kwargs.items()}
        result = gvec_to_xy(**kwargs)
        assert np.allclose(result, expected)

        # Verify that we get the correct answer with a rotation.
        rot = convert_axis_angle_to_rmat(np.r_[0.5, 0.2, 0.6], 1.0)

        rotated_kwargs = copy.deepcopy(kwargs)
        rotated_kwargs['beam_vec'] = np.r_[0.0, 0.0, -1.0]

        # The following are not rotated:
        # gvec_c are relative to the crystal frame
        # rMat_c is in sample frame
        # tvec_c is relative to sample frame
        to_rotate = ['rmat_d', 'rmat_s', 'tvec_d', 'tvec_s', 'beam_vec']
        for k in to_rotate:
            rotated_kwargs[k] = rot @ rotated_kwargs[k]

        result = gvec_to_xy(**rotated_kwargs)
        assert np.allclose(result, expected)


class TestGvecXY:
    """Test gvec_to_xy and xy_to_gvec"""

    # Base Case: sample and crystal align with lab frame;
    # detector distance = 10; gvec = beam, no diffraction.
    base = GvecXYData(
        tvec_c = np.zeros(3),
        tvec_d = np.array([0., 0., -10]),
        tvec_s = np.zeros(3),
        rmat_c = np.identity(3),
        rmat_d = np.identity(3),
        rmat_s = np.identity(3),
        gvec_c = np.array([0, 0, -1.0]),
        beam_vec = np.array([0, 0, -1.0]),
        xy = np.array((np.nan, np.nan))
    )

    @staticmethod
    def to_array(a):
        return np.array(a, dtype=float)

    @classmethod
    def run_test(cls, prob):
        """Run a test problem"""
        print("prob: ", prob)
        xy_d = gvec_to_xy(
            cls.to_array(prob.gvec_c),
            cls.to_array(prob.rmat_d),
            cls.to_array(prob.rmat_s),
            cls.to_array(prob.rmat_c),
            cls.to_array(prob.tvec_d),
            cls.to_array(prob.tvec_s),
            cls.to_array(prob.tvec_c),
            beam_vec=cls.to_array(prob.beam_vec),
        )
        assert np.allclose(xy_d, prob.xy, equal_nan=True)


    @staticmethod
    def gvec_dvec(theta_deg, eta_deg):
        """Return diffraction vector

        PARAMETERS
        ----------
        theta_deg: float
           diffraction angle theta, in degrees
        eta_degrees: float
           azimuthal angle, in degrees

        RETURNS
        -------
        gvec: array(3):
           diffraction vector relative to beam in -z-direction
        dvec: array(3):
           direction (unit vector) of diffracted beam
        """
        eta = np.radians(eta_deg)
        ce, se = np.cos(eta), np.sin(eta)

        # Diffraction vector makes angle theta with plane normal.
        normal_ang = np.radians(90 - theta_deg)
        cna, sna = np.cos(normal_ang), np.sin(normal_ang)
        gvec = np.array((ce * sna, se * sna, cna))

        # Diffracted beam direction makes angle 2-theta with beam.
        two_theta = np.radians(2 * theta_deg)
        c2t, s2t = np.cos(two_theta), np.sin(two_theta)
        dvec = np.array((ce * s2t, se * s2t, -c2t))

        return gvec, dvec

    @staticmethod
    def make_rmat(angle_deg, axis):
        """Make a rotation matrix

        PARAMETERS
        ----------
        angle: float
           rotation angle in degrees
        axis: array(3)
           axis of rotation (not normalized to unit vector)

        RETURNS
        -------
        array(3, 3):
           rotation matrix
        """
        return convert_axis_angle_to_rmat(
            np.array(axis), np.radians(angle_deg)
        )

    @classmethod
    def test_theta_eta(cls):
        """Vary diffraction angle in simple case

        TEST PARAMETERS
        ---------------
        theta_deg: float
           diffraction angle in degrees
        eta_deg: float, default=0
           azimuthal angle in degrees
        """
        ThetaEta = namedtuple("ThetaEta", ["theta_deg", "eta_deg"])
        nan_tests = [
            ThetaEta(0, 0), ThetaEta(46, 0),
        ]

        tests = [
            ThetaEta(10, 0), ThetaEta(44.9, 0),
            ThetaEta(10, 45), ThetaEta(10, -45),
        ]

        p0_l = (0, 0, 0)
        d0_l = cls.base.tvec_d
        nv_l = (0, 0, 1)
        for t in tests:
            print("test: ", t)
            gvec, dvec = cls.gvec_dvec(t.theta_deg, t.eta_deg)
            det_x = line_plane_intersect(
                p0_l, dvec, d0_l, nv_l
            )
            if t.theta_deg <= 0:
                answer =  np.array((np.nan, np.nan))
            else:
                answer = det_x[:2]

            cls.run_test(cls.base._replace(
                gvec_c=gvec, xy=answer
            ))

    @classmethod
    def test_beam(cls):
        """Vary beam direction

        The beam is rotated by a specified rotation matrix, determined
        from an angle/axis pair.

        TEST PARAMETERS
        ---------------
        aa: 2-tuple
           angle/axis pairs
        """
        TestData = namedtuple("TestData", ["angle", "axis"])
        tests =[
            TestData(90, (1, 0, 0)),
            TestData(90, (0, 1, 0)),
            TestData(45, (1, 0, 0)),
            TestData(45, (0, 0, 1))
        ]

        theta_deg, eta_deg = 5, 0
        gvec, dvec = cls.gvec_dvec(theta_deg, eta_deg)

        p0_l = (0, 0, 0)
        d0_l = cls.base.tvec_d
        nv_d = (0, 0, 1)
        for t in tests:
            print("test: ", t)
            rmat_d = cls.make_rmat(t.angle, t.axis)
            gvec_l = rmat_d @ gvec
            dvec_l = rmat_d @ dvec
            tvec_l = rmat_d @ cls.base.tvec_d
            beam = rmat_d @ cls.base.beam_vec
            det_x = line_plane_intersect(
                p0_l, dvec_l, tvec_l, rmat_d @ nv_d
            )
            x_d = rmat_d.T @ (det_x - tvec_l)

            cls.run_test(cls.base._replace(
                beam_vec=beam, tvec_d=tvec_l, gvec_c=gvec_l,
                rmat_d=rmat_d, xy=x_d[:2],
            ))

    @classmethod
    def test_translations(cls):
        """Vary translation vectors

        TEST PARAMETERS
        ---------------
        td: array(3)/3-tuple
           detector translation in lab frame
        ts:
           sample translation in lab frame
        tc:
           crystal translation in sample frame
        """
        _flds = ["td", "ts", "tc"]
        TvecData = namedtuple("TvecData", _flds)

        tests = [
            TvecData(td=(0, 0, -10), ts=(0, 0, 0), tc=(0, 0, 0)),
            TvecData(td=(0, 0, -10), ts=(0, 0, 7), tc=(0, 0, 0)),
            TvecData(td=(2, 3, -10), ts=(0, 0, 0), tc=(0, 0, -3)),
            TvecData(td=(0, 0, -10), ts=(0, 1, 2), tc=(2, 3, -4)),
        ]

        nv_l = (0, 0, 1)
        theta_deg, eta_deg = 7.5, 0
        gvec, dvec = cls.gvec_dvec(theta_deg, eta_deg)
        for t in tests:
            print("test data:\n", t)
            td = np.array(t.td)
            ts = np.array(t.ts)
            tc = np.array(t.tc)
            p0_l = ts + tc
            det_x = line_plane_intersect(
                p0_l, dvec, td, nv_l
            )
            xy = det_x[:2] - td[:2]

            cls.run_test(cls.base._replace(
                gvec_c=gvec, tvec_d=td, tvec_s=ts, tvec_c=tc,
                xy=xy
            ))


def unit_vector(v):
    return v/np.linalg.norm(v)


def make_unit_vector(theta, phi):
    """Make a unit vector from spherical coordinates

    PARAMETERS
    ----------
    theta: float
       azimuth angle
    phi: float
       angle with north pole

    RETURNS
    -------
    array(3)
       unit vector
    """
    return (
        np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)
    )


def line_plane_intersect(p0, dvec, d0, nvec):
    """Solve line/plane intersection

    PARAMETERS
    ----------
    p0: array/3-tuple of floats
        a point on the line
    dvec: array/3-tuple of floats
        direction (unit) vector of line
    d0: array/3-tuple of floats
        origin of detector
    nvec: array/3-tuple of floats
        normal vector to detector

    RETURNS
    -------
    xyz: 3-tuple
        intersection point

    NOTES
    -----
    From geometry, t = (d0 - p0).n/(d.n), and x = p0 + td
    """
    p0_, d = np.array([p0, dvec])
    t = np.dot((d0 - p0_), nvec)/np.dot(d, nvec)
    x = p0_ + t * d
    return x

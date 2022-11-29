import numpy as np

class sampleRFZ:

    """This class samples the rodrigues fundamental zone
    of a point group uniformly in the density sense and
    returns a list of orientations which are spaced,
    on an average, to the user specified angular spacing
    @author Saransh Singh, LLNL
    @date   11/28/2022 1.0 original

     Note
    ----
    Details can be found in:
    S. Singh and M. De Graef, "Orientation sampling for 
    dictionary-based diffraction pattern indexing methods". 
    MSMSE 24, 085013 (2016)

    Attributes
    ----------
    pgnum: int
        point group number of crystal
    average_angular_spacing" float
        average angular spacing of sampling (in degrees)

    """

    def __init__(self,
                 pgnum,
                 sampling_type='default',
                 average_angular_spacing=3.0):
        """__init__ method of the sampleRFZ class.


        Parameters
        ----------
        pgnum : int
            point group number
        sampling_type : str
            default sampling with origin
            special which is shifted to mid-points
            of each cubochoric cell
        average_angular_spacing : float
            average angular spacing in degrees

        """

        self.pgnum = pgnum
        self.sampling_type = sampling_type
        self.avg_ang_spacing = average_angular_spacing


    def sampling_N(self):
        """Get the number of sampling steps in the cubochoric
        cube based on the average angular spacing requested.
        Uses eqns. 9 and 10 of S. Singh and M. De Graef MSMSE 24,
        085013 (2016)

        """
        if self.sampling_type.lower() == 'default':
            return np.rint(131.97049 / (self.avg_ang_spacing - 0.03732)).astype(np.int32)
        elif self.sampling_type.lower() == 'special':
            return np.rint(125.70471 / (self.avg_ang_spacing - 0.07127)).astype(np.int32)

    @property
    def cubN(self):
        return self.sampling_N()
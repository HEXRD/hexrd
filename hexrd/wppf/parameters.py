import h5py
import numpy as np
import yaml

class Parameters:
    """
    ==================================================================================
    ==================================================================================

    >> @AUTHOR:     Saransh Singh, Lanwrence Livermore National Lab,
                    saransh1@llnl.gov
    >> @DATE:       05/18/2020 SS 1.0 original
    >> @DETAILS:    this is the parameter class which handles all refinement parameters
        for both the Rietveld and the LeBail refimentment problems

        ===============================================================================
        ===============================================================================
    """

    def __init__(self,
                 name=None,
                 vary=False,
                 value=0.0,
                 lb=-np.Inf,
                 ub=np.Inf):

        self.param_dict = {}

        if(name is not None):
            self.add(name=name,
                     vary=vary,
                     value=value,
                     lb=min,
                     ub=max)

    def add(self,
            name,
            vary=False,
            value=0.0,
            lb=-np.Inf,
            ub=np.Inf):
        """
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       05/18/2020 SS 1.0 original
            >> @DETAILS:    add a single named parameter
        """
        self[name] = Parameter(name=name, vary=vary, value=value, lb=lb, ub=ub)

    def add_many(self,
                 names,
                 varies,
                 values,
                 lbs,
                 ubs):
        """
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       05/18/2020 SS 1.0 original
            >> @DETAILS:    load a list of named parameters
        """
        assert len(names) == len(varies), "lengths of tuples not consistent"
        assert len(names) == len(values), "lengths of tuples not consistent"
        assert len(names) == len(lbs), "lengths of tuples not consistent"
        assert len(names) == len(ubs), "lengths of tuples not consistent"

        for i, n in enumerate(names):
            self.add(n, vary=varies[i], value=values[i], lb=lbs[i], ub=ubs[i])

    def load(self, fname):
        """
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       05/18/2020 SS 1.0 original
            >> @DETAILS:    load parameters from yaml file
        """
        with open(fname) as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)

        for k in dic.keys():
            v = dic[k]
            self.add(k, value=np.float(v[0]), lb=np.float(v[1]),
                     ub=np.float(v[2]), vary=np.bool(v[3]))

    def dump(self, fname):
        """
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       05/18/2020 SS 1.0 original
            >> @DETAILS:    dump the class to a yaml looking file. name is the key and the list
                            has [value, lb, ub, vary] in that order
        """
        dic = {}
        for k in self.param_dict.keys():
            dic[k] = [self[k].value, self[k].lb, self[k].ub, self[k].vary]

        with open(fname, 'w') as f:
            data = yaml.dump(dic, f, sort_keys=False)

    def dump_hdf5(self, file):
        """
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       01/15/2021 SS 1.0 original
            >> @DETAILS:    dump the class to a hdf5 file. the file argument could either be a 
                            string or a h5.File instance. If it is a filename, then HDF5 file
                            is created, a parameter group is created and data is written out
                            with data names being the parameter name. Else data written to Parameter
                            group in existing file object
        """
        if(isinstance(file, str)):
            fexist = path.isfile(file)
            if(fexist):
                fid = h5py.File(file, 'r+')
            else:
                fid = h5py.File(file, 'x')

        elif(isinstance(file, h5py.File)):
            fid = file

        else:
            raise RuntimeError(
                'Parameters: dump_hdf5 Pass in a \
                 filename string or h5py.File object')

        if("/Parameters" in fid):
            del(fid["Parameters"])
        gid_top = fid.create_group("Parameters")

        for p in self:
            param = self[p]
            gid = gid_top.create_group(p)

            # write the value, lower and upper bounds and vary status
            did = gid.create_dataset("value", (1, ), dtype=np.float64)
            did.write_direct(np.array(param.value, dtype=np.float64))

            did = gid.create_dataset("lb", (1, ), dtype=np.float64)
            did.write_direct(np.array(param.lb, dtype=np.float64))

            did = gid.create_dataset("ub", (1, ), dtype=np.float64)
            did.write_direct(np.array(param.ub, dtype=np.float64))

            did = gid.create_dataset("vary", (1, ), dtype=np.bool)
            did.write_direct(np.array(param.vary, dtype=np.bool))

    def __getitem__(self, key):
        if(key in self.param_dict.keys()):
            return self.param_dict[key]
        else:
            raise ValueError('variable with name not found')

    def __setitem__(self, key, parm_cls):

        if(key in self.param_dict.keys()):
            warnings.warn(
                'variable already in parameter list. overwriting ...')
        if(isinstance(parm_cls, Parameter)):
            self.param_dict[key] = parm_cls
        else:
            raise ValueError('input not a Parameter class')

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if(self.n < len(self.param_dict.keys())):
            res = list(self.param_dict.keys())[self.n]
            self.n += 1
            return res
        else:
            raise StopIteration

    def __str__(self):
        retstr = 'Parameters{\n'
        for k in self.param_dict.keys():
            retstr += self[k].__str__()+'\n'

        retstr += '}'
        return retstr


class Parameter:
    """
    ===================================================================================
    ===================================================================================

    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       05/18/2020 SS 1.0 original
    >> @DETAILS:    the parameters class (previous one) is a collection of this
                    parameter class indexed by the name of each variable

    ================================================================================
    =================================================================================
    """

    def __init__(self,
                 name=None,
                 vary=False,
                 value=0.0,
                 lb=-np.Inf,
                 ub=np.Inf):

        self.name = name
        self.vary = vary
        self.value = value
        self.lb = lb
        self.ub = ub

    def __str__(self):
        retstr = '< Parameter \''+self.name+'\'; value : ' + \
            str(self.value)+'; bounds : ['+str(self.lb)+',' + \
            str(self.ub)+' ]; vary :'+str(self.vary)+' >'

        return retstr

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if(isinstance(name, str)):
            self._name = name

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, minval):
        self._min = minval

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, maxval):
        self._max = maxval

    @property
    def vary(self):
        return self._vary

    @vary.setter
    def vary(self, vary):
        if(isinstance(vary, bool)):
            self._vary = vary

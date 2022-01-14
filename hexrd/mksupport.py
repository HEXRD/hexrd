from hexrd.symbols import pstr_Elements, sitesym, \
    tworig, PrintPossibleSG, TRIG, pstr_spacegroup,\
    pstr_mkxtal
import h5py
import os
import numpy as np
import datetime
import getpass
from hexrd.unitcell import _StiffnessDict, _pgDict

def mk(filename, xtalname):

    # print some initial information for the user
    print(pstr_mkxtal)

    # get the crystal system. legend is printed above
    xtal_sys, bool_trigonal, bool_hexset = GetXtalSystem()

    lat_param = GetLatticeParameters(xtal_sys, bool_trigonal)

    # get the space group number. legend will be printed above
    space_group, iset = GetSpaceGroup(xtal_sys, bool_trigonal, bool_hexset)

    AtomInfo = GetAtomInfo()
    AtomInfo.update({'file': filename, 'xtalname': xtalname,
                     'xtal_sys': xtal_sys, 'SG': space_group,\
                     'SGsetting': iset})

    Write2H5File(AtomInfo, lat_param)


def GetXtalSystem():

    xtal_sys = input("Crystal System (1-7 use the legend above): ")
    if(not xtal_sys.isdigit()):
        raise ValueError(
            "Invalid value. \
            Please enter valid number between 1-7 using the legend above.")
    else:
        xtal_sys = int(xtal_sys)
        if(xtal_sys < 1 or xtal_sys > 7):
            raise ValueError(
                "Value outside range. \
                Please enter numbers between 1 and 7 using legend above")

    btrigonal = False
    bhexset = False

    if(xtal_sys == 5):
        print(" 1. Hexagonal setting \n 2. Rhombohedral setting")
        hexset = input("(1/2)? :    ")

        if(not hexset.isdigit()):
            raise ValueError("Invalid value.")
        else:
            hexset = int(hexset)
            if(not hexset in [1, 2]):
                raise ValueError(
                    "Invalid value of integer. Only 1 or 2 is acceptable.")

        btrigonal = True

        if(hexset == 1):
            bhexset = True
            xtal_sys = 4
            # only temporarily set to 4 so that the correct
            # lattice parameter can be queried next

        elif(hexset == 2):
            bhexset = False

    return xtal_sys, btrigonal, bhexset


def GetLatticeParameters(xtal_sys, bool_trigonal):

    a = input("a [nm] : ")
    if(not a.replace('.', '', 1).isdigit()):
        raise ValueError("Invalid floating point value.")
    else:
        a = float(a)

    b = a
    c = a
    alpha = 90.0
    beta = 90.0
    gamma = 90.0
    lat_param = {'a': a, 'b': b, 'c': c,
                 'alpha': alpha, 'beta': beta, 'gamma': gamma}

    # cubic symmetry
    if (xtal_sys == 1):
        pass

    # tetragonal symmetry
    elif(xtal_sys == 2):

        c = input("c [nm] : ")
        if(not c.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value.")
        else:
            c = float(c)

        lat_param['c'] = c

    # orthorhombic symmetry
    elif(xtal_sys == 3):
        b = input("b [nm] : ")
        if(not b.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value.")
        else:
            b = float(b)

        c = input("c [nm] : ")
        if(not c.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value.")
        else:
            c = float(c)

        lat_param['b'] = c
        lat_param['c'] = c

    # hexagonal system
    elif(xtal_sys == 4):
        c = input("c [nm] : ")
        if(not c.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value.")
        else:
            c = float(c)

        lat_param['c'] = c
        lat_param['gamma'] = 120.0

        if(bool_trigonal):
            xtal_sys = 5

    # rhombohedral system
    elif(xtal_sys == 5):
        alpha = input("alpha [deg] :    ")
        if(not alpha.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value.")
        else:
            alpha = float(alpha)

        lat_param['alpha'] = alpha
        lat_param['beta'] = alpha
        lat_param['gamma'] = alpha

    # monoclinic system
    elif(xtal_sys == 6):
        b = input("b [nm] : ")
        if(not b.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value.")
        else:
            b = float(b)

        c = input("c [nm] : ")
        if(not c.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value.")
        else:
            c = float(c)

        beta = input("beta [deg] :  ")
        if(not beta.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value.")
        else:
            beta = float(beta)

        lat_param['b'] = b
        lat_param['c'] = c
        lat_param['beta'] = beta

    # triclinic system
    elif(xtal_sys == 7):
        b = input("b [nm] : ")
        if(not b.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value.")
        else:
            b = float(b)

        c = input("c [nm] : ")
        if(not c.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value.")
        else:
            c = float(c)

        alpha = input("alpha [deg] :    ")
        if(not alpha.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value.")
        else:
            alpha = float(alpha)

        beta = input("beta [deg] :  ")
        if(not beta.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value.")
        else:
            beta = float(beta)

        gamma = input("gamma [deg] :    ")
        if(not gamma.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value.")
        else:
            gamma = float(gamma)

        lat_param['b'] = b
        lat_param['c'] = c
        lat_param['alpha'] = alpha
        lat_param['beta'] = beta
        lat_param['gamma'] = gamma

    print("\n")
    return lat_param


def GetSpaceGroup(xtal_sys, btrigonal, bhexset):

    if(btrigonal):
        xtal_sys = 5

    if(btrigonal and (not bhexset)):
        print("\n The space groups below correspond to the ")
        print("second (rhombohedral) setting.")
        print(" Please select one of these space groups.\n")

        for i in range(0, 7):
            pstr = str(TRIG[i]) + ":" + pstr_spacegroup[TRIG[i]]
            if ((i + 1) % 4 == 0 or i == 6):
                print(pstr)
            else:
                print(pstr, end='')

        print(50*"-"+"\n")

    else:
        sgmin, sgmax = PrintPossibleSG(xtal_sys)

    sg = input("Space group number (use legend above): ")
    if(not sg.isdigit()):
        raise ValueError(
            "Invalid value. Please enter valid number between \
            1 and 230 using the legend above.")
    else:
        sg = int(sg)

    if(btrigonal and (not bhexset)):
        if(not sg in TRIG):
            raise ValueError(
                "Invalid space group entered. \
                Please use one of the space groups from the legend above")
        if (sg == 146):
            sg = 231
        if (sg == 148):
            sg = 232
        if (sg == 155):
            sg = 233
        if (sg == 160):
            sg = 234
        if (sg == 161):
            sg = 235
        if (sg == 166):
            sg = 236
        if (sg == 167):
            sg = 237
    else:
        if(sg < sgmin or sg > sgmax):
            raise ValueError(
                "Value outside range. Please enter numbers between \
                 {} and {} using legend above".format(sgmin, sgmax))

    iset = SpaceGroupSetting(sg)

    return sg, iset


def SpaceGroupSetting(sgnum):

    iset = 1
    if(sgnum in tworig):
        idx = tworig.index(sgnum)
        print(' ---------------------------------------------')
        print(' This space group has two origin settings.')
        print(' The first setting has site symmetry    : ' +
              sitesym[2*idx - 2])
        print(' the second setting has site symmetry   : ' +
              sitesym[2*idx - 1])
        iset = input(' Which setting do you wish to use (1/2) : ')
        if(not iset.isdigit()):
            raise ValueError("Invalid integer value for atomic number.")
        else:
            iset = int(iset)
            print(iset)
            if(not iset in [1, 2]):
                raise ValueError(" Value entered for setting must be 1 or 2 !")

    return iset-1


def GetAtomInfo():

    print(pstr_Elements)
    ctr = 0
    Z = []
    APOS = []
    DW = []
    ani = 0
    stiffness = np.zeros([6, 6])

    ques = 'y'
    while(ques.strip().lower() == 'y' or ques.strip().lower() == 'yes'):
        tmp = input("Enter atomic number of species :   ")

        if(not tmp.isdigit()):
            raise ValueError("Invalid integer value for atomic number.")
        else:
            tmp = int(tmp)

        Z.append(tmp)
        asym, U, ani = GetAsymmetricPositions(ani)
        APOS.append(asym)
        DW.append(U)
        ques = input("Another atom? (y/n) : ")

    return {'Z': Z, 'APOS': APOS, 'U': DW, 'stiffness': stiffness}


def GetAsymmetricPositions(aniU):

    asym = input(
        "Enter asymmetric position of atom in unit cell \
         separated by comma (fractional coordinates) :  ")
    asym = [x.strip() for x in asym.split(',')]

    for i, x in enumerate(asym):
        tmp = x.split('/')
        if(len(tmp) == 2):
            if(tmp[1].strip() != '0'):
                asym[i] = str(float(tmp[0])/float(tmp[1]))
            else:
                raise ValueError("Division by zero in fractional coordinates.")
        else:
            pass

    if(len(asym) != 3):
        raise ValueError("Need 3 coordinates in x,y,z fractional coordinates.")

    for i, x in enumerate(asym):
        if(not x.replace('.', '', 1).isdigit()):
            raise ValueError(
                "Invalid floating point value in fractional coordinates.")
        else:
            asym[i] = float(x)
            if(asym[i] < 0.0 or asym[i] >= 1.0):
                raise ValueError(
                    " fractional coordinates only in the \
                    range [0,1) i.e. 1 excluded")

    occ, dw = GetOccDW(aniU=aniU)
    if isinstance(dw, float):
        aniU = 1
    else:
        aniU = 2

    asym.extend([occ])
    return asym, dw, aniU


def GetOccDW(aniU=0):

    occ = input("Enter site occupation :    ")
    if(not occ.replace('.', '', 1).isdigit()):
        raise ValueError(
            "Invalid floating point value in fractional coordinates.")
    else:
        occ = float(occ)
        if(occ > 1.0 or occ <= 0.0):
            raise ValueError(
                "site occupation can only in range (0,1.0] i.e. 0 excluded")

    if(aniU != 0):
        ani = aniU
    else:
        ani = input(
            "Isotropic or anisotropic Debye-Waller factor? \n \
             1 for isotropic, 2 for anisotropic : ")

        if(not ani.isdigit()):
            raise ValueError("Invalid integer value for atomic number.")
        else:
            ani = int(ani)

    if(ani == 1):

        dw = input("Enter isotropic Debye-Waller factor [nm^(-2)] : ")
        if(not dw.replace('.', '', 1).isdigit()):
            raise ValueError(
                "Invalid floating point value in fractional coordinates.")
        else:
            dw = float(dw)

        return [occ, dw]
    elif(ani == 2):

        U = np.zeros([6, ])

        U11 = input("Enter U11 [nm^2] : ")
        if(not U11.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value in U11.")
        else:
            U[0] = float(U11)

        U22 = input("Enter U22 [nm^2] : ")
        if(not U22.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value in U22.")
        else:
            U[1] = float(U22)

        U33 = input("Enter U33 [nm^2] : ")
        if(not U33.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value in U33.")
        else:
            U[2] = float(U33)

        U12 = input("Enter U12 [nm^2] : ")
        if(not U12.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value in U12.")
        else:
            U[3] = float(U12)

        U13 = input("Enter U13 [nm^2] : ")
        if(not U13.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value in U13.")
        else:
            U[4] = float(U13)

        U23 = input("Enter U23 [nm^2] : ")
        if(not U23.replace('.', '', 1).isdigit()):
            raise ValueError("Invalid floating point value in U23.")
        else:
            U[5] = float(U23)

        return [occ, U]

    else:
        raise ValueError("Invalid input. Only 1 or 2 acceptable.")

# write to H5 file


def Write2H5File(AtomInfo, lat_param, path=None):
    # Should we close the file? Only do so if we opened it
    close = False

    if isinstance(AtomInfo['file'], str):
        # first check if file exists
        fexist = os.path.isfile(AtomInfo['file'])

        if(fexist):
            fid = h5py.File(AtomInfo['file'], 'r+')
        else:
            Warning('File doesn''t exist. creating it')
            fid = h5py.File(AtomInfo['file'], 'x')

        close = True
    # Have we been past a h5py.File?
    elif isinstance(AtomInfo['file'], h5py.File):
        fid = AtomInfo['file']
    else:
        raise TypeError('Unexpected file type.')

    WriteH5Data(fid, AtomInfo, lat_param, path)

    if close:
        fid.close()


def WriteH5Data(fid, AtomInfo, lat_param, path=None):
    """
    @DATE 02/09/2021 SS added hkls, exclusions and dmin to
    material.h5 file output
    @Date 01/14/2022 SS added tThWidth to materials file
    """

    # Add the path prefix if we have been given one
    if path is not None:
        path = f"{path}/{AtomInfo['xtalname']}"
    else:
        path = AtomInfo['xtalname']

    node = f'/{path}'
    if node in fid:
        print("crystal already exists. overwriting...\n")
        del fid[node]

    gid = fid.create_group(path)

    did = gid.create_dataset(
        "Atomtypes", (len(AtomInfo['Z']), ), dtype=np.int32)
    did.write_direct(np.array(AtomInfo['Z'], dtype=np.int32))

    did = gid.create_dataset("CrystalSystem", (1,), dtype=np.int32)
    did.write_direct(np.array(AtomInfo['xtal_sys'], dtype=np.int32))

    did = gid.create_dataset("Natomtypes", (1,), dtype=np.int32)
    did.write_direct(np.array([len(AtomInfo['Z'])], dtype=np.int32))

    did = gid.create_dataset("SpaceGroupNumber", (1,), dtype=np.int32)
    did.write_direct(np.array([AtomInfo['SG']], dtype=np.int32))

    did = gid.create_dataset("SpaceGroupSetting", (1,), dtype=np.int32)
    did.write_direct(np.array([AtomInfo['SGsetting']], dtype=np.int32))

    did = gid.create_dataset("LatticeParameters", (6,), dtype=np.float64)
    did.write_direct(np.array(list(lat_param.values()), dtype=np.float64))

    did = gid.create_dataset("stiffness", (6, 6), dtype=np.float64)
    did.write_direct(np.array(AtomInfo['stiffness'], dtype=np.float64))


    if "tThWidth" in AtomInfo: 
        did = gid.create_dataset("tThWidth", (1,), dtype=np.float64)
        did.write_direct(np.array([AtomInfo['tThWidth']], dtype=np.float64))

    if 'hkls' in AtomInfo:
        did = gid.create_dataset("hkls",
                                 AtomInfo['hkls'].shape,
                                 dtype=np.int32)
        did.write_direct(AtomInfo['hkls'])

    if 'dmin' in AtomInfo:
        did = gid.create_dataset("dmin", (1,), dtype=np.float64)
        did.write_direct(np.array(AtomInfo['dmin'], dtype=np.float64))

    did = gid.create_dataset(
        "AtomData", (4, len(AtomInfo['Z'])), dtype=np.float64)
    # this is done for contiguous c-allocation
    arr = np.array(AtomInfo['APOS'], dtype=np.float32).transpose()
    arr2 = arr.copy()
    did.write_direct(arr2)

    if 'charge' in AtomInfo:
        data = np.array(AtomInfo['charge'], dtype=object)
        dt = h5py.special_dtype(vlen=str)
        did = gid.create_dataset("ChargeStates",
                                 (len(AtomInfo['Z']),),
                                 dtype=dt)
        did.write_direct(data)

    '''
        handling of isotropic vs
        anisotropic debye waller factors
    '''

    if not isinstance(AtomInfo['U'][0], np.floating):
        did = gid.create_dataset(
            "U", (6, len(AtomInfo['Z'])), dtype=np.float64)
        arr = np.array(AtomInfo['U'], dtype=np.float32).transpose()
        arr2 = arr.copy()
        did.write_direct(arr2)
    else:
        did = gid.create_dataset("U", (len(AtomInfo['Z']),), dtype=np.float32)
        did.write_direct(np.array(AtomInfo['U'], dtype=np.float64))

    # variable length string type
    dt = h5py.special_dtype(vlen=str)

    date = datetime.date.today().strftime("%B %d, %Y")
    did = gid.create_dataset("CreationDate", data=date, dtype=dt)

    time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    did = gid.create_dataset("CreationTime", data=time, dtype=dt)

    creator = getpass.getuser()
    did = gid.create_dataset("Creator", data=creator, dtype=dt)

    pname = "ProgramName"
    did = gid.create_dataset(pname, data="heXRD", dtype=dt)

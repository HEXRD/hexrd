
pstr_mkxtal = "\n\n    This is a program to create a HDF5 file for storing crystallographic information.\n "
pstr_mkxtal = pstr_mkxtal + " This format is the same format as used in the EMsoft (electron microscoy) suite.\n "
pstr_mkxtal = pstr_mkxtal + " The following inputs are required:\n "
pstr_mkxtal = pstr_mkxtal + "         Crystal System:\n"
pstr_mkxtal = pstr_mkxtal + "                 1. Cubic\n"
pstr_mkxtal = pstr_mkxtal + "                 2. Tetragonal\n"
pstr_mkxtal = pstr_mkxtal + "                 3. Orthorhombic\n"
pstr_mkxtal = pstr_mkxtal + "                 4. Hexagonal\n"
pstr_mkxtal = pstr_mkxtal + "                 5. Trigonal\n"
pstr_mkxtal = pstr_mkxtal + "                 6. Monoclinic\n"
pstr_mkxtal = pstr_mkxtal + "                 7. Triclinic\n\n"
pstr_mkxtal = pstr_mkxtal + "         Space group number\n"
pstr_mkxtal = pstr_mkxtal + "         Atomic number (Z) for all species in unit cell\n"
pstr_mkxtal = pstr_mkxtal + "         Asymmetric positions for all atoms in unit cell\n"
pstr_mkxtal = pstr_mkxtal + "         Debye-Waller factors for all atoms in the unit cell\n"
pstr_mkxtal = pstr_mkxtal + "         You'll be prompted for these values now\n\n"
pstr_mkxtal = pstr_mkxtal + "\n Note about the trigonal system:\n"
pstr_mkxtal = pstr_mkxtal + " -------------------------------\n"
pstr_mkxtal = pstr_mkxtal + " Primitive trigonal crystals are defined with respect to a HEXAGONAL\n"
pstr_mkxtal = pstr_mkxtal + " reference frame.  Rhombohedral crystals can be referenced with\n"
pstr_mkxtal = pstr_mkxtal + " respect to a HEXAGONAL basis (first setting), or with respect to\n"
pstr_mkxtal = pstr_mkxtal + " a RHOMBOHEDRAL basis (second setting).  The default setting for\n"
pstr_mkxtal = pstr_mkxtal + " trigonal symmetry is the hexagonal setting.  When you select\n"
pstr_mkxtal = pstr_mkxtal + " crystal system 5 above, you will be prompted for the setting. \n"

pstr_spacegroup = [ " P  1      " ," P -1      ", \
# MONOCLINIC SPACE GROUPS
" P 2       " ," P 21      " ," C 2       " ," P m       ", \
" P c       " ," C m       " ," C c       " ," P 2/m     ", \
" P 21/m    " ," C 2/m     " ," P 2/c     " ," P 21/c    ", \
" C 2/c     ", \
# ORTHORHOMBIC SPACE GROUPS
" P 2 2 2   " ," P 2 2 21  " ," P 21 21 2 " ," P 21 21 21", \
" C 2 2 21  " ," C 2 2 2   " ," F 2 2 2   " ," I 2 2 2   ", \
" I 21 21 21" ," P m m 2   " ," P m c 21  " ," P c c 2   ", \
" P m a 2   " ," P c a 21  " ," P n c 2   " ," P m n 21  ", \
" P b a 2   " ," P n a 21  " ," P n n 2   " ," C m m 2   ", \
" C m c 21  " ," C c c 2   " ," A m m 2   " ," A b m 2   ", \
" A m a 2   " ," A b a 2   " ," F m m 2   " ," F d d 2   ", \
" I m m 2   " ," I b a 2   " ," I m a 2   " ," P m m m   ", \
" P n n n   " ," P c c m   " ," P b a n   " ," P m m a   ", \
" P n n a   " ," P m n a   " ," P c c a   " ," P b a m   ", \
" P c c n   " ," P b c m   " ," P n n m   " ," P m m n   ", \
" P b c n   " ," P b c a   " ," P n m a   " ," C m c m   ", \
" C m c a   " ," C m m m   " ," C c c m   " ," C m m a   ", \
" C c c a   " ," F m m m   " ," F d d d   " ," I m m m   ", \
" I b a m   " ," I b c a   " ," I m m a   ", \
# TETRAGONAL SPACE GROUPS  
" P 4       " ," P 41      " ," P 42      " ," P 43      ", \
" I 4       " ," I 41      " ," P -4      " ," I -4      ", \
" P 4/m     " ," P 42/m    " ," P 4/n     " ," P 42/n    ", \
" I 4/m     " ," I 41/a    " ," P 4 2 2   " ," P 4 21 2  ", \
" P 41 2 2  " ," P 41 21 2 " ," P 42 2 2  " ," P 42 21 2 ", \
" P 43 2 2  " ," P 43 21 2 " ," I 4 2 2   " ," I 41 2 2  ", \
" P 4 m m   " ," P 4 b m   " ," P 42 c m  " ," P 42 n m  ", \
" P 4 c c   " ," P 4 n c   " ," P 42 m c  " ," P 42 b c  ", \
" I 4 m m   " ," I 4 c m   " ," I 41 m d  " ," I 41 c d  ", \
" P -4 2 m  " ," P -4 2 c  " ," P -4 21 m " ," P -4 21 c ", \
" P -4 m 2  " ," P -4 c 2  " ," P -4 b 2  " ," P -4 n 2  ", \
" I -4 m 2  " ," I -4 c 2  " ," I -4 2 m  " ," I -4 2 d  ", \
" P 4/m m m " ," P 4/m c c " ," P 4/n b m " ," P 4/n n c ", \
" P 4/m b m " ," P 4/m n c " ," P 4/n m m " ," P 4/n c c ", \
" P 42/m m c" ," P 42/m c m" ," P 42/n b c" ," P 42/n n m", \
" P 42/m b c" ," P 42/m n m" ," P 42/n m c" ," P 42/n c m", \
" I 4/m m m " ," I 4/m c m " ," I 41/a m d" ," I 41/a c d", \
# RHOMBOHEDRAL SPACE GROUPS  
" P 3       " ," P 31      " ," P 32      " ," R 3       ", \
" P -3      " ," R -3      " ," P 3 1 2   " ," P 3 2 1   ", \
" P 31 1 2  " ," P 31 2 1  " ," P 32 1 2  " ," P 32 2 1  ", \
" R 3 2     " ," P 3 m 1   " ," P 3 1 m   " ," P 3 c 1   ", \
" P 3 1 c   " ," R 3 m     " ," R 3 c     " ," P -3 1 m  ", \
" P -3 1 c  " ," P -3 m 1  " ," P -3 c 1  " ," R -3 m    ", \
" R -3 c    ", \
# HEXAGONAL SPACE GROUPS   
" P 6       " ," P 61      " ," P 65      " ," P 62      ", \
" P 64      " ," P 63      " ," P -6      " ," P 6/m     ", \
" P 63/m    " ," P 6 2 2   " ," P 61 2 2  " ," P 65 2 2  ", \
" P 62 2 2  " ," P 64 2 2  " ," P 63 2 2  " ," P 6 m m   ", \
" P 6 c c   " ," P 63 c m  " ," P 63 m c  " ," P -6 m 2  ", \
" P -6 c 2  " ," P -6 2 m  " ," P -6 2 c  " ," P 6/m m m ", \
" P 6/m c c " ," P 63/m c m" ," P 63/m m c", \
#CUBIC SPACE GROUPS
" P 2 3     " ," F 2 3     " ," I 2 3     " ," P 21 3    ", \
" I 21 3    " ," P m 3     " ," P n 3     " ," F m 3     ", \
" F d 3     " ," I m 3     " ," P a 3     " ," I a 3     ", \
" P 4 3 2   " ," P 42 3 2  " ," F 4 3 2   " ," F 41 3 2  ", \
" I 4 3 2   " ," P 43 3 2  " ," P 41 3 2  " ," I 41 3 2  ", \
" P -4 3 m  " ," F -4 3 m  " ," I -4 3 m  " ," P -4 3 n  ", \
" F -4 3 c  " ," I -4 3 d  " ," P m 3 m   " ," P n 3 n   ", \
" P m 3 n   " ," P n 3 m   " ," F m 3 m   " ," F m 3 c   ", \
" F d 3 m   " ," F d 3 c   " ," I m 3 m   " ," I a 3 d   ", \
# TRIGONAL GROUPS RHOMBOHEDRAL SETTING
" R 3   |146" ," R -3  |148" ," R 3 2 |155" ," R 3 m |160", \
" R 3 c |161" ," R -3 m|166" ," R -3 c|167"]

xtal_dict = {1:'Cubic', 2:'Tetragonal', 3:'Orthorhombic', 4:'Hexagonal', 5:'Trigonal', 6:'Monoclinic', 7:'Triclinic'}

pstr_pointgroup = ['    1','   -1','    2','    m','  2/m','  222', \
'  mm2','  mmm','    4','   -4','  4/m','  422', \
'  4mm',' -42m','4/mmm','    3','   -3','   32', \
'   3m','  -3m','    6','   -6','  6/m','  622', \
'  6mm',' -6m2','6/mmm','   23','   m3','  432', \
' -43m',' m-3m','  532','  822',' 1022',' 1222' ]

TRIG = [146,148,155,160,161,166,167]

# symbols and Z for all elements
pstr_Elements = ' ------------------------------------ Periodic Table of the Elements --------------------------------------' + "\n" \
'1:H                                                                                                    2:He' + "\n" \
'3:Li  4:Be                                                               5:B   6:C   7:N   8:O   9:F  10:Ne' + "\n" \
'11:Na 12:Mg                                                             13:Al 14:Si 15:P  16:S  17:Cl 18:Ar' + "\n" \
'19:K  20:Ca 21:Sc 22:Ti 23:V  24:Cr 25:Mn 26:Fe 27:Co 28:Ni 29:Cu 30:Zn 31:Ga 32:Ge 33:As 34:Se 35:Br 36:Kr' + "\n" \
'37:Rb 38:Sr 39:Y  40:Zr 41:Nb 42:Mo 43:Tc 44:Ru 45:Rh 46:Pd 47:Ag 48:Cd 49:In 50:Sn 51:Sb 52:Te 53: I 54:Xe' + "\n" \
'55:Cs 56:Ba ----- 72:Hf 73:Ta 74:W  75:Re 76:Os 77:Ir 78:Pt 79:Au 80:Hg 81:Tl 82:Pb 83:Bi 84:Po 85:At 86:Rn' + "\n" \
'87:Fr 88:Ra -----' + "\n" \
'57:La 58:Ce 59:Pr 60:Nd 61:Pm 62:Sm 63:Eu 64:Gd 65:Tb 66:Dy 67:Ho 68:Er 69:Tm 70:Yb 71:Lu' + "\n" \
'89:Ac 90:Th 91:Pa 92:U' + "\n" \
' ----------------------------------------------------------------------------------------------------------'

sitesym = ['222    ',' -1    ','222/n  ',' -1    ','mm2/n  ',' -1    ', \
'222    ',' -1    ','222    ',' -1    ','-4     ',' -1    ', \
'-4     ',' -1    ','-4     ',' -1    ','422    ','2/m    ', \
'422/n  ',' -1    ','-4m2   ','2/m    ','-4/ncn ',' -1    ', \
'-4121/c',' -1    ','-42m   ','2/m    ','-4m2/n ',' -1    ', \
'-4cg   ','2/m    ','-4m2   ','2/m    ','-4c21  ',' -1    ', \
'23     ',' -3    ','23     ',' -3    ','432    ',' -3    ', \
'-43m   ','-3m    ','-43m   ','-3m    ','23     ',' -3    ']

tworig = [48,50,59,68,70,85,86,88,125,126,129,130,133,134,137,138,\
141,142,201,203,222,224,227,228]


def PrintPossibleSG(xtal_sys):
        if(xtal_sys == 1):
                sgmax = 230
                sgmin = 195
        elif(xtal_sys == 2):
                sgmax = 142
                sgmin = 75
        elif(xtal_sys == 3):
                sgmax = 74
                sgmin = 16
        elif(xtal_sys == 4):
                sgmax = 194
                sgmin = 168
        elif(xtal_sys == 5):
                sgmax = 167
                sgmin = 143
        elif(xtal_sys == 6):
                sgmax = 15
                sgmin = 3
        elif(xtal_sys == 7):
                sgmax = 2
                sgmin = 1

        for i in range(sgmin,sgmax+1):
                j = i - sgmin + 1
                pstr = str(i) + ":" + pstr_spacegroup[i-1] + "\t"
                if(j % 4 == 0 or j == sgmax):
                        print(pstr)
                else:
                        print(pstr, end='')
        print("\n")

        return sgmin, sgmax
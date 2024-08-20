
pstr_mkxtal = (
    f'\n\n This is a program to create a HDF5 file for storing '
    f'crystallographic information.\n'
    f' The following inputs are required:\n '
    f'         Crystal System:\n'
    f'                 1. Cubic\n'
    f'                 2. Tetragonal\n'
    f'                 3. Orthorhombic\n'
    f'                 4. Hexagonal\n'
    f'                 5. Trigonal\n'
    f'                 6. Monoclinic\n'
    f'                 7. Triclinic\n\n'
    f'         Space group number\n'
    f'         Atomic number (Z) for all species in unit cell\n'
    f'         Asymmetric positions for all atoms in unit cell\n'
    f'         Debye-Waller factors for all atoms in the unit cell\n'
    f'         You\'ll be prompted for these values now \n\n'
    f'\n Note about the trigonal system:\n'
    f' -------------------------------\n'
    f' Primitive trigonal crystals are defined with respect to a HEXAGONAL\n'
    f' reference frame.  Rhombohedral crystals can be referenced with\n'
    f' respect to a HEXAGONAL basis (first setting), or with respect to\n'
    f' a RHOMBOHEDRAL basis (second setting).  The default setting for\n'
    f' trigonal symmetry is the hexagonal setting.  When you select\n'
    f' crystal system 5 above, you will be prompted for the setting. \n')


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

xtal_dict = {1:'cubic', 2:'tetragonal', 3:'orthorhombic', 4:'hexagonal', 5:'trigonal', 6:'monoclinic', 7:'triclinic'}
xtal_sys_dict = {'cubic':1, 'tetragonal':2, 'orthorhombic':3, 'hexagonal':4, 'trigonal':5, 'monoclinic':6, 'triclinic':7}


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


#
# Hall Symbols copied from:
#
HALL_STR = r"""
     1         P 1
     2        -P 1
     3:b       P 2y
     3:c       P 2
     3:a       P 2x
     4:b       P 2yb
     4:c       P 2c
     4:a       P 2xa
     5:b1      C 2y
     5:b2      A 2y
     5:b3      I 2y
     5:c1      A 2
     5:c2      B 2
     5:c3      I 2
     5:a1      B 2x
     5:a2      C 2x
     5:a3      I 2x
     6:b       P -2y
     6:c       P -2
     6:a       P -2x
     7:b1      P -2yc
     7:b2      P -2yac
     7:b3      P -2ya
     7:c1      P -2a
     7:c2      P -2ab
     7:c3      P -2b
     7:a1      P -2xb
     7:a2      P -2xbc
     7:a3      P -2xc
     8:b1      C -2y
     8:b2      A -2y
     8:b3      I -2y
     8:c1      A -2
     8:c2      B -2
     8:c3      I -2
     8:a1      B -2x
     8:a2      C -2x
     8:a3      I -2x
     9:b1      C -2yc
     9:b2      A -2yac
     9:b3      I -2ya
     9:-b1     A -2ya
     9:-b2     C -2ybc
     9:-b3     I -2yc
     9:c1      A -2a
     9:c2      B -2bc
     9:c3      I -2b
     9:-c1     B -2b
     9:-c2     A -2ac
     9:-c3     I -2a
     9:a1      B -2xb
     9:a2      C -2xbc
     9:a3      I -2xc
     9:-a1     C -2xc
     9:-a2     B -2xbc
     9:-a3     I -2xb
    10:b      -P 2y
    10:c      -P 2
    10:a      -P 2x
    11:b      -P 2yb
    11:c      -P 2c
    11:a      -P 2xa
    12:b1     -C 2y
    12:b2     -A 2y
    12:b3     -I 2y
    12:c1     -A 2
    12:c2     -B 2
    12:c3     -I 2
    12:a1     -B 2x
    12:a2     -C 2x
    12:a3     -I 2x
    13:b1     -P 2yc
    13:b2     -P 2yac
    13:b3     -P 2ya
    13:c1     -P 2a
    13:c2     -P 2ab
    13:c3     -P 2b
    13:a1     -P 2xb
    13:a2     -P 2xbc
    13:a3     -P 2xc
    14:b1     -P 2ybc
    14:b2     -P 2yn
    14:b3     -P 2yab
    14:c1     -P 2ac
    14:c2     -P 2n
    14:c3     -P 2bc
    14:a1     -P 2xab
    14:a2     -P 2xn
    14:a3     -P 2xac
    15:b1     -C 2yc
    15:b2     -A 2yac
    15:b3     -I 2ya
    15:-b1    -A 2ya
    15:-b2    -C 2ybc
    15:-b3    -I 2yc
    15:c1     -A 2a
    15:c2     -B 2bc
    15:c3     -I 2b
    15:-c1    -B 2b
    15:-c2    -A 2ac
    15:-c3    -I 2a
    15:a1     -B 2xb
    15:a2     -C 2xbc
    15:a3     -I 2xc
    15:-a1    -C 2xc
    15:-a2    -B 2xbc
    15:-a3    -I 2xb
    16         P 2 2
    17         P 2c 2
    17:cab     P 2a 2a
    17:bca     P 2 2b
    18         P 2 2ab
    18:cab     P 2bc 2
    18:bca     P 2ac 2ac
    19         P 2ac 2ab
    20         C 2c 2
    20:cab     A 2a 2a
    20:bca     B 2 2b
    21         C 2 2
    21:cab     A 2 2
    21:bca     B 2 2
    22         F 2 2
    23         I 2 2
    24         I 2b 2c
    25         P 2 -2
    25:cab     P -2 2
    25:bca     P -2 -2
    26         P 2c -2
    26:ba-c    P 2c -2c
    26:cab     P -2a 2a
    26:-cba    P -2 2a
    26:bca     P -2 -2b
    26:a-cb    P -2b -2
    27         P 2 -2c
    27:cab     P -2a 2
    27:bca     P -2b -2b
    28         P 2 -2a
    28:ba-c    P 2 -2b
    28:cab     P -2b 2
    28:-cba    P -2c 2
    28:bca     P -2c -2c
    28:a-cb    P -2a -2a
    29         P 2c -2ac
    29:ba-c    P 2c -2b
    29:cab     P -2b 2a
    29:-cba    P -2ac 2a
    29:bca     P -2bc -2c
    29:a-cb    P -2a -2ab
    30         P 2 -2bc
    30:ba-c    P 2 -2ac
    30:cab     P -2ac 2
    30:-cba    P -2ab 2
    30:bca     P -2ab -2ab
    30:a-cb    P -2bc -2bc
    31         P 2ac -2
    31:ba-c    P 2bc -2bc
    31:cab     P -2ab 2ab
    31:-cba    P -2 2ac
    31:bca     P -2 -2bc
    31:a-cb    P -2ab -2
    32         P 2 -2ab
    32:cab     P -2bc 2
    32:bca     P -2ac -2ac
    33         P 2c -2n
    33:ba-c    P 2c -2ab
    33:cab     P -2bc 2a
    33:-cba    P -2n 2a
    33:bca     P -2n -2ac
    33:a-cb    P -2ac -2n
    34         P 2 -2n
    34:cab     P -2n 2
    34:bca     P -2n -2n
    35         C 2 -2
    35:cab     A -2 2
    35:bca     B -2 -2
    36         C 2c -2
    36:ba-c    C 2c -2c
    36:cab     A -2a 2a
    36:-cba    A -2 2a
    36:bca     B -2 -2b
    36:a-cb    B -2b -2
    37         C 2 -2c
    37:cab     A -2a 2
    37:bca     B -2b -2b
    38         A 2 -2
    38:ba-c    B 2 -2
    38:cab     B -2 2
    38:-cba    C -2 2
    38:bca     C -2 -2
    38:a-cb    A -2 -2
    39         A 2 -2c
    39:ba-c    B 2 -2c
    39:cab     B -2c 2
    39:-cba    C -2b 2
    39:bca     C -2b -2b
    39:a-cb    A -2c -2c
    40         A 2 -2a
    40:ba-c    B 2 -2b
    40:cab     B -2b 2
    40:-cba    C -2c 2
    40:bca     C -2c -2c
    40:a-cb    A -2a -2a
    41         A 2 -2ac
    41:ba-c    B 2 -2bc
    41:cab     B -2bc 2
    41:-cba    C -2bc 2
    41:bca     C -2bc -2bc
    41:a-cb    A -2ac -2ac
    42         F 2 -2
    42:cab     F -2 2
    42:bca     F -2 -2
    43         F 2 -2d
    43:cab     F -2d 2
    43:bca     F -2d -2d
    44         I 2 -2
    44:cab     I -2 2
    44:bca     I -2 -2
    45         I 2 -2c
    45:cab     I -2a 2
    45:bca     I -2b -2b
    46         I 2 -2a
    46:ba-c    I 2 -2b
    46:cab     I -2b 2
    46:-cba    I -2c 2
    46:bca     I -2c -2c
    46:a-cb    I -2a -2a
    47        -P 2 2
    48:1       P 2 2 -1n
    48:2      -P 2ab 2bc
    49        -P 2 2c
    49:cab    -P 2a 2
    49:bca    -P 2b 2b
    50:1       P 2 2 -1ab
    50:2      -P 2ab 2b
    50:1cab    P 2 2 -1bc
    50:2cab   -P 2b 2bc
    50:1bca    P 2 2 -1ac
    50:2bca   -P 2a 2c
    51        -P 2a 2a
    51:ba-c   -P 2b 2
    51:cab    -P 2 2b
    51:-cba   -P 2c 2c
    51:bca    -P 2c 2
    51:a-cb   -P 2 2a
    52        -P 2a 2bc
    52:ba-c   -P 2b 2n
    52:cab    -P 2n 2b
    52:-cba   -P 2ab 2c
    52:bca    -P 2ab 2n
    52:a-cb   -P 2n 2bc
    53        -P 2ac 2
    53:ba-c   -P 2bc 2bc
    53:cab    -P 2ab 2ab
    53:-cba   -P 2 2ac
    53:bca    -P 2 2bc
    53:a-cb   -P 2ab 2
    54        -P 2a 2ac
    54:ba-c   -P 2b 2c
    54:cab    -P 2a 2b
    54:-cba   -P 2ac 2c
    54:bca    -P 2bc 2b
    54:a-cb   -P 2b 2ab
    55        -P 2 2ab
    55:cab    -P 2bc 2
    55:bca    -P 2ac 2ac
    56        -P 2ab 2ac
    56:cab    -P 2ac 2bc
    56:bca    -P 2bc 2ab
    57        -P 2c 2b
    57:ba-c   -P 2c 2ac
    57:cab    -P 2ac 2a
    57:-cba   -P 2b 2a
    57:bca    -P 2a 2ab
    57:a-cb   -P 2bc 2c
    58        -P 2 2n
    58:cab    -P 2n 2
    58:bca    -P 2n 2n
    59:1       P 2 2ab -1ab
    59:2      -P 2ab 2a
    59:1cab    P 2bc 2 -1bc
    59:2cab   -P 2c 2bc
    59:1bca    P 2ac 2ac -1ac
    59:2bca   -P 2c 2a
    60        -P 2n 2ab
    60:ba-c   -P 2n 2c
    60:cab    -P 2a 2n
    60:-cba   -P 2bc 2n
    60:bca    -P 2ac 2b
    60:a-cb   -P 2b 2ac
    61        -P 2ac 2ab
    61:ba-c   -P 2bc 2ac
    62        -P 2ac 2n
    62:ba-c   -P 2bc 2a
    62:cab    -P 2c 2ab
    62:-cba   -P 2n 2ac
    62:bca    -P 2n 2a
    62:a-cb   -P 2c 2n
    63        -C 2c 2
    63:ba-c   -C 2c 2c
    63:cab    -A 2a 2a
    63:-cba   -A 2 2a
    63:bca    -B 2 2b
    63:a-cb   -B 2b 2
    64        -C 2bc 2
    64:ba-c   -C 2bc 2bc
    64:cab    -A 2ac 2ac
    64:-cba   -A 2 2ac
    64:bca    -B 2 2bc
    64:a-cb   -B 2bc 2
    65        -C 2 2
    65:cab    -A 2 2
    65:bca    -B 2 2
    66        -C 2 2c
    66:cab    -A 2a 2
    66:bca    -B 2b 2b
    67        -C 2b 2
    67:ba-c   -C 2b 2b
    67:cab    -A 2c 2c
    67:-cba   -A 2 2c
    67:bca    -B 2 2c
    67:a-cb   -B 2c 2
    68:1       C 2 2 -1bc
    68:2      -C 2b 2bc
    68:1ba-c   C 2 2 -1bc
    68:2ba-c  -C 2b 2c
    68:1cab    A 2 2 -1ac
    68:2cab   -A 2a 2c
    68:1-cba   A 2 2 -1ac
    68:2-cba  -A 2ac 2c
    68:1bca    B 2 2 -1bc
    68:2bca   -B 2bc 2b
    68:1a-cb   B 2 2 -1bc
    68:2a-cb  -B 2b 2bc
    69        -F 2 2
    70:1       F 2 2 -1d
    70:2      -F 2uv 2vw
    71        -I 2 2
    72        -I 2 2c
    72:cab    -I 2a 2
    72:bca    -I 2b 2b
    73        -I 2b 2c
    73:ba-c   -I 2a 2b
    74        -I 2b 2
    74:ba-c   -I 2a 2a
    74:cab    -I 2c 2c
    74:-cba   -I 2 2b
    74:bca    -I 2 2a
    74:a-cb   -I 2c 2
    75         P 4
    76         P 4w
    77         P 4c
    78         P 4cw
    79         I 4
    80         I 4bw
    81         P -4
    82         I -4
    83        -P 4
    84        -P 4c
    85:1       P 4ab -1ab
    85:2      -P 4a
    86:1       P 4n -1n
    86:2      -P 4bc
    87        -I 4
    88:1       I 4bw -1bw
    88:2      -I 4ad
    89         P 4 2
    90         P 4ab 2ab
    91         P 4w 2c
    92         P 4abw 2nw
    93         P 4c 2
    94         P 4n 2n
    95         P 4cw 2c
    96         P 4nw 2abw
    97         I 4 2
    98         I 4bw 2bw
    99         P 4 -2
   100         P 4 -2ab
   101         P 4c -2c
   102         P 4n -2n
   103         P 4 -2c
   104         P 4 -2n
   105         P 4c -2
   106         P 4c -2ab
   107         I 4 -2
   108         I 4 -2c
   109         I 4bw -2
   110         I 4bw -2c
   111         P -4 2
   112         P -4 2c
   113         P -4 2ab
   114         P -4 2n
   115         P -4 -2
   116         P -4 -2c
   117         P -4 -2ab
   118         P -4 -2n
   119         I -4 -2
   120         I -4 -2c
   121         I -4 2
   122         I -4 2bw
   123        -P 4 2
   124        -P 4 2c
   125:1       P 4 2 -1ab
   125:2      -P 4a 2b
   126:1       P 4 2 -1n
   126:2      -P 4a 2bc
   127        -P 4 2ab
   128        -P 4 2n
   129:1       P 4ab 2ab -1ab
   129:2      -P 4a 2a
   130:1       P 4ab 2n -1ab
   130:2      -P 4a 2ac
   131        -P 4c 2
   132        -P 4c 2c
   133:1       P 4n 2c -1n
   133:2      -P 4ac 2b
   134:1       P 4n 2 -1n
   134:2      -P 4ac 2bc
   135        -P 4c 2ab
   136        -P 4n 2n
   137:1       P 4n 2n -1n
   137:2      -P 4ac 2a
   138:1       P 4n 2ab -1n
   138:2      -P 4ac 2ac
   139        -I 4 2
   140        -I 4 2c
   141:1       I 4bw 2bw -1bw
   141:2      -I 4bd 2
   142:1       I 4bw 2aw -1bw
   142:2      -I 4bd 2c
   143         P 3
   144         P 31
   145         P 32
   146:H       R 3
   146:R       P 3*
   147        -P 3
   148:H      -R 3
   148:R      -P 3*
   149         P 3 2
   150         P 3 2"
   151         P 31 2c (0 0 1)
   152         P 31 2"
   153         P 32 2c (0 0 -1)
   154         P 32 2"
   155:H       R 3 2"
   155:R       P 3* 2
   156         P 3 -2"
   157         P 3 -2
   158         P 3 -2"c
   159         P 3 -2c
   160:H       R 3 -2"
   160:R       P 3* -2
   161:H       R 3 -2"c
   161:R       P 3* -2n
   162        -P 3 2
   163        -P 3 2c
   164        -P 3 2"
   165        -P 3 2"c
   166:H      -R 3 2"
   166:R      -P 3* 2
   167:H      -R 3 2"c
   167:R      -P 3* 2n
   168         P 6
   169         P 61
   170         P 65
   171         P 62
   172         P 64
   173         P 6c
   174         P -6
   175        -P 6
   176        -P 6c
   177         P 6 2
   178         P 61 2 (0 0 -1)
   179         P 65 2 (0 0 1)
   180         P 62 2c (0 0 1)
   181         P 64 2c (0 0 -1)
   182         P 6c 2c
   183         P 6 -2
   184         P 6 -2c
   185         P 6c -2
   186         P 6c -2c
   187         P -6 2
   188         P -6c 2
   189         P -6 -2
   190         P -6c -2c
   191        -P 6 2
   192        -P 6 2c
   193        -P 6c 2
   194        -P 6c 2c
   195         P 2 2 3
   196         F 2 2 3
   197         I 2 2 3
   198         P 2ac 2ab 3
   199         I 2b 2c 3
   200        -P 2 2 3
   201:1       P 2 2 3 -1n
   201:2      -P 2ab 2bc 3
   202        -F 2 2 3
   203:1       F 2 2 3 -1d
   203:2      -F 2uv 2vw 3
   204        -I 2 2 3
   205        -P 2ac 2ab 3
   206        -I 2b 2c 3
   207         P 4 2 3
   208         P 4n 2 3
   209         F 4 2 3
   210         F 4d 2 3
   211         I 4 2 3
   212         P 4acd 2ab 3
   213         P 4bd 2ab 3
   214         I 4bd 2c 3
   215         P -4 2 3
   216         F -4 2 3
   217         I -4 2 3
   218         P -4n 2 3
   219         F -4c 2 3
   220         I -4bd 2c 3
   221        -P 4 2 3
   222:1       P 4 2 3 -1n
   222:2      -P 4a 2bc 3
   223        -P 4n 2 3
   224:1       P 4n 2 3 -1n
   224:2      -P 4bc 2bc 3
   225        -F 4 2 3
   226        -F 4c 2 3
   227:1       F 4d 2 3 -1d
   227:2      -F 4vw 2vw 3
   228:1       F 4d 2 3 -1cd
   228:2      -F 4cvw 2vw 3
   229        -I 4 2 3
   230        -I 4bd 2c 3
"""
# Hermann-Mauguin notation
HM_STR = r"""
     1        P 1
     2        P -1
     3:b      P 1 2 1
     3:c      P 1 1 2
     3:a      P 2 1 1
     4:b      P 1 21 1
     4:c      P 1 1 21
     4:a      P 21 1 1
     5:b1     C 1 2 1
     5:b2     A 1 2 1
     5:b3     I 1 2 1
     5:c1     A 1 1 2
     5:c2     B 1 1 2
     5:c3     I 1 1 2
     5:a1     B 2 1 1
     5:a2     C 2 1 1
     5:a3     I 2 1 1
     6:b      P 1 m 1
     6:c      P 1 1 m
     6:a      P m 1 1
     7:b1     P 1 c 1
     7:b2     P 1 n 1
     7:b3     P 1 a 1
     7:c1     P 1 1 a
     7:c2     P 1 1 n
     7:c3     P 1 1 b
     7:a1     P b 1 1
     7:a2     P n 1 1
     7:a3     P c 1 1
     8:b1     C 1 m 1
     8:b2     A 1 m 1
     8:b3     I 1 m 1
     8:c1     A 1 1 m
     8:c2     B 1 1 m
     8:c3     I 1 1 m
     8:a1     B m 1 1
     8:a2     C m 1 1
     8:a3     I m 1 1
     9:b1     C 1 c 1
     9:b2     A 1 n 1
     9:b3     I 1 a 1
     9:-b1    A 1 a 1
     9:-b2    C 1 n 1
     9:-b3    I 1 c 1
     9:c1     A 1 1 a
     9:c2     B 1 1 n
     9:c3     I 1 1 b
     9:-c1    B 1 1 b
     9:-c2    A 1 1 n
     9:-c3    I 1 1 a
     9:a1     B b 1 1
     9:a2     C n 1 1
     9:a3     I c 1 1
     9:-a1    C c 1 1
     9:-a2    B n 1 1
     9:-a3    I b 1 1
    10:b      P 1 2/m 1
    10:c      P 1 1 2/m
    10:a      P 2/m 1 1
    11:b      P 1 21/m 1
    11:c      P 1 1 21/m
    11:a      P 21/m 1 1
    12:b1     C 1 2/m 1
    12:b2     A 1 2/m 1
    12:b3     I 1 2/m 1
    12:c1     A 1 1 2/m
    12:c2     B 1 1 2/m
    12:c3     I 1 1 2/m
    12:a1     B 2/m 1 1
    12:a2     C 2/m 1 1
    12:a3     I 2/m 1 1
    13:b1     P 1 2/c 1
    13:b2     P 1 2/n 1
    13:b3     P 1 2/a 1
    13:c1     P 1 1 2/a
    13:c2     P 1 1 2/n
    13:c3     P 1 1 2/b
    13:a1     P 2/b 1 1
    13:a2     P 2/n 1 1
    13:a3     P 2/c 1 1
    14:b1     P 1 21/c 1
    14:b2     P 1 21/n 1
    14:b3     P 1 21/a 1
    14:c1     P 1 1 21/a
    14:c2     P 1 1 21/n
    14:c3     P 1 1 21/b
    14:a1     P 21/b 1 1
    14:a2     P 21/n 1 1
    14:a3     P 21/c 1 1
    15:b1     C 1 2/c 1
    15:b2     A 1 2/n 1
    15:b3     I 1 2/a 1
    15:-b1    A 1 2/a 1
    15:-b2    C 1 2/n 1
    15:-b3    I 1 2/c 1
    15:c1     A 1 1 2/a
    15:c2     B 1 1 2/n
    15:c3     I 1 1 2/b
    15:-c1    B 1 1 2/b
    15:-c2    A 1 1 2/n
    15:-c3    I 1 1 2/a
    15:a1     B 2/b 1 1
    15:a2     C 2/n 1 1
    15:a3     I 2/c 1 1
    15:-a1    C 2/c 1 1
    15:-a2    B 2/n 1 1
    15:-a3    I 2/b 1 1
    16        P 2 2 2
    17        P 2 2 21
    17:cab    P 21 2 2
    17:bca    P 2 21 2
    18        P 21 21 2
    18:cab    P 2 21 21
    18:bca    P 21 2 21
    19        P 21 21 21
    20        C 2 2 21
    20:cab    A 21 2 2
    20:bca    B 2 21 2
    21        C 2 2 2
    21:cab    A 2 2 2
    21:bca    B 2 2 2
    22        F 2 2 2
    23        I 2 2 2
    24        I 21 21 21
    25        P m m 2
    25:cab    P 2 m m
    25:bca    P m 2 m
    26        P m c 21
    26:ba-c   P c m 21
    26:cab    P 21 m a
    26:-cba   P 21 a m
    26:bca    P b 21 m
    26:a-cb   P m 21 b
    27        P c c 2
    27:cab    P 2 a a
    27:bca    P b 2 b
    28        P m a 2
    28:ba-c   P b m 2
    28:cab    P 2 m b
    28:-cba   P 2 c m
    28:bca    P c 2 m
    28:a-cb   P m 2 a
    29        P c a 21
    29:ba-c   P b c 21
    29:cab    P 21 a b
    29:-cba   P 21 c a
    29:bca    P c 21 b
    29:a-cb   P b 21 a
    30        P n c 2
    30:ba-c   P c n 2
    30:cab    P 2 n a
    30:-cba   P 2 a n
    30:bca    P b 2 n
    30:a-cb   P n 2 b
    31        P m n 21
    31:ba-c   P n m 21
    31:cab    P 21 m n
    31:-cba   P 21 n m
    31:bca    P n 21 m
    31:a-cb   P m 21 n
    32        P b a 2
    32:cab    P 2 c b
    32:bca    P c 2 a
    33        P n a 21
    33:ba-c   P b n 21
    33:cab    P 21 n b
    33:-cba   P 21 c n
    33:bca    P c 21 n
    33:a-cb   P n 21 a
    34        P n n 2
    34:cab    P 2 n n
    34:bca    P n 2 n
    35        C m m 2
    35:cab    A 2 m m
    35:bca    B m 2 m
    36        C m c 21
    36:ba-c   C c m 21
    36:cab    A 21 m a
    36:-cba   A 21 a m
    36:bca    B b 21 m
    36:a-cb   B m 21 b
    37        C c c 2
    37:cab    A 2 a a
    37:bca    B b 2 b
    38        A m m 2
    38:ba-c   B m m 2
    38:cab    B 2 m m
    38:-cba   C 2 m m
    38:bca    C m 2 m
    38:a-cb   A m 2 m
    39        A b m 2
    39:ba-c   B m a 2
    39:cab    B 2 c m
    39:-cba   C 2 m b
    39:bca    C m 2 a
    39:a-cb   A c 2 m
    40        A m a 2
    40:ba-c   B b m 2
    40:cab    B 2 m b
    40:-cba   C 2 c m
    40:bca    C c 2 m
    40:a-cb   A m 2 a
    41        A b a 2
    41:ba-c   B b a 2
    41:cab    B 2 c b
    41:-cba   C 2 c b
    41:bca    C c 2 a
    41:a-cb   A c 2 a
    42        F m m 2
    42:cab    F 2 m m
    42:bca    F m 2 m
    43        F d d 2
    43:cab    F 2 d d
    43:bca    F d 2 d
    44        I m m 2
    44:cab    I 2 m m
    44:bca    I m 2 m
    45        I b a 2
    45:cab    I 2 c b
    45:bca    I c 2 a
    46        I m a 2
    46:ba-c   I b m 2
    46:cab    I 2 m b
    46:-cba   I 2 c m
    46:bca    I c 2 m
    46:a-cb   I m 2 a
    47        P m m m
    48:1      P n n n:1
    48:2      P n n n:2
    49        P c c m
    49:cab    P m a a
    49:bca    P b m b
    50:1      P b a n:1
    50:2      P b a n:2
    50:1cab   P n c b:1
    50:2cab   P n c b:2
    50:1bca   P c n a:1
    50:2bca   P c n a:2
    51        P m m a
    51:ba-c   P m m b
    51:cab    P b m m
    51:-cba   P c m m
    51:bca    P m c m
    51:a-cb   P m a m
    52        P n n a
    52:ba-c   P n n b
    52:cab    P b n n
    52:-cba   P c n n
    52:bca    P n c n
    52:a-cb   P n a n
    53        P m n a
    53:ba-c   P n m b
    53:cab    P b m n
    53:-cba   P c n m
    53:bca    P n c m
    53:a-cb   P m a n
    54        P c c a
    54:ba-c   P c c b
    54:cab    P b a a
    54:-cba   P c a a
    54:bca    P b c b
    54:a-cb   P b a b
    55        P b a m
    55:cab    P m c b
    55:bca    P c m a
    56        P c c n
    56:cab    P n a a
    56:bca    P b n b
    57        P b c m
    57:ba-c   P c a m
    57:cab    P m c a
    57:-cba   P m a b
    57:bca    P b m a
    57:a-cb   P c m b
    58        P n n m
    58:cab    P m n n
    58:bca    P n m n
    59:1      P m m n:1
    59:2      P m m n:2
    59:1cab   P n m m:1
    59:2cab   P n m m:2
    59:1bca   P m n m:1
    59:2bca   P m n m:2
    60        P b c n
    60:ba-c   P c a n
    60:cab    P n c a
    60:-cba   P n a b
    60:bca    P b n a
    60:a-cb   P c n b
    61        P b c a
    61:ba-c   P c a b
    62        P n m a
    62:ba-c   P m n b
    62:cab    P b n m
    62:-cba   P c m n
    62:bca    P m c n
    62:a-cb   P n a m
    63        C m c m
    63:ba-c   C c m m
    63:cab    A m m a
    63:-cba   A m a m
    63:bca    B b m m
    63:a-cb   B m m b
    64        C m c a
    64:ba-c   C c m b
    64:cab    A b m a
    64:-cba   A c a m
    64:bca    B b c m
    64:a-cb   B m a b
    65        C m m m
    65:cab    A m m m
    65:bca    B m m m
    66        C c c m
    66:cab    A m a a
    66:bca    B b m b
    67        C m m a
    67:ba-c   C m m b
    67:cab    A b m m
    67:-cba   A c m m
    67:bca    B m c m
    67:a-cb   B m a m
    68:1      C c c a:1
    68:2      C c c a:2
    68:1ba-c  C c c b:1
    68:2ba-c  C c c b:2
    68:1cab   A b a a:1
    68:2cab   A b a a:2
    68:1-cba  A c a a:1
    68:2-cba  A c a a:2
    68:1bca   B b c b:1
    68:2bca   B b c b:2
    68:1a-cb  B b a b:1
    68:2a-cb  B b a b:2
    69        F m m m
    70:1      F d d d:1
    70:2      F d d d:2
    71        I m m m
    72        I b a m
    72:cab    I m c b
    72:bca    I c m a
    73        I b c a
    73:ba-c   I c a b
    74        I m m a
    74:ba-c   I m m b
    74:cab    I b m m
    74:-cba   I c m m
    74:bca    I m c m
    74:a-cb   I m a m
    75        P 4
    76        P 41
    77        P 42
    78        P 43
    79        I 4
    80        I 41
    81        P -4
    82        I -4
    83        P 4/m
    84        P 42/m
    85:1      P 4/n:1
    85:2      P 4/n:2
    86:1      P 42/n:1
    86:2      P 42/n:2
    87        I 4/m
    88:1      I 41/a:1
    88:2      I 41/a:2
    89        P 4 2 2
    90        P 42 1 2
    91        P 41 2 2
    92        P 41 21 2
    93        P 42 2 2
    94        P 42 21 2
    95        P 43 2 2
    96        P 43 21 2
    97        I 4 2 2
    98        I 41 2 2
    99        P 4 m m
   100        P 4 b m
   101        P 42 c m
   102        P 42 n m
   103        P 4 c c
   104        P 4 n c
   105        P 42 m c
   106        P 42 b c
   107        I 4 m m
   108        I 4 c m
   109        I 41 m d
   110        I 41 c d
   111        P -4 2 m
   112        P -4 2 c
   113        P -4 21 m
   114        P -4 21 c
   115        P -4 m 2
   116        P -4 c 2
   117        P -4 b 2
   118        P -4 n 2
   119        I -4 m 2
   120        I -4 c 2
   121        I -4 2 m
   122        I -4 2 d
   123        P 4/m m m
   124        P 4/m c c
   125:1      P 4/n b m:1
   125:2      P 4/n b m:2
   126:1      P 4/n n c:1
   126:2      P 4/n n c:2
   127        P 4/m b m
   128        P 4/m n c
   129:1      P 4/n m m:1
   129:2      P 4/n m m:2
   130:1      P 4/n c c:1
   130:2      P 4/n c c:2
   131        P 42/m m c
   132        P 42/m c m
   133:1      P 42/n b c:1
   133:2      P 42/n b c:2
   134:1      P 42/n n m:1
   134:2      P 42/n n m:2
   135        P 42/m b c
   136        P 42/m n m
   137:1      P 42/n m c:1
   137:2      P 42/n m c:2
   138:1      P 42/n c m:1
   138:2      P 42/n c m:2
   139        I 4/m m m
   140        I 4/m c m
   141:1      I 41/a m d:1
   141:2      I 41/a m d:2
   142:1      I 41/a c d:1
   142:2      I 41/a c d:2
   143        P 3
   144        P 31
   145        P 32
   146:H      R 3
   146:R      R 3
   147        P -3
   148:H      R -3
   148:R      R -3
   149        P 3 1 2
   150        P 3 2 1
   151        P 31 1 2
   152        P 31 2 1
   153        P 32 1 2
   154        P 32 2 1
   155:H      R 32
   155:R      R 32
   156        P 3 m 1
   157        P 3 1 m
   158        P 3 c 1
   159        P 3 1 c
   160:H      R 3 m
   160:R      R 3 m
   161:H      R 3 c
   161:R      R 3 c
   162        P -3 1 m
   163        P -3 1 c
   164        P -3 m 1
   165        P -3 c 1
   166:H      R -3 m
   166:R      R -3 m
   167:H      R -3 c
   167:R      R -3 c
   168        P 6
   169        P 61
   170        P 65
   171        P 62
   172        P 64
   173        P 63
   174        P -6
   175        P 6/m
   176        P 63/m
   177        P 6 2 2
   178        P 61 2 2
   179        P 65 2 2
   180        P 62 2 2
   181        P 64 2 2
   182        P 63 2 2
   183        P 6 m m
   184        P 6 c c
   185        P 63 c m
   186        P 63 m c
   187        P -6 m 2
   188        P -6 c 2
   189        P -6 2 m
   190        P -6 2 c
   191        P 6/m m m
   192        P 6/m c c
   193        P 63/m c m
   194        P 63/m m c
   195        P 2 3
   196        F 2 3
   197        I 2 3
   198        P 21 3
   199        I 21 3
   200        P m -3
   201:1      P n -3:1
   201:2      P n -3:2
   202        F m -3
   203:1      F d -3:1
   203:2      F d -3:2
   204        I m -3
   205        P a -3
   206        I a -3
   207        P 4 3 2
   208        P 42 3 2
   209        F 4 3 2
   210        F 41 3 2
   211        I 4 3 2
   212        P 43 3 2
   213        P 41 3 2
   214        I 41 3 2
   215        P -4 3 m
   216        F -4 3 m
   217        I -4 3 m
   218        P -4 3 n
   219        F -4 3 c
   220        I -4 3 d
   221        P m -3 m
   222:1      P n -3 n:1
   222:2      P n -3 n:2
   223        P m -3 n
   224:1      P n -3 m:1
   224:2      P n -3 m:2
   225        F m -3 m
   226        F m -3 c
   227:1      F d -3 m:1
   227:2      F d -3 m:2
   228:1      F d -3 c:1
   228:2      F d -3 c:2
   229        I m -3 m
   230        I a -3 d
"""

def _buildDict(hstr):
    """build the dictionaries from the notation string

    Returns two dictionaries:  one taking sg number to string
    and the inverse

    Takes the first Hall symbol it finds.  This is desired
    for the rhombohedral lattices so that they use the hexagonal
    convention.
    """
    d  = dict()
    di = dict()
    hs = hstr.split('\n')
    for l in hs:
        li = l.strip()
        if li:
            nstr, hstr = li.split(None, 1)
            nstr = nstr.split(':', 1)[0]
            n = int(nstr)
            hstr = hstr.split(':', 1)[0]
            hstr = hstr.replace(" ", "")
            if n not in d:
                d[n] = hstr
            di[hstr] = n

    return d, di


lookupHall, Hall_to_sgnum = _buildDict(HALL_STR)
lookupHM,     HM_to_sgnum = _buildDict(HM_STR)

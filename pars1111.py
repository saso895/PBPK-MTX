#def get_params():
BW = 73
QPlas = 390.0
QL = 99.45
QK = 74.1
QRest = QPlas - QL - QK
VPlas = 5.6
VL = 1.8
VK = 0.31
VRest = 62.29

MKC = 0.0084
PL = 4.66
PK = 2.9
PRest = 0.2
MW = 454.439
Free = 0.58
Vmax_baso_invitro = 242.75
Km_baso = 17.814
protein = 2.0e-6
GFR = 13.9
Kreab = 0.1
Kbile = 3.3
Kurine = 0.063

PTC = MKC * 6e7 * 1000
Vmax_basoC = (Vmax_baso_invitro * PTC * protein * 60 * (MW / 1e12) * 1000)
Vmax_baso = Vmax_basoC * BW**0.75

params = {
    "QRest": QRest,
    "QK": QK,
    "QL": QL,
    "QPlas": QPlas,
    "VRest": VRest,
    "VK": VK,
    "VL": VL,
    "VPlas": VPlas,
    "PRest": PRest,
    "PK": PK,
    "PL": PL,
    "Kbile": Kbile,
    "GFR": GFR,
    "Free": Free,
    "Vmax_baso": Vmax_baso,
    "Km_baso": Km_baso,
    "Kurine": Kurine,
    "Kreab": Kreab,
}

    #params = [QRest, QK, QL, QPlas, VRest, VK, VL, VPlas, PRest, PK, PL, Kbile, GFR, Free, Vmax_baso, Km_baso, Kurine, Kreab]
    #return params

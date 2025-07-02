import pandas as pd

BW                  =  73          #  kg, Bodyweight (EPA Factors Handbook, 2011)



QRest               =  216.4500    #  L/h Qplas-QL-QK Rest of body output (Brown 1997) 
QK                  =  74.1000     #  L/h 0.19*6.5*60 kidneys output (Brown 1997) 
QL                  =  99.4500     #  L/h 0.255*6.5*60 Liver output (Brown 1997)  
QPlas               =  390.0000    #  L/h 6.5*60 Cardiac output (Brown 1997) 
VRest               =  62.2900     #  L Rest of body volume (Brown 1997) 
VK                  =  0.3100      #  L kidneys volume(Brown 1997)
VL                  =  1.8000      #  L Liver volume (Brown 1997) 
VPlas               =  5.6000      #  L Cardiac output (Brown 1997) 

PRest               =  0.2      # Restofbody/plasma PC; (Average of fat and non fat) 
PK                  =  2.9      # Kidney/plasma PC; (calKP_PT.R)
PL                  =  4.66     # Liver/plasma PC; (calKP_PT.R) 
Kbile               =  3.3      # L/hr
GFR                 =  13.9     # L/hr (Kayode Ogungbenro，2014)
Free                =  0.58     # MTX unbound fractions in plasma (Giuseppe Pesenti et al., 2021)

MKC                 =  0.0084   # Fraction mass of kidny (percent of BW); Brown, 1997 
MW                  =  454.439  # g/mol, MTX molecular mass 
protein             =  2.0e-6   # mg protein/proximal tubuel cell, Amount of protein in proximal tubule cells
PTC = MKC*6e7*1000#; // cells/kg BW, Number of PTC (cells/kg BW) (based on 60 million PTC/gram kidney,Hsu et al., 2014); Revised original equation (PTC = MKC*6e7) from Worley et al. (2015)

Vmax_baso_invitro   =  242.75   # pmol/mg protein/min, Average of OAT3 and Rfc1
Vmax_basoC = Vmax_baso_invitro*PTC*protein*60*(MW/1e12)*1000 #; // mg/h/kg BW^0.75, Vmax of basolateral transporters (average Oat1 and Oat3)


Vmax_baso = Vmax_basoC*BW**0.75#; // mg/h
Km_baso             =  17.814   # mg/L, Km of basolateral transpoter, Average of OAT3 and Rfc1
Kurine              =  0.063    # L/h, Rate of urine elimination from urine storage (male) (fit to data)
Kreab               =  0.1      # L/hr





# 创建参数表
# 将所有参数存储在字典 init_pars 中
init_pars = {
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

# 保存为文件
# MTXpars_calcu.to_csv("MTXpars_calcu_6_24.csv", index=False)

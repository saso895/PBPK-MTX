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

# === MOD BEGIN 2025‑07‑23 新增“顶侧外排 & 胆汁饱和”参数 =========================
# 近曲小管顶侧外排（MRP2/BCRP 方向，肾小管→尿）
Vmax_apical         = 120.0        #  mg/h   经验初值，可后续拟合
Km_apical           =  20.0        #  mg/L   经验初值，可后续拟合

# 肝胆汁转运饱和（取代/补充线性 Kbile）
Vmax_bile           =  30.0        #  mg/h   经验初值，可后续拟合
Km_bile             =  15.0        #  mg/L   经验初值，可后续拟合
# === MOD END =====================================================================



# 创建参数表
# 将所有参数存储在字典 init_pars 中
init_pars = {
    # ‑‑ 血流与容积
    "QRest": QRest,
    "QK": QK,
    "QL": QL,
    "QPlas": QPlas,
    "VRest": VRest,
    "VK": VK,
    "VL": VL,
    "VPlas": VPlas,
    # ‑‑ 分配与线性转运
    "PRest": PRest,
    "PK": PK,
    "PL": PL,
    "Kbile": Kbile,
    "GFR": GFR,
    "Free": Free,
    # ‑‑ 肾主动/被动过程
    "Vmax_baso": Vmax_baso,
    "Km_baso": Km_baso,
    "Kurine": Kurine,
    "Kreab": Kreab,
    # === MOD BEGIN 2025‑07‑23 新增参数登记 ===
    "Vmax_apical": Vmax_apical, "Km_apical": Km_apical,
    "Vmax_bile":   Vmax_bile,   "Km_bile":   Km_bile,
    # === MOD END ============================
}

# 保存为文件
# MTXpars_calcu.to_csv("MTXpars_calcu_6_24.csv", index=False)

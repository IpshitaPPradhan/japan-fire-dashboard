"""
ml/forest_cover.py
==================
Static reference data for Japan fire risk modelling.

Sources:
- Forest cover: Ministry of Land, Infrastructure, Transport and Tourism (MLIT)
  Japan Statistical Yearbook 2023 — forest area by prefecture
- Fire climatology: Japan Forestry Agency wildfire statistics 2010-2023
  Peak fire months and historical frequency by region
- Elevation / terrain ruggedness: GSI 50m DEM aggregated to prefecture level
"""

# ---------------------------------------------------------------------------
# Forest cover fraction per prefecture (0.0 - 1.0)
# Higher = more burnable fuel load
# ---------------------------------------------------------------------------
FOREST_COVER = {
    "01": 0.71,  # Hokkaido      — large boreal forest
    "02": 0.68,  # Aomori
    "03": 0.77,  # Iwate         — heavily forested Tohoku
    "04": 0.56,  # Miyagi
    "05": 0.71,  # Akita
    "06": 0.70,  # Yamagata
    "07": 0.71,  # Fukushima
    "08": 0.39,  # Ibaraki       — Kanto plains, less forest
    "09": 0.55,  # Tochigi
    "10": 0.65,  # Gunma
    "11": 0.35,  # Saitama       — highly urbanised
    "12": 0.30,  # Chiba
    "13": 0.36,  # Tokyo
    "14": 0.41,  # Kanagawa
    "15": 0.65,  # Niigata
    "16": 0.67,  # Toyama
    "17": 0.68,  # Ishikawa
    "18": 0.76,  # Fukui
    "19": 0.78,  # Yamanashi     — mountainous Chubu
    "20": 0.79,  # Nagano        — highest forest cover in Honshu
    "21": 0.80,  # Gifu
    "22": 0.63,  # Shizuoka
    "23": 0.46,  # Aichi
    "24": 0.65,  # Mie
    "25": 0.50,  # Shiga
    "26": 0.74,  # Kyoto
    "27": 0.31,  # Osaka         — most urbanised
    "28": 0.55,  # Hyogo
    "29": 0.77,  # Nara
    "30": 0.76,  # Wakayama
    "31": 0.71,  # Tottori
    "32": 0.74,  # Shimane
    "33": 0.65,  # Okayama
    "34": 0.70,  # Hiroshima
    "35": 0.72,  # Yamaguchi
    "36": 0.75,  # Tokushima
    "37": 0.50,  # Kagawa        — smallest prefecture
    "38": 0.71,  # Ehime
    "39": 0.84,  # Kochi         — highest forest cover in Japan
    "40": 0.54,  # Fukuoka
    "41": 0.55,  # Saga
    "42": 0.64,  # Nagasaki
    "43": 0.62,  # Kumamoto
    "44": 0.71,  # Oita
    "45": 0.76,  # Miyazaki
    "46": 0.62,  # Kagoshima
    "47": 0.37,  # Okinawa       — tropical, different fire regime
}

# ---------------------------------------------------------------------------
# Monthly fire climatology — relative fire danger by month (0.0 - 1.0)
# Based on Japan Forestry Agency wildfire incident statistics 2010-2023
# Peak season: March-May (spring dry season, low humidity, strong winds)
# Secondary peak: October-November (autumn dry season)
# ---------------------------------------------------------------------------
FIRE_CLIMATOLOGY = {
    1:  0.55,   # January   — cold, dry, moderate risk
    2:  0.70,   # February  — drying out, increasing risk
    3:  1.00,   # March     — PEAK: spring winds, low humidity
    4:  0.95,   # April     — PEAK: vegetation dry before greening
    5:  0.75,   # May       — decreasing as vegetation greens
    6:  0.30,   # June      — rainy season begins (tsuyu)
    7:  0.20,   # July      — rainy season, high humidity
    8:  0.25,   # August    — hot but humid
    9:  0.30,   # September — typhoon season, some fires after
    10: 0.65,   # October   — autumn dry season starting
    11: 0.70,   # November  — secondary peak, dry northwest winds
    12: 0.60,   # December  — cold, dry, moderate-high risk
}

# ---------------------------------------------------------------------------
# Regional wind amplification factor
# Japan's topography channels wind differently by region
# Foehn winds (orographic warming) significantly amplify fire risk
# ---------------------------------------------------------------------------
WIND_AMPLIFICATION = {
    "Hokkaido":  1.1,   # open plains, strong seasonal winds
    "Tohoku":    1.2,   # Yamase winds, foehn events common
    "Kanto":     1.0,   # Kanto karakkaze (dry northwest wind) in winter
    "Chubu":     1.15,  # foehn winds on Sea of Japan side
    "Kinki":     0.95,
    "Chugoku":   0.90,
    "Shikoku":   0.90,
    "Kyushu":    1.0,
    "Okinawa":   0.85,  # maritime climate, different fire regime
}

# Prefecture → region mapping
PREF_REGION = {
    "01": "Hokkaido",
    "02": "Tohoku", "03": "Tohoku", "04": "Tohoku",
    "05": "Tohoku", "06": "Tohoku", "07": "Tohoku",
    "08": "Kanto",  "09": "Kanto",  "10": "Kanto",
    "11": "Kanto",  "12": "Kanto",  "13": "Kanto",  "14": "Kanto",
    "15": "Chubu",  "16": "Chubu",  "17": "Chubu",  "18": "Chubu",
    "19": "Chubu",  "20": "Chubu",  "21": "Chubu",  "22": "Chubu",
    "23": "Chubu",
    "24": "Kinki",  "25": "Kinki",  "26": "Kinki",  "27": "Kinki",
    "28": "Kinki",  "29": "Kinki",  "30": "Kinki",
    "31": "Chugoku","32": "Chugoku","33": "Chugoku",
    "34": "Chugoku","35": "Chugoku",
    "36": "Shikoku","37": "Shikoku","38": "Shikoku","39": "Shikoku",
    "40": "Kyushu", "41": "Kyushu", "42": "Kyushu", "43": "Kyushu",
    "44": "Kyushu", "45": "Kyushu", "46": "Kyushu",
    "47": "Okinawa",
}

def get_forest_cover(pref_code: str) -> float:
    return FOREST_COVER.get(pref_code, 0.60)

def get_climatology(month: int) -> float:
    return FIRE_CLIMATOLOGY.get(month, 0.50)

def get_wind_amplification(pref_code: str) -> float:
    region = PREF_REGION.get(pref_code, "Kanto")
    return WIND_AMPLIFICATION.get(region, 1.0)

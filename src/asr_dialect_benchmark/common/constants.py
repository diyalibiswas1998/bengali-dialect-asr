DIALECTS = ["barishal", "chittagong", "khulna", "rajshahi"]
DEFAULT_AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".m4a", ".ogg"]

# ──────────────────────────────────────────────────────────────────────────────
# Vaani dataset – West Bengal district configurations
# ──────────────────────────────────────────────────────────────────────────────
VAANI_DISTRICT_CONFIGS = [
    "WestBengal_Alipurduar",
    "WestBengal_CoochBehar",
    "WestBengal_Darjeeling",
    "WestBengal_Jalpaiguri",
    "WestBengal_Jhargram",
    "WestBengal_PaschimMedinipur",
    "WestBengal_Purulia",
    "WestBengal_Malda",
    "WestBengal_DakshinDinajpur",
    "WestBengal_North24Parganas",
    "WestBengal_Kolkata",
]

# Maps config name -> integer label index (used by VaaniDataset)
VAANI_DISTRICT_TO_IDX: dict = {
    config: idx for idx, config in enumerate(VAANI_DISTRICT_CONFIGS)
}

"""Versioned constants for the West Bengal Vaani benchmark."""

DIALECT_GROUPS = [
    "Rarhi",
    "Varendri",
    "Rajbanshi/Kamrupi",
    "Manbhumi/Jharkhandi",
]
DIALECTS = DIALECT_GROUPS  # Backwards-compatible public name.
DIALECT_TO_IDX = {name: index for index, name in enumerate(DIALECT_GROUPS)}

DISTRICT_TO_DIALECT = {
    "Kolkata": "Rarhi",
    "North24Parganas": "Rarhi",
    "Malda": "Varendri",
    "DakshinDinajpur": "Varendri",
    "Alipurduar": "Rajbanshi/Kamrupi",
    "CoochBehar": "Rajbanshi/Kamrupi",
    "Jalpaiguri": "Rajbanshi/Kamrupi",
    "Darjeeling": "Rajbanshi/Kamrupi",
    "Jhargram": "Manbhumi/Jharkhandi",
    "PaschimMedinipur": "Manbhumi/Jharkhandi",
    "Purulia": "Manbhumi/Jharkhandi",
}

BOUNDARY_DISTRICTS = ("Darjeeling", "North24Parganas")
DIALECT_MAPPING_VERSION = "west-bengal-proxy-v1"
DIALECT_MAPPING_REFERENCE = (
    "https://ruralindiaonline.org/te/library/resource/"
    "linguistic-survey-of-india---west-bengal-part-i/"
)

VAANI_DISTRICT_CONFIGS = [f"WestBengal_{district}" for district in DISTRICT_TO_DIALECT]
VAANI_DISTRICT_TO_IDX = {
    config: index for index, config in enumerate(VAANI_DISTRICT_CONFIGS)
}
DEFAULT_AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".m4a", ".ogg"]

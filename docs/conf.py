from importlib.metadata import version as package_version

project = "expax"
author = "Seth D. Axen"
copyright = "2026, Seth D. Axen"
release = package_version("expax")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
autodoc_typehints = "signature"
autodoc_typehints_format = "short"

html_theme = "furo"
exclude_patterns = ["_build", "superpowers"]

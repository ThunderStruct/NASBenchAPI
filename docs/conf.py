project = "NASBenchAPI"
author = "NASBenchAPI Contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "separated"

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "furo"
html_title = "NASBenchAPI Documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#345995",
        "color-brand-content": "#1f2933",
        "color-background-secondary": "#f4f6fb",
        "color-admonition-background": "#edf3ff",
    },
    "dark_css_variables": {
        "color-brand-primary": "#8cb9ff",
        "color-brand-content": "#f8fafc",
        "color-admonition-background": "#111827",
    },
}

pygments_style = "friendly"
pygments_dark_style = "native"

"""
Library Fallback Handler for Diagram Generation
Provides robust alternatives when preferred libraries are unavailable
"""

import importlib
import logging
from typing import Dict, List

class LibraryFallbackHandler:
    """Handles library availability and provides fallback options"""
    
    def __init__(self):
        self.available_libraries = {}
        self.fallback_map = {
            'graphviz': ['matplotlib', 'networkx'],
            'schemdraw': ['matplotlib', 'plotly'],
            'pygraphviz': ['networkx', 'matplotlib'],
            'anytree': ['networkx', 'matplotlib'],
            'ete3': ['networkx', 'matplotlib'],
            'control': ['scipy', 'matplotlib'],
            'bokeh': ['plotly', 'matplotlib'],
            'kaleido': ['matplotlib'],
            'scapy': ['matplotlib', 'networkx'],
            'statsmodels': ['scipy', 'seaborn'],
        }
        self._check_library_availability()
    
    def _check_library_availability(self):
        """Check which libraries are actually available"""
        libraries_to_check = [
            'matplotlib', 'numpy', 'pandas', 'seaborn', 'plotly', 'scipy',
            'networkx', 'graphviz', 'sklearn', 'sympy', 'bokeh', 'statsmodels',
            'schemdraw', 'pygraphviz', 'anytree', 'ete3', 'control', 'scapy', 'kaleido'
        ]
        
        for lib in libraries_to_check:
            try:
                if lib == 'sklearn':
                    importlib.import_module('sklearn.datasets')
                elif lib == 'scapy':
                    importlib.import_module('scapy.all')
                elif lib == 'statsmodels':
                    importlib.import_module('statsmodels.api')
                else:
                    importlib.import_module(lib)
                self.available_libraries[lib] = True
            except ImportError:
                self.available_libraries[lib] = False
    
    def get_available_libraries_for_domain(self, domain: str) -> List[str]:
        """Get list of available libraries for a specific domain - supports all 24+ academic domains"""
        
        # Comprehensive domain mapping for all 24+ academic subjects
        domain_libraries = {
            # Computer Science Domains
            "Computer Science - Operating Systems": ["graphviz", "matplotlib", "networkx", "plotly", "seaborn", "schemdraw"],
            "Computer Science - Networks": ["graphviz", "matplotlib", "networkx", "plotly", "scapy", "pygraphviz", "bokeh"],
            "Computer Science - Algorithms": ["graphviz", "matplotlib", "networkx", "plotly", "seaborn", "pygraphviz", "anytree", "ete3"],
            "Computer Science - Database": ["networkx", "matplotlib", "plotly", "pandas", "seaborn", "graphviz"],
            "Computer Science - AI/ML": ["matplotlib", "seaborn", "plotly", "sklearn", "numpy", "pandas", "networkx"],
            "Computer Science - Software Engineering": ["graphviz", "matplotlib", "networkx", "plotly", "schemdraw"],
            
            # Electronics & Engineering
            "Electronics - Digital Logic": ["matplotlib", "numpy", "plotly", "scipy", "schemdraw", "graphviz"],
            "Electronics - Analog Circuits": ["matplotlib", "numpy", "scipy", "plotly", "schemdraw", "control"],
            "Switching & Circuit Design": ["matplotlib", "numpy", "scipy", "plotly", "schemdraw", "control"],
            "General Engineering": ["matplotlib", "numpy", "scipy", "plotly", "networkx"],
            
            # Mathematics Domains
            "Mathematics - Calculus": ["matplotlib", "numpy", "scipy", "sympy", "plotly"],
            "Mathematics - Statistics": ["matplotlib", "numpy", "scipy", "seaborn", "plotly", "statsmodels"],
            "Mathematics - Linear Algebra": ["matplotlib", "numpy", "scipy", "seaborn", "plotly"],
            
            # Physics Domains
            "Physics - Mechanics": ["matplotlib", "numpy", "scipy", "plotly"],
            "Physics - Thermodynamics": ["matplotlib", "numpy", "scipy", "plotly", "seaborn"],
            "Physics - Electromagnetism": ["matplotlib", "numpy", "scipy", "plotly", "schemdraw"],
            
            # Chemistry Domains
            "Chemistry - Organic": ["matplotlib", "numpy", "plotly", "networkx"],
            "Chemistry - Inorganic": ["matplotlib", "numpy", "plotly", "networkx", "seaborn"],
            
            # Biology Domains
            "Biology - Cell Biology": ["matplotlib", "numpy", "plotly", "networkx"],
            "Biology - Genetics": ["matplotlib", "numpy", "plotly", "networkx", "ete3", "anytree"],
            
            # Economics Domains
            "Economics - Microeconomics": ["matplotlib", "numpy", "plotly", "seaborn"],
            "Economics - Macroeconomics": ["matplotlib", "numpy", "plotly", "seaborn", "statsmodels"],
            
            # Business & General
            "Business": ["matplotlib", "numpy", "plotly", "seaborn", "pandas"],
            "General Science": ["matplotlib", "numpy", "plotly", "seaborn"],
            
            # Legacy support for old domain names
            "Operating Systems": ["graphviz", "matplotlib", "networkx", "plotly", "seaborn"],
            "DCCN / Networks": ["networkx", "matplotlib", "plotly", "numpy", "scapy"],
            "Digital Logic / Control": ["matplotlib", "numpy", "plotly", "scipy", "schemdraw"],
            "Algorithms / Architecture": ["matplotlib", "seaborn", "networkx", "plotly", "numpy", "graphviz"],
            "Database Systems": ["networkx", "matplotlib", "plotly", "pandas", "graphviz"],
            "Machine Learning / AI": ["matplotlib", "seaborn", "networkx", "sklearn", "numpy", "pandas"]
        }
        
        # Get libraries for the domain
        domain_libs = domain_libraries.get(domain, ["matplotlib", "numpy", "plotly"])
        
        # Filter to only available libraries
        available_libs = [lib for lib in domain_libs if self.available_libraries.get(lib, False)]
        
        # If no specific libraries available, use universal fallbacks
        if not available_libs:
            universal_fallbacks = ["matplotlib", "numpy", "plotly", "seaborn", "networkx"]
            available_libs = [lib for lib in universal_fallbacks if self.available_libraries.get(lib, False)]
        
        return available_libs
    
    def get_fallback_library(self, preferred_library: str, domain: str) -> str:
        """Get a fallback library if the preferred one is not available"""
        if self.available_libraries.get(preferred_library, False):
            return preferred_library
        
        # Try fallback options
        fallbacks = self.fallback_map.get(preferred_library, [])
        for fallback in fallbacks:
            if self.available_libraries.get(fallback, False):
                return fallback
        
        # Try domain-specific alternatives
        domain_alternatives = self.get_available_libraries_for_domain(domain)
        if domain_alternatives:
            return domain_alternatives[0]
        
        # Last resort: matplotlib
        return 'matplotlib'

# Global instance
fallback_handler = LibraryFallbackHandler() 
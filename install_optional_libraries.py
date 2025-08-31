#!/usr/bin/env python3
"""
Optional Library Installer for Enhanced Diagram Generation
Helps users install missing libraries for better diagram capabilities
"""

import subprocess
import sys
import importlib
from typing import List, Dict

def check_library(library_name: str, import_name: str = None) -> bool:
    """Check if a library is available"""
    try:
        if import_name:
            importlib.import_module(import_name)
        else:
            importlib.import_module(library_name)
        return True
    except ImportError:
        return False

def install_library(library_name: str) -> bool:
    """Install a library using pip"""
    try:
        print(f"Installing {library_name}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", library_name], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Successfully installed {library_name}")
            return True
        else:
            print(f"❌ Failed to install {library_name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing {library_name}: {e}")
        return False

def main():
    """Main installation function"""
    print("🚀 Enhanced Diagram Libraries Installer")
    print("=" * 50)
    
    # Define optional libraries with their pip names and import names
    optional_libraries = {
        "schemdraw": {
            "pip_name": "schemdraw",
            "import_name": "schemdraw",
            "description": "Circuit and schematic diagrams",
            "category": "Circuit Design"
        },
        "kaleido": {
            "pip_name": "kaleido",
            "import_name": "kaleido",
            "description": "Plotly image export support",
            "category": "Image Export"
        },
        "anytree": {
            "pip_name": "anytree",
            "import_name": "anytree",
            "description": "Tree data structures and visualization",
            "category": "Tree Diagrams"
        },
        "ete3": {
            "pip_name": "ete3",
            "import_name": "ete3",
            "description": "Phylogenetic tree analysis",
            "category": "Advanced Trees"
        },
        "control": {
            "pip_name": "control",
            "import_name": "control",
            "description": "Control systems engineering",
            "category": "Control Systems"
        },
        "pygraphviz": {
            "pip_name": "pygraphviz",
            "import_name": "pygraphviz",
            "description": "Enhanced Graphviz interface",
            "category": "Graph Visualization"
        },
        "bokeh": {
            "pip_name": "bokeh",
            "import_name": "bokeh.plotting",
            "description": "Interactive web visualizations",
            "category": "Interactive Plots"
        }
    }
    
    # Check current status
    print("📊 Checking current library status...")
    missing_libraries = []
    available_libraries = []
    
    for lib_name, lib_info in optional_libraries.items():
        if check_library(lib_name, lib_info["import_name"]):
            available_libraries.append(lib_name)
            print(f"✅ {lib_name}: Available")
        else:
            missing_libraries.append(lib_name)
            print(f"❌ {lib_name}: Missing")
    
    print(f"\n📈 Status: {len(available_libraries)}/{len(optional_libraries)} libraries available")
    
    if not missing_libraries:
        print("🎉 All optional libraries are already installed!")
        return
    
    print(f"\n🔧 Missing libraries: {len(missing_libraries)}")
    for lib_name in missing_libraries:
        lib_info = optional_libraries[lib_name]
        print(f"  • {lib_name} - {lib_info['description']} ({lib_info['category']})")
    
    # Ask user what to install
    print("\n" + "=" * 50)
    print("Installation Options:")
    print("1. Install all missing libraries")
    print("2. Install specific libraries")
    print("3. Show installation commands only")
    print("4. Exit")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled.")
        return
    
    if choice == "1":
        # Install all missing libraries
        print(f"\n🔄 Installing {len(missing_libraries)} missing libraries...")
        success_count = 0
        
        for lib_name in missing_libraries:
            lib_info = optional_libraries[lib_name]
            if install_library(lib_info["pip_name"]):
                success_count += 1
        
        print(f"\n📊 Installation complete: {success_count}/{len(missing_libraries)} succeeded")
        
    elif choice == "2":
        # Install specific libraries
        print("\nSelect libraries to install (enter numbers separated by spaces):")
        for i, lib_name in enumerate(missing_libraries, 1):
            lib_info = optional_libraries[lib_name]
            print(f"{i}. {lib_name} - {lib_info['description']}")
        
        try:
            selections = input("\nEnter selections: ").strip().split()
            selected_libs = []
            
            for sel in selections:
                try:
                    idx = int(sel) - 1
                    if 0 <= idx < len(missing_libraries):
                        selected_libs.append(missing_libraries[idx])
                except ValueError:
                    print(f"Invalid selection: {sel}")
            
            if selected_libs:
                print(f"\n🔄 Installing {len(selected_libs)} selected libraries...")
                success_count = 0
                
                for lib_name in selected_libs:
                    lib_info = optional_libraries[lib_name]
                    if install_library(lib_info["pip_name"]):
                        success_count += 1
                
                print(f"\n📊 Installation complete: {success_count}/{len(selected_libs)} succeeded")
            else:
                print("No valid libraries selected.")
                
        except KeyboardInterrupt:
            print("\n\nInstallation cancelled.")
            return
    
    elif choice == "3":
        # Show installation commands
        print("\n📋 Manual Installation Commands:")
        print("Copy and run these commands in your terminal:\n")
        
        for lib_name in missing_libraries:
            lib_info = optional_libraries[lib_name]
            print(f"# {lib_info['description']}")
            print(f"pip install {lib_info['pip_name']}")
            print()
        
        print("Or install all at once:")
        pip_names = [optional_libraries[lib]["pip_name"] for lib in missing_libraries]
        print(f"pip install {' '.join(pip_names)}")
    
    elif choice == "4":
        print("Installation cancelled.")
        return
    
    else:
        print("Invalid choice.")
        return
    
    # Final status check
    if choice in ["1", "2"]:
        print("\n" + "=" * 50)
        print("🔍 Final Status Check:")
        
        final_available = []
        final_missing = []
        
        for lib_name, lib_info in optional_libraries.items():
            if check_library(lib_name, lib_info["import_name"]):
                final_available.append(lib_name)
                print(f"✅ {lib_name}: Available")
            else:
                final_missing.append(lib_name)
                print(f"❌ {lib_name}: Still missing")
        
        improvement = len(final_available) - len(available_libraries)
        print(f"\n📈 Final Status: {len(final_available)}/{len(optional_libraries)} libraries available")
        
        if improvement > 0:
            print(f"🎉 Improvement: +{improvement} libraries installed!")
        
        if final_missing:
            print(f"\n💡 Tip: Some libraries may require system dependencies.")
            print("   Check the documentation for installation requirements.")

if __name__ == "__main__":
    main() 
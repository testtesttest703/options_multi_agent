#!/usr/bin/env python3
import os
import re
import sys

# Project structure map
PROJECT_STRUCTURE = {
    "root": ["main.py", "constants.py"],
    "env": ["generic_env.py"],
    "agents": ["manager.py", "registry.py"],
    "agents/specialists": ["base_specialist.py", "iron_condor.py", "bull_put.py"],
    "utils": ["data_loader.py", "data_processor.py", "metrics.py"]
}

def analyze_imports(project_root):
    print("Analyzing imports in:", project_root)
    problematic_imports = []
    
    for dirpath, dirnames, filenames in os.walk(project_root):
        for filename in filenames:
            if filename.endswith('.py'):
                filepath = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(filepath, project_root)
                
                # Skip the script itself and any virtual environments
                if "venv" in filepath or "env" in filepath and ".py" not in filepath:
                    continue
                
                # Get the module's directory relative to project root
                module_dir = os.path.dirname(rel_path)
                if not module_dir:
                    module_dir = "root"
                
                with open(filepath, 'r') as f:
                    try:
                        content = f.read()
                        # Extract all import statements
                        imports = re.findall(r'^\s*(from\s+[\.\w]+\s+import\s+.+|import\s+[\.\w]+)', content, re.MULTILINE)
                        
                        for import_stmt in imports:
                            # Check for potential problematic imports
                            if 'from .' in import_stmt:
                                # Check relative imports
                                if module_dir == "root":
                                    problematic_imports.append(f"ERROR: {rel_path} - Relative import in root level: {import_stmt}")
                                elif '..' in import_stmt:
                                    # Going up directories can be tricky
                                    problematic_imports.append(f"WARNING: {rel_path} - Complex relative import, verify: {import_stmt}")
                            
                            # Check specific imports we've had issues with
                            if 'from .utils' in import_stmt and 'env/' in filepath:
                                # The utils is at root level, not in env
                                problematic_imports.append(f"ERROR: {rel_path} - Should use 'from utils' instead of 'from .utils': {import_stmt}")
                            
                            if 'from env.' in import_stmt and not 'from env.constants' in import_stmt and not '/env/' in filepath:
                                # Imports from env module should probably be relative if within env directory
                                problematic_imports.append(f"WARNING: {rel_path} - Check env import: {import_stmt}")
                            
                            if 'from constants import' in import_stmt and 'env/' in filepath:
                                # Constants might be more complicated - we have it at both root and env levels
                                problematic_imports.append(f"NOTICE: {rel_path} - Verify constants import: {import_stmt}")
                            
                            # Check for missing imports based on common patterns
                            if 'extract_options_data' in content and 'data_processor' not in content:
                                problematic_imports.append(f"ERROR: {rel_path} - Uses extract_options_data but doesn't import data_processor")
                            
                    except Exception as e:
                        print(f"Error reading {filepath}: {e}")
    
    return problematic_imports

if __name__ == "__main__":
    # Use current directory if no path is provided
    project_root = sys.argv[1] if len(sys.argv) > 1 else '/home/ubuntu/options_multi_agent'
    
    problems = analyze_imports(project_root)
    
    if problems:
        print("\n===== PROBLEMATIC IMPORTS FOUND =====")
        for problem in problems:
            print(problem)
        print(f"\nFound {len(problems)} potential import issues")
    else:
        print("No problematic imports found!")

    print("\n===== RECOMMENDED IMPORT PATTERNS =====")
    print("From root files (main.py, etc):")
    print("  import constants")
    print("  from env.generic_env import OptionsBaseEnv")
    print("  from utils.data_processor import extract_options_data")
    print("  from agents.registry import SPECIALIST_REGISTRY")
    
    print("\nFrom env/*.py files:")
    print("  import constants  # For root constants")
    print("  from .constants import XYZ  # For env/constants.py")
    print("  from utils.data_processor import extract_options_data  # NOT .utils")
    
    print("\nFrom agents/*.py files:")
    print("  import constants")
    print("  from utils.metrics import calculate_performance")
    print("  from env.generic_env import OptionsBaseEnv  # Absolute import")
    
    print("\nFrom agents/specialists/*.py files:")
    print("  import constants")
    print("  from ..registry import register_specialist  # Relative to agents")
    print("  from utils.data_processor import extract_options_data")

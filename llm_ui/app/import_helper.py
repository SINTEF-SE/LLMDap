import sys
import os

def setup_paths():
    """Add all necessary paths to sys.path to enable imports"""
    # Project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sys.path.append(project_root)
    
    # Profiler directory
    profiler_dir = os.path.join(project_root, 'profiler')
    sys.path.append(profiler_dir)
    
    # Print paths for debugging
    print(f"Project root added to path: {project_root}")
    print(f"Profiler directory added to path: {profiler_dir}")
    
    # Current Python path
    print("Current Python path:")
    for path in sys.path:
        print(f"  {path}")
    
    return project_root, profiler_dir
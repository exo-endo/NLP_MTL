#!/usr/bin/env python3
"""
Run MTL Streamlit app programmatically.
Usage: run with the project's venv python, e.g.

"/path/to/hless/bin/python" run_streamlit_mtl.py

This sets sys.argv and invokes Streamlit's CLI main function in-process.
"""
import os
import sys

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
APP_PATH = os.path.join(BASE_DIR, 'apps', 'slit_offline_mtl.py')

if not os.path.exists(APP_PATH):
    print(f"Error: app not found at {APP_PATH}")
    sys.exit(2)

# Ensure we're in project root so relative paths resolve
os.chdir(BASE_DIR)

# Prepare arguments as if called from command line
sys.argv = ["streamlit", "run", APP_PATH, "--server.port", "8502"]

# Import the streamlit CLI entry point and run it
try:
    # Streamlit >= 1.18 may expose web.cli
    from streamlit.web import cli as stcli
except Exception:
    try:
        # Older versions
        from streamlit import cli as stcli
    except Exception as e:
        print("Could not import Streamlit CLI module:", e)
        sys.exit(3)

# Execute Streamlit CLI (this will block and serve the app)
print("Starting MTL Streamlit app... local URL: http://localhost:8502")
ret = stcli.main()
# stcli.main() may return an exit code
sys.exit(ret if isinstance(ret, int) else 0)

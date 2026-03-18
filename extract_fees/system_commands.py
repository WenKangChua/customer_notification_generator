import subprocess
import sys

def open_file(filepath):
    if sys.platform == "darwin":       # macOS
        subprocess.run(["open", filepath])
    elif sys.platform == "win32":      # Windows
        subprocess.run(["start", filepath], shell=True)
    elif sys.platform.startswith("linux"):  # Linux
        subprocess.run(["xdg-open", filepath])
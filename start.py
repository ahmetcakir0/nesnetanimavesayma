import os
import subprocess

# Uygulamanın bulunduğu dizine git
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Terminali başlat ve streamlit uygulamasını çalıştır
subprocess.run(["streamlit", "run", "nesnetanimavesayma.py"])

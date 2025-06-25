with open("/home/kiranftw/OFFICE/AI-Powered-Solar-Maintenance-System/anomolyDetection.py", "r", encoding="utf-8") as f:
    cleaned = f.read().replace('\u00a0', ' ')
with open("your_script_clean.py", "w", encoding="utf-8") as f:
    f.write(cleaned)

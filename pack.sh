find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

rm -rf doctor-xbt.zip
zip -r doctor-xbt.zip app deepsearch requirements.txt Dockerfile system_prompt.txt
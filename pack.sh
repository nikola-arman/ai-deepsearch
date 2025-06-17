find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

rm -rf doctor-xbt.zip
zip -r doctor-xbt.zip requirements.txt Dockerfile system_prompt.txt app deepsearch
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf
rm -rf ai-deepsearch.zip
zip -r ai-deepsearch.zip app deepsearch requirements.txt Dockerfile system_prompt.txt
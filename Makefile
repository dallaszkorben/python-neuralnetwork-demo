test: init
	python -m unittest discover -s test

init:
	pip install -r requirements.txt


.PHONY: init test
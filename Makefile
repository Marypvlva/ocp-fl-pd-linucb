PYTHON ?= .venv/bin/python

.PHONY: doctor test smoke study index
doctor:
	$(PYTHON) project.py doctor
test:
	$(PYTHON) project.py test
smoke:
	$(PYTHON) project.py smoke
study:
	$(PYTHON) project.py study
index:
	$(PYTHON) project.py index

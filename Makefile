
VIRTUALENV_DIR=${PWD}/env
PIP=${VIRTUALENV_DIR}/bin/pip3
POETRY=${VIRTUALENV_DIR}/bin/poetry
PYTHON=${VIRTUALENV_DIR}/bin/python

all: install-dev install-commit-hooks

prod: install-prod

virtualenv:
	if [ ! -e ${PIP} ]; then python3 -m venv ${VIRTUALENV_DIR}; fi
	${PIP} install --upgrade pip wheel

install-poetry: virtualenv
	${PIP} install poetry==1.0.5
	${POETRY} config virtualenvs.create false
	${POETRY} config virtualenvs.in-project true

install-prod: install-poetry
	${POETRY} install -vvv --no-dev

install-dev: install-poetry
	${POETRY} install -vvv

install-commit-hooks: install-dev
	if [ ! -d ".git" ]; then git init; fi
	${VIRTUALENV_DIR}/bin/pre-commit install -t pre-commit

install-test: install-poetry
	${PIP} install tensorflow==2.*

test: install-test
	${VIRTUALENV_DIR}/bin/pytest tests --cov src/ --cov-report=xml\:coverage.xml --cov-report term-missing -vvv

clean:
	rm -f .coverage
	find . -name '*.pyc' -exec rm -f {} \;
	find . -name '*.pyo' -exec rm -f {} \;
	find . -depth -name '__pycache__' -exec rm -rf {} \;

dist-clean: clean
	rm -rf ${VIRTUALENV_DIR}
	rm -rf dist
	find . -depth -name '*.egg-info' -exec rm -rf {} \;

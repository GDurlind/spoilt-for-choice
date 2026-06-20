PYTHON_VERSION = 3.14.6
POETRY_VERSION = 2.4.1
REPO_NAME = spoilt-for-choice
VENV_DIR = .venv

init:
	pyenv install $(PYTHON_VERSION) --skip-existing
	pyenv local $(PYTHON_VERSION)
	pyenv exec python -m venv $(VENV_DIR)

clean: 
	find . -type d -name .pytest_cache | xargs --no-run-if-empty -t rm -r
	find . -type d -name __pycache__ | xargs --no-run-if-empty -t rm -r
	find . -type d -name dist | xargs --no-run-if-empty -t rm -r
	find . -type d -regex ".*\.egg.*"  | xargs --no-run-if-empty -t rm -r

reqs: clean
	poetry export --without-hashes -f requirements.txt -o requirements.txt --without dev --without-urls

install: 
	poetry install -vv

test:
	poetry run pytest -s tests # --without-integration # need a test to be marked with @pytest.mark.integration_test to be valid

lock: 
	poetry lock -vv

run: 
	python -m options_engine.main

app:
	streamlit run app.py
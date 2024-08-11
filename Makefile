# === Constants ===

# the source directories which are being checked
SRC_DIRS = ./chemotools ./tests

# === Package and Dependencies ===

# Upgrading pip, setuptools and wheel
.PHONY: upgrade-pip
upgrade-pip:
	@echo Upgrading pip, setuptools and wheel ...
	python -m pip install --upgrade pip setuptools wheel

# Installing the required dependencies and building the package
.PHONY: install
install: upgrade-pip
	@echo Installing the required dependencies and building the package ...
	python -m pip install --upgrade . -r requirements.txt

.PHONY: install-dev
install-dev: upgrade-pip
	@echo Installing the required dependencies and building the package for development ...
	python -m pip install --upgrade . -r requirements.txt -r requirements-dev.txt

.PHONY: install-ci
install-ci: upgrade-pip
	@echo Installing the required dependencies for CI ...
	python -m pip install --upgrade . -r requirements.txt -r requirements-dev.txt

# Building the package
.PHONY: build
build:
	@echo Building the package ...
	python -m build

# === Source File Checks ===

# Checking the source files with flake8
.PHONY: lint-flake8
lint-flake8:
	@echo Checking the source files with flake8 ...
	flake8 $(SRC_DIRS) --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 $(SRC_DIRS) --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# === Test Commands ===

# Running a single test
.PHONY: test
test:
	@echo Running specific test with pytest ...
	pytest -k "$(TEST)" -x

# Running the tests
.PHONY: test-htmlcov
test-htmlcov:
	@echo Running the tests with HTML coverage report ...
	pytest --cov=chemotools .\tests -n="auto" --cov-report=html -x

.PHONY: test-xmlcov
test-xmlcov:
	@echo Running the tests with XML coverage report ...
	pytest --cov=chemotools .\tests -n="auto" --cov-report=html -x
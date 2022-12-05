#* Variables
PYTHON := python3
PYTHONPATH := `pwd`
AUTOFLAKE8_ARGS := -r --exclude '__init__.py' --keep-pass-after-docstring

#* Poetry
.PHONY: poetry-download
poetry-download:
	curl -sSL https://install.python-poetry.org/ | $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://install.python-poetry.org/ | $(PYTHON) - --uninstall

#* Installation
.PHONY: install
install:
	poetry install
	# uninstall serial h5py coming from sopht-backend
	pip uninstall -y h5py
	# install parallel h5py
	HDF5_MPI="ON" CC=mpicc pip install --no-binary=h5py h5py
	# install mpi4py-fft
	pip install mpi4py-fft
	# sadly pip ffmpeg doesnt work, hence we use conda for ffmpeg
	conda install -c conda-forge ffmpeg

.PHONY: install_non_python_modules_on_ubuntu
install_non_python_modules_on_ubuntu:
	sudo apt install -y -q openmpi-bin libopenmpi-dev
	sudo apt install libhdf5-mpi-dev
	conda install -c conda-forge fftw

.PHONY: install_non_python_modules_on_macos
install_non_python_modules_on_macos:
	brew install openmpi
	brew install hdf5-mpi
	conda install -c conda-forge fftw

.PHONY: install_with_new_dependency
install_with_new_dependency:
	poetry lock
	poetry install

.PHONY: pre-commit-install
pre-commit-install:
	poetry run pre-commit install

#* Formatters
.PHONY: black
black:
	poetry run black --version
	poetry run black --config pyproject.toml sopht_mpi tests

.PHONY: black-check
black-check:
	poetry run black --version
	poetry run black --diff --check --config pyproject.toml sopht_mpi tests

.PHONY: flake8
flake8:
	poetry run flake8 --version
	poetry run flake8 sopht_mpi tests

.PHONY: autoflake8-check
autoflake8-check:
	poetry run autoflake8 --version
	poetry run autoflake8 $(AUTOFLAKE8_ARGS) sopht_mpi tests
	poetry run autoflake8 --check $(AUTOFLAKE8_ARGS) sopht_mpi tests

.PHONY: autoflake8-format
autoflake8-format:
	poetry run autoflake8 --version
	poetry run autoflake8 --in-place $(AUTOFLAKE8_ARGS) sopht_mpi tests

.PHONY: format-codestyle
format-codestyle: black flake8

.PHONY: check-codestyle
check-codestyle: black-check flake8 autoflake8-check

.PHONY: formatting
formatting: format-codestyle

.PHONY: test
test:
	poetry run mpiexec -n 4 pytest --cache-clear --reruns 2 --with-mpi

.PHONY: update-dev-deps
update-dev-deps:
	poetry add -D pytest@latest coverage@latest pytest-html@latest pytest-cov@latest black@latest

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove ipynbcheckpoints-remove pytestcache-remove

all: format-codestyle cleanup

ci: check-codestyle

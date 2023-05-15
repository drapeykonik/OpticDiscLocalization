initialize_git:
	@echo "Initializing git..."
	git init 
	
install: 
	@echo "Installing..."
	poetry install
	poetry run mim install mmengine
	poetry run mim install "mmcv>=2.0.0"
	cd src/ssd/mmdetection
	poetry run pip install -v -e .
	cd ../../..
	poetry run pre-commit install

activate:
	@echo "Activating virtual environment"
	poetry shell

download_data: src/data/download.py
	@echo "Downloading data..."
	poetry run python src/data/download.py

setup: initialize_git install download_data

process_data: data/raw/localization src/data/process.py
	@echo "Processing data..."
	poetry run python src/data/process.py

data/processed/localization: data/raw/localization src/data/process.py
	poetry run python src/data/process.py

run: data/processed/localization
	poetry run python src/run.py ${model} ${experiment}
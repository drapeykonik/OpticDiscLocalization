initialize_git:
	@echo "Initializing git..."
	git init 
	
install: 
	@echo "Installing..."
	poetry install
	poetry run pre-commit install

activate:
	@echo "Activating virtual environment"
	poetry shell

download_data: src/download.py
	@echo "Downloading data..."
	poetry run python src/download.py

setup: initialize_git install download_data

process_data: data/raw/localization src/process.py
	@echo "Processing data..."
	poetry run python src/process.py

data/processed/localization: data/raw/localization src/process.py
	poetry run python src/process.py
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
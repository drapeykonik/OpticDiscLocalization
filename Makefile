initialize_git:
	@echo "Initializing git..."
	git init 
	
install: 
	@echo "Installing..."
	poetry install
	poetry run mim install mmengine
	poetry run mim install "mmcv>=2.0.0"
	cd mmdetection
	poetry run pip install -v -e .
	cd ..
	poetry run pre-commit install

activate:
	@echo "Activating virtual environment"
	poetry shell

download_data: src/data/download.py
	@echo "Downloading data..."
	poetry run python src/data/download.py

setup: initialize_git install download_data

clear: 
	find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf

process_data: data/raw/localization src/data/process.py
	@echo "Processing data..."
	poetry run python src/data/process.py

data/processed/localization: data/raw/localization src/data/process.py
	poetry run python src/data/process.py

run: data/processed/localization
	poetry run python src/run.py ${model} ${experiment}

evaluate: 
	poetry run python src/evaluate.py ${model} ${experiment} ${image}

ssd_checkpoint: 
	mkdir -p src/ssd/mmdetection/checkpoints
	wget -O src/ssd/mmdetection/checkpoints/ssd512.pth https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20210803_022849-0a47a1ca.pth

coco_annotations: data/processed/localization
	poetry run python src/ssd/dataset.py
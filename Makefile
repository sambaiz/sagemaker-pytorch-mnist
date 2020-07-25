download_datasets:
	python download_datasets.py

upload_datasets: download_datasets
	python upload_datasets.py

local_run: download_datasets
	python training.py
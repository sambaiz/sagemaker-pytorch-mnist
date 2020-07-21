download_datasets:
	python download_datasets.py

local_run: download_datasets
	python main.py
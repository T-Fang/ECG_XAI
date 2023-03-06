
install: requirements.txt
	conda create --name ecg_xai --file requirements.txt
	conda activate ecg_xai

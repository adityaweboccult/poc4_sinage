# POC4(Sinage)

## Create an environment
```sh
conda create -n poc4_sinage python==3.10.12
conda activate poc4_sinage
```

Install libraries

```sh
pip install -r requirements.txt
```

Install onnruntime for GPU
```sh
pip install onnxruntime-gpu
```

## Download all the necessary models
```sh
python gdrive_download_models.py
```

## Setup for tracking
```sh
cd bot_sort/cython_bbox
python setup.py install
```

```sh
cd bot_sort
python setup.py install
```

Change the video path in the `main.py`

Run the main file `main.py`
```sh
python main.py
```

import gdown
import os
file_url = 'https://drive.google.com/uc?id='

folder = "models/"

if not os.path.exists(folder):
    os.makedirs(folder)

botsort_model_path = os.path.join(folder, "mot20_sbs_S50_botsort.onnx")
person_face_model_path = os.path.join(folder, "yolov8x_person_face.onnx")
age_gender_model_path = os.path.join(folder, "modified_mivolo_age_gender.onnx")


if not os.path.exists(botsort_model_path):
    file = gdown.download(file_url+"1ZH-ss1X5ubDqDBxWzVB9Y2hkyyKzgxiK", botsort_model_path)

if not os.path.exists(person_face_model_path):
    file = gdown.download(file_url+"151PskmidYxWggEKadas-NwdwJ9YxeFcY", person_face_model_path)

if not os.path.exists(age_gender_model_path):
    file = gdown.download(file_url+"1EAVTcU2sV9MHRq9FixTJjhQmlE5KOYcJ", age_gender_model_path)






# file = gdown.download(file_url+"1kyR06elzlbilqHe2rOLxGbnSCJmJ4-1K", "onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl")
# file = gdown.download(file_url+"1FPnuT4TUg3EcBlxxNdVeD4rPuva6GD_W", "tensorrt-8.5.2.2-cp38-none-linux_aarch64.whl")
# file = gdown.download(file_url+"1-f0MxBTQhPWSvb91IwQ1W2w4EbiMp8yi",  "torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl")
# file = gdown.download(file_url+"1UXJTlRTckNIbHhh3wC3NieLHgal4CdxZ", "onnxruntime-1.16.0-cp38-cp38-linux_aarch64.whl")

# https://drive.google.com/file/d/18yn2SjZWgEm_T9KynxDP2qo_VDHijYR1/view?usp=sharing wot_age_gender.onnx
# https://drive.google.com/file/d/1Dww9Zs-VSQJoIeERV3oQCvAXMEVyUE5V/view?usp=sharing model-small.onnx
# file = gdown.download(file_url+"1Dww9Zs-VSQJoIeERV3oQCvAXMEVyUE5V", os.path.join(folder, "model-small.onnx"))
# file = gdown.download(file_url+"18yn2SjZWgEm_T9KynxDP2qo_VDHijYR1", os.path.join(folder, "wot_age_gender.onnx"))
# https://drive.google.com/file/d/17oS8WpB77pW1X0aajoDMcFqGEqjrQtRe/view?usp=sharing model-f6b98070.onnx
# file = gdown.download(file_url+"17oS8WpB77pW1X0aajoDMcFqGEqjrQtRe", os.path.join(folder, "model-f6b98070.onnx"))


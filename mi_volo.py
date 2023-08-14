import logging
from typing import Optional

import numpy as np
import torch
from misc import prepare_classification_images
from structures import PersonAndFaceCrops, PersonAndFaceResult
import onnxruntime
from easydict import EasyDict as edict
# from mivolo.model.create_timm_model import create_model
# from timm.data import resolve_data_config


_logger = logging.getLogger("MiVOLO")
has_compile = hasattr(torch, "compile")


class Meta:
    def __init__(self):
        self.min_age = None
        self.max_age = None
        self.avg_age = None
        self.num_classes = None

        self.in_chans = 3
        self.with_persons_model = False
        self.disable_faces = False
        self.use_persons = True
        self.only_age = False

        self.num_classes_gender = 2

    def load_from_ckpt(self, ckpt_path: str, disable_faces: bool = False, use_persons: bool = True) -> "Meta":

        state = torch.load(ckpt_path, map_location="cpu")

        self.min_age = state["min_age"]
        self.max_age = state["max_age"]
        self.avg_age = state["avg_age"]
        self.only_age = state["no_gender"]

        only_age = state["no_gender"]

        self.disable_faces = disable_faces
        if "with_persons_model" in state:
            self.with_persons_model = state["with_persons_model"]
        else:
            self.with_persons_model = True if "patch_embed.conv1.0.weight" in state["state_dict"] else False

        self.num_classes = 1 if only_age else 3
        self.in_chans = 3 if not self.with_persons_model else 6
        self.use_persons = use_persons and self.with_persons_model

        if not self.with_persons_model and self.disable_faces:
            raise ValueError("You can not use disable-faces for faces-only model")
        if self.with_persons_model and self.disable_faces and not self.use_persons:
            raise ValueError("You can not disable faces and persons together")

        return self

    def __str__(self):
        attrs = vars(self)
        attrs.update({"use_person_crops": self.use_person_crops, "use_face_crops": self.use_face_crops})
        return ", ".join("%s: %s" % item for item in attrs.items())

    @property
    def use_person_crops(self) -> bool:
        return self.with_persons_model and self.use_persons

    @property
    def use_face_crops(self) -> bool:
        return not self.disable_faces or not self.with_persons_model



#Mivolo Class for the onnx model 
class MiVOLO:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        half: bool = True,
        disable_faces: bool = False,
        use_persons: bool = True,
        verbose: bool = False,
        torchcompile: Optional[str] = None,
        DEBUG : bool=False
    ):  
        self.verbose = verbose
        self.device = torch.device(device)
        self.half = half and self.device.type != "cpu"

        meta = {'min_age': 1, 'max_age': 95, 'avg_age': 48.0, 'num_classes': 3, 'in_chans': 3, 'with_persons_model': False, 'disable_faces': False,
                'use_persons': False, 'only_age': False, 'num_classes_gender': 2, 'use_person_crops': False, 'use_face_crops': True}
        self.meta = edict(meta)
        if self.verbose:
            _logger.info(f"Model meta:\n{str(self.meta)}")
        print("Runnning in the Mivolo onnx")
        model_name = "mivolo_d1_224"
        providers = ["TensorrtExecutionProvider"]
        if str(device) == "cpu":
            providers.append("CPUExecutionProvider")
            
        else:
            providers.append("CUDAExecutionProvider")
        # providers = ["TensorrtExecutionProvider","",]
        if DEBUG:
            print("ONNX Runtime device in MiVOLO",onnxruntime.get_device())
            print("Providers in MiVOLO",providers)
        # self.model = create_model(
        #     model_name=model_name,
        #     num_classes=self.meta.num_classes,
        #     in_chans=self.meta.in_chans,
        #     pretrained=False,
        #     checkpoint_path=ckpt_path,
        #     filter_keys=["fds."],
        # )
        self.session = onnxruntime.InferenceSession(ckpt_path,providers = providers,verbose=True) #Onnx Session

        #self.param_count = sum([m.numel() for m in self.model.parameters()])
        # _logger.info(f"Model {model_name} created, param count: {self.param_count}")

        # self.data_config = resolve_data_config(
        #     model=self.model,
        #     verbose=verbose,
        #     use_test_size=True,
        # )
        #Data Config of the data
        data_config = {'input_size': (3, 224, 224), 
                       'interpolation': 'bicubic', 
                       'mean': (0.485, 0.456, 0.406), 
                       'std': (0.229, 0.224, 0.225),
                       'crop_pct': 0.96, 'crop_mode': 'center'}
        self.data_config = edict(data_config)
        
        if DEBUG:
            print(f'data_config in the Mivolo class = ',self.data_config)
        self.data_config["crop_pct"] = 1.0
        c, h, w = self.data_config["input_size"]
        assert h == w, "Incorrect data_config"
        self.input_size = w

        
        
        

    def warmup(self, batch_size: int, steps=10):
        if self.meta.with_persons_model:
            input_size = (6, self.input_size, self.input_size)
        else:
            input_size = self.data_config["input_size"]

        input = torch.randn((batch_size,) + tuple(input_size)).to(self.device)

        for _ in range(steps):
            out = self.inference(input)  # noqa: F841

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def inference(self, model_input: torch.tensor) -> torch.tensor:

        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        model_input = model_input.cpu().data.numpy()
        
        output_nodes= [node.name for node in self.session.get_outputs()]
        outputs = self.session.run(None, {'input.1': model_input})
        gemm_result = outputs[0]
        add_result = outputs[1]
        reduce_max = np.max(add_result, axis=1, keepdims=False)
        mul = np.multiply(reduce_max, 0.5)
        add = np.add(mul, gemm_result)

        
        return add

    def custom_predict(self, image: np.ndarray, detected_bboxes: PersonAndFaceResult):
        if detected_bboxes.n_objects == 0:
            return 

        faces_input, person_input, faces_inds, bodies_inds,faces_crops = self.prepare_crops(image, detected_bboxes)
        ages = []
        gender_indx = []
        # if len(faces_crops) > 0:            
        if self.meta.with_persons_model:
            model_input = torch.cat((faces_input, person_input), dim=1)
        else:
            model_input = faces_input
        #print(model_input.shape[0])
        num_batches = model_input.shape[0]
        output = []
        for batch in range(num_batches):
            b = model_input[batch:batch+1] 
            output_batch = self.inference(b).tolist()

            output.append(output_batch)

        output = np.array(output) #converting the list to numpy 

        output = output.reshape(num_batches, -1)  # Reshape to match the desired shape (4, 3)
        if self.device!='cpu':
            output = torch.from_numpy(output)
        
        #print(output)
        
        #print("Reshaped output:\n", output)
        
        #output = self.inference(model_input)

        if self.meta.only_age:
            age_output = output
            gender_probs, gender_indx = None, None
        else:
            age_output = output[:, 2]
            output = output[:,:2]
            if isinstance(output, torch.Tensor):
                gender_output = output[:, :2].softmax(-1)
                gender_probs, gender_indx = gender_output.topk(1)
                
            else:
                gender_output = np.exp(output - np.max(output, axis=1, keepdims=True))
                gender_output /= np.sum(gender_output, axis=1, keepdims=True)
            
                
                gender_probs = np.max(gender_output, axis=1, keepdims=True)
                gender_indx = np.argmax(gender_output, axis=1,keepdims=True)
            
            # print(gender_probs)
            # print(gender_indx)
            
           
           
             
        assert output.shape[0] == len(faces_inds) == len(bodies_inds)

        
        # per face
        for index in range(output.shape[0]):
            face_ind = faces_inds[index]
            body_ind = bodies_inds[index]

            # get_age
            age = age_output[index].item()
            age = age * (self.meta.max_age - self.meta.min_age) + self.meta.avg_age
            age = round(age, 2)
            ages.append(age)

            # print("faces input",faces_crops)
            # print("Length of faces",len(faces_crops))
            # print("faces_inds",faces_inds)

        # print("Age output",ages)
        # print("Gender outputs",gender_output)
        # print("Gender probs",gender_probs)
        # print("Gender indx",gender_indx)
        
        #print(ages)
        
        return ages,gender_indx

        # write gender and age results into detected_bboxes
        self.fill_in_results(output, detected_bboxes, faces_inds, bodies_inds)

    def predict(self, image: np.ndarray, detected_bboxes: PersonAndFaceResult):
        if detected_bboxes.n_objects == 0:
            return

        faces_input, person_input, faces_inds, bodies_inds = self.prepare_crops(image, detected_bboxes)

        if self.meta.with_persons_model:
            model_input = torch.cat((faces_input, person_input), dim=1)
        else:
            model_input = faces_input
        output = self.inference(model_input)

        # write gender and age results into detected_bboxes
        self.fill_in_results(output, detected_bboxes, faces_inds, bodies_inds)

    def fill_in_results(self, output, detected_bboxes, faces_inds, bodies_inds):
        if self.meta.only_age:
            age_output = output
            gender_probs, gender_indx = None, None
        else:
            age_output = output[:, 2]
            gender_output = output[:, :2].softmax(-1)
            gender_probs, gender_indx = gender_output.topk(1)

        assert output.shape[0] == len(faces_inds) == len(bodies_inds)

        # per face
        for index in range(output.shape[0]):
            face_ind = faces_inds[index]
            body_ind = bodies_inds[index]

            # get_age
            age = age_output[index].item()
            age = age * (self.meta.max_age - self.meta.min_age) + self.meta.avg_age
            age = round(age, 2)

            detected_bboxes.set_age(face_ind, age)
            detected_bboxes.set_age(body_ind, age)

            _logger.info(f"\tage: {age}")

            if gender_probs is not None:
                gender = "male" if gender_indx[index].item() == 0 else "female"
                gender_score = gender_probs[index].item()

                _logger.info(f"\tgender: {gender} [{int(gender_score * 100)}%]")

                detected_bboxes.set_gender(face_ind, gender, gender_score)
                detected_bboxes.set_gender(body_ind, gender, gender_score)

    def prepare_crops(self, image: np.ndarray, detected_bboxes: PersonAndFaceResult):

        if self.meta.use_person_crops and self.meta.use_face_crops:
            detected_bboxes.associate_faces_with_persons()

        crops: PersonAndFaceCrops = detected_bboxes.collect_crops(image)
        (bodies_inds, bodies_crops), (faces_inds, faces_crops) = crops.get_faces_with_bodies(
            self.meta.use_person_crops, self.meta.use_face_crops
        )

        if not self.meta.use_face_crops:
            assert all(f is None for f in faces_crops)

        # faces_input = None
        # if len(faces_crops) > 0:
        #     print(len(faces_crops))
        faces_input = prepare_classification_images(
            faces_crops, self.input_size, self.data_config["mean"], self.data_config["std"], device=self.device
        )

        if not self.meta.use_person_crops:
            assert all(p is None for p in bodies_crops)

        person_input = prepare_classification_images(
            bodies_crops, self.input_size, self.data_config["mean"], self.data_config["std"], device=self.device
        )

        _logger.info(
            f"faces_input: {faces_input.shape if faces_input is not None else None}, "
            f"person_input: {person_input.shape if person_input is not None else None}"
        )

        return faces_input, person_input, faces_inds, bodies_inds,faces_crops




#Code of the MiVOLO 


# class MiVOLO:
#     def __init__(
#         self,
#         ckpt_path: str,
#         device: str = "cuda",
#         half: bool = True,
#         disable_faces: bool = False,
#         use_persons: bool = True,
#         verbose: bool = False,
#         torchcompile: Optional[str] = None,
#     ):
#         self.verbose = verbose
#         self.device = torch.device(device)
#         self.half = half and self.device.type != "cpu"

#         self.meta: Meta = Meta().load_from_ckpt(ckpt_path, disable_faces, use_persons)
        
        
        
        
#         print(self.meta)
        
#         if self.verbose:
#             _logger.info(f"Model meta:\n{str(self.meta)}")

#         model_name = "mivolo_d1_224"
#         self.model = create_model(
#             model_name=model_name,
#             num_classes=self.meta.num_classes,
#             in_chans=self.meta.in_chans,
#             pretrained=False,
#             checkpoint_path=ckpt_path,
#             filter_keys=["fds."],
#         )
      
#         self.param_count = sum([m.numel() for m in self.model.parameters()])
#         _logger.info(f"Model {model_name} created, param count: {self.param_count}")

#         self.data_config = resolve_data_config(
#             model=self.model,
#             verbose=verbose,
#             use_test_size=True,
#         )
      
        
        
#         self.data_config["crop_pct"] = 1.0
#         c, h, w = self.data_config["input_size"]
#         assert h == w, "Incorrect data_config"
#         self.input_size = w

#         self.model = self.model.to(self.device)
#         model = self.model
        
#         if torchcompile:
#             assert has_compile, "A version of torch w/ torch.compile() is required for --compile, possibly a nightly."
#             torch._dynamo.reset()
#             self.model = torch.compile(self.model, backend=torchcompile)

#         self.model.eval()
#         if self.half:
#             self.model = self.model.half()

#     def warmup(self, batch_size: int, steps=10):
#         if self.meta.with_persons_model:
#             input_size = (6, self.input_size, self.input_size)
#         else:
#             input_size = self.data_config["input_size"]

#         input = torch.randn((batch_size,) + tuple(input_size)).to(self.device)

#         for _ in range(steps):
#             out = self.inference(input)  # noqa: F841

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()

#     def inference(self, model_input: torch.tensor) -> torch.tensor:

#         with torch.no_grad():
#             if self.half:
#                 model_input = model_input.half()
#             print("model input shape",model_input.size())
#             output = self.model(model_input)
#             print('Output of the mivolo model',output)
#             print("Output of the mivolo model",type(output))
            
#         return output

#     def custom_predict(self, image: np.ndarray, detected_bboxes: PersonAndFaceResult):
#         if detected_bboxes.n_objects == 0:
#             return

#         faces_input, person_input, faces_inds, bodies_inds,faces_crops = self.prepare_crops(image, detected_bboxes)
#         ages = []
#         gender_indx = []
#         # if len(faces_crops) > 0:            
#         if self.meta.with_persons_model:
#             model_input = torch.cat((faces_input, person_input), dim=1)
#         else:
#             model_input = faces_input
#         output = self.inference(model_input)

#         if self.meta.only_age:
#             age_output = output
#             gender_probs, gender_indx = None, None
#         else:
#             age_output = output[:, 2]
#             gender_output = output[:, :2].softmax(-1)
#             gender_probs, gender_indx = gender_output.topk(1)

#         assert output.shape[0] == len(faces_inds) == len(bodies_inds)

        
#         # per face
#         for index in range(output.shape[0]):
#             face_ind = faces_inds[index]
#             body_ind = bodies_inds[index]

#             # get_age
#             age = age_output[index].item()
#             age = age * (self.meta.max_age - self.meta.min_age) + self.meta.avg_age
#             age = round(age, 2)
#             ages.append(age)

#             # print("faces input",faces_crops)
#             # print("Length of faces",len(faces_crops))
#             # print("faces_inds",faces_inds)

#         # print("Age output",ages)
#         # print("Gender outputs",gender_output)
#         # print("Gender probs",gender_probs)
#         # print("Gender indx",gender_indx)

#         return ages,gender_indx

#         # write gender and age results into detected_bboxes
#         self.fill_in_results(output, detected_bboxes, faces_inds, bodies_inds)

#     def predict(self, image: np.ndarray, detected_bboxes: PersonAndFaceResult):
#         if detected_bboxes.n_objects == 0:
#             return

#         faces_input, person_input, faces_inds, bodies_inds = self.prepare_crops(image, detected_bboxes)

#         if self.meta.with_persons_model:
#             model_input = torch.cat((faces_input, person_input), dim=1)
#         else:
#             model_input = faces_input
#         output = self.inference(model_input)

#         # write gender and age results into detected_bboxes
#         self.fill_in_results(output, detected_bboxes, faces_inds, bodies_inds)

#     def fill_in_results(self, output, detected_bboxes, faces_inds, bodies_inds):
#         if self.meta.only_age:
#             age_output = output
#             gender_probs, gender_indx = None, None
#         else:
#             age_output = output[:, 2]
#             gender_output = output[:, :2].softmax(-1)
#             gender_probs, gender_indx = gender_output.topk(1)

#         assert output.shape[0] == len(faces_inds) == len(bodies_inds)

#         # per face
#         for index in range(output.shape[0]):
#             face_ind = faces_inds[index]
#             body_ind = bodies_inds[index]

#             # get_age
#             age = age_output[index].item()
#             age = age * (self.meta.max_age - self.meta.min_age) + self.meta.avg_age
#             age = round(age, 2)

#             detected_bboxes.set_age(face_ind, age)
#             detected_bboxes.set_age(body_ind, age)

#             _logger.info(f"\tage: {age}")

#             if gender_probs is not None:
#                 gender = "male" if gender_indx[index].item() == 0 else "female"
#                 gender_score = gender_probs[index].item()

#                 _logger.info(f"\tgender: {gender} [{int(gender_score * 100)}%]")

#                 detected_bboxes.set_gender(face_ind, gender, gender_score)
#                 detected_bboxes.set_gender(body_ind, gender, gender_score)

#     def prepare_crops(self, image: np.ndarray, detected_bboxes: PersonAndFaceResult):

#         if self.meta.use_person_crops and self.meta.use_face_crops:
#             detected_bboxes.associate_faces_with_persons()

#         crops: PersonAndFaceCrops = detected_bboxes.collect_crops(image)
#         (bodies_inds, bodies_crops), (faces_inds, faces_crops) = crops.get_faces_with_bodies(
#             self.meta.use_person_crops, self.meta.use_face_crops
#         )

#         if not self.meta.use_face_crops:
#             assert all(f is None for f in faces_crops)

#         # faces_input = None
#         # if len(faces_crops) > 0:
#         #     print(len(faces_crops))
#         faces_input = prepare_classification_images(
#             faces_crops, self.input_size, self.data_config["mean"], self.data_config["std"], device=self.device
#         )

#         if not self.meta.use_person_crops:
#             assert all(p is None for p in bodies_crops)

#         person_input = prepare_classification_images(
#             bodies_crops, self.input_size, self.data_config["mean"], self.data_config["std"], device=self.device
#         )

#         _logger.info(
#             f"faces_input: {faces_input.shape if faces_input is not None else None}, "
#             f"person_input: {person_input.shape if person_input is not None else None}"
#         )

#         return faces_input, person_input, faces_inds, bodies_inds,faces_crops


if __name__ == "__main__":
    model = MiVOLO("../pretrained/checkpoint-377.pth.tar", half=True, device="cuda:0")

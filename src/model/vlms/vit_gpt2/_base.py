import time
import cv2
import yaml
import os
import sys
from os.path import join as pjoin
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
print(dir(yaml))

def checkSysPathAndAppend(path, stepBack=0):
  if stepBack > 0:
    for istep in range(stepBack):
      if istep == 0:
        pathStepBack = path
      pathStepBack, filename = os.path.split(pathStepBack)
  else:
    pathStepBack = path

  if not pathStepBack in sys.path:
    sys.path.append(pathStepBack)

  return pathStepBack

folderFile, filename = os.path.split(os.path.realpath(__file__))
FOLDER_PROJECT = checkSysPathAndAppend(folderFile, 4)
FOLDER_CONFIG = os.path.join(FOLDER_PROJECT, 'config')
PATH_CONFIG = pjoin(FOLDER_PROJECT,"src","model","vlms","vit_gpt2","config.yaml")

# sys.path.append(folderFile)
class VlmsVit:
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None

        self.__get_config()
        self.__init_model()
        self.init_param()
        self.init_output()


    def greeting(self):
        print("Hello vlms vits")

    def __get_config(self):
        print("loading config...")
        with open(PATH_CONFIG, "r") as file:
            self.package_info = yaml.safe_load(file)

    def __init_model(self):
        print("loading model...")
        self.name = self.package_info['name']
        vit_path = self.package_info['model_path']
        self.processor =  ViTImageProcessor.from_pretrained(vit_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(vit_path)
        self.tokenizer = AutoTokenizer.from_pretrained(vit_path)
        self.model.config.pad_token_id = self.model.config.eos_token_id

        self.max_length = 16
        self.num_beams = 4
        self.length_penalty = 1.2
        self.gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams, "length_penalty": self.length_penalty}

    def deinit_model(self):
        if self.is_need_deinit_model:
            print("deinitial model...")
            self.model = None

    def reinit_model(self):
        if self.model is None:
            self.__init_model()

    def init_param(self):
        self.t_start = 0
        self.t_predict = 0
        self.is_need_deinit_model = False

    def init_output(self):
        self.results_post = {
            "prompt": None,
            "response": None
        }

    def set_deinit_model(self, deinit_model):
        self.is_need_deinit_model = deinit_model

        if not self.is_need_deinit_model:
            # reload model and not deinit after that
            self.reinit_model()

    def __preprocessing(self):
        self.w_img = self.img.shape[1]
        self.h_img = self.img.shape[0]
        buff_img = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        buff_img = Image.fromarray(buff_img)
        self.img_processed = buff_img
        self.reinit_model()

    def __inference(self, run_async=False):
        print('inferencing...')
        inputs = self.processor(self.img_processed, return_tensors="pt", size=224)
        pixel_values = inputs["pixel_values"]
        attention_mask = torch.ones(pixel_values.shape[:2], dtype=torch.long)
        out = self.model.generate(pixel_values, attention_mask=attention_mask,
                                  **self.gen_kwargs)
        preds = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        caption = [pred.strip() for pred in preds]

        self.results = {
            "input": inputs,
            "out": out,
            "caption": caption[0] if caption else ""
        }

    def __postprocessing(self):
        self.deinit_model()
        self.results_post['response'] = self.results['caption']


    def predict(self, img, preprocess=None, run_async=False):
        print("predicting...")
        self.t_start = time.perf_counter()
        self.img = img
        self.preprocess = preprocess

        self.init_output()
        self.__preprocessing()
        self.__inference(run_async=run_async)
        self.__postprocessing()

        return self.results_post

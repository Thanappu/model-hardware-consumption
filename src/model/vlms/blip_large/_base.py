import time
import cv2
import yaml
import os
import sys
from os.path import join as pjoin
from transformers import BlipProcessor, BlipForConditionalGeneration

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
PATH_CONFIG = pjoin(FOLDER_PROJECT,"src","model","vlms","blip_large","config.yaml")

# sys.path.append(folderFile)
class VlmsBlipLarge:
    def __init__(self):
        self.__get_config()
        self.__init_model()
        self.init_param()
        self.init_output()
        self.model = None
        self.processor = None

    def greeting(self):
        print("Hello vlms")

    def __get_config(self):
        print("loading config...")
        with open(PATH_CONFIG, "r") as file:
            self.package_info = yaml.safe_load(file)


    def __init_model(self):
        print("loading model...")

        blip_path = pjoin(FOLDER_PROJECT,"models",self.package_info['model_path'])
        self.processor = BlipProcessor.from_pretrained(blip_path)
        self.model = BlipForConditionalGeneration.from_pretrained(blip_path)

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
        # buff_img = cv2.resize(self.img)
        buff_img = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        self.img_processed = buff_img
        self.reinit_model()

    def __inference(self, run_async=False):
        print('inferencing...')
        # self.results_objectdet = self.model.predict(self.img_processed, run_async=run_async)
        inputs = self.processor(self.img_processed, return_tensors=self.package_info['return_tensors'])
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=self.package_info['skip_special_tokens'])

        self.results = {
            "input": inputs,
            "out" : out,
            "caption" : caption
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
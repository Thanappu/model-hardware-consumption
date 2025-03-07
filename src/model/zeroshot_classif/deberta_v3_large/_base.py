import time
import yaml
import os
import sys
from os.path import join as pjoin
from transformers import pipeline

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
PATH_CONFIG = pjoin(FOLDER_PROJECT,"src","model","zeroshot_classif","deberta_v3_large","config.yaml")

# sys.path.append(folderFile)
class ZeroshotDebertaLarge:
    def __init__(self):
        self.__get_config()
        self.__init_model()
        self.init_param()
        self.init_output()
        self.model = None

    def greeting(self):
        print("Hello vlms")

    def __get_config(self):
        print("loading config...")
        with open(PATH_CONFIG, "r") as file:
            self.package_info = yaml.safe_load(file)


    def __init_model(self):
        print("loading model...")
        deberta_path = pjoin(FOLDER_PROJECT,"models",self.package_info['model_path'])
        self.model = pipeline("zero-shot-classification", model=deberta_path)

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
            "Result":None,
            "Labels": None,
            "Scores": None,
            "Output":None,
            "Output_score":None
        }

    def set_deinit_model(self, deinit_model):
        self.is_need_deinit_model = deinit_model

        if not self.is_need_deinit_model:
            # reload model and not deinit after that
            self.reinit_model()

    def __preprocessing(self):
        self.reinit_model()

    def __inference(self, run_async=False):
        print('inferencing...')
        # self.results_objectdet = self.model.predict(self.img_processed, run_async=run_async)
        respond = self.model(self.text_input,self.package_info['Category'])
        self.results = {
            "respond" : respond,
            "labels" : respond['labels'],
            "scores" : respond['scores']
        }
    def __postprocessing(self):
        self.deinit_model()
        self.results_post['Result'] = self.results['respond']
        self.results_post['Labels'] = self.results['labels']
        self.results_post['Scores'] = self.results['scores']

        max_index = self.results_post['Scores'].index(max(self.results_post['Scores']))
        self.results_post['Output'] = self.results_post['Labels'][max_index]
        self.results_post['Output_score'] = self.results_post['Scores'][max_index]

        # if self.results_post['Scores'][0] > self.package_info['decision_ratio']:
        #     self.results_post['Output'] = self.results_post['Labels'][0]
        #     self.results_post['Output_score'] = self.results_post['Scores'][0]
        # else:
        #     self.results_post['Output'] = self.results_post['Labels'][1]
        #     self.results_post['Output_score'] = self.results_post['Scores'][1]


    def predict(self, text_input, preprocess=None, run_async=False):
        print("predicting...")
        self.t_start = time.perf_counter()
        self.text_input = text_input
        self.preprocess = preprocess

        self.init_output()
        self.__preprocessing()
        self.__inference(run_async=run_async)
        self.__postprocessing()

        return self.results_post



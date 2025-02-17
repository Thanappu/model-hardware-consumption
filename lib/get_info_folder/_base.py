import yaml
import os
import sys
from os.path import join as pjoin

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
FOLDER_PROJECT = checkSysPathAndAppend(folderFile, 2)
PATH_CONFIG = pjoin(FOLDER_PROJECT,"lib","get_info_folder","config.yaml")

class GetInfoFolder :
    def __init__(self):
        self.package_info = None
        self.__get_config()
        self.init_param()

    def __get_config(self):
        print("loading config...")
        with open(PATH_CONFIG, "r") as file:
            self.package_info = yaml.safe_load(file)

    def get_model_folder(self,path: str):
        folder_names = []
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_dir() and not entry.name.startswith('.'):
                    folder_names.append(entry.name)
        return folder_names, len(folder_names)

    def init_param(self):
        model_path = self.package_info['model_test_path']
        labels_path = self.package_info['labels_test']
        self.num_model = self.get_model_folder(model_path)[1]
        self.list_model = self.get_model_folder(model_path)[0]
        self.num_class = self.get_model_folder(labels_path)[1]
        self.list_model = self.get_model_folder(labels_path)[0]






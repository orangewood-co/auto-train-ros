import os
import shutil
import cv2
import yaml
import json


class RoboflowBB:
    '''
    Transforms the Roboflow dataset to be used with new data formed

    Args:
        - logger (object instance): Logger instance for adding logs
        - prev_folder (str): Path to local previous folder containing dataset in images and labels folder
        - combined_folder (str): Path to local folder to store the new data
        - json_file (str): Path to inputs.json file used for training new data
        - abs_yaml_file (str): Absolute path to the YAML file
    '''
    def __init__(self, logger, prev_folder, combined_folder, json_file, abs_yaml_file):
        self.logger = logger
        self.prev_folder = prev_folder
        self.combined_folder = combined_folder+"/raw_dataset"
        self.json_file = json_file
        self.abs_yaml_file = abs_yaml_file

    def make_copy_folder(self):
        '''
        Copies files from Roboflow folder(prev_folder) to the combined_folder
        '''
        for item in os.listdir(f"{self.prev_folder}/images"):
            source_path = os.path.join(f"{self.prev_folder}/images", item)
            destination_path = os.path.join(f"{self.combined_folder}/images", item)
            shutil.copy(source_path,destination_path)
            self.logger.info(f"Copied images to {self.combined_folder}")
        for item in os.listdir(f"{self.prev_folder}/labels"):
            source_path = os.path.join(f"{self.prev_folder}/labels", item)
            destination_path = os.path.join(f"{self.combined_folder}/labels", item)
            shutil.copy(source_path,destination_path)
            self.logger.info(f"Copied labels to {self.combined_folder}")

    def drawing_bb(self):
        '''
        Draws Bounding boxes on the roboflow images
        '''
        for image in os.listdir(f"{self.combined_folder}/images"):
            image_path = f"{self.combined_folder}/images/{image}"
            image_cv = cv2.imread(image_path)
            ih,iw = image_cv.shape[:2]
            initials = os.path.splitext(image)[0]
            label = f"{initials}.txt"
            if label in os.listdir(f"{self.combined_folder}/labels"):
                with open(f"{self.combined_folder}/labels/{label}", "r") as fl:
                    bb_list = []
                    label_content = fl.read()
                    label_content = label_content.split("\n")
                    for labels in label_content:
                        if len(labels)!=0:
                            xcn, ycn, wn, hn = [float(i) for i in labels.split(" ")[1:]]
                            xc,yc,w,h = xcn*iw, ycn*ih, wn*iw, hn*ih
                            xmax, xmin, ymax, ymin = int((2*xc+w)/2), int((2*xc-w)/2), int((2*yc+h)/2), int((2*yc-h)/2)
                            bb_ind_list = [xmin, ymin, xmax, ymax]
                        bb_list.append(bb_ind_list)
                    for bb in bb_list:
                        image_sh = cv2.rectangle(image_cv, (bb[0],bb[1]), (bb[2],bb[3]), (0,255,0), 2)
                    
                    cv2.imwrite(image_path,image_sh)
    
    def update_json_from_yaml(self):
        '''
        Updates json_file(i.e. inputs.json) from the YAML file 
        '''
        with open(self.abs_yaml_file, 'r') as file:
            yaml_data = yaml.safe_load(file)
        names_list = yaml_data.get('names', [])
        if type(names_list) is dict:
            with open(self.json_file, 'r') as file:
                data = json.load(file)
            data['candidate_labels']=list(names_list.values())
            with open(self.json_file, 'w') as file:
                json.dump(data, file, indent=4)
        else:
            with open(self.json_file, 'r') as file:
                data = json.load(file)
            data['candidate_labels']=names_list
            with open(self.json_file, 'w') as file:
                json.dump(data, file, indent=4)

    def run(self):
        '''
        Copies and annotates the Roboflow dataset, updating the json(inputs.json) file
        Returns the Roboflow dataset as per the new data configurations
        '''
        self.make_copy_folder()
        self.drawing_bb()
        self.logger.info("Stored combined data \n")
        self.update_json_from_yaml()
        self.logger.info("We have updated json file now \n")

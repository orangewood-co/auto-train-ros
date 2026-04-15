import json
import os
import shutil
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

from .roboflow_bb import RoboflowBB
from .new_data import NewData
from .utils_aug import Augment


class AutoTrain:
    '''
    Trains object detection model(YOLOv8) using real time inference data. It is for automating the supervised learning, specifically cutting out the manual labelling task and training the model for it to remember the object as per the label we want.

    Args:
        - node_instance: ROS 2 Node used for creating topic subscriptions
        - image_topic_name (str): ROS topic publishing sensor_msgs/Image frames
        - data_folder (str): Path to local folder to store the new data
        - prev_data_folder (str): Path to local previous folder containing images and labels folder
        - new_weights (boolean): True if to not use any previous data
        - abs_yaml_file (str): Absolute path to the YAML file for given prev_data_folder
        - draw_bb (boolean): True to draw bounding boxes on previous image dataset
        - image_threshold (int): Number of images to capture for creating new dataset
        - number_aug (int): Number of times to apply augmentations
        - epochs (int): Number of epochs for training
        - map_threshold (float): value<=1 ; Threshold to compare mAP50 score
        - inference (boolean): True to perform the inference on live feed
        - inference_threshold (float): value<=1 ; Threshold for inference confidence score
    '''
    def __init__(self, node_instance, image_topic_name, data_folder, prev_data_folder="", new_weights=True, abs_yaml_file=None, draw_bb=False, image_threshold=100, number_aug=3, epochs=69, map_threshold=0.5, inference=False, inference_threshold=0.4) -> None:

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.propagate = False
        # Clear existing handlers to avoid duplicate logs
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Define a StreamHandler to log messages to the console
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        file_handler = RotatingFileHandler('logger.log', mode='a', maxBytes=10 * 1024 * 1024, backupCount=3)  # 10 MB max size, 3 backups
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.node = node_instance
        self.image_topic = image_topic_name

        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        self.data_folder = data_folder
        self.combined_folder = os.path.join(self.data_folder, f"train_v{self.timestamp}" if not new_weights else f"new_weights_v{self.timestamp}")
        self.prev_data_folder = prev_data_folder
        self.new_weights = new_weights
        self.json_file = f"{self.combined_folder}/inputs.json"
        self.draw_bb = draw_bb
        self.abs_yaml_file = abs_yaml_file
        if not self.new_weights and self.abs_yaml_file==None:
            self.logger.error(f"Input abs_yaml_file if new_weights is False")
            raise ValueError(f"Input abs_yaml_file if new_weights is False")
        self.image_threshold = image_threshold
        self.number_aug = number_aug
        self.epochs = epochs
        self.map_threshold = map_threshold
        self.inference = inference
        self.inference_threshold = inference_threshold

    def prev_data(self):
        '''
        Function returns and stores previous data in YOLOv8 format in raw_dataset.
        '''
        rfbb = RoboflowBB(logger=self.logger, prev_folder=self.prev_data_folder, combined_folder=self.combined_folder, json_file=self.json_file, abs_yaml_file=self.abs_yaml_file)
        if self.draw_bb:
            # copies and draws bb in a combined dataset; updates json file as per the yaml file
            rfbb.run()
        else:
            # copies data in combined dataset; and updates json file as per the yaml file
            rfbb.make_copy_folder()
            self.logger.info("Stored combined data \n")
            rfbb.update_json_from_yaml()
            self.logger.info("We have updated json file now \n")
    
    def augment(self):
        '''
        Function augments the images and labels to store the combined dataset for training in aug_dataset.
        '''
        aug = Augment(logger=self.logger, combined_folder=self.combined_folder, json_file=self.json_file)
        imgs = [img for img in os.listdir(self.combined_folder+"/raw_dataset/images") if aug.is_image_by_extension(img)]

        for img_file in imgs:
            image, gt_bboxes, aug_file_name = aug.get_inp_data(img_file)
            for n in range(self.number_aug):
                aug_img, aug_label = aug.get_augmented_results(image, gt_bboxes)
                aug.store_aug(aug_img, aug_label, f"{aug_file_name}_{n+1}")
        self.logger.info("Augmented and saved dataset")

    def new_data(self, object_name, object_specific):
        '''
        Generates new data for the input object, splits it and creates a YAML file for training
        Trains data to generate new weights file

        Args:
            - object (str): Object to be detected
        Returns:
            - new_weights_path (str): Path to the new '.pt' weights file
        '''
        zsl = NewData(logger=self.logger, node=self.node, image_topic=self.image_topic, combined_folder=self.combined_folder, json_file=self.json_file, object_name=object_name, image_threshold=self.image_threshold, epochs=self.epochs, map_threshold=self.map_threshold, inference=self.inference, inference_threshold=self.inference_threshold)
        # Capture, split and store dataset; create yaml file
        zsl.capture_pred(box_threshold=0.6, text_threshold=0.4)
        self.logger.info("Done capturing frames \n")
        raw_img_dir = os.path.join(self.combined_folder, "raw_dataset", "images")
        captured = [f for f in os.listdir(raw_img_dir) if os.path.isfile(os.path.join(raw_img_dir, f))] if os.path.isdir(raw_img_dir) else []
        if not captured:
            self.logger.error("No images were captured — cannot proceed with training.")
            return None
        # update the json file with new class
        with open(self.json_file, 'r') as file:
            data = json.load(file)
        data['candidate_labels'].pop()
        data['candidate_labels'].append(object_specific)
        with open(self.json_file, 'w') as file:
            json.dump(data, file, indent=4)
        # Augment dataset
        self.augment()
        # Update yaml file
        zsl.split_and_yaml()
        # Train on new yaml file and get the MaP50 scores
        new_weights_path, _ = zsl.train()
        return new_weights_path
    
    def run(self):
        '''
        Complete process to get available cameras, and use that to run autotrain and get new weights file.
        '''
        try:
            # check if raw_dataset folder exists or not
            if not os.path.exists(self.combined_folder):
                os.makedirs(self.combined_folder+"/raw_dataset/images")
                os.makedirs(self.combined_folder+"/raw_dataset/labels")
            else:
                raise IOError(f"{self.combined_folder} already exists. Input new name for folder.")
            # check for existence of json file
            if not os.path.exists(self.json_file):
                with open(self.json_file, "w") as f:
                    json_data = {
                        "candidate_labels": []
                    }
                    json.dump(json_data, f, indent=4)

            # get previous data
            if not self.new_weights:
                self.prev_data()

            # give generic name of object to detect
            object_name = input("What object you want to detect: \n") + "."
            # update the json file with new class
            with open(self.json_file, 'r') as file:
                data = json.load(file)
            data['candidate_labels'].append(object_name)
            with open(self.json_file, 'w') as file:
                json.dump(data, file, indent=4)
            
            # Create new data for object specified; and train it and get the MaP50 score
            object_specific = input("What name do you want to give to your trained object: \n")
            new_weights_path = self.new_data(object_name=object_name, object_specific=object_specific)
            return new_weights_path

        except Exception as e:
            self.logger.error("\n Process interrupted!")
            self.logger.error(e)
            if not IOError:
                if os.listdir(f"{self.combined_folder}/raw_dataset"):
                    shutil.rmtree(self.combined_folder)
        except KeyboardInterrupt:
            self.logger.error("Process interrupted in between")
            if os.listdir(f"{self.combined_folder}/raw_dataset"):
                shutil.rmtree(self.combined_folder)


if __name__ == "__main__":
    at = AutoTrain("/home/owl/workspace/auto-train/data/testing", prev_data_folder="/home/owl/workspace/auto-train/data/new_weights_v20250106193615/raw_dataset", new_weights=True, inference=False, image_threshold=10, epochs=2, abs_yaml_file="/home/owl/workspace/auto-train/data/new_weights_v20250106193615/train.yaml")
    new_weights_path = at.run()

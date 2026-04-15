import os
import torch
import cv2
import json
import yaml
import splitfolders
import numpy as np
from PIL import Image
from datetime import datetime
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import YOLO


class NewData:
    '''
    Annotates and stores new data captured

    Args:
        - logger (object instance): Logger instance for adding logs
        - combined_folder (str): Path to local folder to store the new data
        - json_file (str): Path to inputs.json file used for training new data
        - object_name (str): Generic object name to detect
        - image_threshold (int): Number of images to capture for creating new dataset
        - epochs (int): Number of epochs for training
        - map_threshold (float): value<=1 ; Threshold to compare mAP50 score
        - inference (boolean): True to perform the inference on live feed
        - inference_threshold (float): value<=1 ; Threshold for inference confidence score
    '''
    def __init__(self, logger, combined_folder, json_file, object_name, image_threshold, epochs, map_threshold, inference, inference_threshold):
        self.logger = logger
        self.combined_folder = combined_folder
        self.json_file = json_file
        self.object_name = object_name
        self.image_threshold = image_threshold
        self.epochs = epochs
        self.map_threshold = map_threshold
        self.inference = inference
        self.inference_threshold = inference_threshold

        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.device = torch.device(0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(self.device)
        self.model_yolov8 = YOLO('yolov8n.pt')

    def owl_pred_live(self, color_frame, box_threshold=0.6, text_threshold=0.4):
        '''
        Annotate live images using Grounding DINO

        Args:
            - color_frame (numpy.ndarray): Camera input image
            - box_threshold (float): Box threshold for Grounding DINO
            - text_threshold (float): Text threshold for Grounding DINO
        Returns:
            - tuple: A tuple of numpy array for Grounding DINO results, xmin, ymin, xmax, ymax of resulting bounding boxes
        '''
        xmin = ymin = xmax = ymax = None
        image = Image.fromarray(color_frame)  # for live feed
       
        inputs = self.processor(text=self.object_name, images=image, return_tensors="pt").to(self.device)
        torch.cuda.empty_cache()

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold, target_sizes=[image.size[::-1]])
        result = results[0]

        if len(result['labels']) != 0:  # Check if labels list is not empty
            xyxy = result["boxes"][0]
            xmin, ymin, xmax, ymax = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
        return results, xmin, ymin, xmax, ymax

    def capture_pred(self, box_threshold, text_threshold):
        '''
        Capture and store annotated images and labels

        Args:
            - box_threshold (float): Box threshold for Grounding DINO
            - text_threshold (float): Text threshold for Grounding DINO
        '''
        # Capture using cv2
        with open(self.json_file, 'r') as file:
                data = json.load(file)
        cam_index = data['camera_index']
        has_gui = os.environ.get("DISPLAY") is not None
        vid = cv2.VideoCapture(cam_index)
        try:
            img_counter=0
            while True:
                ret, frame = vid.read()
                if has_gui:
                    cv2.imshow('Image Capture', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                if img_counter == self.image_threshold:
                    break

                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results, xmin, ymin, xmax, ymax = self.owl_pred_live(frame_bgr, box_threshold, text_threshold)
                # Store only if object is detected in frame
                if xmin != None:
                    img_folder = self.combined_folder+"/raw_dataset/images"
                    txt_folder = self.combined_folder+"/raw_dataset/labels"
                    if not os.path.exists(img_folder):
                        os.makedirs(img_folder)
                        os.makedirs(txt_folder)
                    img_name = f"image_{img_counter}_{self.timestamp}.jpg"
                    img_path = os.path.join(img_folder, img_name)
                    ih, iw = frame.shape[:2]
                    image_sh = cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,255,0), 2)
                    if has_gui:
                        cv2.imshow("Detected OWL", image_sh)
                    cv2.imwrite(img_path, image_sh)
                    self.logger.info(f"{img_name} written!")
                    img_counter += 1
                    # Create labels.txt
                    txt_path = os.path.join(txt_folder, os.path.splitext(img_name)[0] + ".txt")
                    with open(self.json_file, 'r') as file:
                        data = json.load(file)
                    label_number = len(data['candidate_labels'])-1
                    with open(txt_path, 'w') as file:
                        xc = (xmin + xmax)/2
                        yc = (ymin + ymax)/2
                        w = xmax - xmin
                        h = ymax - ymin
                        data = f"{label_number} {xc/iw} {yc/ih} {w/iw} {h/ih}"
                        file.write(data)
                    
        finally:
            vid.release()
            if has_gui:
                cv2.destroyAllWindows()

    def split_and_yaml(self):
        '''
        Splits and creates YAML file for training
        '''
        upper_folder = self.combined_folder+"/split_dataset"
        if not os.path.exists(upper_folder):
            os.makedirs(upper_folder)
        splitfolders.ratio(self.combined_folder+"/aug_dataset", output=upper_folder, ratio=(0.7, 0.3))
        self.logger.info("Training and validation sets ready")
        # Create yaml file
        f = open(self.json_file)
        candidate_labels = json.load(f)['candidate_labels']
        f.close()
        path = os.path.abspath(self.combined_folder)
        split_path = path.split("/{}".format(self.combined_folder))[0]
        yaml_content={
            'path': split_path,
            'train': f"{upper_folder}/train",
            'val': f"{upper_folder}/val",
            'names': {index: value for index,value in enumerate(candidate_labels)}
        }
        yaml_path = self.combined_folder+"/train.yaml"
        if not os.path.exists(yaml_path):
            with open(yaml_path,'w') as file:
                yaml.dump(yaml_content, file)
        self.logger.info("YAML file created")

    def train(self):
        '''
        Trains and returns new weight file for new dataset
        '''
        results = self.model_yolov8.train(data=f"{self.combined_folder}/train.yaml", epochs=self.epochs, device=self.device, project=self.combined_folder)
        rdict = results.__dict__
        new_weights_path = str(rdict["save_dir"])+"/weights/best.pt"
        # Get MaP50 Score
        map = rdict['box'].__dict__
        map50 = map['all_ap'][0][0]
        # Put threshold on MaP50 score
        if map50>=self.map_threshold:
            new_weights_path = new_weights_path
            self.logger.info("Trained and stored the new weights")
        else:
            new_weights_path = None
            self.logger.error('Try with more images and training more epochs')
        # Start live inference
        has_gui = os.environ.get("DISPLAY") is not None
        if self.inference and new_weights_path!=None:
            with open(self.json_file, 'r') as file:
                data = json.load(file)
            cam_index = data['camera_index']

            vid = cv2.VideoCapture(cam_index)
            new_yolov8 = YOLO(new_weights_path).to(self.device)
            while True:
                _, frame = vid.read()
                if has_gui:
                    cv2.imshow('Image Capture', frame)
                results = new_yolov8(frame, conf=self.inference_threshold, device=self.device, stream=True)                

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        confidence = round(float(box.conf[0]) * 100, 2)
                        cls = int(box.cls[0])
                        label = f"{data['candidate_labels'][cls]}: {confidence}%"
                        cv2.putText(frame, label, [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

                if has_gui:
                    cv2.imshow('Inference', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        break

        return new_weights_path, map50

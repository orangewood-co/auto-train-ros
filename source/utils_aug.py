import albumentations as A
import cv2
import os
import shutil
import json


class Augment():
    '''
    Augments the data to create more data for training

    Args:
        - logger (object instance): Logger instance for adding logs
        - combined_folder (str): Path to local folder to store the new data
        - json_file (str): Path to inputs.json file used for training new data
    '''
    def __init__(self, logger, combined_folder, json_file):
        self.logger = logger
        self.combined_folder = combined_folder
        self.json_file = json_file

    def is_image_by_extension(self, file_name):
        '''
        Check if the given file has a recognized image extension.

        Args:
            - file_name (str): Name of the file.
        Returns:
            - bool: True if the file has a recognized image extension, False otherwise.
        '''
        # List of common image extensions
        image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']
        # Get the file extension
        file_extension = file_name.lower().split('.')[-1]
        # Check if the file has a recognized image extension
        return file_extension in image_extensions

    def get_inp_data(self, img_file):
        '''
        Get input data for image processing.

        Args:
            - img_file (str): Name of the input image file.
        Returns:
            - tuple: A tuple containing the image, ground truth bounding boxes, and augmented file name.
        '''
        file_name = os.path.splitext(img_file)[0]
        aug_file_name = f"{file_name}_aug_out"
        image = cv2.imread(os.path.join(self.combined_folder+"/raw_dataset/images", img_file))
        lab_pth = os.path.join(self.combined_folder+"/raw_dataset/labels", f"{file_name}.txt")
        with open(self.json_file, 'r') as file:
            data = json.load(file)
        gt_bboxes = self.get_bboxes_list(lab_pth, data["candidate_labels"])
        return image, gt_bboxes, aug_file_name

    def get_album_bb_list(self, yolo_bbox, class_names):
        '''
        Extracts bounding box information for a single object from YOLO format.

        Args:
            - yolo_bbox (str): YOLO format string representing bounding box information.
            - class_names (list): List of class names corresponding to class numbers.
        Returns:
            - list: A list containing [x_center, y_center, width, height, class_name].
        '''
        str_bbox_list = yolo_bbox.split()
        class_number = int(str_bbox_list[0])
        class_name = class_names[class_number]
        bbox_values = list(map(float, str_bbox_list[1:]))
        album_bb = bbox_values + [class_name]
        return album_bb

    def get_album_bb_lists(self, yolo_str_labels, classes):
        '''
        Extracts bounding box information for multiple objects from YOLO format.

        Args:
            - yolo_str_labels (str): YOLO format string containing bounding box information for multiple objects.
            - classes (list): List of class names corresponding to class numbers.
        Returns:
            - list: A list of lists, each containing [x_center, y_center, width, height, class_name].
        '''
        album_bb_lists = []
        yolo_list_labels = yolo_str_labels.split('\n')
        for yolo_str_label in yolo_list_labels:
            if yolo_str_label:
                album_bb_list = self.get_album_bb_list(yolo_str_label, classes)
                album_bb_lists.append(album_bb_list)
        return album_bb_lists

    def get_bboxes_list(self, inp_lab_pth, classes):
        '''
        Reads YOLO format labels from a file and returns bounding box information.

        Args:
            - inp_lab_pth (str): Path to the YOLO format labels file.
            - classes (list): List of class names corresponding to class numbers.
        Returns:
            - list: A list of lists, each containing [x_center, y_center, width, height, class_name].
        '''
        yolo_str_labels = open(inp_lab_pth, "r").read()

        if not yolo_str_labels:
            self.logger.info("No object")
            return []

        lines = [line.strip() for line in yolo_str_labels.split("\n") if line.strip()]
        album_bb_lists = self.get_album_bb_lists("\n".join(lines), classes) if len(lines) > 1 else [self.get_album_bb_list("\n".join(lines), classes)]
        return album_bb_lists

    def get_augmented_results(self, image, bboxes):
        '''
        Apply data augmentation to an input image and bounding boxes.

        Args:
            - image (numpy.ndarray): Input image.
            - bboxes (list): List of bounding boxes in YOLO format [x_center, y_center, width, height, class_name].
        Returns:
            - tuple: A tuple containing the augmented image and the transformed bounding boxes.
        '''
        bbox_coords = [b[:4] for b in bboxes]
        class_labels = [b[4] for b in bboxes]

        transform = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0),
            A.Affine(scale=0.4, p=0.6),
            A.Blur(blur_limit=(3, 7), p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', clip=True, label_fields=['class_labels']))

        transformed = transform(image=image, bboxes=bbox_coords, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = [list(b) + [l] for b, l in zip(transformed['bboxes'], transformed['class_labels'])]
        return transformed_image, transformed_bboxes

    def make_copy_folder(self, new_combined_folder):
        '''
        Copies data folder from combined_folder to new folder specified.

        Args:
            - new_combined_folder (str): Path where augmented results will be stored.\
        '''
        if not os.path.exists(new_combined_folder):
            os.makedirs(new_combined_folder+"/images")
            os.makedirs(new_combined_folder+"/labels")

        img_folder = self.combined_folder+"/raw_dataset/images"
        txt_folder = self.combined_folder+"/raw_dataset/labels"

        for item in os.listdir(img_folder):
            source_path = os.path.join(img_folder, item)
            destination_path = os.path.join(f"{new_combined_folder}/images", item)
            shutil.copy(source_path,destination_path)
        for item in os.listdir(txt_folder):
            source_path = os.path.join(txt_folder, item)
            destination_path = os.path.join(f"{new_combined_folder}/labels", item)
            shutil.copy(source_path,destination_path)

    def store_aug(self, aug_img, aug_label, aug_file_name):
        '''
        Stores augmented data, combined with original data to combined_folder.

        Args:
            - aug_img (numpy.ndarray): Augmented Image to store
            - aug_label (list): List of bounding boxes in YOLOv8 format
            - aug_file_name (str): Path to augmented file name
        '''
        new_path = os.path.join(self.combined_folder, 'aug_dataset')
        self.make_copy_folder(new_path)

        aug_img_pth = os.path.join(self.combined_folder+"/aug_dataset/images" ,aug_file_name+".jpg")
        cv2.imwrite(aug_img_pth, aug_img)
        with open(self.json_file, 'r') as file:
            data = json.load(file)
        aug_lab_pth = os.path.join(self.combined_folder+"/aug_dataset/labels" ,aug_file_name+".txt")
        with open(aug_lab_pth,'w') as out:
            for bbox in aug_label:
                label_name = bbox[-1]
                label_index = data["candidate_labels"].index(label_name)
                upd_bbox = f"{label_index} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"
                out.write(upd_bbox+"\n")

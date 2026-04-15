import cv2
import json

class AvailableCam():
    '''
    Lists the available cameras in the given range to choose from

    Args:
        - logger (object instance): Logger instance for adding logs
        - json_file (str): Path to inputs.json file used for training new data
        - camera_range (int): Range of camera indices to check
    '''
    def __init__(self, logger, json_file, camera_range):
        self.logger = logger
        self.camera_range = camera_range
        self.json_file = json_file

    def get_available_cameras(self):
        '''
        Function retirns the list of available camera indeices within the specified range
        '''
        available_cameras = []
        # Check for cameras 
        for i in range(self.camera_range):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras

    def select_camera(self):
        '''
        Function inputs the camera index to use and stores that in the json file
        '''
        cameras = self.get_available_cameras()
        #check for present cameras
        if cameras:
            #for multiple cameras
            if len(cameras)>1:
                self.logger.info("Available Cameras:", cameras)
                while True:
                    cam_index = input('Enter camera index to use: ')
                    if int(cam_index) in cameras:
                        #store the cam index as dict in input.json
                        with open(self.json_file, 'r') as file:
                            data = json.load(file)
                        data['camera_index'] = cam_index
                        with open(self.json_file, 'w') as file:
                            json.dump(data, file, indent=4)
                        self.logger.info(f'Camera accessed: {cam_index}')
                        break
                    else:
                        print('Choose the camera from the indexes given above')
            else:
                #store the cam index as dict in input.json
                with open(self.json_file, 'r') as file:
                    data = json.load(file)
                data['camera_index'] = cameras[0]
                self.logger.info(f'Camera accessed: {cameras[0]}')
                with open(self.json_file, 'w') as file:
                    json.dump(data, file, indent=4)
        else:
            self.logger.error("No cameras found.")
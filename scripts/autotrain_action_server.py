#!/usr/bin/env python3

import json
import os
import shutil
import traceback

import rclpy
from auto_train_ros.action import AutoTrain as AutoTrainAction
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from source.auto_train import AutoTrain


class AutoTrainActionServer(Node):
    def __init__(self):
        super().__init__("auto_train_action_server")
        self._action_server = ActionServer(self, AutoTrainAction, "auto_train", self.execute_callback)
        self.get_logger().info("AutoTrain Action Server is ready.")

    def _publish_feedback(self, goal_handle, message):
        self.get_logger().info(f"Feedback: {message}")
        feedback = AutoTrainAction.Feedback()
        feedback.status_message = message
        goal_handle.publish_feedback(feedback)

    def execute_callback(self, goal_handle):
        req = goal_handle.request
        self.get_logger().info("Received AutoTrain goal.")

        abs_yaml = req.abs_yaml_file if req.abs_yaml_file else None

        try:
            at = AutoTrain(
                node_instance=self,
                image_topic_name=req.image_topic_name,
                data_folder=req.data_folder,
                prev_data_folder=req.prev_data_folder,
                new_weights=req.new_weights,
                abs_yaml_file=abs_yaml,
                draw_bb=req.draw_bb,
                image_threshold=req.image_threshold,
                number_aug=req.number_aug,
                epochs=req.epochs,
                map_threshold=req.map_threshold,
                inference=req.inference,
                inference_threshold=req.inference_threshold,
            )
        except ValueError as e:
            self.get_logger().error(f"Invalid parameters: {e}")
            goal_handle.abort()
            return AutoTrainAction.Result(new_weights_path="", success=False)

        try:
            # --- Replicate AutoTrain.run() without interactive input() calls ---

            # 1. Create directory structure
            if not os.path.exists(at.combined_folder):
                os.makedirs(at.combined_folder + "/raw_dataset/images")
                os.makedirs(at.combined_folder + "/raw_dataset/labels")
            else:
                raise IOError(f"{at.combined_folder} already exists.")
            self._publish_feedback(goal_handle, "Directory structure created")

            # 2. Initialise the inputs JSON file
            if not os.path.exists(at.json_file):
                with open(at.json_file, "w") as f:
                    json.dump({"candidate_labels": []}, f, indent=4)
            self._publish_feedback(
                goal_handle,
                f"Subscribing to image topic '{req.image_topic_name}'",
            )

            # 3. Load previous data if continuing from existing weights
            if not req.new_weights:
                at.prev_data()
                self._publish_feedback(goal_handle, "Previous data loaded")

            # 4. Append the generic object name to candidate_labels
            object_name = req.object_name.rstrip(".") + "."
            with open(at.json_file, "r") as f:
                data = json.load(f)
            data["candidate_labels"].append(object_name)
            with open(at.json_file, "w") as f:
                json.dump(data, f, indent=4)
            self._publish_feedback(
                goal_handle,
                f"Object name '{req.object_name}' registered, starting data collection",
            )

            # 5. Collect new data, augment, and train
            new_weights_path = at.new_data(object_name=object_name, object_specific=req.object_label)

            result = AutoTrainAction.Result()
            result.new_weights_path = new_weights_path or ""
            result.success = new_weights_path is not None

            if result.success:
                self._publish_feedback(goal_handle, "Training complete")
                goal_handle.succeed()
            else:
                self._publish_feedback(goal_handle, "Failed — no images captured or mAP below threshold")
                goal_handle.abort()
            self.get_logger().info(f"Result: weights={result.new_weights_path}")
            return result

        except Exception as e:
            self.get_logger().error(f"AutoTrain failed: {e}\n{traceback.format_exc()}")
            # Clean up partial data on failure
            if os.path.exists(at.combined_folder) and os.listdir(f"{at.combined_folder}/raw_dataset"):
                shutil.rmtree(at.combined_folder)
            goal_handle.abort()
            return AutoTrainAction.Result(new_weights_path="", success=False)


def main(args=None):
    rclpy.init(args=args)
    node = AutoTrainActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().warn("Shutting down AutoTrain Action Server...")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

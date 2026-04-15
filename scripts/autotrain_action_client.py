#!/usr/bin/env python3

import json
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from auto_train_ros.action import AutoTrain as AutoTrainAction


class AutoTrainActionClient(Node):
    def __init__(self, vision_config_path=""):
        super().__init__("auto_train_action_client")
        self._action_client = ActionClient(self, AutoTrainAction, "auto_train")
        self._vision_config_path = vision_config_path

    def send_goal(
        self,
        data_folder,
        object_name,
        object_label,
        image_topic_name,
        prev_data_folder="",
        new_weights=True,
        abs_yaml_file="",
        draw_bb=False,
        image_threshold=100,
        number_aug=3,
        epochs=69,
        map_threshold=0.5,
        inference=False,
        inference_threshold=0.4,
    ):
        self.get_logger().info("Waiting for AutoTrain Action Server...")
        self._action_client.wait_for_server()
        self.get_logger().info("Server available, sending goal.")

        goal = AutoTrainAction.Goal()
        goal.data_folder = data_folder
        goal.prev_data_folder = prev_data_folder
        goal.new_weights = new_weights
        goal.abs_yaml_file = abs_yaml_file
        goal.draw_bb = draw_bb
        goal.image_threshold = image_threshold
        goal.number_aug = number_aug
        goal.epochs = epochs
        goal.map_threshold = map_threshold
        goal.inference = inference
        goal.inference_threshold = inference_threshold
        goal.object_name = object_name
        goal.object_label = object_label
        goal.image_topic_name = image_topic_name

        self._send_goal_future = self._action_client.send_goal_async(
            goal, feedback_callback=self._feedback_callback
        )
        self._send_goal_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal was rejected by the server.")
            return
        self.get_logger().info("Goal accepted by server.")
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        result = future.result().result
        if result.success:
            self.get_logger().info(f"Training succeeded! New weights: {result.new_weights_path}")
            self._update_vision_config(result.new_weights_path)
        else:
            self.get_logger().error("Training failed or mAP threshold not met.")
        rclpy.shutdown()

    def _update_vision_config(self, weights_path):
        if not self._vision_config_path:
            self.get_logger().warn("No vision_config_path provided, skipping config update.")
            return
        try:
            with open(self._vision_config_path, "r") as f:
                config = json.load(f)
            config["new_weights"] = weights_path
            with open(self._vision_config_path, "w") as f:
                json.dump(config, f, indent=4)
            self.get_logger().info(f"Updated vision config '{self._vision_config_path}' with new weights.")
        except Exception as e:
            self.get_logger().error(f"Failed to update vision config: {e}")

    def _feedback_callback(self, feedback_msg):
        self.get_logger().info(f"Feedback: {feedback_msg.feedback.status_message}")


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="AutoTrain Action Client")
    parser.add_argument(
        "--vision-config-path",
        type=str,
        required=True,
        help="Absolute path to the vision_config.json file",
    )
    parsed, remaining = parser.parse_known_args()

    rclpy.init(args=remaining)

    client = AutoTrainActionClient(vision_config_path=parsed.vision_config_path)

    # --- Example goal: adjust these to your use-case ---
    client.send_goal(
        data_folder="/home/owl/auto_train_data",
        object_name="rectangular yellow object",
        object_label="duster",
        image_topic_name="/camera/color/image_raw",
        new_weights=True,
        image_threshold=100,
        number_aug=3,
        epochs=20,
        map_threshold=0.0,
    )

    rclpy.spin(client)
    client.destroy_node()


if __name__ == "__main__":
    main()

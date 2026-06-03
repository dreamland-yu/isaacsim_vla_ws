#!/usr/bin/env python3
import io

import cv2
import numpy as np
import rclpy
import zmq
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from .robot_configs import SO100_JOINT_NAMES


class VLAObservationSender3CamNode(Node):
    def __init__(self):
        super().__init__("vla_observation_sender_3cam_node")

        self.bridge = CvBridge()

        self.obs_context = zmq.Context()
        self.obs_socket = self.obs_context.socket(zmq.PUB)
        self.obs_socket.bind("tcp://127.0.0.1:5556")

        self.cam1_sub = Subscriber(self, Image, "/camera1_img")
        self.cam2_sub = Subscriber(self, Image, "/camera2_img")
        self.cam3_sub = Subscriber(self, Image, "/camera3_img")
        self.joint_sub = Subscriber(self, JointState, "/joint_states")

        self.sync = ApproximateTimeSynchronizer(
            [self.cam1_sub, self.cam2_sub, self.cam3_sub, self.joint_sub],
            queue_size=10,
            slop=0.03,
        )
        self.sync.registerCallback(self.callback_observation)

        self.last_send = self.get_clock().now()
        self.get_logger().info("Start to send 3 camera images and joint states ......")

    def callback_observation(
        self,
        cam1_msg: Image,
        cam2_msg: Image,
        cam3_msg: Image,
        joint_msg: JointState,
    ):
        now = self.get_clock().now()
        if (now - self.last_send).nanoseconds < 1e9 / 10:
            return
        self.last_send = now

        timestamp = cam1_msg.header.stamp.sec + cam1_msg.header.stamp.nanosec * 1e-9

        img1 = self.bridge.imgmsg_to_cv2(cam1_msg, desired_encoding="bgr8")
        img2 = self.bridge.imgmsg_to_cv2(cam2_msg, desired_encoding="bgr8")
        img3 = self.bridge.imgmsg_to_cv2(cam3_msg, desired_encoding="bgr8")

        ok1, buf1 = cv2.imencode(".jpg", img1, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        ok2, buf2 = cv2.imencode(".jpg", img2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        ok3, buf3 = cv2.imencode(".jpg", img3, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not (ok1 and ok2 and ok3):
            self.get_logger().warn("JPEG encoding failed, skipping frame")
            return

        if joint_msg.name != SO100_JOINT_NAMES:
            self.get_logger().warn("The joint names don't match robot definition!")
            return
        curr_joint_states = np.array(joint_msg.position, dtype=np.float32)

        observation = self.pack_observation(
            img1_bytes=buf1.tobytes(),
            img2_bytes=buf2.tobytes(),
            img3_bytes=buf3.tobytes(),
            joints=curr_joint_states,
            ts=timestamp,
        )
        self.obs_socket.send(observation, flags=0)

    def pack_observation(
        self,
        img1_bytes,
        img2_bytes,
        img3_bytes,
        joints: np.ndarray,
        ts: float,
    ):
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            ts=np.array([ts], dtype=np.float64),
            joints=joints,
            img1=np.frombuffer(img1_bytes, dtype=np.uint8),
            img1_encoded=np.array([1], dtype=np.int8),
            img2=np.frombuffer(img2_bytes, dtype=np.uint8),
            img2_encoded=np.array([1], dtype=np.int8),
            img3=np.frombuffer(img3_bytes, dtype=np.uint8),
            img3_encoded=np.array([1], dtype=np.int8),
        )
        return buf.getvalue()


def main(args=None):
    rclpy.init(args=args)
    node = VLAObservationSender3CamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

import os
import time
import sys
import importlib
import threading
import numpy as np
import pyarrow as pa
from dora import Node

# ROS Imports
import rospy
# from cv_bridge import CvBridge # <--- 移除这个导致冲突的库

class ROSDoraBridge:
    def __init__(self):
        # 1. 从环境变量读取配置
        self.topic = os.getenv("ROS_TOPIC")
        self.msg_type_str = os.getenv("MSG_TYPE")
        self.output_name = os.getenv("OUTPUT_NAME", "data")

        # 处理类型
        self.data_mode = os.getenv("DATA_TYPE", "ATTRIBUTE") 
        self.extract_path = os.getenv("EXTRACT_PATH", "")
        self.target_encoding = os.getenv("ENCODING", "rgb8")

        if not self.topic or not self.msg_type_str:
            raise ValueError("Environment variables ROS_TOPIC and MSG_TYPE must be set.")

        # 2. 初始化工具 (移除 self.bridge)
        self.lock = threading.Lock()
        self.latest_msg = None
        self.latest_meta = {}

        # 3. 动态加载 ROS 消息类型
        self.msg_class = self._load_msg_class(self.msg_type_str)

        # 4. 初始化 ROS 节点
        rospy.init_node(f"dora_bridge_{os.getpid()}", anonymous=True, disable_signals=True)

        # 5. 建立订阅
        rospy.Subscriber(self.topic, self.msg_class, self._ros_callback)
        print(f"[{os.getpid()}] Subscribed to {self.topic} ({self.msg_type_str})", flush=True)

    def _load_msg_class(self, type_str):
        try:
            pkg, name = type_str.split('/')
            module = importlib.import_module(f"{pkg}.msg")
            return getattr(module, name)
        except Exception as e:
            raise ImportError(f"Could not load ROS message type '{type_str}': {e}")

    def _get_nested_attr(self, obj, path):
        if not path: return obj
        for attr in path.split('.'):
            obj = getattr(obj, attr)
        return obj

    def _ros_callback(self, msg):
        with self.lock:
            self.latest_msg = msg
            self.latest_meta = {
                "timestamp": time.time_ns(),
                "topic": self.topic
            }

    def _manual_image_to_numpy(self, msg):
        """
        纯 Numpy 实现 ROS Image -> cv2/numpy 转换
        避开 cv_bridge 的动态链接库冲突
        """
        try:
            dtype_class = np.uint8
            channels = 1

            # 简单的编码判断
            if "8UC1" in msg.encoding or "mono8" in msg.encoding:
                channels = 1
            elif "8UC3" in msg.encoding or "rgb8" in msg.encoding or "bgr8" in msg.encoding:
                channels = 3
            elif "16UC1" in msg.encoding or "mono16" in msg.encoding:
                dtype_class = np.uint16
                channels = 1

            # 构造 numpy 数组
            # msg.data 是 bytes，需要 frombuffer
            img = np.frombuffer(msg.data, dtype=dtype_class)
            img = img.reshape(msg.height, msg.width, channels)

            return img
        except Exception as e:
            raise RuntimeError(f"Failed to convert image manually: {e}")

    def process_and_send(self, node):
        msg = None
        meta = {}
        with self.lock:
            if self.latest_msg is None:
                return 
            msg = self.latest_msg
            meta = self.latest_meta.copy()

        try:
            storage = None

            # === 模式 A: 图像处理 ===
            if self.data_mode == "IMAGE" or self.msg_type_str == "sensor_msgs/Image":
                # 使用手动转换替代 cv_bridge
                cv_img = self._manual_image_to_numpy(msg)

                # 处理编码转换 (BGR -> RGB)
                # 假设 ROS 发出来的是 bgr8 (常见情况)，如果你需要 rgb8
                if self.target_encoding == "rgb8" and msg.encoding == "bgr8":
                    cv_img = cv_img[:, :, ::-1] # BGR to RGB
                elif self.target_encoding == "bgr8" and msg.encoding == "rgb8":
                    cv_img = cv_img[:, :, ::-1] # RGB to BGR

                # 补充 Metadata
                meta["width"] = cv_img.shape[1]
                meta["height"] = cv_img.shape[0]
                meta["encoding"] = self.target_encoding

                storage = pa.array(cv_img.ravel())

            # === 模式 B: 通用属性提取 ===
            else:
                val = self._get_nested_attr(msg, self.extract_path)

                if isinstance(val, (list, tuple)):
                    np_val = np.array(val)

                # 针对 geometry_msgs/Point, Vector3 (x,y,z)
                elif hasattr(val, 'x') and hasattr(val, 'y') and hasattr(val, 'z'):
                    if hasattr(val, 'w'): 
                        # 针对 geometry_msgs/Quaternion (x,y,z,w)
                        np_val = np.array([val.x, val.y, val.z, val.w])
                    else:
                        # 针对 Point / Vector3
                        np_val = np.array([val.x, val.y, val.z])

                else:
                    # 针对标量 (Float32, Int32, Bool 等)
                    np_val = np.array([val]) 

                storage = pa.array(np_val.ravel())

            if storage is not None:
                node.send_output(self.output_name, storage, meta)

        except Exception as e:
            # 打印详细错误栈，方便调试
            import traceback
            traceback.print_exc()
            print(f"Error processing frame: {e}", file=sys.stderr)

def main():
    bridge = ROSDoraBridge()
    node = Node()
    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "tick":
                bridge.process_and_send(node)
        elif event["type"] == "STOP":
            break

if __name__ == "__main__":
    main()
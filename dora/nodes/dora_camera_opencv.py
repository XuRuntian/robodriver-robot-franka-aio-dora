"""TODO: Add docstring."""
import argparse
import os
import time
import re
import cv2
import numpy as np
import pyarrow as pa
from dora import Node

RUNNER_CI = True if os.getenv("CI") == "true" else False
FLIP = os.getenv("FLIP", "")

# ========== 解析符号链接→真实数字索引 ==========
def resolve_video_index(input_path):
    """
    解析输入为真实视频设备的数字索引（纯数字）
    支持输入格式：
    1. 符号链接路径：/dev/video81 → 解析为真实路径 /dev/video16 → 提取索引 16
    2. 真实路径：/dev/video15 → 提取索引 15
    3. 纯数字：15 → 直接返回 15
    返回：有效索引（int）/ None（失败）
    """
    # 1. 纯数字直接返回
    if isinstance(input_path, int):
        return input_path
    if isinstance(input_path, str):
        if input_path.isnumeric():
            return int(input_path)
    
    # 2. 处理设备路径（补全/dev/前缀）
    if isinstance(input_path, str):
        dev_path = input_path if input_path.startswith("/dev/") else f"/dev/{input_path}"
        
        # 检查路径是否存在
        if not os.path.exists(dev_path):
            print(f"[ERROR] 设备路径不存在：{dev_path}")
            return None
        
        # 解析符号链接到真实路径
        real_path = os.path.realpath(dev_path)
        if dev_path != real_path:
            print(f"[DEBUG] 符号链接解析：{dev_path} → 真实路径：{real_path}")
        
        # 提取数字索引
        match = re.search(r"/video(\d+)", real_path)
        if match:
            real_index = int(match.group(1))
            print(f"[DEBUG] 提取真实数字索引：{real_path} → {real_index}")
            return real_index
        else:
            print(f"[ERROR] 无法从路径提取索引：{real_path}")
            return None
    
    print(f"[ERROR] 不支持的输入格式：{input_path}")
    return None

def main():
    """TODO: Add docstring."""
    parser = argparse.ArgumentParser(
        description="OpenCV Video Capture: This node is used to capture video from a camera.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        help="The name of the node in the dataflow.",
        default="opencv-video-capture",
    )
    parser.add_argument(
        "--path",
        type=int,
        required=False,
        help="The path of the device to capture (e.g. /dev/video1, or an index like 0, 1...",
        default=0,
    )
    parser.add_argument(
        "--image-width",
        type=int,
        required=False,
        help="The width of the image output. Default is 1280.",
        default=1280,  # 设为相机支持的默认宽度
    )
    parser.add_argument(
        "--image-height",
        type=int,
        required=False,
        help="The height of the camera. Default is 720.",
        default=720,   # 设为相机支持的默认高度
    )
    args = parser.parse_args()

    # ========== 核心修正：解析符号链接 + 鱼眼相机索引逻辑 ==========
    # 1. 读取环境变量/参数，解析真实索引
    video_capture_path = os.getenv("CAPTURE_PATH", args.path)
    real_index = resolve_video_index(video_capture_path)
    
    # 2. 校验索引有效性
    if real_index is None:
        print(f"[FATAL] 无法解析有效相机索引：{video_capture_path}")
        return
    
    # 3. 鱼眼相机：奇数索引用于捕获，偶数用于控制（自动修正为奇数）
    capture_index = real_index if real_index % 2 == 0 else real_index - 1
    print(f"[DEBUG] 鱼眼相机捕获索引修正：{real_index} → {capture_index}")

    # 4. 校验捕获索引是否存在
    if not os.path.exists(f"/dev/video{capture_index}"):
        print(f"[ERROR] 捕获索引{capture_index}对应的设备不存在，尝试兜底修正")
        capture_index = real_index + 1
    if not os.path.exists(f"/dev/video{capture_index}"):
        print(f"[ERROR] 捕获索引{capture_index}对应的设备不存在，尝试兜底修正")
        capture_index = real_index - 1
    # if not os.path.exists(f"/dev/video{capture_index}"):
    #     print(f"[ERROR] 捕获索引{capture_index}对应的设备不存在，尝试兜底修正")
    #     capture_index = real_index + 2
    # if not os.path.exists(f"/dev/video{capture_index}"):
    #     print(f"[ERROR] 捕获索引{capture_index}对应的设备不存在，尝试兜底修正")
    #     capture_index = real_index - 2
        if not os.path.exists(f"/dev/video{capture_index}"):
            print(f"[FATAL] 无可用捕获设备（{real_index} +- 2区间内 均不存在）")
            return

    # ========== 关键：指定V4L2后端 + 强制MJPG格式 ==========
    capture_path = f"/dev/video{capture_index}"
    video_capture = cv2.VideoCapture(capture_path, cv2.CAP_V4L2)
    time.sleep(2)  # 加长初始化延迟（鱼眼相机需要）

    # 校验是否打开成功
    if not video_capture.isOpened():
        print(f"[FATAL] 无法打开相机设备：{capture_path}")
        # 兜底：用索引打开
        video_capture = cv2.VideoCapture(capture_index, cv2.CAP_V4L2)
        if not video_capture.isOpened():
            print(f"[FATAL] 索引方式也无法打开：{capture_index}")
            return
    print(f"[SUCCESS] 成功打开相机：{capture_path}")

    # ========== 强制配置相机格式/分辨率/帧率（适配video15） ==========
    # 强制设为MJPG格式（相机主要支持，帧率更高）
    video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # 设置帧率（30fps，兼容所有分辨率）
    video_capture.set(cv2.CAP_PROP_FPS, 30)

    image_width = os.getenv("IMAGE_WIDTH", 640)
    if image_width is not None:
        image_width = int(image_width) if str(image_width).isnumeric() else 1280
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)

    image_height = os.getenv("IMAGE_HEIGHT", 480)
    if image_height is not None:
        image_height = int(image_height) if str(image_height).isnumeric() else 720
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

    # 读取实际分辨率（验证设置）
    actual_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[DEBUG] 相机实际分辨率：{actual_width}x{actual_height}")

    encoding = os.getenv("ENCODING", "rgb8")
    node = Node(args.name)
    start_time = time.time()
    pa.array([])  # initialize pyarrow array

    # ========== 主循环（优化帧读取/错误处理） ==========
    for event in node:
        if RUNNER_CI and time.time() - start_time > 10:
            break

        if event["type"] == "INPUT" and event["id"] == "tick":
            ret, frame = video_capture.read()
            if not ret:
                print(f"[ERROR] 无法读取相机帧（{capture_path}），返回空帧")
                frame = np.zeros((actual_height, actual_width, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    f"Error: Camera {capture_path} read failed",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    1,
                    1,
                )
            else:
                # 翻转处理
                if FLIP == "VERTICAL":
                    frame = cv2.flip(frame, 0)
                elif FLIP == "HORIZONTAL":
                    frame = cv2.flip(frame, 1)
                elif FLIP == "BOTH":
                    frame = cv2.flip(frame, -1)

                # 分辨率调整（仅当需要时）
                if (image_width and image_height) and (
                    frame.shape[1] != image_width or frame.shape[0] != image_height
                ):
                    frame = cv2.resize(frame, (image_width, image_height))

            # 编码转换
            if encoding == "rgb8":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif encoding == "yuv420":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
            elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                ret, frame = cv2.imencode("." + encoding, frame)
                if not ret:
                    print("[ERROR] 图像编码失败")
                    continue

            # 发送数据
            metadata = event["metadata"]
            metadata["encoding"] = encoding
            metadata["width"] = int(frame.shape[1]) if len(frame.shape)>=2 else actual_width
            metadata["height"] = int(frame.shape[0]) if len(frame.shape)>=2 else actual_height
            storage = pa.array(frame.ravel())
            node.send_output("image", storage, metadata)

        elif event["type"] == "ERROR":
            raise RuntimeError(event["error"])

    # 释放资源
    video_capture.release()
    print("[INFO] 相机资源已释放")

if __name__ == "__main__":
    main()

import os
import time
import requests
import pyarrow as pa
from dora import Node
import numpy as np
def post(url, route, json=None):
    resp = requests.post(f"{url}/{route}", json=json)
    resp.raise_for_status()
    return resp 

def get_arm_data(url):
    """统一获取机械臂关节、夹爪、位姿数据，返回字典格式"""
    arm_data = {
        "jointstate": None,
        # "gripper": None,  # 格式：[当前距离(mm), 是否稳定夹持(0/1)]
        "pose": None,
        "quaternion": None,
        "success": False
    }

    try:
        # 获取关节角度
        joint_resp = requests.post(f"{url}getq", timeout=0.1)
        if joint_resp.status_code == 200:
            arm_data["jointstate"] = joint_resp.json()["q"]

        # gripper = requests.post(f"{url}get_pika_gripper", json={}, timeout=0.1)
        # if gripper.status_code == 200:
        #     arm_data["gripper"] = [gripper.json()["gripper_mm"]]
        # 获取末端位姿
        pose_resp = requests.post(f"{url}getpos_euler", timeout=0.1)
        if pose_resp.status_code == 200:
            pose = pose_resp.json()["pose"]
            arm_data["pose"] = [pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]]

        pose_quaternion_resp = requests.post(f"{url}getpos", timeout=0.1)
        if pose_quaternion_resp.status_code == 200:
            arm_data["quaternion"] = pose_quaternion_resp.json()["pose"][3:]
        # 标记数据是否完整（包含夹爪数据）
        
        if all(v is not None for v in [arm_data["jointstate"], arm_data["pose"], arm_data["quaternion"]]):
            arm_data["success"] = True

    except requests.exceptions.RequestException as e:
        print(f"机械臂API请求失败: {e}")

    return arm_data


def main():
    arm_url = os.getenv("url", "http://127.0.0.1:5000/")
    print(f"机械臂API地址: {arm_url}")


    last_gripper_data = 0  # 用于缓存最新的夹爪数据
    node = Node()
    ctrl_frame = 0

    # 事件循环
    for event in node:
        event_type = event["type"]

        if event_type == "INPUT":
            if "action" in event["id"]:
                pass

            if event["id"] == "action_joint":
                if ctrl_frame > 0:
                    continue
                try:
                    joint = event["value"].to_numpy()
                    goal_eef_quat = joint[:-1].astype(float)
                    # gripper_val = float(joint[-1])  # 目标开合度（mm）
                    
                    post(arm_url, "pose", {"arr": goal_eef_quat.tolist()})

                except Exception as e:
                    print(f"执行 'action_joint' 失败: {e}")
            
            elif event["id"] == "action_joint_ctrl":        
                try:
                    joint = event["value"].to_numpy()
                    goal_eef_quat = joint[:-1]
                    # gripper_val = joint[-1]  # 目标开合度（mm）
                    
                    post(arm_url, "pose", {"arr": goal_eef_quat.tolist()})

                except Exception as e:
                    print(f"执行 'action_joint_ctrl' 失败: {e}")

            elif event["id"] == "get_joint":
                # 传入夹爪辅助类，获取夹爪数据
                arm_data = get_arm_data(arm_url)
                if arm_data["success"]:
                    combined_list = (
                        arm_data["jointstate"]
                        +[last_gripper_data]
                        + arm_data["pose"]
                        + arm_data["quaternion"]
                    )
                    # print(f"发送{combined_list}")
                    node.send_output(
                        "jointstate",
                        pa.array(combined_list, type=pa.float32()),
                        {"timestamp": time.time_ns()}
                    )
                
            elif event["id"] == "gripper":
                try:
                    # 缓存夹爪数据，供后续 get_joint 使用
                    # 假设 value 是 pyarrow 数组，转为 list 或 numpy
                    val = event["value"][0]
                    last_gripper_data = float(val)
                except Exception as e:
                    print(f"处理 gripper 数据出错: {e}")
        elif event["id"] == "stop":
            print("收到停止指令，停止机械臂...")

    print("Dora节点退出，清理资源...")

if __name__ == "__main__":
    main()

import os
import time
import requests
import pyarrow as pa
from dora import Node
def post(url, route, json=None):
    resp = requests.post(f"{url}/{route}", json=json)
    resp.raise_for_status()
    return resp 

def get_arm_data(url):
    """统一获取机械臂关节、夹爪、位姿数据，返回字典格式"""
    arm_data = {
        "quaternion": None,
        "success": False
    }

    try:

        pose_quaternion_resp = requests.post(f"{url}get_sense_pika_eepose", json={}, timeout=0.1)
        if pose_quaternion_resp.status_code == 200:
            arm_data["quaternion"] = pose_quaternion_resp.json()['pika_sense_eepose']
        # 标记数据是否完整（包含夹爪数据）
        
        if all(v is not None for v in [arm_data["quaternion"]]):
            arm_data["success"] = True

    except requests.exceptions.RequestException as e:
        print(f"机械臂API请求失败: {e}")

    return arm_data


def main():
    arm_url = os.getenv("url", "http://127.0.0.1:5000/")
    print(f"机械臂API地址: {arm_url}")


    node = Node()

    # 事件循环
    for event in node:
        event_type = event["type"]

        if event_type == "INPUT":
            if "action" in event["id"]:
                pass

            elif event["id"] == "get_joint":
                arm_data = get_arm_data(arm_url)
                if arm_data["success"]:
                    combined_list = (
                        arm_data["quaternion"]
                    )
                    node.send_output(
                        "jointstate",
                        pa.array(combined_list, type=pa.float32()),
                        {"timestamp": time.time_ns()}
                    )
                
        elif event["id"] == "stop":
            print("收到停止指令，停止pika...")

    print("Dora节点退出，清理资源...")

if __name__ == "__main__":
    main()

# robodriver-robot-franka-aio-dora
## Get Start
Clone the repository
```
git clone --recurse-submodules https://github.com/BAAI-EI-DATA/robodriver-robot-franka-aio-dora.git && cd robodriver-robot-franka-aio-dora
```

Install the project to Robodriver
```
conda activate robodriver
pip install -e .
```

Configure the dataflow.yml
```
cd /path/to/your/robodriver-robot-franka-aio-dora
cd dora
```
Open the dataflow.yml file, then you can configure your sensor

## Start collecting
Activate environment
```
conda activate robodriver
```
Start Dora
```
dora up
```
Start dataflow
```
cd /path/to/your/robodriver-robot-franka-aio-dora
dora start dora/dataflow.yml 
```
Launch RoboXStudio
```
cd /path/to/your/RoboDriver
python robodriver/scripts/run.py \
  --robot.type=franka_aio_dora 
```

## Bug Fixes
1. Rerun video replay failure:  
   Edit `RoboDriver/robodriver/core/coordinator.py`, change `visual_worker(mode="distant")` to `mode="local"`.

2. OpenCV cvShowImage error on launch (`python robodriver/scripts/run.py --robot.type=franka_aio_dora`):  
   Comment out `cv2.imshow(key, img)` and `cv2.waitKey(1)` in `robodriver/scripts/run.py`.


## Data Information
franka robotic arm data is transmitted by the Dora node. Each robotic arm node sends **14-dimensional information** with the following composition:

- Joint angles: Values of joints 1-7 (7 dimensions)
- Gripper state: Gripper opening/closing degree 
- End-effector pose: End Euler angles (3 dimensions, representing rotation around X/Y/Z axes) \
**When you need to modify the data information, you can modify `dataflow.yml` and `config.py`**

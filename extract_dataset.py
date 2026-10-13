import os
import glob
import csv
import numpy as np
import cv2
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

def get_rosbag_options(path, storage_id='sqlite3'):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    return storage_options, converter_options

def extract_all():
    bag_base_dir = os.path.expanduser('~/rosbag_data')
    output_dir = os.path.expanduser('~/dataset')
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, 'data.csv')

    bag_paths = sorted(glob.glob(os.path.join(bag_base_dir, 'auto_drive_dataset_*')))
    if not bag_paths:
        print("[ERROR] No bag files found in ~/rosbag_data matching 'auto_drive_dataset_*'")
        return

    print(f"[INFO] Found {len(bag_paths)} datasets. Starting extraction...")

    img_type = get_message('sensor_msgs/msg/Image')
    cmd_type = get_message('geometry_msgs/msg/Twist')

    total_frames = 0
    records = []

    for bag_path in bag_paths:
        bag_name = os.path.basename(bag_path)
        print(f"--> Processing: {bag_name}")
        
        storage_options, converter_options = get_rosbag_options(bag_path)
        reader = rosbag2_py.SequentialReader()
        try:
            reader.open(storage_options, converter_options)
        except Exception as e:
            print(f"    [WARN] Failed to open {bag_name}: {e}")
            continue

        images = []
        cmd_vels = []

        while reader.has_next():
            topic, data, t = reader.read_next()
            if topic == '/camera/image_raw':
                msg = deserialize_message(data, img_type)
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
                if msg.encoding == 'rgb8':
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                images.append((t, img))
            elif topic == '/cmd_vel':
                msg = deserialize_message(data, cmd_type)
                cmd_vels.append((t, msg.linear.x, msg.angular.z))

        if not images or not cmd_vels:
            print(f"    [SKIP] Empty data in {bag_name}")
            continue

        cmd_times = np.array([c[0] for c in cmd_vels])

        for t_img, img in images:
            idx = np.argmin(np.abs(cmd_times - t_img))
            _, lin_x, ang_z = cmd_vels[idx]

            frame_filename = f"frame_{total_frames:06d}.jpg"
            img_save_path = os.path.join(img_dir, frame_filename)
            cv2.imwrite(img_save_path, img)

            records.append((f"images/{frame_filename}", round(lin_x, 4), round(ang_z, 4)))
            total_frames += 1

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'linear_x', 'angular_z'])
        writer.writerows(records)

    print("="* 50)
    print("[SUCCESS] Extraction completed!")
    print(f"  Total Extracted Frames : {total_frames}")
    print(f"  Output Directory       : {output_dir}")
    print(f"  Labels CSV             : {csv_file}")

if __name__ == '__main__':
    extract_all()
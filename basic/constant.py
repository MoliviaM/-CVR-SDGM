import os
import logging

import os

if 'HOME' in os.environ:
    ROOT_PATH = os.path.join(os.environ['HOME'], 'workplace', 'disk3', 'video-retrieval')
elif 'USERPROFILE' in os.environ:  # Windows 兼容
    ROOT_PATH = os.path.join(os.environ['USERPROFILE'], 'workplace', 'disk3', 'video-retrieval')
else:
    # 手动设置一个默认路径，避免 KeyError
    ROOT_PATH = "D:/workspace/workplace/disk3/video-retrieval"

print(f"ROOT_PATH set to: {ROOT_PATH}")  # 打印出来，确认是否正确


logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)
# logger.setLevel(logging.INFO)


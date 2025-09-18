from pathlib import Path
import ultralytics
from ultralytics.utils import LOGGER
import logging

LOGGER.setLevel(logging.ERROR) 
import tqdm


if __name__ == "__main__":
    yolo_model = ultralytics.YOLO("yolov8n.pt")
    data_root = Path().cwd().joinpath("data","waymo","processed","training")
    for scene_folder in data_root.glob("*"):
        if not scene_folder.joinpath("images").exists():
            print(f'data incomplete in scene {scene_folder.name}, skip')
        else:
            print(f'processing scene {scene_folder.name}')
            for img_path in tqdm.tqdm(list(scene_folder.joinpath("images").glob("*.jpg"))):
                results = yolo_model.predict(source=str(img_path), save_txt=True, conf=0.5, project=str(scene_folder), name="bboxes", exist_ok=True, verbose=False)
            print(f'Finished processing scene {scene_folder.name}')
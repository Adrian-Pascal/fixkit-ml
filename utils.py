import os
from pathlib import Path
import shutil
import yaml
import cv2

ACCEPTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

def restructure_data_for_yolo(input_path, output_path):
        assert os.path.exists(input_path), f"Path {input_path} does not exist."

        input = Path(input_path)
        output = Path(output_path)
        images = output / "images"
        labels = output / "labels"
        classes_map = {}

        for i, article in enumerate(input.iterdir()):
            if not article.is_dir():
                print(f"/{article.name} is not a directory, skipping...")
                continue

            classes_map[i] = article.name

            for subdir in article.iterdir():
                if not subdir.is_dir():
                    print(f"/{article.name}/{subdir.name} is not a directory, skipping...")
                    continue

                for file in subdir.iterdir():
                    if file.suffix.lower() not in ACCEPTED_EXTENSIONS:
                        print(f"File {file.name} has unsupported extension, skipping...")
                        continue

                    target_dir = images / subdir.name / article.name 
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file, target_dir / file.name)

                    label_dir = labels / subdir.name / article.name
                    label_dir.mkdir(parents=True, exist_ok=True)
                    label_file = label_dir / (file.stem + ".txt")
                    with open(label_file, 'w') as f:
                        f.write(f"{i} 0.5 0.5 0.7 0.6\n")

        with open(output / "data.yaml", 'w') as f:
            yaml.dump({
                'path': '.',
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'nc': len(classes_map),
                'names': classes_map
            }, f)


def visualize_labels(input_path):
    assert os.path.exists(input_path), f"Path {input_path} does not exist."

    input = Path(input_path)
    images = input / "images"
    labels = input / "labels"

    for subdir in images.iterdir():
        if not subdir.is_dir():
            continue

        for classdir in subdir.iterdir():
            if not classdir.is_dir():
                continue

            for image_file in classdir.glob("*"):
                if image_file.suffix.lower() not in ACCEPTED_EXTENSIONS:
                    continue

                label_file = labels / subdir.name / classdir.name / (image_file.stem + ".txt")
                if not label_file.exists():
                    print(f"No label file for {image_file.name}, skipping...")
                    continue

                img = cv2.imread(str(image_file))
                height, width, _ = img.shape
                print(f"Processing {image_file.name}...")
                print(f"Image dimensions: {width}x{height}")

                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    id, x_center, y_center, w, h = map(float, lines[0].strip().split())
                    print(f"Label data: id={id}, x_center={x_center}, y_center={y_center}, w={w}, h={h}")

                    x1 = int((x_center - w / 2) * width)
                    y1 = int((y_center - h / 2) * height)
                    x2 = int((x_center + w / 2) * width)
                    y2 = int((y_center + h / 2) * height)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 10)
                    cv2.putText(img, str(id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.namedWindow(f"Image: {image_file.name}", cv2.WINDOW_NORMAL)
                cv2.imshow(f"Image: {image_file.name}", img)
                cv2.resizeWindow(f"Image: {image_file.name}", 800, 600)
                cv2.waitKey(0)

if __name__ == "__main__":

    # Example usage
    input_path = "./fix-kit-split"
    output_path = "./dataset"
    restructure_data_for_yolo(input_path, output_path)
from pathlib import Path
import shutil
# 原始数据集
SOURCE = Path("dataset")
# 新的三分类数据集
DEST = Path("dataset_3class")
# 原类别编号 -> 新类别编号
CLASS_MAP = {
    2: 0,  # book
    5: 1,  # laptop
    4: 2,  # cell phone -> phone
}
SPLITS = ["train", "val", "test"]
for split in SPLITS:
    src_images = SOURCE / "images" / split
    src_labels = SOURCE / "labels" / split
    dst_images = DEST / "images" / split
    dst_labels = DEST / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    kept_images = 0
    kept_objects = 0
    for label_path in src_labels.glob("*.txt"):
        new_lines = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                old_class = int(parts[0])
                if old_class not in CLASS_MAP:
                    continue
                new_class = CLASS_MAP[old_class]
                parts[0] = str(new_class)
                new_lines.append(" ".join(parts))
        # 如果这张图片至少包含一个我们需要的目标
        if new_lines:
            # 保存新的标签文件
            new_label_path = dst_labels / label_path.name
            with open(new_label_path, "w") as f:
                f.write("\n".join(new_lines) + "\n")
            # 找对应图片
            image_found = False
            for suffix in [".jpg", ".jpeg", ".png"]:
                image_path = (
                    src_images
                    / f"{label_path.stem}{suffix}"
                )
                if image_path.exists():
                    shutil.copy2(
                        image_path,
                        dst_images / image_path.name,
                    )
                    image_found = True
                    break
            if image_found:
                kept_images += 1
                kept_objects += len(new_lines)
    print(
        f"{split}: "
        f"{kept_images} images, "
        f"{kept_objects} objects"
    )
# 创建新的 data.yaml
yaml_content = """path: dataset_3class
train: images/train
val: images/val
test: images/test
nc: 3
names:
  0: book
  1: laptop
  2: phone
"""
with open(DEST / "data.yaml", "w") as f:
    f.write(yaml_content)
print("Done.")
print("Three-class dataset saved to dataset_3class/")

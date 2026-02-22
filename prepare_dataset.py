import os
import shutil

# Must download dataset
root = "CUB_200_2011"
images_folder = os.path.join(root, "images")

split_file = open(os.path.join(root, "train_test_split.txt")).read().splitlines()
label_file = open(os.path.join(root, "image_class_labels.txt")).read().splitlines()
images_file = open(os.path.join(root, "images.txt")).read().splitlines()
classes_file = open(os.path.join(root, "classes.txt")).read().splitlines()

id_to_class = {int(x.split()[0]): x.split()[1] for x in classes_file}
imgid_to_label = {int(x.split()[0]): int(x.split()[1]) for x in label_file}
imgid_to_name = {int(x.split()[0]): x.split()[1] for x in images_file}
imgid_to_split = {int(x.split()[0]): int(x.split()[1]) for x in split_file}

for img_id in imgid_to_name:
    img_path = os.path.join(images_folder, imgid_to_name[img_id])
    class_name = id_to_class[imgid_to_label[img_id]]

    split = "train" if imgid_to_split[img_id] == 1 else "test"

    out_dir = os.path.join("dataset", split, class_name)
    os.makedirs(out_dir, exist_ok=True)

    shutil.copy(img_path, out_dir)

print("Done!")


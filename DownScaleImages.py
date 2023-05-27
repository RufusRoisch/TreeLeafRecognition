import cv2

newHeight = 200
newWidth = 100

def down_size(path: str, name: str):
    img = cv2.imread(path)
    res = cv2.resize(img, dsize=(newWidth, newHeight), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("LeafPicsDownScaled/" + name, res)

source_dir = "LeafPictures/Leaf"
check_int = 0
for treeClass in range(1, 16, 1):
    for leafNum in range(1, 76, 1):
        img_name = "l" + str(treeClass) + "nr" + f"{leafNum:03d}" + ".tif"
        img_path = source_dir + str(treeClass) + "/" + img_name
        down_size(img_path, img_name)
        check_int += 1

print(f"Done resizing -{check_int}- pictures!")
print(f"Should be {75 * 15}")
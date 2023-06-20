import cv2


# resulting width and height in pixels
newWidth = 150
newHeight = 300


# loads -> downsizes -> saves single image
def down_size(path: str, name: str):
    # loads original
    img = cv2.imread(path)

    # resizes image
    res = cv2.resize(img, dsize=(newWidth, newHeight), interpolation=cv2.INTER_CUBIC)

    # saves resized image to different directory
    cv2.imwrite("LeafPicsDownScaled/" + name, res)


source_dir = "LeafPictures/Leaf"
# used for checking if all pictures has been resized
amount_pictures_resized = 0

# iterates over all leaf pictures of every tree class
for treeClass in range(1, 16, 1):
    for leafNum in range(1, 76, 1):
        # constructs path to current leaf file
        img_name = "l" + str(treeClass) + "nr" + f"{leafNum:03d}" + ".tif"
        img_path = source_dir + str(treeClass) + "/" + img_name

        # downsizes that picture
        down_size(img_path, img_name)

        # increases count for images resized
        amount_pictures_resized += 1

# shoes if all pictures where resized
print(f"Done resizing -{amount_pictures_resized}- pictures!")
print(f"Should be {75 * 15}")
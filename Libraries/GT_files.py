# Loading the Ground Truth Image Data from the the .txt file
def load_GT(GT_path):
    GT_text = open(GT_path, "r").readlines()
    GT = []
    count = 0
    for text in GT_text:
        row = text.split()
        row1 = list(map(float, row[1:]))
        GT.append([row[0]] + row1)
        count = count + 1
    return GT

# Displaying the Image along with the Ground Truth bounding boxes
def add_GTLabels(img,GT):
    from cv2 import rectangle
    for gt in GT:
        if gt[0] != 'DontCare':
            #print(gt[0])
            #print(int(gt[4]))
            start = (int(gt[4]), int(gt[5]))
            end = (int(gt[6]), int(gt[7]))
            #print(start)
            #print(end)
            color = (255, 0, 0)
            rectangle(img, start, end, color, 2)

import numpy as np
import cv2

mask_path = "test/output/first/mask/000000.jpg"
mask = cv2.imread(mask_path)

# compute box similiar to https://github.com/mikeqzy/3dgs-avatar-release/blob/main/train.py
mask[mask == 255] = 0
print(mask)
mask_new = np.where(mask)
print(mask)
y1, y2 = mask_new[1].min(), mask_new[1].max() + 1
x1, x2 = mask_new[0].min(), mask_new[0].max() + 1
mask[x1:x2,y1:y2,:] = 100
cv2.imwrite("box.jpg",mask)
print((x1+x2)/2)
print((y1+y2)/2)

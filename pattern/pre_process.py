import cv2
import numpy as np
image = cv2.imread('./image/000.png')
image = cv2.resize(image, (224, 224))
print(image.shape)
cv2.imwrite('./image/001.png', image)
image_save = image.transpose((2, 0, 1))
with open("image_pattern.txt", "w") as f:
    f.write(' '.join(map(str, image_save.flatten().tolist())) )

with open("image_pattern.txt", "r") as f:
    data = list(map(int, f.read().split()))

image_rc = np.array(data).reshape(3, 224, 224).transpose(1, 2, 0)
image_rc = np.uint8(image_rc)
print(image_rc.shape)
cv2.imwrite('./reconstructed_image.png', image_rc)
cv2.imshow('Reconstructed Image', image_rc)
cv2.waitKey(0)
cv2.destroyAllWindows()

import matplotlib
import matplotlib.pyplot as plt
import cv2


def plot_img_and_mask(img, mask, rate_image):
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    fr = lambda x, r: (int(x.shape[1]/r), int(x.shape[0]/r))
    mask = cv2.resize(mask, fr(img, 1) )
    mask[:, :, -1] = mask[:, :, -1] * 255
    img_ = cv2.addWeighted(img, rate_image, mask, 1-rate_image, 0)
    img_ = cv2.resize(img_, fr(img, 3) )
    cv2.imshow('result', img_)
    d = cv2.waitKey(0)
    if d == ord('q'):
        cv2.destroyAllWindows()
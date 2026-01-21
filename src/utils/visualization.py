import matplotlib.pyplot as plt

def overlay(image, mask, alpha=0.3):
    plt.imshow(image, cmap='gray')
    plt.imshow(mask, cmap='Reds', alpha=alpha)
    plt.show()

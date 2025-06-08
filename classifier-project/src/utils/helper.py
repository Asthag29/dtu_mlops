import matplotlib.pyplot as plt
import numpy as np

def view_classify(img, ps, version="Fashion"):
    """Function for viewing an image and its predicted classes."""
    ps = ps.data.numpy().flatten()
    img = img.data.numpy().reshape(28, 28)

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 6), nrows=1, ncols=2)
    ax1.imshow(img, cmap='gray')
    ax1.axis('off')
    
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title(f'Class Probability\n{version}')
    ax2.set_xlim(0, 1.0)
    
    plt.tight_layout()
    plt.show()

### importing required packages


```python
from keras.datasets import cifar10
import numpy as np 
import matplotlib.pyplot as plt
%matplotlib inline
```

    Using Theano backend.


### Load the dataset


```python

(X_train,y_train),(X_test,y_test) = cifar10.load_data()
```

### loading the reconstructed image .npy files


```python
conv1 = np.load("conv1_out.npy")
conv2 = np.load("conv2_out.npy")
conv3 = np.load("conv3_out.npy")
conv4 = np.load("conv4_out.npy")
```

### display a sample image


```python
plt.imshow(conv3[1].transpose(1,2,0))
plt.show()
```


![png](output_7_0.png)


### Displaying randomlely chosen 10 images from convolutional layer 1


```python

for i in range(10):
    n = np.random.randint(len(conv1))
    plt.subplot(1, 2, 1)
    plt.imshow(X_test[n].transpose(1,2,0))
    plt.title("True Image")
    plt.xticks([]); plt.yticks([])
    
    plt.subplot(1, 2, 2)
    plt.imshow(conv1[n].transpose(1,2,0))
    plt.title("predicted mage")
    plt.xticks([]); plt.yticks([])
    plt.show()
```


![png](output_9_0.png)



![png](output_9_1.png)



![png](output_9_2.png)



![png](output_9_3.png)



![png](output_9_4.png)



![png](output_9_5.png)



![png](output_9_6.png)



![png](output_9_7.png)



![png](output_9_8.png)



![png](output_9_9.png)


### Displaying randomlely chosen 10 images from convolutional layer 2


```python
for i in range(10):
    n = np.random.randint(len(conv2))
    plt.subplot(1, 2, 1)
    plt.imshow(X_test[n].transpose(1,2,0))
    plt.title("True Image")
    plt.subplot(1, 2, 2)
    plt.imshow(conv2[n].transpose(1,2,0))
    plt.title("predicted mage")
    plt.show()
    
```


![png](output_11_0.png)



![png](output_11_1.png)



![png](output_11_2.png)



![png](output_11_3.png)



![png](output_11_4.png)



![png](output_11_5.png)



![png](output_11_6.png)



![png](output_11_7.png)



![png](output_11_8.png)



![png](output_11_9.png)


### Displaying randomlely chosen 10 images from convolutional layer 3


```python

for i in range(10):
    n = np.random.randint(len(conv3))
    plt.subplot(1, 2, 1)
    plt.imshow(X_test[n].transpose(1,2,0))
    plt.title("True Image")
    plt.xticks([]); plt.yticks([])
    
    plt.subplot(1, 2, 2)
    plt.imshow(conv3[n].transpose(1,2,0))
    plt.title("predicted mage")
    plt.xticks([]); plt.yticks([])
    
    plt.show()
    
```


![png](output_13_0.png)



![png](output_13_1.png)



![png](output_13_2.png)



![png](output_13_3.png)



![png](output_13_4.png)



![png](output_13_5.png)



![png](output_13_6.png)



![png](output_13_7.png)



![png](output_13_8.png)



![png](output_13_9.png)


### Displaying randomlely chosen 10 images from convolutional layer 4


```python

for i in range(10):
    n = np.random.randint(len(conv4))
    plt.subplot(1, 2, 1)
    plt.imshow(X_test[n].transpose(1,2,0))
    plt.title("True Image")
    plt.xticks([]); plt.yticks([])
    
    plt.subplot(1, 2, 2)
    plt.imshow(conv4[n].transpose(1,2,0))
    plt.title("predicted mage")
    plt.xticks([]); plt.yticks([])
    
    plt.show()
```


![png](output_15_0.png)



![png](output_15_1.png)



![png](output_15_2.png)



![png](output_15_3.png)



![png](output_15_4.png)



![png](output_15_5.png)



![png](output_15_6.png)



![png](output_15_7.png)



![png](output_15_8.png)



![png](output_15_9.png)


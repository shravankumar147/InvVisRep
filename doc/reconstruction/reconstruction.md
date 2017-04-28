
### importing required packages
```python
import numpy as np 
import matplotlib.pyplot as plt
%matplotlib inline
```

### loading .npy files
```python
conv2 = np.load("conv2_out.npy")
conv3 = np.load("conv3_out.npy")
conv4 = np.load("conv4_out.npy")
```

### example to display any particular image
```python
plt.imshow(conv3[1].transpose(1,2,0))
plt.show()
```


![png](output_2_0.png)


### Displaying randomlely chosen 10 images from convolutional layer 2
```python

for i in range(10):
    n = np.random.randint(len(conv2))
    plt.imshow(conv3[n].transpose(1,2,0))
    plt.show()
    
```


![png](output_3_0.png)



![png](output_3_1.png)



![png](output_3_2.png)



![png](output_3_3.png)



![png](output_3_4.png)



![png](output_3_5.png)



![png](output_3_6.png)



![png](output_3_7.png)



![png](output_3_8.png)



![png](output_3_9.png)

### Displaying randomlely chosen 10 images from convolutional layer 3

```python

for i in range(10):
    n = np.random.randint(len(conv3))
    plt.imshow(conv3[n].transpose(1,2,0))
    plt.show()
    
    
```


![png](output_4_0.png)



![png](output_4_1.png)



![png](output_4_2.png)



![png](output_4_3.png)



![png](output_4_4.png)



![png](output_4_5.png)



![png](output_4_6.png)



![png](output_4_7.png)



![png](output_4_8.png)



![png](output_4_9.png)


### Displaying randomlely chosen 10 images from convolutional layer 4
```python

for i in range(10):
    n = np.random.randint(len(conv4))
    plt.imshow(conv3[n].transpose(1,2,0))
    plt.show()
```


![png](output_5_0.png)



![png](output_5_1.png)



![png](output_5_2.png)



![png](output_5_3.png)



![png](output_5_4.png)



![png](output_5_5.png)



![png](output_5_6.png)



![png](output_5_7.png)



![png](output_5_8.png)



![png](output_5_9.png)



```python

```

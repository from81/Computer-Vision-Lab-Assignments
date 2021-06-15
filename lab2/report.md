## Task 1: Corner Detection

### 1.2, 1.3

```python
def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    img = np.pad(img, pad_width=pad, mode='constant', constant_values=0)
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    return result


def fspecial(shape=(3, 3), sigma=0.5):
    """
    Create gaussian filter based on specified parameter, shape and sigma.
    The 2DGaussian will be a function of distance from the center and the provided sigma.
    
    P ~ Norm(0, sigma)
    h[i, j] = P(d) where P(d) is the probability density function of the specified gaussian distribution, 
    and d is distance from center (0,0) to the posiution i, j.
    
    The closer d is to 0, the higher h[i, j] is.
    The output matrix (kernel) will be normalized to have sum of 1
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0: # shouldn't this be if sumh != 1?
        h /= sumh
    return h



# Parameters, add more if needed
sigma = 5
# thresh = 0.01
harris_kernel_shape = (5, 5)
k = 0.05

# Derivative masks
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
dy = dx.transpose()

try:
    bw = plt.imread('0.png')
except:
    bw = plt.imread('Task1/Harris-1.jpg')
    bw = cv2.cvtColor(bw, cv2.COLOR_RGB2GRAY)

    
bw = np.array(bw * 255, dtype=int)

# computer x and y derivatives of image
Ix = conv2(bw, dx)
Iy = conv2(bw, dy)

# gaussian filter
g = fspecial(
    (
        max(1, np.floor(3 * sigma) * 2 + 1), 
        max(1, np.floor(3 * sigma) * 2 + 1)
    ), 
    sigma
)

Iy2 = conv2(np.power(Iy, 2), g)
Ix2 = conv2(np.power(Ix, 2), g)
Ixy = conv2(Ix * Iy, g)

######################################################################
# Task: Compute the Harris Cornerness
######################################################################

# if w(x, y) is (1, 1), don't need to apply convolution
if harris_kernel_shape == (1, 1):
    lambda_1 = Ix2
    lambda_2 = Iy2
# else, for each pixel position, we will add up Ix2 and Iy2 of each pixel that overlaps with the fitler
else:
    kernel = np.ones(harris_kernel_shape)
    lambda_1 = conv2(Ix2, kernel)
    lambda_2 = conv2(Iy2, kernel)

# calculate determinant and trace of matrix M for each pixel in the image, and finally compute R
det_m = lambda_1 * lambda_2 - np.square(conv2(Ixy, kernel))
trace_m = lambda_1 + lambda_2
R = det_m - (k * np.square(trace_m))

######################################################################
# Task: Perform non-maximum suppression and
#       thresholding, return the N corner points
#       as an Nx2 matrix of x and y coordinates
######################################################################

def non_maximum_suppression(R):
    """
    Perform non-maximum suppression by finding pixels that have intensity greater than 8 neighbors.
    
    input:
        R: 2-d numpy array of corner response values    
	returns:
        corners: list of tuples indicating corners ditected
        arr: 2-d numpy binary array where arr[i, j] == 1 indicates corner as detected by non-maximum suppression
    """
    
    corners = []
    for i in range(1, len(R)-1):
        for j in range(1, len(R[0])-1):
            # check if R[i,j] is a corner by comparing each pixel to 8 neighbors
            # collect corners
            if (
                (R[i,j] >= R[i+1,j]) & (R[i,j] >= R[i-1,j]) & (R[i,j] >= R[i,j+1]) & (R[i,j] >= R[i,j-1]) & 
                (R[i,j] >= R[i+1,j+1]) & (R[i,j] >= R[i-1,j-1]) & (R[i,j] >= R[i-1,j+1]) & (R[i,j] >= R[i-1,j-1])
            ):
                corners += (i, j),

    # create a binary array from corners array to visualize corners
    arr = np.zeros(R.shape)
    for corner in corners:
        arr[corner[0], corner[1]] = 1
    return corners, arr

corners, Nx2 = non_maximum_suppression(R)
```



### 1.4

Fixed parameters

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download.png" alt="download" style="zoom:60%;" />

Variable parameters

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-1.png" alt="download-1" style="zoom:60%;" />

### 1.5. Compare results with `cv2.cornerHarris()`

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-1.png" alt="download-2" style="zoom:60%;" />

#### Factors that affect the performance of Harris corner detection

Harris corner detection is invariant to location, but not scale. As such, the performance will depend on the degree of gaussian smoothing determined by `sigma`, the kernel size of the sobel filter, the size of neighborhood considered when determining whether `img[i][j]` is a corner or not, and the `k` value in calculating the cornerness score $R$.

### 1.6

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-3.png" alt="download-3" style="zoom:60%;" />

#### Analyse the results why we cannot get corners by discussing and visualising your corner response scores of the image.

Cornerness R visualized using `plt.imshow()` with cmap `RdBu_r` (blue is lower).

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-4.png" alt="download-4" style="zoom:40%;" />

R value is approximately 0 for almost all pixel positions in the image. The only pixels with high R scores are at top and bottom left corners, and top and bottom center because the right half of the image is all 0s. Corners were detected on the left corners only because of the 0 padding of width 1 that's applied to the image in `conv2()` function. Corners were also detected in the top and bottom center because of the boundary between the white and black regions, in addition to the 0 padding. Aside from the influence from 0 padding, no corners were detected because there are no corners in the image.

    Top left corner
    [[1.95029123 2.09187119 0.11344908]
     [2.09187119 2.41787998 0.51097243]
     [0.11344908 0.51097243 0.17606731]]
     
    Top right corner
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
     
    Bottom left corner
    [[0.11344908 0.51097243 0.17606731]
     [2.09187119 2.41787998 0.51097243]
     [1.95029123 2.09187119 0.11344908]]
     
    Bottom right corner
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
     
    Top center
    [[ 0.17165154  2.65415093  3.26432971  0.96915069  0.06661764]
     [ 0.57056862  2.97199509  3.20967762  0.64287179 -0.0273949 ]
     [ 0.19054354  0.57293854 -0.33744827 -1.04884253 -0.27012009]]
     
    Bottom center
    [[ 0.19054354  0.57293854 -0.33744827 -1.04884253 -0.27012009]
     [ 0.57056862  2.97199509  3.20967762  0.64287179 -0.0273949 ]
     [ 0.17165154  2.65415093  3.26432971  0.96915069  0.06661764]]

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-5.png" alt="download-5" style="zoom:60%;" />

### 1.7

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-6.png" alt="download-6" style="zoom:60%;" />

## Task 2: KMeans

### 2.1 KMeans

```python
class KMeans:
    def __init__(self, X: np.ndarray, k: int):
        """initialize parameters and randomly create centroids in the domain of input data"""
        self.X = X
        self.k = k
        self.centroids = np.random.uniform(size=(self.k, self.X.shape[1]))
        self.pred = np.zeros(X.shape[0])

    @staticmethod
    def euclidean_distance(X, centroids):
        """calculate pair-wise euclidean distance"""
        k = centroids.shape[0]
        
        diff = X[np.newaxis,:,:] - centroids[:,np.newaxis,:]        
        assert diff.shape == (k, X.shape[0], X.shape[1])

        dist = np.sqrt(np.sum(diff**2, axis=-1))
        dist = dist.transpose()
        assert dist.shape == (X.shape[0], k)
        
        return dist
        
    def predict(self, verbose=False):
        i = 1
        start_time = time.time()
        
        while True:
            # calculate distance between every point and every centroid
            dist = KMeans.euclidean_distance(self.X, self.centroids)
            
            # assign each point to a class with nearest centroid
            pred = np.argmin(dist, axis=1)

            # calculate new centroid and wss for each cluster
            new_centroids = []
            wss = []
            for c in range(self.k):
                if len(self.X[pred==c]) == 0:
                    new_centroid = np.random.uniform(size=self.centroids.shape[1])
                else:
                    new_centroid = np.mean(self.X[pred==c], axis=0)
                new_centroids += new_centroid,

                # within-cluster sum of squared difference
                wss += np.sum((self.X[pred==c] - new_centroid) ** 2),

            new_centroids = np.array(new_centroids)
            assert self.centroids.shape == new_centroids.shape
            self.centroids = new_centroids
            self.wss = np.array(wss)
                        
            if all(pred == self.pred):
                return time.time() - start_time
            else:                    
                if i % 5 == 0 and verbose:
                    elapsed = time.time() - start_time
                    print(f'iteration: {i:>3}\twss: {self.wss.sum():^10.2f}\ttime in seconds: {elapsed:>5.2f}')
                    
                last_wss = self.wss
                self.pred = pred
                i += 1
                
def my_kmeans(data, start, end):
    """
    Run KMeans for various k values in the given range and return the scores and time to converge
    """
    scores = []
    models = []
    conv_time = []
    
    assert start < end
    
    for k in range(start, end):
        kmeans = KMeans(data, k=k)
        t_elapsed = kmeans.predict()
            
        wss = kmeans.wss.sum()
        scores += wss,
        models += kmeans,
        conv_time += t_elapsed,
    return np.array(scores), conv_time
```

### 2.2

#### With pixel coordinates

##### Image 1: M&M

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-7.png" alt="download-7" style="zoom:60%;" />

Using `k=8`

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-8.png" alt="download-8" style="zoom:60%;" />

##### Image 2: Peppers

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-9.png" alt="download-9" style="zoom:60%;" />

Using `k=8`

![download-10](/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-10.png)

#### Without pixel coordinates

##### Image 1: M&M

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-11.png" alt="download-11" style="zoom:60%;" />

Using `k=7`

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-12.png" alt="download-12" style="zoom:60%;" />

##### Image 2: Peppers

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-13.png" alt="download-13" style="zoom:60%;" />

Using `k=7`

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-14.png" alt="download-14" style="zoom:60%;" />

**Why the model that did not use pixel coordinates performed better**:

The model trained using pixel coordinates performed worse than the model that was trained without pixel coordinates. In cases where a certain object appears without overlapping, or are far apart from each other, they are less likely to be clustered together in the former model. The reason is that the former model places weight on proximity of the pixel coordinates, which is a wrong assumption to make in object recognition. Furthermore, placing weight on pixel coordinate also means reducing the weight placed on the euclidean space in the CIE-Lab space. This is evident in both examples from the former model that was trained with pixel coordinates, as the background is unnaturally divided into three or four blocks of equal size, despite having somewhat consistent color. The background is much better captured as one or two classes in the latter model that did not use pixel coordinates.

### 2.3 KMeans++

#### Key differences

The key difference between (Naive) KMeans and KMeans++ is in the initialization of the centroids.
While KMeans randomly initializes centroids in the input data space using uniform distribution, KMeans++ randomly picks one data point as the first centroid, and picks the rest of k-1 centroids by choosing the data point that is most distant from the closest centroid.

```python
class KMeansPP:
    def __init__(self, X: np.ndarray, k: int):
        self.X = X
        self.k = k
        self.initialize_centroids()
        self.pred = np.zeros(self.X.shape[0])

    @staticmethod
    def euclidean_distance(X, centroids):
        """calculate pair-wise euclidean distance"""
        k = centroids.shape[0]
        
        diff = X[np.newaxis,:,:] - centroids[:,np.newaxis,:]        
        assert diff.shape == (k, X.shape[0], X.shape[1])

        dist = np.sqrt(np.sum(diff**2, axis=-1))
        dist = dist.transpose()
        assert dist.shape == (X.shape[0], k)
        
        return dist

    def initialize_centroids(self):
        X = copy.deepcopy(self.X)
        idx = np.random.randint(low=0, high=X.shape[0])
        centroid = X[idx]
        centroids = np.array([centroid])
        X = np.delete(X, idx, 0)
        
        for i in range(1, self.k):
            # calculate distance to the nearest centroid for all points that are not used as a centroid
            dist = KMeansPP.euclidean_distance(X, centroids)
            dist_to_nearest_centroid = np.min(dist, 1)
            
            # use the point furthest from the nearest cluster as the next centroid
            idx = np.argmax(dist_to_nearest_centroid ** 2)
            centroid = X[idx]
            centroids = np.vstack((centroids, centroid))
            X = np.delete(X, idx, 0)
            
        self.centroids = centroids

    def predict(self, verbose=False):
        i = 1
        start_time = time.time()
        
        while True:
            # calculate distance between every point and every centroid
            dist = KMeans.euclidean_distance(self.X, self.centroids)
            
            # assign each point to a class with nearest centroid
            pred = np.argmin(dist, axis=1)

            # calculate new centroid and wss for each cluster
            new_centroids = []
            wss = []
            for c in range(self.k):
                if len(self.X[pred==c]) == 0:
                    new_centroid = np.random.uniform(size=self.centroids.shape[1])
                else:
                    new_centroid = np.mean(self.X[pred==c], axis=0)
                new_centroids += new_centroid,

                # within-cluster sum of squared difference
                wss += np.sum((self.X[pred==c] - new_centroid) ** 2),

            new_centroids = np.array(new_centroids)
            assert self.centroids.shape == new_centroids.shape
            self.centroids = new_centroids
            self.wss = np.array(wss)
                        
            if all(pred == self.pred):
                return time.time() - start_time
            else:                    
                if i % 5 == 0 and verbose:
                    elapsed = time.time() - start_time
                    print(f'iteration: {i:>3}\twss: {self.wss.sum():^10.2f}\ttime in seconds: {elapsed:>5.2f}')
                    
                self.pred = pred
                i += 1

def my_kmeanspp(data, start, end):
    scores = []
    models = []
    conv_time = []
    
    assert start < end
    
    for k in range(start, end):        
        kmeanspp = KMeansPP(data, k=k)
        t_elapsed = kmeanspp.predict()
            
        wss = kmeanspp.wss.sum()
        scores += wss,
        models += kmeanspp,
        conv_time += t_elapsed,
    return np.array(scores), conv_time
```



<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/newplot.png" alt="newplot" style="zoom:80%;" />

#### Image 1: M&M

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-15.png" alt="download-15" style="zoom:60%;" />

Using `k=7`

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-16.png" alt="download-16" style="zoom:60%;" />

#### Image 2: Peppers

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-17.png" alt="download-17" style="zoom:60%;" />

Using `k=8`

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-18.png" alt="download-18" style="zoom:60%;" />

#### Model evaluation & comparison

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-19.png" alt="download-19" style="zoom:60%;" />

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-20.png" alt="download-20" style="zoom:60%;" />



## Task 3: Eigenface

### 3.1 Explain why alignment is necessary for eigen-face

Alignment is necessary because face detection using eigenface is not invariant to position or scale. Each pixel coordinate position i, j is essentially treated as a distinct random variable (that may or may not correlate with others). If the images are not aligned, it will be difficult to find correlation or direction of maximum variance.

### 3.2. Perform PCA and show mean face

```python
import numpy as np
from numpy import linalg as LA

class PCA:
    def fit_transform(self, X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        input:
            X: n by m where n is the number of data points and m is the number of features
        returns:
        	Xapprox: rank k approximation of X where k = n_components
        """
        self.X = X
    
        # create mean face by averaging the pixel intensity values across all our flattened face images
        self.mean = self.X.mean(0)
        self.X_centered = X - self.mean
        
        U, s, VT = LA.svd(X, full_matrices=False)
        self.svd_output = (U, s, VT)
        self.ev = self.explained_variance(s)
        
        S = np.diag(s)
        Xapprox = U[:,:n_components] @ S[:n_components,:n_components] @ VT[:n_components,:]
        return Xapprox + self.mean
    
    def explained_variance(self, s):
        """calculate explained variance"""
        ev = s ** 2 / np.sum(s ** 2)
        return ev
    
k = 15
model = PCA()
A_approx = model.fit_transform(A, n_components=k)
```

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-21.png" alt="download-21" style="zoom:40%;" />

#### Given the high dimensionality, directly performing eigen value decomposition of the covariance matrix would be slow. Find a faster way to compute eigen values and vectors, explain the reason.

Assumption: We don't care about eigenvalues that are approximately 0.

There are two ways (among other more complex methods) to compute eigenvalues and eigenvectors for non-square matrices with high dimensionality.

Let $A$ be a real matrix $A \in \mathbb{R}^{n \times m}$ where $m \gt n$

1.  Eigendecomposition

	If $A$ is a real matrix, then $AA^T$ and $A^TA$ are symmetric square matrices.

	Matrix $AA^T = L \in \mathbb{R}^{n \times n}$ and $A^TA = L \in \mathbb{R}^{m \times m}$.

	Since the number of nonzero eigenvectors of $AA^T$ is not greater than the number of eigenvectors of $A^TA$, we can perform eigendecomposition of $AA^T = L \in \mathbb{R}^{n \times n}$ to get eigenvectors and eigenvalues of $A^TA = L \in \mathbb{R}^{m \times m}$. Since $AA^T = L \in \mathbb{R}^{n \times n}$ is smaller (135 x 135) than (45045, 45045), it is much faster.

	$\therefore$ If $v$ is an eigenvector with nonzero eigenvalue $\lambda$ of $A^TA$, then $Av$ is an eigenvector with the same eigenvalue of $AA^T$.

2. Singular Value Decomposition

	Perform SVD directly on $A$, taking advantage of existing algorithms that can obtain eigenvalues and eigenvectors of the covariance matrix of $A$ without having to compute $A^TA$.

### 3.3 Determine top k principal components and visualize top k eigenfaces

x axis represents the number of principal components, and y axis is the cumulative explained variance ratio.

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/image-20210430164830661.png" alt="image-20210430164830661" style="zoom:50%;" />

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-22.png" alt="download-22" style="zoom:60%;" />

### 3.4 Project test data to facespace spanned by first k eigenvectors and perform nearest neighbor search

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/image-20210430165141073.png" alt="image-20210430165141073" style="zoom:50%;" />

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-23.png" alt="download-23" style="zoom:60%;" />

Framing it as a classification problem, the accuracy is $\frac{3}{3}=100\%$

### 3.5 Repeat the above step with a photo of self

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/image-20210430165240832.png" alt="image-20210430165240832" style="zoom:50%;" />

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-24.png" alt="download-24" style="zoom:60%;" />

### 3.6 Repeat the above, with rest of selfies used as part of training set

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/image-20210430165355954.png" alt="image-20210430165355954" style="zoom:50%;" />

<img src="/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab2/report.assets/download-25-9765885.png" alt="download-25" style="zoom:60%;" />




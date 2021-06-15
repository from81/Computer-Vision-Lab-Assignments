Kai Hirota

SID: u7233149

## Task 1: 3D-2D Camera Calibration

### 1.1 Code for calibration

```python
# helper functions not included: to_homogenous(), to_heterogenous()

def calibrate(im: np.ndarray, XYZ: np.ndarray, uv: np.ndarray):
    """
    Compute the 3x4 camera calibration matrix C such that xi = C @ Xi, where Xi is the world coordinate and xi is the image pixel coordinate

    Parameters:
        im: Image of the calibration target.
        XYZ: N x 3 array of XYZ coordinates of the calibration target points.
        uv: N x 2 array of image coordinates of the calibration target points.

    Returns:
        C (np.ndarray): 3 x 4 camera calibration matrix
    """
    assert XYZ.shape[0] >= 6
    assert uv.shape[0] == XYZ.shape[0]

    n = uv.shape[0]
    
    if uv.shape[1] == 2:
        uv = to_homogenous_coord(uv)
    
    if XYZ.shape[1] == 3:
        XYZ = to_homogenous_coord(XYZ)

    # construct matrix A and get the eigenvec corresponding to the smallest non-zero eigenvalue
    A = get_A(uv, XYZ)
    U, S, VT = LA.svd(A)
    p = VT[-1]
    
    # normalize p to unit length
    p = p / (p @ p)
    C = p.reshape(3, 4)
    
    return C

def get_A(uv: np.ndarray, xyz: np.ndarray):
    """Construct matrix A"""
    n = uv.shape[0]
    A = np.vstack([build_A(uv[i], xyz[i]) for i in range(n)])
    assert A.shape == (2*n, 12)
    return A

def build_A(img_coord, world_coord):
    """Build A 2 rows at a time"""
    if len(img_coord) == 3:
        u, v, _ = img_coord
    else:
        u, v = img_coord

    if len(world_coord) == 4:
        x, y, z, _ = world_coord
    else:
        x, y, z = world_coord

    # build matrix A of 2n x 12
    A = np.array([
        [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u],
        [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v]
    ])
        
    return A

####### Normalized DLT #######
h, w, _ = img.shape

Tnorm = np.array([
    [w+h, 0, w/2],
    [0, w+h, h/2],
    [0, 0, 1]
])

eigvals, eigvecs = LA.eig( (XYZ - XYZ.mean(0)) @ (XYZ - XYZ.mean(0)).T )
eigvals = np.diag(1 / eigvals[:3])
eigvecs = eigvecs[:3, :3]

left_component = eigvecs @ eigvals @ LA.inv(eigvecs)
right_component = -eigvecs @ eigvals @ LA.inv(eigvecs) @ XYZ.mean(0)
right_component = right_component.reshape(len(right_component), -1)

Snorm = np.hstack([left_component, right_component])
Snorm = np.vstack([Snorm, [0,0,0,1]])

uv_norm = Tnorm @ to_homogenous_coord(uv).T
uv_norm = uv_norm.T

XYZ_norm = Snorm @ to_homogenous_coord(XYZ).T
XYZ_norm = XYZ_norm.T

##############################

# how to use the code
P = calibrate(img, XYZ, uv)

# if using Normalized DLT
# P = calibrate(img, XYZ_norm, uv_norm)
# P = LA.inv(Tnorm) @ P @ Snorm
# P = P / P[-1,-1]

# project coordinates from world to img
xyz_proj = P @ XYZ.T
xyz_proj = to_heterogenous_coord(xyz_proj.T)
```



### 1.2 Selected Image & Correspondence Points

![download](/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab3/report.assets/download-1344860.png)



### 1.3 Compute Calibration Matrix P and project correspondence points to pixel coordinate system

$$
P = \displaystyle \left[\begin{matrix}
0.0089 & -0.0041 & -0.0133 & 0.6928\\
-0.0004 & -0.0156 & 0.0023 & 0.7208\\
0 & 0 & 0 & 0.0021
\end{matrix}\right]
$$

![download-1](/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab3/report.assets/download-1-1344914.png)

Mean Squared Error (pixels): 0.5426



### 1.4 Extrinsic and Intrinsic Parameters

$$
K = \displaystyle \left[\begin{matrix}908.7838 & -1.4141 & 382.2355\\0 & 892.3563 & 293.5467\\0 & 0 & 1\end{matrix}\right]
$$

$$
R = \displaystyle \left[\begin{matrix}0.8174 & -0.1021 & -0.567\\0.1568 & -0.9076 & 0.3894\\-0.5543 & -0.4072 & -0.7259\end{matrix}\right]
$$

$$
t = \displaystyle \left[\begin{matrix}76.4326 & 56.7049 & 85.5121\end{matrix}\right]
$$



### 1.5 Focal length & pitch angle

#### Focal Length

Since $K_{1,1} \ne K_{2,2}$ and $K_{1,2} \ne 0$, we have an intrinsic parameter matrix where the aspect ratio is not 1.
$$
K = \displaystyle 

\left[\begin{matrix}
\alpha & \gamma & x_0 \\
0 & \beta & y_0\\
0 & 0 & 1
\end{matrix}\right]
$$
where

- $(x_0, y_0)=$ Principal point
- $\alpha = \alpha_x$
- $\beta = \frac{\alpha_y}{\sin\theta}$
- $\alpha_x,\ \alpha_y$ are horizontal and vertical focal length in pixel units
- $\theta=\pi/2 \implies \beta=\alpha_y$

Assuming $\theta=\pi/2$, $\alpha_x=f_x$ and $\alpha_y = f_y$. Referring to $K$ defined in (2) in [section 1.4](#1.4-extrinsic-and-intrinsic-parameters), we have

- $f_x = 908.7838$ (pixels)
- $f_y=892.3563$ (pixels)

#### Pitch Angle with respect to X-Z plane

The position, ${\displaystyle C}$, of the camera expressed in world coordinates is ${\displaystyle C=-R^{-1}T=-R^{T}T}$. Let $\vec p$ be a vector that lie on the X-Z plane and $\vec c$ be the position of the camera, with both vectors having the same basis (coordinate system). We have the geometric property of vectors:

$$
\cos \theta = 
\frac{\vec{c} \cdot \vec{p}}{\left\Vert\vec{c}\right\Vert  \cdot \left\Vert\vec{p}\right\Vert}
$$

where $\vec c, \vec p \in \mathbb{R}^{3}$

The angle $\theta$ between $C$ and the X-Z plane can be calculated as $\theta = \arccos (\cos (\theta))$.

```python
C = -R.T @ t.reshape(len(t), -1)
C.T
>> array([[-23.96402805,  94.08800595,  83.32400034]])

plane = np.array([7,0,7])
theta = np.arccos((C.T @ plane) / (LA.norm(C) * LA.norm(plane)))
theta
>> 1.23654328

np.degrees(theta)
>> 70.84871102
```

The pitch angle, or the angle between the camera's optical axis and the ground-plane, is approximately 70.85 degrees.



### 1.6 Resize

![download-1](/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab3/report.assets/download-1.png)

Resized image



#### Calibration Matrix P' and Parameters K', R', t'

$$
P' = 
\displaystyle \left[\begin{matrix}0.0089 & -0.0041 & -0.0133 & 0.693\\-0.0002 & -0.0158 & 0.0026 & 0.7206\\0.0 & 0.0 & 0.0 & 0.0043\end{matrix}\right]
$$

$$
K' = 
\displaystyle \left[\begin{matrix}442.1749 & 1.4406 & 182.1194\\0.0 & 442.5788 & 135.2311\\0.0 & 0.0 & 1.0\end{matrix}\right]
$$

$$
R' = 
\displaystyle \left[\begin{matrix}0.8133 & -0.1007 & -0.5731\\0.1607 & -0.9077 & 0.3876\\-0.5592 & -0.4073 & -0.7221\end{matrix}\right]
$$

$$
t' = \displaystyle \left[\begin{matrix}73.4314 & 58.4466 & 83.2765\end{matrix}\right]
$$

- Intrinsic parameters $K$
	- Focal length $K'_{1,1}, K'_{2,2}$ are approximately half of $K_{1,1}, K_{2,2}$ due to width and height being halved.
	- Skew parameter $|K_{1,2}| \approx |K'_{1,2}|$ since the width and height were scaled by the same factor. In other words, resizing while keeping the aspect ratio should not skew the image.
	- Principal point $K'_{1,3}, K'_{2,3}$ are approximately half of $K_{1,3}, K_{2,3}$ due to the width and height being halved.
- Extrinsic parameters
	- $R \approx R'$ since rotation is scale-invariant if width and height are scaled by the same factor.
	- $t \approx t'$ for the same reason as $R, R'$ above.



## Task 2: Two-View DLT based homography estimation

### 2.1 Code, images used, and correspondence points

```python
def homography(u2Trans: List[float], v2Trans: List[float], 
               uBase: List[float], vBase: List[float]):
    """
    Computes the homography H applying the Direct Linear Transformation
    
    Parameters:
        u2Trans : Vectors with coordinates u and v of the transformed image
        v2Trans : point (p')
        uBase : vectors with coordinates u and v of the original base
        vBase : image point p
    """
    uv_src = np.vstack([u2Trans, v2Trans]).T
    uv_dest = np.vstack([uBase, vBase]).T
    return _homography(uv_src, uv_dest)

def _homography(uv_src: np.ndarray, uv_dest: np.ndarray):
    assert uv_src.shape[0] >= 6
    assert uv_src.shape[0] == uv_dest.shape[0]

    if uv_src.shape[1] == 2:
        uv_src = to_homogenous_coord(uv_src)
    
    if uv_dest.shape[1] == 2:
        uv_dest = to_homogenous_coord(uv_dest)
        
    A = get_A(uv_src, uv_dest)
    U, S, VT = LA.svd(A)
    p = VT[-1]
    
    # normalize p to unit length
    p = p / (p @ p)
    H = p.reshape(3, 3)
    
    return H

def get_A(uv_src: np.ndarray, uv_dest: np.ndarray):
    """Iteratively build matrix A for homography"""
    n = uv_src.shape[0]
    A = np.vstack([build_A_homography(uv_src[i], uv_dest[i]) for i in range(n)])
    assert A.shape == (2*n, 9)
    return A
    
def build_A_homography(uv_src, uv_dest):
    """Build A 2 rows at a time"""
    assert len(uv_src) == 3 and len(uv_dest) == 3
    
    u, v, _ = uv_src
    x, y, _ = uv_dest

    A = np.array([
        [u, v, 1, 0, 0, 0, -x*u, -x*v, -x],
        [0, 0, 0, u, v, 1, -y*u, -y*v, -y]
    ])

    return A

# how to use the code
# uv_left, uv_right: numpy array containing xy coordinates of correspondence points
H = homography(
    u2Trans=uv_left[:, 0].tolist(), 
    v2Trans=uv_left[:, 1].tolist(), 
    uBase=uv_right[:, 0].tolist(), 
    vBase=uv_right[:, 1].tolist()
)

# warp the source image l using H
# l = Left.jpg = warp source
# r = Right.jpg = warp destination
x, y, _ = r.shape
size = (y, x)
l_warped = cv2.warpPerspective(l, H, dsize=size)

# project the correspondence points as well
uv_left = to_homogenous_coord(uv_left)
uv_left_proj = H @ uv_left.T
uv_left_proj = to_heterogenous_coord(uv_left_proj.T)
```



![download-5](/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab3/report.assets/download-5-1345195.png)



### 2.2 Compute homography matrix H

$$
H = 
\displaystyle \left[\begin{matrix}-0.0149 & 0.0004 & 0.9997\\-0.0024 & -0.0065 & 0.0169\\0 & 0 & -0.0047\end{matrix}\right]
$$



### 2.3 Warp image using H

![download](/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab3/report.assets/download-1345354.png)

![download-1](/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab3/report.assets/download-1-1345455.png)



![download](/Users/Kai/Dropbox/Documents/Classes/engn6528_computer_vision/lab3/report.assets/download.png)

```python
# euclidean distance between target and warped correspondence points in pixels
dist = np.sqrt(np.sum(np.square(uv_right - uv_left_proj), 1))
dist
>> array([0.48353272, 0.59920041, 0.27162193, 1.05412312, 0.54916902,
       0.2332485 ])
```

Mean Squared Error (pixels): 0.3556

The small distance between the target and warped points are result of manual input of correspondence points using `plt.ginput()`.


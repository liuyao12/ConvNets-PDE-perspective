# ConvNets from the PDE perspective

The 3x3 conv can be seen as a differential operator (of order ≤2): the so-called Sobel filters are partial derivatives in the x- and y-directions of the image, and the Gaussian kernel is (1+) the Laplacian.

$$ \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1\end{bmatrix} \sim \frac{\partial}{\partial x} $$ 
$$ \frac{1}{16}\begin{bmatrix}1&2&1\\2&4&2\\1&2&1\end{bmatrix} \sim 1+\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2} $$

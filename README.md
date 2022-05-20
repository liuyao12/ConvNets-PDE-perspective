# ConvNets from the PDE perspective

The 3x3 conv can be seen as a differential operator (of order ≤2): the so-called Sobel filters are partial derivatives in the x- and y-directions of the image, and the Gaussian kernel is (1+) the Laplacian.

$$
\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1\end{bmatrix} \sim \frac{\partial}{\partial x}
$$ 

$$
\frac{1}{16}\begin{bmatrix}1&2&1\\2&4&2\\1&2&1\end{bmatrix} \sim 1+\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2}
$$

By compounding multiple layers of 3x3 conv, with "identity skip connections" (ResNet), you are effectively solving a partial differential equation, numerically.

$$
\displaystyle \frac{\partial u}{\partial t} = \sigma (Lu), \quad L= \alpha + \beta \frac{\partial}{\partial x} + \gamma \frac{\partial}{\partial y} + \delta\frac{\partial^2}{\partial x\partial y} +\cdots
$$

\[Indeed, with $u_n$ the solution at the discrete time $t_n, we'd compute the next time-slice simply by 

$$
u_{n+1} = u_n+ \sigma (L u_n) \cdot \Delta t
$$
the so-called "forward Euler" method.] With the nonlinear activation $\sigma$, this is a nonlinear PDE, which is known for complicated behavior (chaos). But ReLU is rather mild, so perhaps some of the information is being passed down like a linear PDE, which is better understood. For example, compounding Sobel can "shift" the image in one direction, at a rate of one pixel per layer.

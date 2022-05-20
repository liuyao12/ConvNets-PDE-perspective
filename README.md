# ConvNets from the PDE perspective

---- *This started as a response to ConvNext, first on Twitter, then on Weights & Biases. If anything could be called the "ConvNets of the 2020s", it'd be, in my opinion, those that are designed from the PDE perspective. Everyone is invited to **open a discussion**, to ask for clarification, suggest and develop new ideas, a la the Polymath project, and to PR any code that you would like to share.*

The 3x3 conv can be seen as a **differential operator** (of order ≤2): the so-called Sobel filters are partial derivatives in the x- and y-directions of the image, and the Gaussian kernel is (1+) the Laplacian.

$$
\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1\end{bmatrix} \sim \frac{\partial}{\partial x}
$$ 

$$
\frac{1}{16}\begin{bmatrix}1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix} \sim 1+\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2}
$$

By compounding multiple layers of 3x3 conv, with "identity skip connections" (ResNet), you are effectively solving a **partial differential equation**, *numerically*.

$$
\displaystyle \frac{\partial u}{\partial t} = \sigma (Lu), \quad L= \alpha + \beta \frac{\partial}{\partial x} + \gamma \frac{\partial}{\partial y} + \delta\frac{\partial^2}{\partial x\partial y} +\cdots
$$

\[Indeed, with $u_n$ the solution at the discrete time $t_n$, we'd compute the next time-slice simply by 

$$
u_{n+1} = u_n+ \sigma (L u_n) \cdot \Delta t
$$

the so-called "forward Euler" method.] With the nonlinear activation $\sigma$, this is a *nonlinear* PDE, which is known for complicated behavior (chaos). But ReLU is rather mild, so perhaps some of the information is being passed down like a linear PDE, which is better understood. For example, compounding Sobel can "shift" the image in one direction, at a rate of one pixel per layer.

In fact, with multiple channels this is technically a system of PDEs, or a PDE with matrix coefficients. The coefficients $\alpha, \beta,\ldots$ are nothing but the weights that are being updated and optimized for the classification layer. The connection may be summarized in the form of a table:

Convolutional Neural Nets | Partial Differential Equations
:----:|:-------:
input layer | initial condition
feed forward | solving the equation numerically
hidden layers | solution at intermediate times
output (penultimate) layer | solution at final time
**convolution** with 3×3 kernel | **differential operator** of order ≤ 2
weights | coefficients
boundary handling <br> (padding) | boundary condition
multiple channels <br> [e.g. 16×16×3×3 kernel; <br> 16×16×1×1 kernel] | system of (coupled) PDEs <br> [16×16 matrix of differential operators; <br> 16×16 matrix of constants]
groups (=g) | matrix is block-diagonal (direct sum of g blocks)

The training of a ConvNet would be an **inverse problem**: we know the solutions (dataset), and look for the PDE that would yield those solutions. On the grand scheme of things, it is part of the **continuous formulation** of deep neural nets, which supplements the more traditional "statistical learning" interpretation. If you thought the curse of dimensionality was bad enough, you might find relief in optimization over an *infinite-dimensional* space, aka the *calculus of variations*. (I have very little to say about backpropagation, and will focus on the "feed forward" of ConvNets.)

![](/translation.gif)

If you think about it, a full 3x3 conv from (say) 64 channels to 64 channels is rather wasteful, for there can only be at most 9 different kernels (or rather, 9 linearly independent kernels). One way to address this is to take each of the 64 channels, convolve with 9 different kernels, and take linear combinations; in other words, 64 channels go into 64x9 channels (with groups=64), followed by a 1x1 conv. This is awkwardly named "depthwise separable convolution".

In fact, doing x9 is also rather wasteful, for the space of second-order differential operators is 6 or perhaps 5 dimensional, so I'd go with x4. That x1 already works so well (in MobileNet, etc.) is kind of surprising. Now, the

	64 --(3x3, groups=64) --> 64x4 --(1x1)--> 64

block is a lot like the Bottleneck block in ResNet. (I don't have a good intuition with the placement of the extra 1x1.)

What I'm saying is that the perspective of PDE can get you quite a lot of the design of the modern ConvNets — consisting almost entirely of 3x3 and 1x1, with skip connections — more naturally and more directly. Moreover, it suggests other ways of tweaking the architecture:

1. In such PDEs, one always needs to impose a *boundary condition* (BC), which is simply padding in conv. It seems that only "padding with 0" (Dirichlet BC) is ever used. One could instead try to implement the Neumann BC (by using `padding_mode="reflect"`), which could make traveling waves bounce back and thereby retain more information.

1. One special type of PDEs, called **symmetric hyperbolic systems** (Maxwell and Dirac equations are prominent linear examples), would be interesting to implement. Or instead, we could help make the ConvNet more "hyperbolic" by putting a suitable regularization term in the loss function.

1. The PDEs here are all "constant coefficients" (i.e., the coefficients are constant in the x and y variables, but do vary in t). What if we make them vary in x and y as well? That is, after the standard 3x3 conv, multiply the result by the "coordinate function" of the form $ax+by+c$. Taking the latest torchvision code for ResNet, here are the relevant changes that can be made (easily adaptable to other ConvNets):

``` python

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.


    expansion: int = 4


    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # self.conv2 = conv3x3(width, width, stride, groups, dilation)    # This is the original
        # self.bn2 = norm_layer(width)                                    # This is the original
        # self.conv3 = conv1x1(width, planes * self.expansion)            # This is the original
        self.conv2 = conv3x3(width, width * 4, stride, width, dilation)   # Modified
        self.bn2 = norm_layer(width * 4)                                  # Modified
        self.XY = None                                                    # Modified
        self.mix = conv1x1(6, width * 4)                                  # Modified
        self.conv3 = conv1x1(width * 4, planes * self.expansion)          # Modified
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x: Tensor) -> Tensor:
        identity = x


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)
        if self.XY is None or self.XY.size()[2] != out.size()[2]:            # Added
            N, C, H, W = out.size()                                          # Added
            XX = torch.from_numpy(np.indices((1, 1, H, W))[3] * 2 / W - 1)   # Added
            YY = torch.from_numpy(np.indices((1, 1, H, W))[2] * 2 / H - 1)   # Added
            ones = torch.from_numpy(np.ones((1, 1, H, W)))                   # Added
            self.XY = torch.cat([ones, XX, YY, XX*XX, XX*YY, YY*YY],         # Added
                                dim=1).type(out.dtype).to(out.device)        # Added
        out = out * self.mix(self.XY)                                        # Added
        out = self.bn2(out)
	out = self.relu(out)


        out = self.conv3(out)
        out = self.bn3(out)


        if self.downsample is not None:
            identity = self.downsample(x)


        out += identity
        out = self.relu(out)


        return out

```

One could motivate the addition of variable coefficients as enabling the ConvNet to learn to *rotate* and *scale* the image, just like how the Sobel can shift the image, but by different amounts for different parts of the image. But whether or not it actually *learns* these transformations is not guaranteed, nor easy to verify. At any rate, a better explanation may be that it at least expands the "expressive power" of the network.

I hope someone with resources can put this to more thorough tests on ImageNet, and share the results. It seems that only with solid results will it convince more people to take this perspective seriously.

I'd bet that Yann LeCun did understand PDEs well when he introduced the ConvNet, but purposefully framed it in terms of convolutions. It's a bit unfortunate that, without the guide of PDE, the field had missed many opportunities to improve the architecture design, or did so with ad hoc reasoning. The first to note the connection between ResNet and differential equations or dynamical systems is perhaps Weinan E, an applied mathematician from Princeton. The Neural ODE paper also starts out from the same observation, but it treats each pixel as a dynamical variable (hence ODE), interacting with its immediate neighbors; it's more natural to speak of PDEs, if somewhat limited to ConvNets, so that both the depth (t) and the image dimensions (x, y) are continuous. To this day, the PDE perspective is still not widely adopted among mainstream AI researchers; see, for example, A ConvNet for the 2020s. The mathematics isn't complicated; I recommend 3blue1brown's excellent 2-part introduction, focusing on the heat equation. 

Last but not least, the PDE or dynamical systems perspective also provides a partial answer (though somewhat useless) to the problem of interpretability. The "hypothesis space" is now a class of PDEs that seems to be rich enough for traces of *integrability within chaos* — analogous to the long-term stability and predictability of the solar system, despite the fact that the three-body problem is chaotic — and that gradient descent is somehow quite effective in finding them.


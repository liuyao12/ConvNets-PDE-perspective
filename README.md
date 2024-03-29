# ConvNets from the PDE perspective

* This version started out as a reaction to [ConvNext: A ConvNet for the 2020s](https://github.com/facebookresearch/ConvNeXt), first on Twitter, then expanded [on Weights & Biases](https://wandb.ai/liuyao12/posts/ConvNets-from-the-PDE-perspective--VmlldzoxNzY2NDE2). In my (very biased) opinion, if anything is to be called the "ConvNets of the 2020s", it would be those that are designed from the PDE perspective.
* A newer version is now published as the appendix in [A Novel Convolutional Neural Network Architecture with a Continuous Symmetry](https://arxiv.org/abs/2308.01621).
* Everyone is invited to **open [a discussion](https://github.com/liuyao12/ConvNets-PDE-perspective/discussions)**, to offer criticisms, ask for clarifications, suggest or develop new ideas and experiments, à la the Polymath projects (open collaborations of research math that work quite well for iterations of "small" improvements, rather than major conceptual breakthroughs); and of course to PR any code or results that you would like to share. The hope is to allow *anyone* from *anywhere* to contribute, to move the field forward, without the inefficient cycle of published papers that repeat many of the same things over, while sweeping other things under the rug. Unlike the Polymath projects, there doesn't appear to be a clearly defined goal here.

The key observation is that the 3x3 conv can be seen as a **differential operator** (of order ≤2): the so-called Sobel filters are partial derivatives in the $x$- and $y$-directions of the image, and the Gaussian kernel is (1+) the Laplacian.

$$
\begin{bmatrix} -1 & 0 & 1 \\\\ -2 & 0 & 2 \\\\ -1 & 0 & 1\end{bmatrix} \sim \frac{\partial}{\partial x}
$$ 

$$
\frac{1}{16}\begin{bmatrix}1 & 2 & 1 \\\\ 2 & 4 & 2 \\\\ 1 & 2 & 1 \end{bmatrix} \sim 1+\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2}
$$

By compounding multiple layers of 3x3 conv, with "identity skip connections" (ResNet), you are effectively solving a **partial differential equation**, *numerically*.

$$
\displaystyle \frac{\partial u}{\partial t} = \sigma (Lu), \quad L= \alpha + \beta \frac{\partial}{\partial x} + \gamma \frac{\partial}{\partial y} + \delta\frac{\partial^2}{\partial x\partial y} +\cdots
$$

\[Indeed, with $u_n$ the solution at the discrete time $t_n$, we'd compute the next time-slice simply by 

$$
u_{n+1} = u_n+ \sigma (L u_n) \cdot \Delta t
$$

the so-called "forward Euler" method.] With the nonlinear activation $\sigma$, this is a *nonlinear* PDE, which is known for complicated behavior (chaos). But ReLU is rather mild, so perhaps some of the information is being passed down like a linear PDE, which is better understood. For example, compounding Sobel "shifts" the image in one direction, at a rate of one pixel per layer:

![translation](translation.gif)

In fact, with multiple channels this is technically a *system* of PDEs, or a PDE with matrix coefficients. The coefficients $\alpha, \beta,\ldots$ are nothing but the weights that are being updated and optimized for the classification layer. This perspective may be summarized in the form of a dictionary:

Convolutional Neural Nets | Partial Differential Equations
:----:|:-------:
`x=x+Conv(x)` | $\frac{\partial u}{\partial t} = \Delta u$
input layer | initial condition
feed-forward | solving the equation numerically
hidden layers | solution at intermediate times
output layer | solution at final time
**convolution** with 3×3 kernel | **differential operator** of order ≤ 2
weights | coefficients
boundary handling <br> (padding) | boundary condition
multiple channels <br> [e.g. 16×16×3×3 kernel; <br> 16×16×1×1 kernel] | system of (coupled) PDEs <br> [16×16 matrix of differential operators; <br> 16×16 matrix of constants]
groups (=g) | matrix is block-diagonal (direct sum of g blocks)

The training of a ConvNet would be an **inverse problem**: we know the solutions (dataset), and look for the PDE that would yield those solutions. On the grand scheme of things, it is part of the **continuous formulation** of deep neural nets, which supplements the more traditional "statistical learning" interpretation. If you thought the curse of dimensionality was bad enough, you might find relief in optimization over an *infinite*-dimensional space, aka the *calculus of variations*. (I have very little to say about backpropagation, and will focus on the "feed forward" or the "architecture" of ConvNets.)

If you think about it, a full 3x3 conv from (say) 64 channels to 64 channels is rather wasteful, for there can only be at most 9 different kernels (or rather, 9 linearly independent kernels). One way to address this is to take each of the 64 channels, convolve with 9 different kernels, and take linear combinations; in other words, 64 channels go into 64x9 channels (3x3 with `groups=64`), followed by a 1x1 conv. This is awkwardly named "depthwise separable convolution".

In fact, doing x9 is also rather wasteful, for the space of second-order differential operators is 6 or perhaps 5 dimensional, so I'd go with x4. That x1 already works so well (in MobileNet, etc.) is kind of surprising. Now, the

	64 --(3x3, groups=64) --> 64x4 --(1x1)--> 64

block is a lot like the Bottleneck block in ResNet. (I don't have a good intuition with the placement of the other 1x1.)

What I'm saying is that the perspective of PDE can get you quite a lot of the design of the modern ConvNets — consisting almost entirely of 3x3 and 1x1, with skip connections — more naturally and more directly. Moreover, it suggests other ways of tweaking the architecture:

1. In such PDEs, one always needs to impose a *boundary condition* (BC), which is simply padding in conv. It seems that only "padding with 0" (Dirichlet BC) is ever used. One could instead try to implement the Neumann BC (by using `padding_mode="reflect"`), which could make traveling waves bounce back and thereby retain more information.

1. One important class of PDEs, called **symmetric hyperbolic systems** (Maxwell and Dirac equations are prominent *linear* examples), would be interesting to implement. Or instead, we could help make the ConvNet more "hyperbolic" by putting a suitable regularization term in the loss function. In a sense, hyperbolic equations are better suited for numerical simulation due to the "finite speed of propagation".

1. The PDEs here are all "constant coefficients" (i.e., the coefficients are constant in the $x$ and $y$ variables, but do vary in $t$). What if we make them vary in $x$ and $y$ as well? That is, after the standard 3x3 conv, multiply the result by the "coordinate function" of the form $ax+by+c$. Taking the latest torchvision code for ResNet, here are the relevant changes that can be made (easily adaptable to other ConvNets):

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
        self.mix = conv1x1(5, width * 4, bias=True)                       # Modified
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
        if self.XY is None or self.XY.size()[2] != out.size()[2]:                  # Added
            N, C, H, W = out.size()                                                # Added
            XX = torch.Tensor(np.indices((1,1,H,W))[3]*2/(W-1)-1)                  # Added
            YY = torch.Tensor(np.indices((1,1,H,W))[2]*2/(H-1)-1)                  # Added
            self.XY = torch.cat([XX, YY,  XX*XX, XX*YY, YY*YY],                    # Added
                                   dim=1).type(out.dtype).to(out.device)           # Added
        out = out * self.mix(self.XY)                                              # Added
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

One could motivate the addition of variable coefficients as enabling the ConvNet to *rotate* and *scale* the image—just like how the Sobel can shift the image, but by different amounts for different parts of the image. But whether or not it actually *learns* these transformations is not guaranteed, nor easy to verify. At any rate, a better explanation may be that it at least expands the "expressive power" of the network.

![](rotation.gif) ![](dilation.gif)

I hope someone with resources can put this to more thorough tests on ImageNet, and share the results. It seems that only with solid results will it convince more people to take this perspective seriously.

I'd bet that Yann LeCun did understand PDEs well when he introduced the first ConvNet, but purposefully framed it in terms of convolutions. It's a bit unfortunate that, without the guide of PDE, the field had missed many opportunities to improve the architecture design, or did so with ad hoc reasoning. The first to note the connection between ResNet and differential equations or dynamical systems is perhaps Weinan E, an applied mathematician from Princeton. The Neural ODE paper also starts out from the same observation, but it treats each pixel as a dynamical variable (hence ODE), interacting with its immediate neighbors; it's more natural to speak of PDEs, if somewhat limited to Computer Vision, so that both the depth ($t$) and the image dimensions ($x$, $y$) become continuous. To this day, the PDE perspective is still not widely adopted among mainstream AI researchers; see, for example, [A ConvNet for the 2020s](https://github.com/facebookresearch/ConvNeXt). The mathematics isn't complicated; I recommend 3blue1brown's excellent 2-part introduction, focusing on the heat equation. 

https://youtu.be/ly4S0oi3Yz8

Last but not least, the PDE or dynamical systems perspective points to a partial answer (though somewhat useless one) to the problem of interpretability. The "hypothesis space" is now a class of PDEs that seems to be rich enough for traces of *integrability amidst chaos* — analogous to the long-term stability and predictability of the solar system, despite the fact that the three-body problem is chaotic — and that gradient descent is somehow quite effective in finding them. Though it would be nice to "prove" such a *universal integrability hypothesis*, it's also fine to assume it and see what new ideas it would inspire.

It should be added that this perspective is getting some exposure in recent years, with several workshops at top venues on it:
* 2020 ICLR workshop: ![Integration of Deep Neural Models and Differential Equations](https://iclr.cc/virtual_2020/workshops_5.html)
* 2021 NeurIPS workshop: ![Symbiosis of Deep Learning and Differential Equations (DLDE)](https://dl-de.github.io/)


## Appendix
One thing that might be hard to find in a PDE course is how repeated *local* operations (first-order differential operators) give rise to a *global* transformation. This is "well known", but under different names in different areas: infinitesimal generator and the exponential map (Lie group/Lie algebra), semigroup of unbounded operators (functional analysis), the method of characteristics (linear PDEs), flow of a vector field (differential geometry), and even "observables" or self-adjoint operators in quantum mechanics. The simplest way to understand it is by way of the **exponential** — the master key for solving *any* linear differential equation — of $\frac{d}{dx}$ (in one dimension)

$$
e^{a\frac{d}{dx}}f(x)=\left(1+a\frac{d}{dx}+\frac{1}{2!}a^2\frac{d^2}{dx^2}+\cdots\right)f(x)=f(x+a)
$$

which is nothing but the formula for Taylor expansion. We would say that translation (in $x$-direction) is *generated* by the operator $\frac{d}{dx}$. One might worry that $a$ has to be within the radius of convergence, but amazingly this "works" even for non-smooth functions. To get a feel for it, recall that the exponential can also be defined by

$$
e^L = \lim_{n\to\infty} (1+\frac{L}{n})^n
$$

which actually involves two limiting processes, and if we take the two limits simultaneously, differentiability of $f$ is not required after all. That may explain why the gifs above look so nice, better than one might expect. By the way, the rotation and scaling (centered at the origin) are generated by

$$
L=-y\frac{\partial}{\partial x}+x\frac{\partial}{\partial y} \quad \text{and}\quad L=x\frac{\partial}{\partial x}+y\frac{\partial}{\partial y}
$$
 
respectively. Note that these are differential operators with *variable* coefficients.

IMO, this is the kind of thing that would be hard to come up with, or to justify, *without* the continuous formulation of neural nets. Unfortunately, the mathematics is just beyond what the standard CS and ML training would prepare you, but should and can be made more accessible with little or no jargon.

# Optimisation-based registration

---

## General problem

Most classical registration software cast the problem as optimising (let's say minimizing) a functional of the form

$$
\mathcal{L} = \underbrace{\mathcal{S}\left(\mathbf{f}, g\circ\boldsymbol\phi ~\mid~ \boldsymbol\theta\right)}\_{\text{similarity}} ~~~+ \underbrace{\mathcal{R}(\boldsymbol\theta)}\_{\text{regularity}} ~.
$$

> [!NOTE]
> - $\mathcal{S}$ is the similarity metric, which measures how well the fixed and moved images match each other.
> - $\mathcal{R}$ is the regulariser, which measures the regularity of the transformation.
> - $\mathbf{f} \in \mathbb{R}^N$ is the fixed image, disceretised into $N$ voxels.
> - $g \in \mathcal{C}\left(\mathbb{R}^3 \rightarrow \mathbb{R}\right)$ is the moving image, defined as a continuous function of space. Think B-spline encoding.
> - $\boldsymbol\phi \in \mathbb{R}^{N \times 3}$ is the transformation field evaluated at the fixed image voxels.
>   - More formally, I could have defined $\varphi \in \mathcal{C}\left(\mathbb{R}^3 \rightarrow \mathbb{R}^3\right)$, the continuous spatial transformation, 
>     and $\boldsymbol\phi = \left\\{ \varphi\left( \mathbf{x}\_{n} \right) \right\\}_{1 \leqslant n \leqslant N}$ the discretized transfomation.
>   - Abusing notations, we have $\left(g\circ\boldsymbol\phi\right) \in \mathbb{R}^N$.
> - $\boldsymbol\theta \in \mathbb{R}^K$ is the parameterisation of $\boldsymbol\phi$ &mdash; that's what we optimize.


> [!TIP]
> A nice property of this framework is that when the similarity is well chosen, it can be seen as finding a maximum _a posteriori_ value for $\boldsymbol\theta$, with
> 
> $$
> \mathcal{L} = -\ln\underbrace{p\left(\boldsymbol\theta ~\mid~ \mathbf{f}, g\right)}\_{\text{posterior}} = -\ln\underbrace{p\left(\mathbf{f}  ~\mid~ g, \boldsymbol\theta\right)}\_{\text{likelihood}} ~-~ \ln\underbrace{p(\boldsymbol\theta)}\_{\text{prior}} + \ln p\left(\mathbf{f}  ~\mid~ g\right) ~.
> $$

---

## Flavoured Gradient Descent

To simplify things, let us keep only references to $\boldsymbol\theta$, since that's what we optimise for:

$$
\mathcal{L}\left(\boldsymbol\theta\right) = \mathcal{S}\left(\boldsymbol\theta\right) + \mathcal{R}(\boldsymbol\theta) ~.
$$

In 99% of the cases, the regulariser is actually quadratic in the parameters, so we have 

$$
 \mathcal{R}(\boldsymbol\theta) = \frac{1}{2}\boldsymbol\theta^{\mathrm{T}}\mathbf{R}\boldsymbol\theta ~~~~\text{with}~~~~ \mathbf{R} \in \mathbb{R}^{K \times K} ~.
$$

Classically finite-difference-based regularisation is used, in which case (assuming circulant boundary conditions), the matrix $\mathbf{R}$ is Toeplitz-circulant, and its inverse $\mathbf{R}^{-1}$ can be computed using Fourier transforms. Let us further note that the gradient and Hessian of the regularisation term are
- $\boldsymbol\nabla\mathcal{R}(\boldsymbol\theta) = \mathbf{R}\boldsymbol\theta$,
- $\mathcal{H}\mathcal{R}(\boldsymbol\theta) = \mathbf{R}$.

When it comes to the similarity term, let us also write its gradient and Hessian as
- $\boldsymbol\nabla\mathcal{S}(\boldsymbol\theta) = \mathbf{g}_{\boldsymbol\theta}$,
- $\mathcal{H}\mathcal{S}(\boldsymbol\theta) = \mathbf{H}_{\boldsymbol\theta}$.

> [!NOTE]
> In general the similarity is not quadratic in $\boldsymbol\theta$, so the gradient and Hessian both depend on the point at which they are evaluated in some unknown way.

Let's optimise! The simplest optimisation scheme is gradient descent with step size $\gamma$:

$$
\boldsymbol\theta^{\text{new}} = \boldsymbol\theta - \gamma\left(\mathbf{g}_{\boldsymbol\theta} + \mathbf{R}\boldsymbol\theta\right) ~.
$$

On the other end of the spectrum, if the Hessian of the similarity term is known, we can use [Newton-Raphson](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization) (also known simply as _Newton's optimisation method_):

$$
\boldsymbol\theta^{\text{new}} = \boldsymbol\theta - \left(\mathbf{H}\_{\boldsymbol\theta} + \mathbf{R}\right)^{-1}\left(\mathbf{g}_{\boldsymbol\theta} + \mathbf{R}\boldsymbol\theta\right) ~.
$$

However, they both belong to the broader family of [preconditioned gradient descent methods](https://en.wikipedia.org/wiki/Preconditioner#Preconditioning_in_optimization), of the form

$$
\boldsymbol\theta^{\text{new}} = \boldsymbol\theta - \mathbf{P}\left(\mathbf{g}_{\boldsymbol\theta} + \mathbf{R}\boldsymbol\theta\right) ~,
$$

where $\mathbf{P}\_{\boldsymbol\theta}$ is known as the preconditioner. In Newton's case, the preconditioner changes at each step, with $\mathbf{P}\_{\boldsymbol\theta} = \left(\mathbf{H}\_{\boldsymbol\theta} + \mathbf{R}\right)^{-1}$, whereas in gradient descent, the preconditioner is fixed and $\mathbf{P} = \gamma\mathbf{I}_K$. [This chapter ](https://web.eecs.umich.edu/~fessler/book/c-opt.pdf) in Jeff Fessler's unpublished textbook made me realise that most optimisation techniques are variants of preconditioned GD. Now, in general we don't know $\mathbf{H}\_{\boldsymbol\theta}$ (and even when we know it, it may not be the best choice). Designing a good optimiser therefore reduces to finding a good preconditioner, that is both efficient to compute&mdash;so that each step is computationaly efficient&mdash;and close enough to the true Hessian&mdash;so that each step is close to optimal in some sense. 

---

## Preconditioners

Since the regulariser is quadratic, all preconditioners will have the shape $\mathbf{P} = \left(\mathbf{H}^{\text{approx}} + \mathbf{R}\right)$, where $\mathbf{H}^{\text{approx}}$ is some alternative to the true Hessian of the similarity term. In this section, we will focus on variants of $\mathbf{H}^{\text{approx}}$.

[Gauss-Newton](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm) is generally presented in the context of nonlinear least-squares optimization. Keeping it general, let's assume that

$$
\mathcal{S}(\boldsymbol\theta) = \frac{1}{2} \mathcal{F}(\boldsymbol\theta)^{\mathbf{T}} \mathbf{W} \mathcal{F}(\boldsymbol\theta)
$$

where $\mathbf{W} \in \mathbb{R}^{M \times M}$ is a positive semi-definite matrix of mixing weights (generally diagonal) and $\mathcal{F} \in \mathcal{C}(\mathbb{R}^K \rightarrow \mathbb{R}^M)$ is some nonlinear function. By application of the chain rule, its gradient and Hessian are
- $\boldsymbol\nabla\mathcal{S}(\boldsymbol\theta) = \boldsymbol\nabla\mathcal{F}(\boldsymbol\theta)^{\mathrm{T}} \mathbf{W} \mathcal{F}(\boldsymbol\theta)$.
- $\mathcal{H}\mathcal{S}(\boldsymbol\theta) = \boldsymbol\nabla\mathcal{F}(\boldsymbol\theta)^{\mathrm{T}} \mathbf{W} \boldsymbol\nabla\mathcal{F}(\boldsymbol\theta) + \sum_{m}\mathcal{H}\mathcal{F}_m(\boldsymbol\theta) \cdot \mathbf{w}_m^{\text{T}} \mathcal{F}(\boldsymbol\theta)$.

Ths idea is that you're essentially discarding the second term of the Hessian, giving us the preconditioner

$$
\mathbf{H}^{\text{approx}} = \boldsymbol\nabla\mathcal{F}(\boldsymbol\theta)^{\mathrm{T}} \mathbf{W} \boldsymbol\nabla\mathcal{F}(\boldsymbol\theta)
$$

Discarding the second term of the Hessian derives from the use of [Fisher's scoring](https://en.wikipedia.org/wiki/Scoring_algorithm), where the assumption $\boldsymbol\nabla\mathcal{S}(\boldsymbol\theta) = \mathbf{0}$ is plugged back into the Hessian. In this case, this yields $\mathcal{F}(\boldsymbol\theta) = \mathbf{0}$ and therefore the nulling of the second term. One of its main advantages is that it is always positive semi-definite, even in cases when the true Hessian $\mathcal{H}\mathcal{S}(\boldsymbol\theta)$ is negative semi-definite (which can happen!).

> [!NOTE]
> Fisher's scoring can be applied in more general contexts than nonlinear least-squares,
> and is a common way to obtain approximate Hessians to use as preconditioners.
> 
> For example, JA uses it when optimizing bias fields in unified segmentation. As you probably know, he does not
> log transform the data, and enforces bias field positivity by wrapping it in an exponential. The Gaussian log-likelihood in a cluster is
>
> $$
> \ln p(x \mid \beta, \mu, \sigma^2) = -\frac{1}{2\sigma^2} (\exp(\beta) x - \mu)^2 + \beta ~,
> $$
> 
> where the trailing $\beta$ comes from the log-determinant of the variance. The gradient and Hessian of the negative log-likelihhod with respect to $\beta$ are
> - $\nabla\mathcal{L}(\beta) = \frac{\exp(\beta)x}{\sigma_2} (\exp(\beta) x - \mu) - 1$
> - $\mathcal{H}\mathcal{L}(\beta) = 1 + \nabla\mathcal{L}(\beta) + \frac{\left(\exp(\beta)x\right)^2}{\sigma^2}$
>
> Adding the condition $\nabla\mathcal{L}(\beta) = 0$ and plugging it in the Hessian gives us
> - $\mathcal{H}^{\text{Fisher}}\mathcal{L}(\beta) = 1 + \frac{\left(\exp(\beta)x\right)^2}{\sigma^2}$
>
> This si typically a case where the true Hessian can be negative, but Fisher's Hessian is not!

One problem with Fisher's scoring is that the approximate Hessian can be _less positive-definite_ than the true Hessian. Before delving into it, let's better define what we mean by "more or less positive-definite".

> [!TIP]
> A partial order on the cone of positive-semidefinite (PSD) matrices can be defined, and it is known as [Loewner's order](https://en.wikipedia.org/wiki/Loewner_order).
>
> Given two PSD matrices $\mathbf{X}$ and $\mathbf{Y}$, we say that $\mathbf{X}$ _majorises_ $\mathbf{Y}$ (_i.e._, is more positive-definite), noted $\mathbf{X} \succeq \mathbf{Y}$, if $\mathbf{X} - \mathbf{Y}$ is positive semi-definite (also written $\mathbf{X} - \mathbf{Y} \succeq \mathbf{0}$).

> [!IMPORTANT]
> In optimization, Loewner's order matters because, for a convex function $\mathcal{L}$, using a preconditioner whose inverse is more positive definite than all Hessians of $\mathcal{L}$ ensures that preconditioned gradient descent converges. If you're familiar with maximization-minimization framework, it's quite easy to see that with such a preconditioner the quadratic
> 
> $$
> \mathcal{Q}(\boldsymbol\theta) = \mathcal{L}(\boldsymbol\theta_0) + \boldsymbol\nabla\mathcal{L}(\boldsymbol\theta_0) \left(\boldsymbol\theta - \boldsymbol\theta_0\right) + \frac{1}{2}\left(\boldsymbol\theta - \boldsymbol\theta_0\right)^{\mathrm{T}}\mathbf{P}^{-1}\left(\boldsymbol\theta - \boldsymbol\theta_0\right) ~,
> $$
>
> which the preconditioned gradient descent step happens to minimize, is above $\mathcal{L}(\boldsymbol\theta)$ everywhere.

> [!NOTE]
> In [Model-based multiparameter mapping](https://arxiv.org/pdf/2102.01604.pdf) (which is not very well written, sorry...) we show that an approximate Hessian of the form
>
> $$
> \mathbf{H}^{\text{aprox}} = \boldsymbol\nabla\mathcal{F}(\boldsymbol\theta)^{\mathrm{T}} \mathbf{W} \boldsymbol\nabla\mathcal{F}(\boldsymbol\theta) + \text{diag}\left(\left|\sum\_{m}\mathcal{H}\mathcal{F}_m(\boldsymbol\theta) \cdot \mathbf{w}\_m^{\text{T}} \mathcal{F}(\boldsymbol\theta)\right| \mathbf{1}\right)~,
> $$
>
> which is a majorizer of the true Hessian, works better than the Gauss-Newton preconditioner.

[Levenberg-Marquardt](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) simply consists of choosing a preconditioner that interpolates between gradient descent and Gauss-Newton. Their preconditioners are:
- $\mathbf{H}^{\text{Levenberg}} = \mathbf{H}^{\text{Fisher}} + \lambda \mathbf{I}$
- $\mathbf{H}^{\text{Marquardt}} = \mathbf{H}^{\text{Fisher}} + \lambda \text{diag}\left( \mathbf{H}^{\text{Fisher}} \right)$

Another preconditioner that sometimes work well (and sometimes doesn't) is the diagonal of the true Hessian (or the diagonal of Fisher's Hessian), _i.e_, $\mathbf{H}^{\text{Jacobi}} = \text{diag}(\mathbf{H})$. It is the preconditioner used in [Jacobi's method](https://en.wikipedia.org/wiki/Jacobi_method) for diagonally-dominant systems.

Finally, in registration (on generally in regulatised problems), we may set $\mathbf{H}^{\text{approx}} = \mathbf{0}$, which means our preconditioner is the inverse of the regulariser:

$$
\mathbf{P}^{\text{Hilbert}} = \gamma\mathbf{R}^{-1}
$$

giving the update step

$$
\boldsymbol{\theta}^{\text{new}} = \boldsymbol\theta - \gamma\mathbf{R}^{-1}\left(\mathbf{g}_{\boldsymbol\theta} + \mathbf{R}\boldsymbol\theta\right) = \boldsymbol\theta - \gamma\left(\mathbf{R}^{-1}\mathbf{g}\_{\boldsymbol\theta} + \boldsymbol\theta\right) = (1 - \gamma)\boldsymbol\theta  - \gamma\mathbf{R}^{-1}\mathbf{g}\_{\boldsymbol\theta} ~.
$$

These Hilbert gradients were introduced in [Beg et al. (2005)&mdash;section 6.1](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=ede22f3682bc33b361f5c67a17476a46ea3d3f4c). Interestingly, only $\mathbf{R}^{-1}$ appears in the update equation, not $\mathbf{R}$. While (as said before) $\mathbf{R}^{-1}$ can be computed from  $\mathbf{R}$ if one uses something like the membrane or bending energy, most LDDMM folks prefer to directly implement $\mathbf{R}^{-1}$ as a convolution with a Gaussian kernel (_i.e._, in other words, $\mathbf{R}^{-1}$ is the squared-exponential covariance matrix commonly used in Gaussian processes), even though that means no analytical expression for $\mathbf{R}$.

---
title:
    A crash course in machine learning
---

One hugely important area in which my knowledge is close to zero is machine
learning and artificial intelligence. It's not only a terrible but also pitiful
idea to finish a graduate degree in computer science without knowing at least
somewhat how this stuff works, so I decided to follow the openly available
[Stanford CS229 machine learning
course](https://www.youtube.com/watch?v=jGwO_UgTS7I) and provide a summary for
each chapter here since exercise material and lecture notes appear to have been
taken down. I try to apply as much as possible and follow along using Rust. The
code can be found under
[Baseng0815/ml-implementations](https://github.com/Baseng0815/ml-implementations).

# Linear Regression

## Objectives

Linear regression allows us to make predictions by training a model on a given
set of training data and then applying this model to new values. More formally,
given training data in the form of $m$ pairs $(x^{(i)},y^{(i)})$ of feature
vectors $x^{(i)}\in\mathbb{R}^d$ and values $y^{(i)}\in\mathbb{R}$, we try to
find parameters $\theta$ so that the prediction function
$h_\theta:\mathbb{R}^d\rightarrow\mathbb{R}$ is as accurate as possible w.r.t.
some loss metric $\mathcal{J}:\theta\rightarrow\mathbb{R}$:

\begin{gather}
    arg\min_\theta\mathcal{J(\theta)}
\end{gather}

Now the question of how to choose $h_\theta$ and $\mathcal{J}$ naturally
arises. Since we are doing linear regression, we will choose a linear model
which will make a prediction by linear combination of features using weights
$\theta_i$. To account for bias, we will also add a dummy feature $x_0=1$ so
that $\theta_0 x_0=\theta_0$. Conveniently, this allows us to rewrite
predictions as matrix-vector multiplication and opens the door to all the neat
linear algebra tools:

\begin{gather}
    h_\theta(x)=\sum_{j=0}^d \theta^{(i)}x^{(i)}=\theta^Tx^{(i)}
\end{gather}

The loss metric usually employed is mean-squared error (MSE) which is defined
as follows:

\begin{gather}
    \mathcal{J}(\theta)=\frac{1}{m}\sum_{i=1}^d\mathcal{J}_{x^{(i)}}(\theta)=\frac{1}{m}\sum_{i=1}^d (h_\theta(x^{(i)})-y^{(i)})^2
\end{gather}

Why did we define the loss metric this way? The simple answer you might have
already come across is that we want to avoid negative loss values and taking the
absolute value is bad since it leaves $\mathcal{J}$ non-continuous and thus
non-differentiable. This explanation alone (at least for me) was unsatisfactory
since we could have also chosen
$\mathcal{J}_{x^{(i)}}(\theta)=(h_\theta(x^{(i)})-y^{(i)})^4$ or basically any
other positive, differentiable function. A more technical explanation can be
found in the assumption of the distribution of noise in the data. Since almost
all real-life data exhibits some form of measuring error and other unmodeled
randomness, our prediction can never be fully accurate. Let $\Sigma^{(i)}$
denote the error terms; the situation then looks like this:

\begin{gather}
    y^{(i)}=\theta^{(i)}x^{(i)}+\Sigma^{(i)}
\end{gather}

We now assume $\Sigma^{(i)}$ to be iid (independently and identically
distributed) following a Gaussian distribution with mean $\mu=0$ and some
unknown standard deviation $\sigma$:

\begin{align}
    \Pr(\Sigma^{(i)})=\mathcal{N}(0,\sigma^2)&=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(\Sigma^{(i)})^2}{2\sigma^2}) \\
    \Pr(\Sigma^{(i)}\vert x^{(i)};\theta)&=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})
\end{align}

Using this, we can define the likelihood $\mathcal{L}:\mathbb{R}^{d+1}\rightarrow\mathbb{R}$
as the probability for observing certain outcomes $y^{(i)}$ to be realized
given input features $x^{(i)}$ and parameters $\theta$:

\begin{align}
    \mathcal{L}(\theta)=\Pr(y\vert x;\theta)&=\Pr(y^{(1)}\vert x^{(1)};\theta)\cdot\Pr(y^{(2)}\vert x^{(2)};\theta)\cdot\dotsc\cdot\Pr(y^{(m)}\vert x^{(m)};\theta) \\
                                            &=\prod_{i=1}^m \Pr(y^{(i)}\vert x^{(i)};\theta) \\
                                            &=\prod_{i=1}^m \frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})
\end{align}

Maximizing $\mathcal{L}$ is equivalent to maximizing the probability to predict
correctly. Since the above function is inconvenient to work with, we can
transform it and maximize the log likelihood
$l(\theta)=\log\mathcal{L}(\theta)$ instead. This is possible since $\log$ is a
strictly increasing function:

\begin{align}
l(\theta)=\log\mathcal{L}(\theta)=&\log\prod_{i=1}^m \frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}) \\
                                  &=\sum_{i=1}^m \log\frac{1}{\sqrt{2\pi}\sigma}-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2} \\
                                  &=m\log(\frac{1}{\sqrt{2\pi}\sigma})-\frac{1}{2\sigma^2}\sum_{i=1}^m (y^{(i)}-\theta^Tx^{(i)})^2
\end{align}

It is now easy to see that maximizing $l(\theta)$ is equivalent to minimizing
$\sum_{i=1}^m (y^{(i)}-\theta^Tx^{(i)})^2$ which is equivalent to minimizing
the MSE since the result stays invariant under constant factors.

## Finding parameters $\theta$

We have discussed the objective, but the important problem of how to find
parameters to maximize $\mathcal{L}(\theta)$/minimize $\mathcal{J(\theta)}$. A
very common iterative method for finding minima of differentiable functions is
gradient descent: starting at point $\theta^{(t)}$, the most direct path to the
nearest minimum is the path of steepest descent. Imagine being lost on a foggy
mountain and trying to find your way back down: even though you can only see a
few meters at best, you can easily find your way down by looking for the
steepest slope near you, taking a step, then looking again and so on.

In the simple one-dimensional case for $f:\mathbb{R}\rightarrow\mathbb{R}$, the
steepest descent at point $x$ is given by the derivative $f'(x)$. Since the
error function $\mathcal{J}$ we are trying to minimize is vector-valued, we
instead look at the gradient
$\nabla_\theta\mathcal{J}:\mathbb{R}^{d+1}\rightarrow\mathbb{R}$ which is
nothing more than a row vector containing all partial derivatives of
$\mathcal{J}$ with respect to $\theta_j$:

\begin{gather}
    \nabla_\theta\mathcal{J}=\left(\frac{\partial\mathcal{J}}{\partial\theta_0},\frac{\partial\mathcal{J}}{\partial\theta_1},\dots,\frac{\partial\mathcal{J}}{\partial\theta_d}\right)
\end{gather}

Parameters are then iteratively updated by calculating the gradient of the
training data, multiplying it by a value $\alpha$ and then subtracting it from
the previous parameters:

\begin{gather}
    \theta^{(t+1)}\leftarrow \theta^{(t)}-\alpha\nabla_\theta\mathcal{J}(\theta^{(t)})
\end{gather}

This is repeated until $\theta^{(t)}$ converges. $\alpha$ is called a
hyperparameter and in this case determines as the learning rate: these kinds of
values are common in machine learning algorithms and control the behavior in a
multitude of ways. Set $\alpha$ too small and $\theta^{(t)}$ will take too long
to converge. Set it too big and the model might overfit or parameters will
oscillate around the optimum without ever coming closer to it. The correct
value depends on the problem at hand and is best determined experimentally (see
below for an example).

Because calculating the gradient can be expensive for large datasets, an
approximative algorithm called stochastic gradient descent is often used in
practice. It works by extracting random fixed-sized batches from the training
data and using these to approximate the gradient for the whole training set.

Deriving the gradient is very straightforward since we are only dealing with
linear functions:

\begin{alignat}{2}
    \nabla_\theta\mathcal{J}(\theta^{(t)})&=\nabla_\theta\frac{1}{m}\sum_{i=1}^m\mathcal{J}_{x^{(i)}}(\theta)&&=\frac{1}{m}\sum_{i=1}^m\nabla_\theta\mathcal{J}_{x^{(i)}}(\theta^{(t)}) \\
                                          &=\frac{1}{m}\sum_{i=1}^m\nabla_\theta(h_\theta(x^{(i)})-y^{(i)})^2&&=\frac{2}{m}\sum_{i=1}^m x^{(i)}(h_\theta(x^{(i)})-y^{(i)})
\end{alignat}

Although this already yields a nice formula that can be used as-is, we can
simplify notation even more by packing all $m$ training examples into a matrix
$X\in\mathbb{R}^{m\times (d+1)}$. We then have

\begin{gather}
    X\theta=\begin{bmatrix}x^{(1)}_0&\cdots&x^{(1)}_d\\\vdots&\ddots&\vdots\\x^{(m)}_0&\cdots&x^{(m)}_d\end{bmatrix}\begin{bmatrix}\theta_0\\\vdots\\\theta_d\end{bmatrix}=
    \begin{bmatrix}(x^{(1)})^T\theta\\\vdots\\(x^{(m)})^T\theta\end{bmatrix}=\begin{bmatrix}\theta^T x^{(1)}\\\vdots\\\theta^T x^{(m)}\end{bmatrix}=\begin{bmatrix}h_\theta(x^{(1)})\\\vdots\\h_\theta(x^{(m)})\end{bmatrix}=Y
\end{gather}

Rewriting $\mathcal{J}(\theta)$ and deriving $\nabla_\theta\mathcal{J}$ using
basic algebra from which I will spare you:

\begin{align}
    \mathcal{J}(\theta)&=\frac{1}{m}(X\theta-Y)^T(X\theta-Y) \\
    \nabla_\theta\mathcal{J}(\theta)&=\nabla_\theta\frac{1}{m}(X\theta-Y)^T(X\theta-Y)=\frac{1}{m}\nabla_\theta(\theta^TX^T-Y^T)(X\theta-Y)=\dots=X^TX\theta-X^TY
\end{align}

We have now reduced the gradient to a closed form. This brings with it one last
optimization: we can now directly find the optimal $\theta$ without needing to
do any iterations at all! Since the loss function $\mathcal{J}$ is convex, it
has one and only one local minimum and no maxima. This allows us to set the
gradient to zero and directly solve for $\theta$:

\begin{gather}
    \nabla_\theta\mathcal{J}(\theta)\stackrel{!}{=}0\Rightarrow\theta=(X^TX)^{-1}X^TY
\end{gather}

The matrix $(X^TX)^{-1}X^T$ is also known as the [Moore-Penrose
pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) and
is used to calculate a least squares solution to a system of linear equations
which is exactly what we were doing in the first place. It can happen that
$(X^TX)$ is not invertible if we have linearly dependent data points. This can
be fixed by removing these data points or by computing an approximation to the
pseudoinverse which also works very well.

## An illustrated example

I'll use the [Boston housing
dataset](https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset)
as an example for linear regression. The goal is to predict housing prices
based on 13 parameters. The data was preprocessed by removing all samples not
in the $[0.05, 0.95]$ range and by normalizing all features to the range
$[0,1]$ so large features don't overwhelm smaller ones. A few features with
their optimal regression lines look like this:

![](/res/machine-learning/housing_features.jpg)

Implementing the aforementioned algorithms yields the following curves for batch
gradient descent and stochastic gradient descent respectively:

![](/res/machine-learning/regression_loss_bgd.jpg)
![](/res/machine-learning/regression_loss_sgd.jpg)

All curves converge to the minimal loss of $21.8948$ as given by the normal
equation, but some do so more quickly. Using higher learning rates than $1$
results in divergence.

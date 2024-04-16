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
denote the error terms; the true situation then looks like this:

\begin{gather}
    y^{(i)}=\theta^{(i)}x^{(i)}+\Sigma^{(i)}
\end{gather}

We now assume $\Sigma^{(i)}$ to be iid (independently and identically
distributed) following a Gaussian distribution with mean $\mu=0$ and some
unknown standard deviation $\sigma$:

\begin{align}
    p(\Sigma^{(i)})=\mathcal{N}(0,\sigma^2)&=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(\Sigma^{(i)})^2}{2\sigma^2}) \\
    p(\Sigma^{(i)}\vert x^{(i)};\theta)&=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})
\end{align}

Using this, we can define the likelihood
$\mathcal{L}:\mathbb{R}^{d+1}\rightarrow\mathbb{R}$ as the probability for
certain outcomes $y^{(i)}$ to be realized given input features $x^{(i)}$ and
parameters $\theta$:

\begin{align}
    \mathcal{L}(\theta)=p(y\vert x;\theta)&=p(y^{(1)}\vert x^{(1)};\theta)\cdot p(y^{(2)}\vert x^{(2)};\theta)\cdot\dotsc\cdot p(y^{(m)}\vert x^{(m)};\theta) \\
                                            &=\prod_{i=1}^m p(y^{(i)}\vert x^{(i)};\theta) \\
                                            &=\prod_{i=1}^m \frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})
\end{align}

Maximizing $\mathcal{L}$ is equivalent to maximizing the probability to predict
correctly. Since the above function is inconvenient to work with, we can
transform it and maximize the log likelihood
$l(\theta)=\ln\mathcal{L}(\theta)$ instead. This is possible since $\ln$ is a
strictly increasing function:

\begin{align}
l(\theta)=\ln\mathcal{L}(\theta)=&\ln\prod_{i=1}^m \frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}) \\
                                  &=\sum_{i=1}^m \ln\frac{1}{\sqrt{2\pi}\sigma}-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2} \\
                                  &=m\ln(\frac{1}{\sqrt{2\pi}\sigma})-\frac{1}{2\sigma^2}\sum_{i=1}^m (y^{(i)}-\theta^Tx^{(i)})^2
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

# Logistic regression

Our goal in the previous example was to predict a continuous value from a
feature vector. We now turn to the problem of classification, i.e. assigning
distinct classes to features. Logistic regression allows us to do binary
classification in a way not too dissimilar to linear regression.

Linear regression itself is ill-suited for classification since it produces
unbounded continuous values while bounded values $h_\theta\in[0,1]$ are more
desirable for obvious reasons. It also tends to deal poorly with new data and
outliers. We can solve both of these problems by taking the output of the
linear regression model and putting it through the logistic function
$g:\mathbb{R}\rightarrow[0,1],z\mapsto\frac{1}{1+e^{-z}}$:

\begin{gather}
    h_\theta(x)=g(\theta^T x)=\frac{1}{1+e^{-\theta^T x}}
\end{gather}

<div>
<script type="text/javascript">
window.PlotlyConfig = { MathJaxConfig: "local" };
</script>
<script
charset="utf-8"
src="https://cdn.plot.ly/plotly-2.30.0.min.js"
></script>
<div
id="812141df-a4db-4e96-9c65-ee1e2fdf68ad"
class="plotly-graph-div"
style="height: 500px; width: 100%"
></div>
<script type="text/javascript">
window.PLOTLYENV = window.PLOTLYENV || {};
if (document.getElementById("812141df-a4db-4e96-9c65-ee1e2fdf68ad")) {
Plotly.newPlot(
"812141df-a4db-4e96-9c65-ee1e2fdf68ad",
[
{
hovertemplate:
"x=%{x}\u003cbr\u003ey=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [
-10.0, -9.9, -9.8, -9.700000000000001, -9.600000000000001,
-9.500000000000002, -9.400000000000002, -9.300000000000002,
-9.200000000000003, -9.100000000000003, -9.000000000000004,
-8.900000000000004, -8.800000000000004, -8.700000000000005,
-8.600000000000005, -8.500000000000005, -8.400000000000006,
-8.300000000000006, -8.200000000000006, -8.100000000000007,
-8.000000000000007, -7.9000000000000075, -7.800000000000008,
-7.700000000000008, -7.6000000000000085, -7.500000000000009,
-7.400000000000009, -7.30000000000001, -7.20000000000001,
-7.10000000000001, -7.000000000000011, -6.900000000000011,
-6.800000000000011, -6.700000000000012, -6.600000000000012,
-6.500000000000012, -6.400000000000013, -6.300000000000013,
-6.2000000000000135, -6.100000000000014, -6.000000000000014,
-5.900000000000015, -5.800000000000015, -5.700000000000015,
-5.600000000000016, -5.500000000000016, -5.400000000000016,
-5.300000000000017, -5.200000000000017, -5.100000000000017,
-5.000000000000018, -4.900000000000018, -4.8000000000000185,
-4.700000000000019, -4.600000000000019, -4.5000000000000195,
-4.40000000000002, -4.30000000000002, -4.200000000000021,
-4.100000000000021, -4.000000000000021, -3.9000000000000217,
-3.800000000000022, -3.7000000000000224, -3.6000000000000227,
-3.500000000000023, -3.4000000000000234, -3.300000000000024,
-3.200000000000024, -3.1000000000000245, -3.000000000000025,
-2.9000000000000252, -2.8000000000000256, -2.700000000000026,
-2.6000000000000263, -2.5000000000000266, -2.400000000000027,
-2.3000000000000274, -2.2000000000000277, -2.100000000000028,
-2.0000000000000284, -1.9000000000000288, -1.8000000000000291,
-1.7000000000000295, -1.6000000000000298, -1.5000000000000302,
-1.4000000000000306, -1.300000000000031, -1.2000000000000313,
-1.1000000000000316, -1.000000000000032, -0.9000000000000323,
-0.8000000000000327, -0.700000000000033, -0.6000000000000334,
-0.5000000000000338, -0.4000000000000341,
-0.30000000000003446, -0.20000000000003482,
-0.10000000000003517, -3.552713678800501e-14,
0.09999999999996412, 0.19999999999996376, 0.2999999999999634,
0.39999999999996305, 0.4999999999999627, 0.5999999999999623,
0.699999999999962, 0.7999999999999616, 0.8999999999999613,
0.9999999999999609, 1.0999999999999606, 1.1999999999999602,
1.2999999999999599, 1.3999999999999595, 1.4999999999999591,
1.5999999999999588, 1.6999999999999584, 1.799999999999958,
1.8999999999999577, 1.9999999999999574, 2.099999999999957,
2.1999999999999567, 2.2999999999999563, 2.399999999999956,
2.4999999999999556, 2.5999999999999552, 2.699999999999955,
2.7999999999999545, 2.899999999999954, 2.999999999999954,
3.0999999999999535, 3.199999999999953, 3.2999999999999527,
3.3999999999999524, 3.499999999999952, 3.5999999999999517,
3.6999999999999513, 3.799999999999951, 3.8999999999999506,
3.9999999999999503, 4.09999999999995, 4.1999999999999496,
4.299999999999949, 4.399999999999949, 4.4999999999999485,
4.599999999999948, 4.699999999999948, 4.799999999999947,
4.899999999999947, 4.999999999999947, 5.099999999999946,
5.199999999999946, 5.299999999999946, 5.399999999999945,
5.499999999999945, 5.599999999999945, 5.699999999999944,
5.799999999999944, 5.8999999999999435, 5.999999999999943,
6.099999999999945, 6.1999999999999424, 6.29999999999994,
6.399999999999942, 6.499999999999943, 6.599999999999941,
6.699999999999939, 6.79999999999994, 6.899999999999942,
6.99999999999994, 7.0999999999999375, 7.199999999999939,
7.29999999999994, 7.399999999999938, 7.499999999999936,
7.5999999999999375, 7.699999999999939, 7.799999999999937,
7.899999999999935, 7.999999999999936, 8.099999999999937,
8.199999999999935, 8.299999999999933, 8.399999999999935,
8.499999999999936, 8.599999999999934, 8.699999999999932,
8.799999999999933, 8.899999999999935, 8.999999999999932,
9.09999999999993, 9.199999999999932, 9.299999999999933,
9.399999999999931, 9.499999999999929, 9.59999999999993,
9.699999999999932, 9.79999999999993, 9.899999999999928,
],
xaxis: "x",
y: [
4.539786870243442e-5, 5.017216468376423e-5,
5.544852472279494e-5, 6.12797396166024e-5,
6.772414961977015e-5, 7.484622751061113e-5,
8.271722285166628e-5, 9.141587385216132e-5,
0.00010102919390777258, 0.00011165334062956242,
0.00012339457598623134, 0.0001363703270794966,
0.00015071035805975695, 0.00016655806477733527,
0.0001840719049634231, 0.00020342697805520555,
0.0002248167702332942, 0.00024845508183933307,
0.0002745781561013311, 0.00030344703002891716,
0.00033535013046647583, 0.0003706061406263941,
0.00040956716498604734, 0.00045262222324053155,
0.0005002011070795601, 0.0005527786369235948,
0.0006108793594343958, 0.0006750827306328318,
0.0007460288338366899, 0.0008244246863982869,
0.000911051194400636, 0.0010067708200856265,
0.0011125360328603092, 0.0012293986212774065,
0.0013585199504289429, 0.0015011822567369735,
0.0016588010801744015, 0.0018329389424927814,
0.0020253203890498554, 0.0022378485212763027,
0.00247262315663474, 0.0027319607630110214,
0.0030184163247083803, 0.003334807307413295,
0.0036842398994359313, 0.0040701377158960635,
0.004496273160941109, 0.00496680165005688,
0.005486298899450314, 0.006059801491584011,
0.00669285092428474, 0.007391541344281842,
0.008162571153159747, 0.009013298652847659,
0.009951801866904135, 0.010986942630592971,
0.012128434984274005, 0.013386917827664511,
0.01477403169327276, 0.016302499371440606,
0.017986209962091184, 0.01984030573407709,
0.021881270936130008, 0.02412702141766868,
0.02659699357686527, 0.029312230751355667,
0.03229546469844978, 0.03557118927263536,
0.039165722796763454, 0.043107254941085124,
0.047425873177565664, 0.0521535630784165,
0.057324175898867374, 0.06297335605699497,
0.06913842034334514, 0.07585818002124169, 0.08317269649392033,
0.09112296101485388, 0.09975048911968266, 0.10909682119561022,
0.1192029220221146, 0.1301084743629946, 0.14185106490048427,
0.15446526508353087, 0.16798161486607135, 0.18242552380635185,
0.19781611144141342, 0.2141650169574362, 0.2314752165009768,
0.24973989440487648, 0.26894142136998883, 0.2890504973749894,
0.31002551887238056, 0.3318122278318266, 0.35434369377419694,
0.3775406687981375, 0.40131233988753984, 0.42555748318833253,
0.4501660026875135, 0.47502081252105116, 0.4999999999999911,
0.524979187478931, 0.549833997312469, 0.57444251681165,
0.5986876601124431, 0.6224593312018458, 0.6456563062257868,
0.6681877721681576, 0.6899744811276042, 0.710949502624996,
0.7310585786299971, 0.7502601055951101, 0.7685247834990105,
0.7858349830425518, 0.8021838885585753, 0.8175744761936375,
0.8320183851339188, 0.8455347349164599, 0.8581489350995071,
0.8698915256369975, 0.8807970779778779, 0.8909031788043829,
0.9002495108803109, 0.9088770389851402, 0.9168273035060743,
0.9241418199787533, 0.9308615796566504, 0.9370266439430008,
0.9426758241011287, 0.94784643692158, 0.9525741268224311,
0.9568927450589121, 0.9608342772032339, 0.9644288107273622,
0.967704535301548, 0.9706877692486423, 0.9734030064231328,
0.9758729785823296, 0.9781187290638684, 0.9801596942659214,
0.9820137900379076, 0.9836975006285582, 0.9852259683067263,
0.9866130821723345, 0.9878715650157252, 0.9890130573694063,
0.9900481981330951, 0.9909867013471517, 0.9918374288468397,
0.9926084586557177, 0.9933071490757148, 0.9939401985084155,
0.9945137011005493, 0.9950331983499427, 0.9955037268390586,
0.9959298622841037, 0.9963157601005639, 0.9966651926925865,
0.9969815836752914, 0.9972680392369888, 0.9975273768433651,
0.9977621514787236, 0.9979746796109501, 0.998167061057507,
0.9983411989198255, 0.998498817743263, 0.9986414800495709,
0.9987706013787224, 0.9988874639671396, 0.9989932291799142,
0.9990889488055994, 0.9991755753136017, 0.9992539711661631,
0.999324917269367, 0.9993891206405654, 0.9994472213630764,
0.9994997988929205, 0.9995473777767595, 0.9995904328350139,
0.9996293938593735, 0.9996646498695334, 0.9996965529699712,
0.9997254218438986, 0.9997515449181605, 0.9997751832297667,
0.9997965730219448, 0.9998159280950366, 0.9998334419352227,
0.9998492896419403, 0.9998636296729204, 0.9998766054240137,
0.9998883466593704, 0.9998989708060922, 0.9999085841261478,
0.9999172827771484, 0.9999251537724895, 0.9999322758503801,
0.9999387202603833, 0.9999445514752772, 0.9999498278353162,
],
yaxis: "y",
type: "scatter",
},
],
{
template: { data: { scatter: [{ type: "scatter" }] } },
xaxis: { anchor: "y", domain: [0.0, 1.0], title: { text: "z" }, fixedrange: true },
yaxis: { anchor: "x", domain: [0.0, 1.0], title: { text: "g(z)" }, fixedrange: true },
legend: { tracegroupgap: 0 },
title: { text: "$\\text{The logistic function } g(z)=\\frac{1}{1+e^{-z}}$", font: { size: 30 } },
height: 500,
},
{ responsive: true }
);
}
</script>
</div>

We pursue the same strategy as before and find a maximum likelihood estimate
$\theta$. Let $y=0$ and $y=1$ indicate membership of $y$ to the first resp. the
second class. Assume $h_\theta(x)$ outputs values s.t.

\begin{align}
    \Pr[y=1\vert x;\theta]&=h_\theta(x) \\
    \Pr[y=0\vert x;\theta]&=1-h_\theta(x) \\
    p(y\vert x;\theta)&=h_\theta(x)^y(1-h_\theta)^{(1-y)}
\end{align}

Since $y^{(i)}$ are assumed to be iid, the probability of observing all given
outcomes $\vec y$ given $\vec x$ under $\theta$ is precisely equal to the
product of observing each outcome separately:

\begin{gather}
    \mathcal{L}(\theta)=p(\vec y\vert\vec x;\theta)=\prod_{i=1}^m p(y^{(i)}\vert x^{(i)};\theta)=\prod_{i=1}^m h_\theta(x^{(i)})^{y^{(i)}}(1-h_\theta(x^{(i)}))^{1-y^{(i)}}
\end{gather}

## Iterative approach 1: gradient ascent

Unfortunately, no normal equation exists for logistic regression so we resort
to an iterative approach. Since we want to maximize $\mathcal{L}(\theta)$, we
can do gradient ascent which is pretty much the same as gradient descent but
with a flipped sign. We will once again utilize the log likelihood
$l(\theta)=\ln\mathcal{L}(\theta)$ instead for simpler algebra:

\begin{gather}
    \theta^{t+1}\leftarrow\theta^t+\nabla_\theta\mathcal{l}(\theta^t)
\end{gather}

Deriving the gradient:

\begin{align}
    \nabla_\theta l(\theta)&=\nabla_\theta\ln\prod_{i=1}^m h_\theta(x^{(i)})^{y^{(i)}}(1-h_\theta(x^{(i)}))^{1-y^{(i)}} \\
    &=\nabla_\theta\sum_{i=1}^m \ln(h_\theta(x^{(i)})^{y^{(i)}})+\ln((1-h_\theta(x^{(i)}))^{1-y^{(i)}}) \\
    &=\nabla_\theta\sum_{i=1}^m y^{(i)}\ln(h_\theta(x^{(i)}))+(1-y^{(i)})\ln(1-h_\theta(x^{(i)})) \\
    &=\sum_{i=1}^m y^{(i)}\nabla_\theta\ln\left(\frac{1}{1+e^{-\theta^Tx^{(i)}}}\right)+(1-y^{(i)})\nabla_\theta\ln\left(1-\frac{1}{1+e^{-\theta^Tx^{(i)}}}\right) \\
    &=\sum_{i=1}^m -y^{(i)}\nabla_\theta\ln\left(1+e^{-\theta^Tx^{(i)}}\right)+(1-y^{(i)})\nabla_\theta\ln\left(\frac{e^{-\theta^Tx^{(i)}}}{1+e^{-\theta^Tx^{(i)}}}\right) \\
    &=\sum_{i=1}^m -y^{(i)}\frac{1}{1+e^{-\theta^Tx^{(i)}}}\nabla_\theta(1+e^{-\theta^Tx^{(i)}})+(1-y^{(i)})\nabla_\theta(\ln(e^{-\theta^Tx^{(i)}})-\ln(1+e^{-\theta^Tx^{(i)}})) \\
    &=\sum_{i=1}^m \frac{y^{(i)}e^{-\theta^Tx^{(i)}}}{1+e^{-\theta^Tx^{(i)}}}x^{(i)}+(1-y^{(i)})(\nabla_\theta(-\theta^Tx^{(i)})-\nabla_\theta\ln(1+e^{-\theta^Tx^{(i)}}))) \\
    &=\sum_{i=1}^m \frac{y^{(i)}}{1+e^{\theta^Tx^{(i)}}}x^{(i)}+(1-y^{(i)})(-x^{(i)}+\frac{1}{1+e^{\theta^Tx^{(i)}}}x^{(i)}) \\
    &=\sum_{i=1}^m x^{(i)}\left(y-\frac{1}{1+e^{-\theta^Tx^{(i)}}}\right) \\
    &=\sum_{i=1}^m x^{(i)}(y-g(\theta^Tx^{(i)}))
\end{align}

This yields the following final equation (notice the similarity to linear regression):

\begin{gather}
    \theta^{t+1}\leftarrow\theta^t+\nabla_\theta\mathcal{l}(\theta^t)=\theta^t+\sum_{i=1}^m x^{(i)}(y-h_{\theta^{t}}(x^{(i)}))
\end{gather}

## Iterative approach 2: Newton's method

Newton's method is another iterative method which is faster in many situations.
While gradient descent works by taking a small step towards the minimum or
maximum, Newton's method can be used for finding roots. Since the derivative at
a minimum/maximum is equal to zero, using gradient descent on
$f:\mathbb{R}^n\rightarrow\mathbb{R}$ is equivalent to finding the root of
$f'$.

Consider the polynomial $f(x)=x^3+0.5x^2-4x-1$. We start with some initial $x^0$,
let's say $x^0=0.7$. This is then updated iteratively as follows:

\begin{gather}
    x^{t+1}\leftarrow x^t-\frac{f(x^t)}{f'(x^t)}
\end{gather}

We find the point $(x,y=f(x))$ on the graph and calculate where the tangent
line cuts across the x axis. This intersection point then becomes the basis for
the next iteration. The first three iterations look like this:

<div>
<script type="text/javascript">
window.PlotlyConfig = { MathJaxConfig: "local" };
</script>
<script
charset="utf-8"
src="https://cdn.plot.ly/plotly-2.30.0.min.js"
></script>
<div
id="8a2b8f99-3079-445f-8f75-d29c7f03ee6f"
class="plotly-graph-div"
style="height: 800px; width: 100%"
></div>
<script type="text/javascript">
window.PLOTLYENV = window.PLOTLYENV || {};
if (document.getElementById("8a2b8f99-3079-445f-8f75-d29c7f03ee6f")) {
Plotly.newPlot(
"8a2b8f99-3079-445f-8f75-d29c7f03ee6f",
[
{
hovertemplate:
"x=%{x}\u003cbr\u003ey=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [
-2.5, -2.4, -2.3, -2.1999999999999997, -2.0999999999999996,
-1.9999999999999996, -1.8999999999999995, -1.7999999999999994,
-1.6999999999999993, -1.5999999999999992, -1.4999999999999991,
-1.399999999999999, -1.299999999999999, -1.1999999999999988,
-1.0999999999999988, -0.9999999999999987, -0.8999999999999986,
-0.7999999999999985, -0.6999999999999984, -0.5999999999999983,
-0.4999999999999982, -0.39999999999999813,
-0.29999999999999805, -0.19999999999999796,
-0.09999999999999787, 2.220446049250313e-15,
0.10000000000000231, 0.2000000000000024, 0.3000000000000025,
0.4000000000000026, 0.5000000000000027, 0.6000000000000028,
0.7000000000000028, 0.8000000000000029, 0.900000000000003,
1.000000000000003, 1.1000000000000032, 1.2000000000000033,
1.3000000000000034, 1.4000000000000035, 1.5000000000000036,
1.6000000000000032, 1.7000000000000037, 1.8000000000000043,
1.900000000000004, 2.0000000000000036, 2.100000000000004,
2.2000000000000046, 2.3000000000000043, 2.400000000000004,
],
xaxis: "x",
y: [
-3.5, -2.3439999999999994, -1.3219999999999992,
-0.42799999999999727, 0.3440000000000021, 1.0000000000000027,
1.546000000000002, 1.988000000000003, 2.3320000000000016,
2.5840000000000014, 2.7500000000000013, 2.8360000000000003,
2.848, 2.791999999999999, 2.673999999999998,
2.4999999999999973, 2.2759999999999962, 2.0079999999999956,
1.7019999999999946, 1.363999999999994, 0.9999999999999933,
0.6159999999999928, 0.2179999999999922, -0.18800000000000838,
-0.5960000000000087, -1.0000000000000089, -1.394000000000009,
-1.7720000000000087, -2.1280000000000086, -2.456000000000008,
-2.750000000000007, -3.004000000000006, -3.212000000000005,
-3.368000000000004, -3.466000000000002, -3.5,
-3.4639999999999977, -3.351999999999995, -3.1579999999999924,
-2.875999999999989, -2.499999999999985, -2.023999999999983,
-1.4419999999999762, -0.7479999999999674, 0.06400000000003381,
1.0000000000000355, 2.066000000000047, 3.2680000000000593,
4.6120000000000605, 6.10400000000006,
],
yaxis: "y",
type: "scatter",
},
{
line: { dash: "dot" },
marker: { color: "orange" },
mode: "lines+markers+text",
name: "Iteration 1",
text: "$x_1$",
x: [0.7, 0.7],
y: [0, -3.2119999999999997],
type: "scatter",
},
{
line: { color: "orange" },
mode: "lines",
showlegend: false,
x: [-2.5, 2.5],
y: [2.644000000000001, -6.506],
type: "scatter",
},
{
line: { dash: "dot" },
marker: { color: "red" },
mode: "lines+markers+text",
name: "Iteration 2",
text: "$x_2$",
x: [-1.055191256830601, -1.055191256830601],
y: [0, 2.602599209886551],
type: "scatter",
},
{
line: { color: "red" },
mode: "lines",
showlegend: false,
x: [-2.5, 2.5],
y: [5.080309657506005, -3.494217799270848],
type: "scatter",
},
{
line: { dash: "dot" },
marker: { color: "green" },
mode: "lines+markers+text",
name: "Iteration 3",
text: "$x_3$",
x: [0.462442935260996, 0.462442935260996],
y: [0, -2.6439499812275162],
type: "scatter",
},
{
line: { color: "green" },
mode: "lines",
showlegend: false,
x: [-2.5, 2.5],
y: [5.935274863604935, -8.544708434497998],
type: "scatter",
},
{
marker: { color: "black" },
showlegend: false,
mode: "markers+text",
text: "$x_4$",
x: [-0.4505242715360961],
type: "scatter",
},
],
{
template: { data: { scatter: [{ type: "scatter" }] } },
xaxis: { anchor: "y", domain: [0.0, 1.0], title: { text: "x" }, fixedrange: true },
yaxis: { anchor: "x", domain: [0.0, 1.0], title: { text: "y" }, fixedrange: true },
legend: { tracegroupgap: 0 },
title: { text: "$f(x)=x^3+0.5x^2-4x-1$" },
},
{ responsive: true }
);
}
</script>
</div>

Seven iterations are already enough for my Python implementation to show an
error value of 0.0. This methods converges quadratically, meaning after every
step, the number of correct digits roughly doubles. A disadvantage compared to
gradient descent is that a second order derivative is required since we want to
find $x$ s.t. $f'(x)=0$. If we work with feature vectors
$\theta\in\mathbb{R}^n$, the mathematics stays the same, although we now need
some vector calculus instead of simple derivatives: scalars are replaced by
vectors and we'll use the inverse Jacobian instead of $f'(x)^{-1}$. More
concretely, finding a root $\theta\in\mathbb{R}^n$ for a function
$f:\mathbb{R}^n\rightarrow\mathbb{R}$ can be done as follows:

\begin{gather}
    \theta^{t+1}\leftarrow\theta^t-J_{f}(\theta^t)^{-1}f(\theta^t)
\end{gather}

If $f(\theta)=\mathcal{L}(\theta)$ for a loss function
$\mathcal{L}:\mathbb{R}^n\rightarrow\mathbb{R}$ whose minimum is to be found, we
need to find the root of its derivative (i.e. the gradient) instead. We'll
then use the Hessian $H$,
$H_{ij}=\frac{\partial^2\mathcal{L}}{\partial\theta_i\partial\theta_j}$ instead of the
Jacobian which is the matrix of all second-order partial derivatives:

\begin{gather}
    \theta^{t+1}\leftarrow\theta^t-H_{\mathcal{L}}(\theta^t)^{-1}\nabla_\theta\mathcal{L}(\theta^t)
\end{gather}

# Perceptron

A somewhat simpler model is that of the Perceptron: it uses the Heaviside step
function instead of a logistic curve, yielding a binary 0/1 output without any
probability information:

\begin{align}
    H(z)&=\left\{\begin{array}{lr}0&z<0\\1&z\geq 0\end{array}\right. \\
    h_\theta(x)=H(\theta^Tx)&=\left\{\begin{array}{lr}0&\theta^Tx<0\\1&\theta^Tx\geq 0\end{array}\right.
\end{align}

Iterative training for a data point $(x^{(i)}, y^{(i)})$ with learning rate
$\alpha$ is done as follows:

\begin{gather}
    \theta^{t+1}\leftarrow\theta^t+\alpha(y^{(i)}-h_\theta(x^{(i)}))x^{(i)}=\left\{\begin{array}{lr}\theta^t+\alpha x^{(i)}&y^{(i)}\neq H(\theta^Tx)\\0&y^{(i)}=H(\theta^Tx)\end{array}\right.
\end{gather}

We only update parameters when the prediction is wrong. An intuitive visual
explanation can be found by looking at how an update step moves the decision
boundary: consider model parameters $\theta\in\mathbb{R}^3$ for classifying feature
vectors $x^{(i)}\in\mathbb{R}^2$ (remember we set $x_0^{(i)}=1$ to include a bias term).
The decision boundary is the set of all points $x=(x_0, x_1)$ for which
$\theta^Tx=0$. If we plot $x_1^{(i)}$ on the x-axis and $x_2^{(i)}$ on the
y-axis, we can get a feel for how varying $\theta$ affect this boundary:

<div style="margin: 0 auto; display: flex; justify-content: space-around">
<script type="text/javascript">
window.PlotlyConfig = { MathJaxConfig: "local" };
</script>
<script
charset="utf-8"
src="https://cdn.plot.ly/plotly-2.30.0.min.js"
></script>
<div
id="cebe6851-7d73-49d1-bd02-c88d795bf28f"
class="plotly-graph-div"
style="height: 100%; width: 30%"
></div>
<script type="text/javascript">
window.PLOTLYENV = window.PLOTLYENV || {};
if (document.getElementById("cebe6851-7d73-49d1-bd02-c88d795bf28f")) {
Plotly.newPlot(
"cebe6851-7d73-49d1-bd02-c88d795bf28f",
[
{
hovertemplate:
"t0=-2.0\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2],
yaxis: "y",
type: "scatter",
},
],
{
template: { data: { scatter: [{ type: "scatter" }] } },
xaxis: {
anchor: "y",
domain: [0.0, 1.0],
title: { text: "x1" },
range: [-2, 2],
tickformat: ".2f",
},
yaxis: {
anchor: "x",
domain: [0.0, 1.0],
title: { text: "x2" },
range: [-2, 2],
},
legend: { tracegroupgap: 0 },
margin: { t: 60 },
updatemenus: [
{
buttons: [
{
args: [
null,
{
frame: { duration: 500, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 500, easing: "linear" },
},
],
label: "&#9654;",
method: "animate",
},
{
args: [
[null],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "&#9724;",
method: "animate",
},
],
direction: "left",
pad: { r: 10, t: 70 },
showactive: false,
type: "buttons",
x: 0.1,
xanchor: "right",
y: 0,
yanchor: "top",
},
],
sliders: [
{
active: 0,
currentvalue: { prefix: "t0=" },
len: 0.9,
pad: { b: 10, t: 60 },
steps: [
{
args: [
["-2.0"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "-2.0",
method: "animate",
},
{
args: [
["-1.5"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "-1.5",
method: "animate",
},
{
args: [
["-1.0"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "-1.0",
method: "animate",
},
{
args: [
["-0.5"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "-0.5",
method: "animate",
},
{
args: [
["0.0"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "0.0",
method: "animate",
},
{
args: [
["0.5"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "0.5",
method: "animate",
},
{
args: [
["1.0"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "1.0",
method: "animate",
},
{
args: [
["1.5"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "1.5",
method: "animate",
},
{
args: [
["2.0"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "2.0",
method: "animate",
},
],
x: 0.1,
xanchor: "left",
y: 0,
yanchor: "top",
},
],
},
{ responsive: true }
)
.then(function () {
Plotly.addFrames("cebe6851-7d73-49d1-bd02-c88d795bf28f", [
{
data: [
{
hovertemplate:
"t0=-2.0\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2],
yaxis: "y",
type: "scatter",
},
],
name: "-2.0",
},
{
data: [
{
hovertemplate:
"t0=-1.5\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8],
yaxis: "y",
type: "scatter",
},
],
name: "-1.5",
},
{
data: [
{
hovertemplate:
"t0=-1.0\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
yaxis: "y",
type: "scatter",
},
],
name: "-1.0",
},
{
data: [
{
hovertemplate:
"t0=-0.5\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, -0.0],
yaxis: "y",
type: "scatter",
},
],
name: "-0.5",
},
{
data: [
{
hovertemplate:
"t0=0.0\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [0.4, 0.3, 0.2, 0.1, -0.0, -0.1, -0.2, -0.3, -0.4],
yaxis: "y",
type: "scatter",
},
],
name: "0.0",
},
{
data: [
{
hovertemplate:
"t0=0.5\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [-0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
yaxis: "y",
type: "scatter",
},
],
name: "0.5",
},
{
data: [
{
hovertemplate:
"t0=1.0\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [-0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2],
yaxis: "y",
type: "scatter",
},
],
name: "1.0",
},
{
data: [
{
hovertemplate:
"t0=1.5\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [-0.8, -0.9, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6],
yaxis: "y",
type: "scatter",
},
],
name: "1.5",
},
{
data: [
{
hovertemplate:
"t0=2.0\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [-1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9, -2.0],
yaxis: "y",
type: "scatter",
},
],
name: "2.0",
},
]);
})
.then(function () {
Plotly.animate("cebe6851-7d73-49d1-bd02-c88d795bf28f", null);
});
}
</script>
<script type="text/javascript">
window.PlotlyConfig = { MathJaxConfig: "local" };
</script>
<script
charset="utf-8"
src="https://cdn.plot.ly/plotly-2.30.0.min.js"
></script>
<div
id="7f716991-9cb9-497e-9f00-977467e903df"
class="plotly-graph-div"
style="height: 100%; width: 30%"
></div>
<script type="text/javascript">
window.PLOTLYENV = window.PLOTLYENV || {};
if (document.getElementById("7f716991-9cb9-497e-9f00-977467e903df")) {
Plotly.newPlot(
"7f716991-9cb9-497e-9f00-977467e903df",
[
{
hovertemplate:
"t1=-2.0\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [-8.5, -6.5, -4.5, -2.5, -0.5, 1.5, 3.5, 5.5, 7.5],
yaxis: "y",
type: "scatter",
},
],
{
template: { data: { scatter: [{ type: "scatter" }] } },
xaxis: {
anchor: "y",
domain: [0.0, 1.0],
title: { text: "x1" },
range: [-2, 2],
tickformat: ".2f",
},
yaxis: {
anchor: "x",
domain: [0.0, 1.0],
title: { text: "x2" },
range: [-2, 2],
},
legend: { tracegroupgap: 0 },
margin: { t: 60 },
updatemenus: [
{
buttons: [
{
args: [
null,
{
frame: { duration: 500, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 500, easing: "linear" },
},
],
label: "&#9654;",
method: "animate",
},
{
args: [
[null],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "&#9724;",
method: "animate",
},
],
direction: "left",
pad: { r: 10, t: 70 },
showactive: false,
type: "buttons",
x: 0.1,
xanchor: "right",
y: 0,
yanchor: "top",
},
],
sliders: [
{
active: 0,
currentvalue: { prefix: "t1=" },
len: 0.9,
pad: { b: 10, t: 60 },
steps: [
{
args: [
["-2.0"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "-2.0",
method: "animate",
},
{
args: [
["-1.5"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "-1.5",
method: "animate",
},
{
args: [
["-1.0"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "-1.0",
method: "animate",
},
{
args: [
["-0.5"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "-0.5",
method: "animate",
},
{
args: [
["0.0"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "0.0",
method: "animate",
},
{
args: [
["0.5"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "0.5",
method: "animate",
},
{
args: [
["1.0"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "1.0",
method: "animate",
},
{
args: [
["1.5"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "1.5",
method: "animate",
},
{
args: [
["2.0"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "2.0",
method: "animate",
},
],
x: 0.1,
xanchor: "left",
y: 0,
yanchor: "top",
},
],
},
{ responsive: true }
)
.then(function () {
Plotly.addFrames("7f716991-9cb9-497e-9f00-977467e903df", [
{
data: [
{
hovertemplate:
"t1=-2.0\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [-8.5, -6.5, -4.5, -2.5, -0.5, 1.5, 3.5, 5.5, 7.5],
yaxis: "y",
type: "scatter",
},
],
name: "-2.0",
},
{
data: [
{
hovertemplate:
"t1=-1.5\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [-6.5, -5.0, -3.5, -2.0, -0.5, 1.0, 2.5, 4.0, 5.5],
yaxis: "y",
type: "scatter",
},
],
name: "-1.5",
},
{
data: [
{
hovertemplate:
"t1=-1.0\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5],
yaxis: "y",
type: "scatter",
},
],
name: "-1.0",
},
{
data: [
{
hovertemplate:
"t1=-0.5\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [-2.5, -2.0, -1.5, -1.0, -0.5, -0.0, 0.5, 1.0, 1.5],
yaxis: "y",
type: "scatter",
},
],
name: "-0.5",
},
{
data: [
{
hovertemplate:
"t1=0.0\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
yaxis: "y",
type: "scatter",
},
],
name: "0.0",
},
{
data: [
{
hovertemplate:
"t1=0.5\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [1.5, 1.0, 0.5, -0.0, -0.5, -1.0, -1.5, -2.0, -2.5],
yaxis: "y",
type: "scatter",
},
],
name: "0.5",
},
{
data: [
{
hovertemplate:
"t1=1.0\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [3.5, 2.5, 1.5, 0.5, -0.5, -1.5, -2.5, -3.5, -4.5],
yaxis: "y",
type: "scatter",
},
],
name: "1.0",
},
{
data: [
{
hovertemplate:
"t1=1.5\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [5.5, 4.0, 2.5, 1.0, -0.5, -2.0, -3.5, -5.0, -6.5],
yaxis: "y",
type: "scatter",
},
],
name: "1.5",
},
{
data: [
{
hovertemplate:
"t1=2.0\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [7.5, 5.5, 3.5, 1.5, -0.5, -2.5, -4.5, -6.5, -8.5],
yaxis: "y",
type: "scatter",
},
],
name: "2.0",
},
]);
})
.then(function () {
Plotly.animate("7f716991-9cb9-497e-9f00-977467e903df", null);
});
}
</script>
<script type="text/javascript">
window.PlotlyConfig = { MathJaxConfig: "local" };
</script>
<script
charset="utf-8"
src="https://cdn.plot.ly/plotly-2.30.0.min.js"
></script>
<div
id="f9edc0a2-f908-461d-a534-ff14e9e50d07"
class="plotly-graph-div"
style="height: 100%; width: 30%"
></div>
<script type="text/javascript">
window.PLOTLYENV = window.PLOTLYENV || {};
if (document.getElementById("f9edc0a2-f908-461d-a534-ff14e9e50d07")) {
Plotly.newPlot(
"f9edc0a2-f908-461d-a534-ff14e9e50d07",
[
{
hovertemplate:
"t2=-2.0\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [-0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5, 0.625],
yaxis: "y",
type: "scatter",
},
],
{
template: { data: { scatter: [{ type: "scatter" }] } },
xaxis: {
anchor: "y",
domain: [0.0, 1.0],
title: { text: "x1" },
range: [-2, 2],
tickformat: ".2f",
},
yaxis: {
anchor: "x",
domain: [0.0, 1.0],
title: { text: "x2" },
range: [-2, 2],
},
legend: { tracegroupgap: 0 },
margin: { t: 60 },
updatemenus: [
{
buttons: [
{
args: [
null,
{
frame: { duration: 500, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 500, easing: "linear" },
},
],
label: "&#9654;",
method: "animate",
},
{
args: [
[null],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "&#9724;",
method: "animate",
},
],
direction: "left",
pad: { r: 10, t: 70 },
showactive: false,
type: "buttons",
x: 0.1,
xanchor: "right",
y: 0,
yanchor: "top",
},
],
sliders: [
{
active: 0,
currentvalue: { prefix: "t2=" },
len: 0.9,
pad: { b: 10, t: 60 },
steps: [
{
args: [
["-2.0"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "-2.0",
method: "animate",
},
{
args: [
["-1.5"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "-1.5",
method: "animate",
},
{
args: [
["-1.0"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "-1.0",
method: "animate",
},
{
args: [
["-0.5"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "-0.5",
method: "animate",
},
{
args: [
["0.0"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "0.0",
method: "animate",
},
{
args: [
["0.5"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "0.5",
method: "animate",
},
{
args: [
["1.0"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "1.0",
method: "animate",
},
{
args: [
["1.5"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "1.5",
method: "animate",
},
{
args: [
["2.0"],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
fromcurrent: true,
transition: { duration: 0, easing: "linear" },
},
],
label: "2.0",
method: "animate",
},
],
x: 0.1,
xanchor: "left",
y: 0,
yanchor: "top",
},
],
},
{ responsive: true }
)
.then(function () {
Plotly.addFrames("f9edc0a2-f908-461d-a534-ff14e9e50d07", [
{
data: [
{
hovertemplate:
"t2=-2.0\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [
-0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5,
0.625,
],
yaxis: "y",
type: "scatter",
},
],
name: "-2.0",
},
{
data: [
{
hovertemplate:
"t2=-1.5\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [
-0.5, -0.3333333333333333, -0.16666666666666666, 0.0,
0.16666666666666666, 0.3333333333333333, 0.5,
0.6666666666666666, 0.8333333333333334,
],
yaxis: "y",
type: "scatter",
},
],
name: "-1.5",
},
{
data: [
{
hovertemplate:
"t2=-1.0\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
yaxis: "y",
type: "scatter",
},
],
name: "-1.0",
},
{
data: [
{
hovertemplate:
"t2=-0.5\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
yaxis: "y",
type: "scatter",
},
],
name: "-0.5",
},
{
data: [
{
hovertemplate:
"t2=0.0\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [null, null, null, null, null, null, null, null, null],
yaxis: "y",
type: "scatter",
},
],
name: "0.0",
},
{
data: [
{
hovertemplate:
"t2=0.5\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [1.5, 1.0, 0.5, -0.0, -0.5, -1.0, -1.5, -2.0, -2.5],
yaxis: "y",
type: "scatter",
},
],
name: "0.5",
},
{
data: [
{
hovertemplate:
"t2=1.0\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [
0.75, 0.5, 0.25, -0.0, -0.25, -0.5, -0.75, -1.0, -1.25,
],
yaxis: "y",
type: "scatter",
},
],
name: "1.0",
},
{
data: [
{
hovertemplate:
"t2=1.5\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [
0.5, 0.3333333333333333, 0.16666666666666666, -0.0,
-0.16666666666666666, -0.3333333333333333, -0.5,
-0.6666666666666666, -0.8333333333333334,
],
yaxis: "y",
type: "scatter",
},
],
name: "1.5",
},
{
data: [
{
hovertemplate:
"t2=2.0\u003cbr\u003ex1=%{x}\u003cbr\u003ex2=%{y}\u003cextra\u003e\u003c\u002fextra\u003e",
legendgroup: "",
line: { color: "#1F77B4", dash: "solid" },
marker: { symbol: "circle" },
mode: "lines",
name: "",
orientation: "v",
showlegend: false,
x: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
xaxis: "x",
y: [
0.375, 0.25, 0.125, -0.0, -0.125, -0.25, -0.375, -0.5,
-0.625,
],
yaxis: "y",
type: "scatter",
},
],
name: "2.0",
},
]);
})
.then(function () {
Plotly.animate("f9edc0a2-f908-461d-a534-ff14e9e50d07", null);
});
}
</script>
</div>

Solving $\theta^Tx=0$ for $x_2$ is simple and yields the following which might clarify
the linear nature of the boundary:

\begin{gather}
    x_2=-\frac{\theta_0+\theta_1x_1}{\theta_2}=-\frac{\theta_0}{\theta_2}-\frac{\theta_1}{\theta_2}x_1
\end{gather}

Let's look at a classification example and see how the boundary gets shifted
after every step to more clearly separate the two distinct groups. We will use
the `HarvestTime` and `Weight` attributes of a [Banana Quality
dataset](https://www.kaggle.com/datasets/l3llff/banana). Green markers
represent good and red markers bad quality. We will use 500 samples in total:

<div>
<script type="text/javascript">
window.PlotlyConfig = { MathJaxConfig: "local" };
</script>
<script
charset="utf-8"
src="https://cdn.plot.ly/plotly-2.30.0.min.js"
></script>
<div
id="0f9bce4c-d300-476d-b62b-89155c630e1f"
class="plotly-graph-div"
style="height: 100%; width: 100%"
></div>
<script type="text/javascript">
window.PLOTLYENV = window.PLOTLYENV || {};
if (document.getElementById("0f9bce4c-d300-476d-b62b-89155c630e1f")) {
Plotly.newPlot(
"0f9bce4c-d300-476d-b62b-89155c630e1f",
[
{
marker: {
color: [
"red",
"red",
"green",
"green",
"red",
"red",
"red",
"red",
"red",
"green",
"red",
"green",
"green",
"green",
"green",
"red",
"green",
"green",
"green",
"green",
"red",
"red",
"red",
"green",
"red",
"green",
"red",
"red",
"red",
"red",
"green",
"green",
"green",
"green",
"green",
"green",
"green",
"green",
"red",
"green",
"red",
"green",
"red",
"red",
"green",
"green",
"green",
"red",
"red",
"red",
"green",
"red",
"red",
"red",
"red",
"green",
"green",
"red",
"green",
"green",
"red",
"green",
"green",
"green",
"red",
"red",
"red",
"red",
"green",
"red",
"green",
"green",
"red",
"green",
"green",
"red",
"green",
"red",
"red",
"red",
"red",
"red",
"green",
"green",
"red",
"red",
"red",
"red",
"red",
"red",
"green",
"green",
"red",
"red",
"green",
"red",
"green",
"red",
"red",
"green",
"red",
"red",
"green",
"red",
"red",
"green",
"green",
"red",
"green",
"red",
"red",
"green",
"green",
"green",
"red",
"red",
"green",
"red",
"green",
"red",
"red",
"red",
"red",
"green",
"red",
"red",
"green",
"red",
"green",
"green",
"green",
"red",
"red",
"red",
"green",
"green",
"red",
"green",
"red",
"red",
"green",
"red",
"red",
"red",
"green",
"green",
"green",
"red",
"green",
"red",
"green",
"green",
"green",
"red",
"red",
"green",
"green",
"green",
"green",
"red",
"red",
"red",
"green",
"green",
"green",
"red",
"red",
"red",
"green",
"green",
"red",
"green",
"green",
"red",
"green",
"green",
"red",
"red",
"red",
"green",
"red",
"red",
"green",
"red",
"red",
"green",
"red",
"red",
"green",
"green",
"green",
"green",
"green",
"green",
"green",
"red",
"green",
"green",
"green",
"green",
"red",
"red",
"red",
"green",
"red",
"red",
"red",
"green",
"green",
"red",
"green",
"red",
"red",
"red",
"green",
"red",
"red",
"green",
"red",
"green",
"red",
"red",
"red",
"red",
"red",
"green",
"green",
"red",
"green",
"red",
"red",
"red",
"red",
"green",
"green",
"red",
"green",
"red",
"red",
"red",
"red",
"red",
"red",
"red",
"red",
"red",
"green",
"red",
"green",
"red",
"red",
"red",
"green",
"red",
"green",
"green",
"red",
"red",
"green",
"red",
"red",
"green",
"green",
"green",
"green",
"green",
"green",
"green",
"green",
"green",
"green",
"red",
"green",
"green",
"red",
"red",
"green",
"red",
"green",
"green",
"red",
"green",
"red",
"green",
"red",
"green",
"red",
"green",
"red",
"green",
"green",
"green",
"green",
"green",
"red",
"red",
"green",
"green",
"green",
"green",
"green",
"green",
"green",
"red",
"red",
"red",
"green",
"red",
"red",
"red",
"red",
"red",
"green",
"red",
"green",
"green",
"green",
"red",
"red",
"red",
"green",
"green",
"green",
"red",
"green",
"green",
"green",
"red",
"green",
"red",
"green",
"red",
"red",
"green",
"red",
"green",
"red",
"green",
"green",
"red",
"green",
"red",
"green",
"green",
"green",
"red",
"green",
"red",
"red",
"green",
"red",
"green",
"red",
"green",
"green",
"green",
"red",
"green",
"red",
"red",
"green",
"green",
"green",
"green",
"red",
"green",
"red",
"green",
"red",
"green",
"green",
"red",
"red",
"green",
"red",
"green",
"green",
"red",
"red",
"green",
"green",
"green",
"green",
"red",
"green",
"green",
"red",
"green",
"red",
"red",
"green",
"green",
"green",
"red",
"red",
"green",
"red",
"green",
"green",
"green",
"green",
"red",
"green",
"green",
"red",
"green",
"red",
"red",
"green",
"green",
"green",
"red",
"green",
"green",
"green",
"red",
"red",
"red",
"green",
"green",
"green",
"green",
"red",
"green",
"red",
"green",
"red",
"red",
"red",
"red",
"red",
"red",
"red",
"red",
"green",
"red",
"red",
"green",
"red",
"red",
"green",
"red",
"green",
"green",
"red",
"green",
"red",
"red",
"green",
"red",
"green",
"green",
"green",
"red",
"red",
"green",
"red",
"red",
"red",
"red",
"red",
"green",
"green",
"red",
"red",
"red",
"red",
"red",
"red",
"red",
"green",
"green",
"green",
"red",
"green",
"green",
"red",
"green",
"green",
"green",
"red",
"green",
"red",
"red",
"green",
"red",
"red",
"green",
"green",
"red",
"red",
"red",
"green",
"red",
"red",
"red",
"red",
"green",
"red",
"green",
],
},
mode: "markers",
x: [
-2.8661146, -0.76461077, 0.73299015, 1.4857948, -0.47198698,
2.5236387, -0.27889577, 0.17208149, -1.5552663, -1.9819083,
-0.30845788, 2.1542256, 0.83377033, -1.4282134, 0.50406414,
-2.8568304, 2.6313925, 2.7456653, -1.5660402, -1.2653188,
-1.2652307, 0.099607095, -1.6979195, 2.3105776, 0.09324434,
-1.3795056, -0.5443203, -1.0491182, -4.4782624, -2.191886,
-2.2022934, -2.547049, -0.0120904455, -1.7983992, -0.12371753,
-0.36154962, -2.1299272, 0.33545563, -2.6835854, -1.8310581,
-1.8365955, 1.6591098, -3.6397433, -2.1744232, 2.13511,
-0.20214038, -1.8576926, -2.394633, -3.583523, -3.993356,
-1.3119143, -2.139389, -2.2588084, -2.975183, -3.2680805,
1.9640026, -2.3313777, -0.8863791, 1.9028252, -0.6716628,
-2.9717722, 4.726073, -0.1350994, 0.06375767, -1.7755551,
-1.0604162, 0.5149095, -3.3611777, 3.5470147, -0.6107662,
-1.7780541, -2.1424298, -2.4935436, -2.1188767, 2.00821,
-1.1629738, -1.0738372, -1.9518144, 0.08931943, -2.3904243,
-4.5481887, -0.37572742, -1.9348708, 4.693924, -1.1644317,
-0.6027726, -0.51408494, -3.6163871, -2.94586, 0.37502578,
-0.39984927, 3.1924849, 0.20684125, -1.441941, 2.2247689,
-2.1815941, 1.2938043, -0.81852067, -2.5320504, -3.0943015,
-5.8476043, -2.5158222, 0.09154833, 1.0739439, -1.3343722,
0.3290164, -1.8197408, -3.4271839, -2.6924636, -3.1882832,
0.75569427, -1.9276252, -2.3123548, 1.7467387, -3.6357758,
-2.0021307, 2.9650564, -2.9077537, -0.20197241, -3.4341898,
0.36608893, -1.9845669, -1.9716076, -1.6904236, -1.5373629,
-4.27228, -1.1362016, -1.544048, 0.67013264, -1.242339,
0.4726047, -0.30090076, 1.3776753, -2.2561371, 0.15315741,
1.493338, -1.9117821, -0.43309933, -3.088757, 0.7959783,
1.3691732, -0.44805962, -1.8129942, -1.4638044, -0.6853919,
-1.0482163, -1.4315897, 1.6365131, -1.723828, 1.0641854,
-2.9667165, 1.1222605, -0.58071536, 1.521846, 0.0970223,
1.408092, -2.5428245, -1.3413541, -1.2164019, -1.5111451,
-0.4572687, 0.7298121, 0.57213813, -1.5138155, -2.057181,
-2.1850343, -0.24903333, -3.8767793, 5.6663103, -0.7997878,
-2.0284934, 1.6455438, -1.0601166, -3.1869886, 1.9698415,
-2.200311, -6.2592225, -0.9402797, -2.2954755, 2.3651023,
-2.8567984, -0.11044602, 0.19081959, -0.30779192, -1.1069522,
-1.1161281, -4.1900606, 0.5312796, 2.4846973, -2.2637477,
-3.0293715, -1.0510626, -0.017781936, -1.521068, 1.7506213,
-0.8504297, 0.2414491, -0.8520102, 3.9681873, 2.8383107,
-1.880788, -2.0984008, -3.2682595, 4.591659, -1.3109598,
0.53831816, -3.258068, 2.9742615, -2.419647, 0.5519625,
0.22663634, 2.1468046, -1.1283257, -2.334769, -2.123361,
0.7528358, -0.998888, -0.8447755, 0.44487453, -3.9047296,
-1.5320687, -3.2218482, -0.7414998, -1.661774, -1.4597491,
-2.293642, 3.4937737, -1.6085459, 1.6826656, -2.728848,
-2.5599976, -2.7297306, -0.4496405, 3.125093, -1.7993833,
-3.9573152, 0.21166141, -4.110807, -2.4980328, -2.5221865,
-2.3113048, -1.6957709, -1.7407408, 1.7127231, -3.5605898,
-2.5936723, -0.12092512, -1.6508648, -3.403482, -0.3603593,
-2.7523832, 0.22945508, 0.022585366, -2.0480018, 3.9429078,
4.1359706, 0.28106716, -5.2010365, -1.4112885, -3.1379457,
-1.933354, 2.7994132, 1.4515065, 2.7402725, -0.88641894,
1.451927, 0.18416424, 3.2586496, 0.41631493, 2.864047,
-0.20103, -1.4363364, -1.5155854, -0.55565685, -3.1208982,
-1.9513012, -2.6522799, -4.6121535, 0.96710265, -1.8611852,
0.48223707, 0.96606594, -3.3803015, 1.0252227, -3.931952,
-1.5913963, -5.580557, -0.49414128, -2.3744507, 3.0873713,
2.6482828, -2.0532334, 2.4219851, 2.2684693, -1.7110974,
-2.7069325, -0.7142235, -0.46821395, -1.4159386, -1.8968066,
0.5921024, 3.1649435, -1.566777, -2.1336067, -3.3902194,
-0.45581886, 2.2768323, -4.3397493, -3.69972, -1.3120118,
-1.8566792, 2.2700152, -1.1341056, -2.3494775, 1.1108283,
1.7902676, -0.5137122, -1.1509799, 0.00090650166, -1.2170671,
-0.16447006, 0.034417763, 0.6593304, -3.1721046, 1.490789,
-1.0243345, 0.84550023, 1.4142761, -0.49245808, -0.6715067,
0.38056114, -0.18134238, -1.5971721, 1.8644228, 1.2070907,
-1.503573, -1.3896186, 1.0832114, -0.6864997, -0.2554834,
-1.2394224, -3.2235208, -1.4967052, -1.3003503, 0.114020415,
-0.3259487, -0.27760097, -1.8209496, -0.5369759, -0.5890099,
-0.8381747, 4.1777678, 1.6043104, 2.9759722, 1.8462809,
-1.4002656, -0.65076184, 0.82041717, -0.28356153, 1.2307129,
0.8289383, -1.2417547, 1.3309462, 3.5351605, -0.72330195,
-0.7204721, 0.14962777, -0.34827033, -0.19792755, 2.3207648,
-1.9439071, -1.3627446, -2.2622814, 2.4722228, -1.6565055,
-1.0672115, -1.3652271, -0.039945547, -3.7743363, -0.80291027,
1.0247363, -2.2739406, 1.2163272, -3.0210304, -0.29149663,
1.8940548, -0.52478653, 0.8165017, -2.7389693, -1.2189938,
-0.8062406, -3.2775316, -0.41874555, -1.9756945, 0.56685495,
-0.5275389, -2.1394954, 2.9269314, 2.4815977, 1.5584577,
-0.24891114, -2.838106, -2.1010323, -1.813254, 0.08454918,
3.0590122, -1.707235, -0.31957433, -0.681102, 1.8847642,
-2.116052, -1.5984932, 0.016833125, 3.4010544, 2.0177345,
-1.8235197, -4.6951547, -3.855116, -2.6137528, -3.6144998,
-2.195653, -2.0508451, -1.9448593, -3.0908031, 1.7407668,
-0.9675297, -2.2885287, -0.42639995, 0.53512275, -1.249063,
-1.5131148, -0.4212254, -0.576798, 0.11557154, 1.3314898,
-0.7728848, -5.1016307, -1.5506042, -2.186882, -1.3145161,
0.92687756, -0.41893485, 3.6174712, -1.1071256, 0.5259196,
3.2500265, -1.3222367, -0.62177896, 0.9438046, -2.8821707,
0.6939502, -1.7492058, -2.1910946, -4.1585493, -0.04149286,
-0.6802475, -1.8111968, -0.34704086, 0.011337979, -2.8360023,
-2.7571366, 2.3230004, 2.8526711, -1.7984557, -0.86825234,
-4.5472994, -0.4115, -3.6997406, -0.48499674, -0.048853435,
2.3301253, -0.58282626, 1.5741303, 0.57903767, -2.9956112,
-0.85767996, -0.8890046, 0.095644586, -1.0008069, -1.6907573,
-2.0509799, 1.7970408, -1.7689329, 0.18709378, -2.8417625,
-2.3606057, -3.2703826, 1.9513144, 1.3271769, -2.0983052,
-2.5889091, -3.2214224, -0.6550581, 0.6731124, -1.0704767,
-0.80241233, -1.6671907, 2.6736786, 0.40208346, 0.932349,
],
y: [
-1.6573876, -1.115875, 2.9445722, -2.8685453, -0.26388818,
-2.2588038, 0.8707596, -2.616031, -3.8442528, 0.59545654,
-1.6774757, -3.176759, 1.7688276, 1.5364138, -0.6279205,
-1.2524948, -4.3864965, 0.4661816, 3.5965226, 1.2728935,
-2.0311074, -2.898553, -2.4135258, -1.2724521, -1.3439031,
1.3496836, 1.193122, -2.1224287, 0.1740218, -0.7432217,
2.2659588, 0.9230121, 3.5958033, 2.4314156, 2.5600655,
-2.2852044, 2.4879596, -4.081235, 0.0043280497, 0.078399904,
-2.2094018, -0.55521214, -3.601741, -2.776233, 0.22051112,
3.9266176, 2.516229, -3.3143413, 0.20249395, -3.509904,
0.46899188, -4.3297896, 0.3393716, -2.0446596, -2.1493406,
-1.3588272, 0.42143634, -1.4855208, -5.1584964, 0.17677046,
-3.373348, -4.543538, -1.3402942, -4.686683, -0.9165634,
-0.71541834, 2.257866, -0.90609086, -0.78203243, -0.013541504,
0.37268108, 0.7233512, -3.1430156, 1.2721084, -3.290887,
-2.7758722, 0.4207576, -2.4336782, -5.152493, -3.5412517,
-2.3274915, -1.2561663, -0.29145256, 1.2561446, -2.4485111,
-3.7440822, 0.20981878, -1.4765241, -1.2696879, -2.4994678,
-2.3541024, -0.3686226, -1.0072457, -4.2310348, 1.3145889,
-3.584454, -2.3841064, -0.086155236, -1.1296688, 1.8799393,
-4.8180227, -2.15812, -3.971719, -0.8879012, -1.6976955,
-2.6619537, -0.04494581, -1.0677671, 2.1986482, 0.10668557,
-0.04775443, 3.3580818, 1.6665206, -2.2917733, 1.662635,
-2.7206933, -3.4057918, -4.8673224, 1.8388226, -0.7288469,
-2.1395037, -4.5569453, -0.261996, 3.847965, 1.4284495,
-7.103426, 1.9810203, 2.3582048, 1.6980188, 1.7092083,
-1.6992267, -3.218265, -2.417353, 0.0649613, 3.0194867,
-0.7680226, -1.0192382, -2.2570882, -1.5899317, -2.1469326,
-5.352954, -2.3432925, -1.033908, -5.0857177, 1.8493793,
3.2966454, 3.095668, -1.9640765, 3.2806795, -1.4165988,
1.5523503, -2.548592, 2.3001823, 1.4175395, -0.88765544,
-0.30366004, 1.82151, 2.6805136, 1.4457024, -0.7228551,
-0.79579455, -2.4218118, 2.5915065, 0.8581039, 0.27475291,
-0.36113253, -0.60490006, -2.4374635, -2.0141597, -0.7593392,
-4.1457467, 0.52496505, 1.352214, 0.030233797, -1.9530299,
-0.5057217, 1.7833568, -3.9834852, -1.3524705, -3.5750978,
-0.8163738, -2.2699215, -0.7137241, -0.5263648, 0.36483932,
1.160521, -3.0383685, -0.7839372, -2.5172684, 0.8690368,
2.7020502, -0.5847018, 3.553664, 2.1990147, -2.4497645,
-5.0899835, -0.42388943, -0.5904426, -2.8902779, 0.8533434,
-1.1741241, -0.90366244, -2.1860387, -1.1143174, -2.04948,
0.2989947, -2.6110754, -2.469039, 2.0198166, 0.9675919,
-3.0723042, 1.1936275, -1.5670842, -2.9692762, 0.26193923,
-3.2795424, -0.25513074, 0.8581182, -0.1533284, 0.0005910767,
0.883529, -2.5184875, -0.41535404, -1.1828471, -0.9238011,
0.9177825, 0.7521285, -2.9778478, 0.5265045, -2.2550094,
-4.9512005, -0.8615201, -2.8310673, -1.202684, -1.5210049,
-3.954877, -2.1273303, -1.9464078, -1.2438794, -0.6798828,
-0.021242553, -1.6600323, -1.5516374, -1.5075675, 0.069259174,
-5.523238, 1.9517813, -1.8517587, 0.65329343, -1.4654627,
-2.8653452, -0.25712803, 1.420451, -2.3105218, -0.78509444,
-1.4775121, -3.6790109, -0.42060602, 1.0891815, -0.25740457,
-0.83661693, -2.0875483, 1.5166919, -1.8012048, 1.5771962,
-1.8160366, -0.81849, 0.5497767, -1.8166023, -0.14007282,
0.81645334, 0.012214444, 1.967838, 1.6950226, -2.1817133,
-4.5040126, 2.843291, -2.5749779, -2.2287648, -0.66976815,
-1.5587751, -3.1337192, 0.24424013, -1.1128439, -2.3933892,
1.0870101, -1.386088, 1.7991678, -1.1509533, -1.8911014,
-0.56050223, 2.2504728, 0.18322355, -2.304281, -0.20814478,
0.4469461, -3.4278355, 2.2507317, -1.0022285, 1.5186173,
0.25925222, -2.695596, 2.802962, -1.6680957, -2.3085551,
-3.1698613, -4.1572723, 1.9419452, -2.7320461, -0.96963745,
-3.380801, -2.361004, 2.5515509, -0.21635751, 1.7821505,
-3.8410783, 1.346637, -1.1301717, -2.456458, -2.601244,
1.4021732, 2.4967601, 0.40658078, -2.5257041, -1.7781427,
-0.34003803, -1.0805138, -2.5107486, 1.9231318, -3.1200051,
-3.1499138, -2.6799653, -1.8976176, -2.7329986, -1.4463788,
2.1639624, -0.8847976, 2.0125287, -4.2035766, 0.23766997,
1.5774362, -0.76545095, 0.8859663, 2.6816442, 2.3447897,
-3.1819832, -0.6134325, -2.3058681, 2.3409083, 1.8724184,
-3.0997806, 1.2907028, -1.824753, 0.73167086, -0.21957462,
2.002171, 1.7811475, -1.3660907, -1.185319, -1.3138963,
-1.6884031, -1.1842263, -0.6386285, -2.571182, -3.1335328,
-0.21120319, -3.7123835, -1.2193375, -2.307586, -1.7830101,
1.2248005, -4.5863233, -0.21800223, -1.3595781, -2.2685273,
3.0130498, 2.1028597, -1.5803216, 0.11896703, 3.2545424,
1.089701, 2.481435, -2.5292988, -2.7367852, 2.064319,
-1.0057654, 0.35044533, -0.24440786, -2.623735, -2.35508,
2.4231217, -0.14314315, 1.4006454, -2.6844606, -3.0888834,
1.2828685, -1.045089, -2.7784154, -1.4501271, -0.64575535,
2.5307813, -2.1357055, 2.6450708, 1.2459319, -2.45057,
-1.3365605, -2.2062848, -2.39819, 1.1312141, -1.003564,
-0.10003042, 0.742346, -0.8546614, 0.283987, -2.6093476,
-1.1532081, -3.9156086, -2.897966, 2.2567232, -0.9530375,
0.57780695, 2.4969308, -2.0708961, 0.60217834, -0.17300642,
0.17321406, -2.5636039, -0.38543102, -0.07749376, -0.90938365,
-1.3351517, -1.7003853, 0.26013547, -2.9059055, -2.7207205,
0.2535443, -2.3555493, 1.7502943, -5.191882, -0.24073431,
-2.3310642, -3.843339, 1.258602, 2.534858, -1.3754982,
-0.08792398, -0.010279885, -0.8644623, -2.6528392, -3.0914397,
2.2439253, 1.2106314, 0.09562017, -1.1454213, 1.8492236,
1.5476913, -1.6122991, -1.2205954, -2.058719, -0.9721111,
0.59178174, -0.7421656, -2.0226865, -2.0002, 1.2550695,
-2.500315, -0.10129192, -0.47121057, 1.6060711, -2.3651674,
-1.3086637, -0.41747802, -2.3585312, -1.4413381, 0.8997276,
-2.1358106, -1.8286031, 5.1841984, 1.5983933, 0.27042428,
1.3153375, -0.8291452, -0.48595744, -0.6034076, 0.8114595,
-2.1360006, -1.5950124, -0.9935344, -0.0115941055, -3.0982852,
-0.08551545, -2.1277819, -0.4584839, -4.3785663, -2.34872,
-2.6209826, -2.0568352, -3.1783113, -0.49490714, 2.4165056,
],
type: "scatter",
},
{
line: { color: "black" },
mode: "lines",
name: "BoundaryLine",
type: "scatter",
},
],
{
template: {
data: {
histogram2dcontour: [
{
type: "histogram2dcontour",
colorbar: { outlinewidth: 0, ticks: "" },
colorscale: [
[0.0, "#0d0887"],
[0.1111111111111111, "#46039f"],
[0.2222222222222222, "#7201a8"],
[0.3333333333333333, "#9c179e"],
[0.4444444444444444, "#bd3786"],
[0.5555555555555556, "#d8576b"],
[0.6666666666666666, "#ed7953"],
[0.7777777777777778, "#fb9f3a"],
[0.8888888888888888, "#fdca26"],
[1.0, "#f0f921"],
],
},
],
choropleth: [
{
type: "choropleth",
colorbar: { outlinewidth: 0, ticks: "" },
},
],
histogram2d: [
{
type: "histogram2d",
colorbar: { outlinewidth: 0, ticks: "" },
colorscale: [
[0.0, "#0d0887"],
[0.1111111111111111, "#46039f"],
[0.2222222222222222, "#7201a8"],
[0.3333333333333333, "#9c179e"],
[0.4444444444444444, "#bd3786"],
[0.5555555555555556, "#d8576b"],
[0.6666666666666666, "#ed7953"],
[0.7777777777777778, "#fb9f3a"],
[0.8888888888888888, "#fdca26"],
[1.0, "#f0f921"],
],
},
],
heatmap: [
{
type: "heatmap",
colorbar: { outlinewidth: 0, ticks: "" },
colorscale: [
[0.0, "#0d0887"],
[0.1111111111111111, "#46039f"],
[0.2222222222222222, "#7201a8"],
[0.3333333333333333, "#9c179e"],
[0.4444444444444444, "#bd3786"],
[0.5555555555555556, "#d8576b"],
[0.6666666666666666, "#ed7953"],
[0.7777777777777778, "#fb9f3a"],
[0.8888888888888888, "#fdca26"],
[1.0, "#f0f921"],
],
},
],
heatmapgl: [
{
type: "heatmapgl",
colorbar: { outlinewidth: 0, ticks: "" },
colorscale: [
[0.0, "#0d0887"],
[0.1111111111111111, "#46039f"],
[0.2222222222222222, "#7201a8"],
[0.3333333333333333, "#9c179e"],
[0.4444444444444444, "#bd3786"],
[0.5555555555555556, "#d8576b"],
[0.6666666666666666, "#ed7953"],
[0.7777777777777778, "#fb9f3a"],
[0.8888888888888888, "#fdca26"],
[1.0, "#f0f921"],
],
},
],
contourcarpet: [
{
type: "contourcarpet",
colorbar: { outlinewidth: 0, ticks: "" },
},
],
contour: [
{
type: "contour",
colorbar: { outlinewidth: 0, ticks: "" },
colorscale: [
[0.0, "#0d0887"],
[0.1111111111111111, "#46039f"],
[0.2222222222222222, "#7201a8"],
[0.3333333333333333, "#9c179e"],
[0.4444444444444444, "#bd3786"],
[0.5555555555555556, "#d8576b"],
[0.6666666666666666, "#ed7953"],
[0.7777777777777778, "#fb9f3a"],
[0.8888888888888888, "#fdca26"],
[1.0, "#f0f921"],
],
},
],
surface: [
{
type: "surface",
colorbar: { outlinewidth: 0, ticks: "" },
colorscale: [
[0.0, "#0d0887"],
[0.1111111111111111, "#46039f"],
[0.2222222222222222, "#7201a8"],
[0.3333333333333333, "#9c179e"],
[0.4444444444444444, "#bd3786"],
[0.5555555555555556, "#d8576b"],
[0.6666666666666666, "#ed7953"],
[0.7777777777777778, "#fb9f3a"],
[0.8888888888888888, "#fdca26"],
[1.0, "#f0f921"],
],
},
],
mesh3d: [
{
type: "mesh3d",
colorbar: { outlinewidth: 0, ticks: "" },
},
],
scatter: [
{
fillpattern: {
fillmode: "overlay",
size: 10,
solidity: 0.2,
},
type: "scatter",
},
],
parcoords: [
{
type: "parcoords",
line: { colorbar: { outlinewidth: 0, ticks: "" } },
},
],
scatterpolargl: [
{
type: "scatterpolargl",
marker: { colorbar: { outlinewidth: 0, ticks: "" } },
},
],
bar: [
{
error_x: { color: "#2a3f5f" },
error_y: { color: "#2a3f5f" },
marker: {
line: { color: "#E5ECF6", width: 0.5 },
pattern: {
fillmode: "overlay",
size: 10,
solidity: 0.2,
},
},
type: "bar",
},
],
scattergeo: [
{
type: "scattergeo",
marker: { colorbar: { outlinewidth: 0, ticks: "" } },
},
],
scatterpolar: [
{
type: "scatterpolar",
marker: { colorbar: { outlinewidth: 0, ticks: "" } },
},
],
histogram: [
{
marker: {
pattern: {
fillmode: "overlay",
size: 10,
solidity: 0.2,
},
},
type: "histogram",
},
],
scattergl: [
{
type: "scattergl",
marker: { colorbar: { outlinewidth: 0, ticks: "" } },
},
],
scatter3d: [
{
type: "scatter3d",
line: { colorbar: { outlinewidth: 0, ticks: "" } },
marker: { colorbar: { outlinewidth: 0, ticks: "" } },
},
],
scattermapbox: [
{
type: "scattermapbox",
marker: { colorbar: { outlinewidth: 0, ticks: "" } },
},
],
scatterternary: [
{
type: "scatterternary",
marker: { colorbar: { outlinewidth: 0, ticks: "" } },
},
],
scattercarpet: [
{
type: "scattercarpet",
marker: { colorbar: { outlinewidth: 0, ticks: "" } },
},
],
carpet: [
{
aaxis: {
endlinecolor: "#2a3f5f",
gridcolor: "white",
linecolor: "white",
minorgridcolor: "white",
startlinecolor: "#2a3f5f",
},
baxis: {
endlinecolor: "#2a3f5f",
gridcolor: "white",
linecolor: "white",
minorgridcolor: "white",
startlinecolor: "#2a3f5f",
},
type: "carpet",
},
],
table: [
{
cells: {
fill: { color: "#EBF0F8" },
line: { color: "white" },
},
header: {
fill: { color: "#C8D4E3" },
line: { color: "white" },
},
type: "table",
},
],
barpolar: [
{
marker: {
line: { color: "#E5ECF6", width: 0.5 },
pattern: {
fillmode: "overlay",
size: 10,
solidity: 0.2,
},
},
type: "barpolar",
},
],
pie: [{ automargin: true, type: "pie" }],
},
layout: {
autotypenumbers: "strict",
colorway: [
"#636efa",
"#EF553B",
"#00cc96",
"#ab63fa",
"#FFA15A",
"#19d3f3",
"#FF6692",
"#B6E880",
"#FF97FF",
"#FECB52",
],
font: { color: "#2a3f5f" },
hovermode: "closest",
hoverlabel: { align: "left" },
paper_bgcolor: "white",
plot_bgcolor: "#E5ECF6",
polar: {
bgcolor: "#E5ECF6",
angularaxis: {
gridcolor: "white",
linecolor: "white",
ticks: "",
},
radialaxis: {
gridcolor: "white",
linecolor: "white",
ticks: "",
},
},
ternary: {
bgcolor: "#E5ECF6",
aaxis: {
gridcolor: "white",
linecolor: "white",
ticks: "",
},
baxis: {
gridcolor: "white",
linecolor: "white",
ticks: "",
},
caxis: {
gridcolor: "white",
linecolor: "white",
ticks: "",
},
},
coloraxis: { colorbar: { outlinewidth: 0, ticks: "" } },
colorscale: {
sequential: [
[0.0, "#0d0887"],
[0.1111111111111111, "#46039f"],
[0.2222222222222222, "#7201a8"],
[0.3333333333333333, "#9c179e"],
[0.4444444444444444, "#bd3786"],
[0.5555555555555556, "#d8576b"],
[0.6666666666666666, "#ed7953"],
[0.7777777777777778, "#fb9f3a"],
[0.8888888888888888, "#fdca26"],
[1.0, "#f0f921"],
],
sequentialminus: [
[0.0, "#0d0887"],
[0.1111111111111111, "#46039f"],
[0.2222222222222222, "#7201a8"],
[0.3333333333333333, "#9c179e"],
[0.4444444444444444, "#bd3786"],
[0.5555555555555556, "#d8576b"],
[0.6666666666666666, "#ed7953"],
[0.7777777777777778, "#fb9f3a"],
[0.8888888888888888, "#fdca26"],
[1.0, "#f0f921"],
],
diverging: [
[0, "#8e0152"],
[0.1, "#c51b7d"],
[0.2, "#de77ae"],
[0.3, "#f1b6da"],
[0.4, "#fde0ef"],
[0.5, "#f7f7f7"],
[0.6, "#e6f5d0"],
[0.7, "#b8e186"],
[0.8, "#7fbc41"],
[0.9, "#4d9221"],
[1, "#276419"],
],
},
xaxis: {
gridcolor: "white",
linecolor: "white",
ticks: "",
title: { standoff: 15 },
zerolinecolor: "white",
automargin: true,
zerolinewidth: 2,
},
yaxis: {
gridcolor: "white",
linecolor: "white",
ticks: "",
title: { standoff: 15 },
zerolinecolor: "white",
automargin: true,
zerolinewidth: 2,
},
scene: {
xaxis: {
backgroundcolor: "#E5ECF6",
gridcolor: "white",
linecolor: "white",
showbackground: true,
ticks: "",
zerolinecolor: "white",
gridwidth: 2,
},
yaxis: {
backgroundcolor: "#E5ECF6",
gridcolor: "white",
linecolor: "white",
showbackground: true,
ticks: "",
zerolinecolor: "white",
gridwidth: 2,
},
zaxis: {
backgroundcolor: "#E5ECF6",
gridcolor: "white",
linecolor: "white",
showbackground: true,
ticks: "",
zerolinecolor: "white",
gridwidth: 2,
},
},
shapedefaults: { line: { color: "#2a3f5f" } },
annotationdefaults: {
arrowcolor: "#2a3f5f",
arrowhead: 0,
arrowwidth: 1,
},
geo: {
bgcolor: "white",
landcolor: "#E5ECF6",
subunitcolor: "white",
showland: true,
showlakes: true,
lakecolor: "white",
},
title: { x: 0.05 },
mapbox: { style: "light" },
},
},
xaxis: { title: { text: "Harvest Time" } },
yaxis: { title: { text: "Weight" } },
},
{ responsive: true }
)
.then(function () {
Plotly.addFrames("0f9bce4c-d300-476d-b62b-89155c630e1f", [
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [14.437731282652289, -13.23101295074248],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr0",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-28.74545172771473, 28.74545172771473],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr3",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [14.915500186561097, -13.006458574131097],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr5",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-0.7480365316679164, 0.7480365316679164],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr9",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-9.783615411734088, 11.087688509308567],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr11",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [87.91409459762178, -104.92297714635524],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr12",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [4.65637226198298, -8.043169883620156],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr13",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-10.822660680067541, 13.882037688572447],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr16",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [15.048706395080238, -25.236039843639105],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr18",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.859404019954683, -5.921172468750254],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr21",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [7.5658223506684585, -9.79876090815606],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr26",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-0.2899426020812339, -1.926090584168067],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr31",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-3.7280492708516992, -3.819975945349404],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr35",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [1.1810615175420518, 3.1724664923098076],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr37",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-6.959514190931793, 10.581766572984428],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr38",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-1.3289864209467206, 5.8027367373246115],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr39",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-40.721195673797766, 61.866800886337664],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr40",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [1.6619938748987249, -7.534796331430797],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr46",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [17.96161574659408, -23.393966952099028],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr48",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [9.337910957338309, -14.5331312653145],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr50",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [0.7021795592291484, -5.828034371660947],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr56",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [2.3735835422419838, -5.2193618755564914],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr57",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-25.730965535122476, 40.597319628829325],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr58",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-1567.3776412386073, 2044.316917771108],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr64",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-15.907230767069493, 20.28738153695486],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr66",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-18.595742747843353, 22.120771660023642],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr69",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-14.223044273039019, 19.4950706826425],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr70",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-7.532376144188683, 17.758608947545014],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr71",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [15.123779407868344, -20.20104795043072],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr72",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.6512714239854027, -7.353065563694552],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr73",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-546.612734582579, 831.075085814644],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr74",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [13.645170083191008, -18.04616440991752],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr75",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [10.347312268129237, -12.858094289043954],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr81",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [6.700098713583728, -9.950924033935538],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr82",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [8.572392155867625, -11.444674808058767],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr86",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [22.747306722048343, -33.39080265606342],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr90",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [11.707388139432716, -16.3918985252352],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr92",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [14.65198702790237, -18.254211329598327],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr97",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [1.6546702281675727, -4.093225140504086],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr99",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [50.49835975373576, -143.47091719968208],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr102",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-1.0557920976432205, -8.77732472460097],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr103",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.905181337895401, -6.852135741308384],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr104",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [232.0740306359121, -421.77931556505735],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr105",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-4.577812278127202, -0.7522956427358886],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr108",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [7.5535676596962436, -12.21627667487225],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr109",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [5.087687571802945, -8.73660613464975],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr110",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [77.84251615864736, -89.16764270995742],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr114",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [14.727422571967326, -16.2259432860965],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr120",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [169.28718768574262, -175.7161064166729],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr127",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [20.653127799366057, -22.63302922179972],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr129",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [7.144350149497699, -7.5950385711188],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr132",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [12.49187288881093, -14.326257825553604],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr137",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [5.054083054481214, -5.516243654830771],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr139",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-31.034966145190747, 34.93567745625206],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr140",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [27.628125055687043, -29.14576169589308],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr141",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [5.428039899037065, -6.334348748044737],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr146",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-0.130248282097538, -0.8754748180069414],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr150",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [2.1072529676186718, -4.44830654647316],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr151",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-1.9871362288886847, -1.0132758712623824],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr153",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-1.2987220363655525, -0.08661656433559421],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr154",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [2.586882464962847, -3.140862893459697],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr159",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-2.08963081064638, 1.060026800706145],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr164",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [2.440364159712942, -2.911382264493648],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr165",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [0.8500043100901202, -1.9971938686263055],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr169",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [8.522863316633059, -9.101475224746238],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr173",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.6793760252561363, -5.034928404110086],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr175",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-46.09858410022454, 55.70979134146471],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr179",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [18.631897309303845, -21.062547387798197],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr181",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [7.388202727976789, -9.526347333459151],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr185",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [4.869581412885902, -5.983753871503793],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr187",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [33.8912615127072, -39.483893696223106],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr188",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [8.883118426721895, -13.002844065520373],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr189",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [8.487143623436125, -12.139261861969533],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr205",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [15.589854138273738, -21.513232684103425],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr209",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [10.893988665383004, -7.0354966432067805],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr211",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [4.176265517840403, -4.176265517840403],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr212",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-10.381616851079986, 8.855698615577385],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr214",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-2.9479423495064383, 2.9479423495064383],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr216",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [12.664171172232374, -9.73285752584392],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr220",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-6.692533113866633, 6.692533113866633],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr225",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [6.904885757896593, -7.755153969177377],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr226",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [1.0185331242021183, -5.830887068350061],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr234",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [10.52052446779662, -11.484345769140967],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr238",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [2.5481415494339625, -2.5481415494339625],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr243",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-4.508752490893423, 4.036601645970915],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr248",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.3164339276164836, -4.475572010253341],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr254",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [24.08860081042758, -28.489713376387325],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr261",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [5.918444241874461, -8.320031945458522],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr272",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [23.995618460429053, -33.06702528593947],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr278",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [24.416378455786084, -52.15584127391778],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr279",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.8689651678537382, -8.89062913515431],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr280",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [20.460369867396967, -25.03930352748174],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr282",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [7.768062266332482, -11.296467272524772],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr285",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-25.77423489112755, 45.98667665874816],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr296",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [31.059935048203357, -40.36743848207199],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr303",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [4.718724329614107, -7.047415633101438],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr311",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-72.79982424100923, 97.45026055113148],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr315",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [54.85265483129683, -65.89476247289998],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr317",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [12.80540950515617, -14.691634093287888],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr318",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [11.101612528250916, -13.917602371736994],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr325",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.965974188867205, -5.087119983093573],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr327",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [10.56842162007004, -14.20188816588989],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr330",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [4.005429812416501, -5.650106720610703],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr334",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-14.61956157218039, 29.02246241924777],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr338",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-14.076431883680625, 21.641552129532652],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr339",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [0.317255727546635, -10.51714074110638],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr340",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [17.445773156469695, -21.31730659862912],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr341",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [5.773751172778841, -9.058174222884388],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr342",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [204.14270885782315, -267.4128132580176],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr348",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [8.471939450850835, -9.724047642120292],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr350",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.034263670962536, -3.432720438553859],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr352",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [6.619744093849164, -6.619744093849164],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr356",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [14.421837033174507, -15.490140673017695],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr357",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [5.697951605671116, -5.697951605671116],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr359",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [15.882073318834395, -17.21752546514084],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr360",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [41.00679144674341, -53.770285106131475],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr361",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.143306683243703, -3.6401051503137474],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr366",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.159907457146807, -4.585200804020815],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr367",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [63.50091328296435, -85.15054154244731],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr382",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-310.9912514972791, 365.558300183354],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr386",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [6.531177796907089, -9.084567703896477],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr390",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-5.38057564301269, 1.7552104374133002],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr391",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [1.008432298595711, -2.2351414774643334],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr393",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [12.94437352347635, -16.730962437631366],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr397",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.5257597045603246, -6.50314526671842],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr403",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [9.796159026747853, -12.853906104638966],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr411",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-2.577008439144667, -3.4352523499733953],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr419",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.286577491719272, -5.428957990895863],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr422",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-3.1025700447375515, 0.7964790213540277],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr423",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-5.850055478981952, 4.075963356811394],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr424",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-4.488197982768084, 3.2624022684494745],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr427",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-5.078104124512248, 4.2736431433813316],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr428",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-2.424013869680437, 2.083973646880488],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr429",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-1.537454621749358, 1.537454621749358],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr432",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [0.34786388035615406, -1.0373285804673307],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr434",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.0945853720728023, -3.0945853720728023],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr435",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [46.1970968948239, -52.52235446746391],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr440",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [1.6661149161125428, -3.0691064728420456],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr443",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [5.706461167396316, -6.4054366861046015],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr446",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [109.73361375265044, -128.9190626824402],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr448",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [5.556152256386818, -9.78411635079325],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr451",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-21.42940368918971, 30.729547895807542],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr454",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [20.898677076911408, -22.590440715562185],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr456",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-69.80006548959105, 69.80006548959105],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr468",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [15.95697099628115, -14.926730630164194],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr469",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [16.612610676061564, -16.612610676061564],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr471",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [0.14428498474711113, -0.9695226940989641],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr474",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-26.10665101943864, 12.204899259595171],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr475",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [0.2836402018444656, -1.228669446684746],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr476",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [22.482068763085813, -22.482068763085813],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr480",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-3.551088117295367, 2.310746211871951],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr484",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [6.683483029987904, -6.683483029987904],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr486",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [5.528077313801828, -6.255618008254648],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr492",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-82.88851174737869, 92.20548062024544],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr497",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [508.8116379079629, -539.3073631673867],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr498",
traces: [1],
},
]);
})
.then(function () {
Plotly.animate("0f9bce4c-d300-476d-b62b-89155c630e1f", null);
});
}
</script>
</div>

Notice how the Perceptron struggles to find a boundary since the data is
clearly non-separable in 2D space. We'll later learn how this problem can be
fixed. For now, let's look at another simpler example for which such a boundary
exists:

<div>
<script type="text/javascript">
window.PlotlyConfig = { MathJaxConfig: "local" };
</script>
<script
charset="utf-8"
src="https://cdn.plot.ly/plotly-2.30.0.min.js"
></script>
<div
id="8dde9dda-261c-4420-8eae-1cc6ffd247cd"
class="plotly-graph-div"
style="height: 100%; width: 100%"
></div>
<script type="text/javascript">
window.PLOTLYENV = window.PLOTLYENV || {};
if (document.getElementById("8dde9dda-261c-4420-8eae-1cc6ffd247cd")) {
Plotly.newPlot(
"8dde9dda-261c-4420-8eae-1cc6ffd247cd",
[
{
marker: {
color: [
"red",
"red",
"red",
"red",
"red",
"red",
"red",
"red",
"red",
"red",
"green",
"green",
"green",
"green",
"green",
"green",
"green",
"green",
"green",
"green",
],
},
mode: "markers",
x: [
5, 5, 5, 5, 5, 5, 5, 5, 5, 5, -5, -5, -5, -5, -5, -5, -5, -5,
-5, -5,
],
y: [
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -0.5, -1.5, -2.5, -3.5, -4.5,
-5.5, -6.5, -7.5, -8.5, -9.5,
],
type: "scatter",
},
{
line: { color: "black" },
mode: "lines",
name: "BoundaryLine",
type: "scatter",
},
],
{
template: {
data: {
histogram2dcontour: [
{
type: "histogram2dcontour",
colorbar: { outlinewidth: 0, ticks: "" },
colorscale: [
[0.0, "#0d0887"],
[0.1111111111111111, "#46039f"],
[0.2222222222222222, "#7201a8"],
[0.3333333333333333, "#9c179e"],
[0.4444444444444444, "#bd3786"],
[0.5555555555555556, "#d8576b"],
[0.6666666666666666, "#ed7953"],
[0.7777777777777778, "#fb9f3a"],
[0.8888888888888888, "#fdca26"],
[1.0, "#f0f921"],
],
},
],
choropleth: [
{
type: "choropleth",
colorbar: { outlinewidth: 0, ticks: "" },
},
],
histogram2d: [
{
type: "histogram2d",
colorbar: { outlinewidth: 0, ticks: "" },
colorscale: [
[0.0, "#0d0887"],
[0.1111111111111111, "#46039f"],
[0.2222222222222222, "#7201a8"],
[0.3333333333333333, "#9c179e"],
[0.4444444444444444, "#bd3786"],
[0.5555555555555556, "#d8576b"],
[0.6666666666666666, "#ed7953"],
[0.7777777777777778, "#fb9f3a"],
[0.8888888888888888, "#fdca26"],
[1.0, "#f0f921"],
],
},
],
heatmap: [
{
type: "heatmap",
colorbar: { outlinewidth: 0, ticks: "" },
colorscale: [
[0.0, "#0d0887"],
[0.1111111111111111, "#46039f"],
[0.2222222222222222, "#7201a8"],
[0.3333333333333333, "#9c179e"],
[0.4444444444444444, "#bd3786"],
[0.5555555555555556, "#d8576b"],
[0.6666666666666666, "#ed7953"],
[0.7777777777777778, "#fb9f3a"],
[0.8888888888888888, "#fdca26"],
[1.0, "#f0f921"],
],
},
],
heatmapgl: [
{
type: "heatmapgl",
colorbar: { outlinewidth: 0, ticks: "" },
colorscale: [
[0.0, "#0d0887"],
[0.1111111111111111, "#46039f"],
[0.2222222222222222, "#7201a8"],
[0.3333333333333333, "#9c179e"],
[0.4444444444444444, "#bd3786"],
[0.5555555555555556, "#d8576b"],
[0.6666666666666666, "#ed7953"],
[0.7777777777777778, "#fb9f3a"],
[0.8888888888888888, "#fdca26"],
[1.0, "#f0f921"],
],
},
],
contourcarpet: [
{
type: "contourcarpet",
colorbar: { outlinewidth: 0, ticks: "" },
},
],
contour: [
{
type: "contour",
colorbar: { outlinewidth: 0, ticks: "" },
colorscale: [
[0.0, "#0d0887"],
[0.1111111111111111, "#46039f"],
[0.2222222222222222, "#7201a8"],
[0.3333333333333333, "#9c179e"],
[0.4444444444444444, "#bd3786"],
[0.5555555555555556, "#d8576b"],
[0.6666666666666666, "#ed7953"],
[0.7777777777777778, "#fb9f3a"],
[0.8888888888888888, "#fdca26"],
[1.0, "#f0f921"],
],
},
],
surface: [
{
type: "surface",
colorbar: { outlinewidth: 0, ticks: "" },
colorscale: [
[0.0, "#0d0887"],
[0.1111111111111111, "#46039f"],
[0.2222222222222222, "#7201a8"],
[0.3333333333333333, "#9c179e"],
[0.4444444444444444, "#bd3786"],
[0.5555555555555556, "#d8576b"],
[0.6666666666666666, "#ed7953"],
[0.7777777777777778, "#fb9f3a"],
[0.8888888888888888, "#fdca26"],
[1.0, "#f0f921"],
],
},
],
mesh3d: [
{
type: "mesh3d",
colorbar: { outlinewidth: 0, ticks: "" },
},
],
scatter: [
{
fillpattern: {
fillmode: "overlay",
size: 10,
solidity: 0.2,
},
type: "scatter",
},
],
parcoords: [
{
type: "parcoords",
line: { colorbar: { outlinewidth: 0, ticks: "" } },
},
],
scatterpolargl: [
{
type: "scatterpolargl",
marker: { colorbar: { outlinewidth: 0, ticks: "" } },
},
],
bar: [
{
error_x: { color: "#2a3f5f" },
error_y: { color: "#2a3f5f" },
marker: {
line: { color: "#E5ECF6", width: 0.5 },
pattern: {
fillmode: "overlay",
size: 10,
solidity: 0.2,
},
},
type: "bar",
},
],
scattergeo: [
{
type: "scattergeo",
marker: { colorbar: { outlinewidth: 0, ticks: "" } },
},
],
scatterpolar: [
{
type: "scatterpolar",
marker: { colorbar: { outlinewidth: 0, ticks: "" } },
},
],
histogram: [
{
marker: {
pattern: {
fillmode: "overlay",
size: 10,
solidity: 0.2,
},
},
type: "histogram",
},
],
scattergl: [
{
type: "scattergl",
marker: { colorbar: { outlinewidth: 0, ticks: "" } },
},
],
scatter3d: [
{
type: "scatter3d",
line: { colorbar: { outlinewidth: 0, ticks: "" } },
marker: { colorbar: { outlinewidth: 0, ticks: "" } },
},
],
scattermapbox: [
{
type: "scattermapbox",
marker: { colorbar: { outlinewidth: 0, ticks: "" } },
},
],
scatterternary: [
{
type: "scatterternary",
marker: { colorbar: { outlinewidth: 0, ticks: "" } },
},
],
scattercarpet: [
{
type: "scattercarpet",
marker: { colorbar: { outlinewidth: 0, ticks: "" } },
},
],
carpet: [
{
aaxis: {
endlinecolor: "#2a3f5f",
gridcolor: "white",
linecolor: "white",
minorgridcolor: "white",
startlinecolor: "#2a3f5f",
},
baxis: {
endlinecolor: "#2a3f5f",
gridcolor: "white",
linecolor: "white",
minorgridcolor: "white",
startlinecolor: "#2a3f5f",
},
type: "carpet",
},
],
table: [
{
cells: {
fill: { color: "#EBF0F8" },
line: { color: "white" },
},
header: {
fill: { color: "#C8D4E3" },
line: { color: "white" },
},
type: "table",
},
],
barpolar: [
{
marker: {
line: { color: "#E5ECF6", width: 0.5 },
pattern: {
fillmode: "overlay",
size: 10,
solidity: 0.2,
},
},
type: "barpolar",
},
],
pie: [{ automargin: true, type: "pie" }],
},
layout: {
autotypenumbers: "strict",
colorway: [
"#636efa",
"#EF553B",
"#00cc96",
"#ab63fa",
"#FFA15A",
"#19d3f3",
"#FF6692",
"#B6E880",
"#FF97FF",
"#FECB52",
],
font: { color: "#2a3f5f" },
hovermode: "closest",
hoverlabel: { align: "left" },
paper_bgcolor: "white",
plot_bgcolor: "#E5ECF6",
polar: {
bgcolor: "#E5ECF6",
angularaxis: {
gridcolor: "white",
linecolor: "white",
ticks: "",
},
radialaxis: {
gridcolor: "white",
linecolor: "white",
ticks: "",
},
},
ternary: {
bgcolor: "#E5ECF6",
aaxis: {
gridcolor: "white",
linecolor: "white",
ticks: "",
},
baxis: {
gridcolor: "white",
linecolor: "white",
ticks: "",
},
caxis: {
gridcolor: "white",
linecolor: "white",
ticks: "",
},
},
coloraxis: { colorbar: { outlinewidth: 0, ticks: "" } },
colorscale: {
sequential: [
[0.0, "#0d0887"],
[0.1111111111111111, "#46039f"],
[0.2222222222222222, "#7201a8"],
[0.3333333333333333, "#9c179e"],
[0.4444444444444444, "#bd3786"],
[0.5555555555555556, "#d8576b"],
[0.6666666666666666, "#ed7953"],
[0.7777777777777778, "#fb9f3a"],
[0.8888888888888888, "#fdca26"],
[1.0, "#f0f921"],
],
sequentialminus: [
[0.0, "#0d0887"],
[0.1111111111111111, "#46039f"],
[0.2222222222222222, "#7201a8"],
[0.3333333333333333, "#9c179e"],
[0.4444444444444444, "#bd3786"],
[0.5555555555555556, "#d8576b"],
[0.6666666666666666, "#ed7953"],
[0.7777777777777778, "#fb9f3a"],
[0.8888888888888888, "#fdca26"],
[1.0, "#f0f921"],
],
diverging: [
[0, "#8e0152"],
[0.1, "#c51b7d"],
[0.2, "#de77ae"],
[0.3, "#f1b6da"],
[0.4, "#fde0ef"],
[0.5, "#f7f7f7"],
[0.6, "#e6f5d0"],
[0.7, "#b8e186"],
[0.8, "#7fbc41"],
[0.9, "#4d9221"],
[1, "#276419"],
],
},
xaxis: {
gridcolor: "white",
linecolor: "white",
ticks: "",
title: { standoff: 15 },
zerolinecolor: "white",
automargin: true,
zerolinewidth: 2,
},
yaxis: {
gridcolor: "white",
linecolor: "white",
ticks: "",
title: { standoff: 15 },
zerolinecolor: "white",
automargin: true,
zerolinewidth: 2,
},
scene: {
xaxis: {
backgroundcolor: "#E5ECF6",
gridcolor: "white",
linecolor: "white",
showbackground: true,
ticks: "",
zerolinecolor: "white",
gridwidth: 2,
},
yaxis: {
backgroundcolor: "#E5ECF6",
gridcolor: "white",
linecolor: "white",
showbackground: true,
ticks: "",
zerolinecolor: "white",
gridwidth: 2,
},
zaxis: {
backgroundcolor: "#E5ECF6",
gridcolor: "white",
linecolor: "white",
showbackground: true,
ticks: "",
zerolinecolor: "white",
gridwidth: 2,
},
},
shapedefaults: { line: { color: "#2a3f5f" } },
annotationdefaults: {
arrowcolor: "#2a3f5f",
arrowhead: 0,
arrowwidth: 1,
},
geo: {
bgcolor: "white",
landcolor: "#E5ECF6",
subunitcolor: "white",
showland: true,
showlakes: true,
lakecolor: "white",
},
title: { x: 0.05 },
mapbox: { style: "light" },
},
},
xaxis: { title: { text: "Harvest Time" } },
yaxis: { title: { text: "Weight" } },
},
{ responsive: true }
)
.then(function () {
Plotly.addFrames("8dde9dda-261c-4420-8eae-1cc6ffd247cd", [
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-0.393939393939394, 0.4141414141414142],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr0",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-0.8041237113402062, 0.8453608247422681],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr1",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-1.24468085106383, 1.3085106382978726],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr2",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-1.7333333333333336, 1.8222222222222226],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr3",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-2.294117647058824, 2.4117647058823533],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr4",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-2.962025316455697, 3.113924050632912],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr5",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-3.791666666666668, 3.986111111111112],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr6",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-4.875000000000001, 5.125000000000002],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr7",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-6.381818181818184, 6.709090909090911],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr8",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-8.666666666666668, 9.111111111111114],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr9",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-10.911392405063294, 11.367088607594939],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr15",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-14.750000000000007, 15.250000000000007],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr17",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-22.800000000000015, 23.422222222222242],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr19",
traces: [1],
},
]);
})
.then(function () {
Plotly.animate("8dde9dda-261c-4420-8eae-1cc6ffd247cd", null);
});
}
</script>
</div>

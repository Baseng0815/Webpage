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

<div style="height: 800px;">
<script type="text/javascript">
window.PlotlyConfig = { MathJaxConfig: "local" };
</script>
<script
charset="utf-8"
src="https://cdn.plot.ly/plotly-2.30.0.min.js"
></script>
<div
id="2170b78f-d199-42ba-a2a3-084da382dcd8"
class="plotly-graph-div"
style="height: 100%; width: 100%"
></div>
<script type="text/javascript">
window.PLOTLYENV = window.PLOTLYENV || {};
if (document.getElementById("2170b78f-d199-42ba-a2a3-084da382dcd8")) {
Plotly.newPlot(
"2170b78f-d199-42ba-a2a3-084da382dcd8",
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
template: { data: { scatter: [{ type: "scatter" }] } },
xaxis: { title: { text: "Harvest Time" } },
yaxis: { title: { text: "Weight" } },
updatemenus: [
{
buttons: [
{
args: [
null,
{
frame: { duration: 500, redraw: false },
fromcurrent: true,
transition: {
duration: 300,
easing: "quadratic-in-out",
},
},
],
label: "Play",
method: "animate",
},
{
args: [
[null],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
transition: { duration: 0 },
},
],
label: "Pause",
method: "animate",
},
],
direction: "left",
pad: { r: 10, t: 87 },
showactive: false,
type: "buttons",
x: 0.1,
xanchor: "right",
y: 0,
yanchor: "top",
},
],
},
{ responsive: true }
)
.then(function () {
Plotly.addFrames("2170b78f-d199-42ba-a2a3-084da382dcd8", [
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [0.11207860969058463, -0.13266926187609132],
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
y: [0.14665928336933443, -0.14665928336933443],
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
y: [0.32606409474770803, -0.34755007915914726],
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
y: [0.3610781003632175, -0.40434191585474305],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr14",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [0.6067363944536402, -0.6748643922139915],
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
y: [0.8170737299222335, -0.9092427429604276],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr23",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [0.8910122737918211, -0.9611025015260688],
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
y: [0.8687333509695416, -0.9647501647426486],
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
y: [0.9347267206250875, -1.060929582821194],
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
y: [0.7365098915381351, -0.887803631724898],
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
y: [0.8975267273651374, -1.0752803681983596],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr41",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [1.1445904320526932, -1.2976100411344254],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr52",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [1.3556830200096406, -1.5373537546414282],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr55",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [1.6507415441527802, -1.8732604639422878],
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
y: [2.3085278898244086, -2.5757469669532775],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr61",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [2.323876402305011, -2.6268141442504915],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr62",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [2.4934629460023454, -2.8521577230935633],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr63",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [2.5359615154132826, -2.8745106669555813],
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
y: [2.263968233714369, -2.63403770119882],
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
y: [2.6649152006445562, -3.092285329605731],
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
y: [2.3838447515071564, -2.8492444458030373],
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
y: [2.484693551789064, -2.9159128229717384],
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
y: [2.5156572630655805, -3.003443496295522],
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
y: [2.817084063184739, -3.366988576248925],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr96",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.049713118766059, -3.6887423137381004],
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
y: [3.269889016836249, -3.9924934206988745],
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
y: [2.921534628949824, -3.690081836401703],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr106",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [2.8044559702629295, -3.5270145476275205],
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
y: [3.2664383803295864, -4.076052914409917],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr113",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [4.147042398915289, -4.940444102920454],
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
y: [5.144809538809985, -6.065548367103323],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr116",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [5.726469382487424, -6.6279180244951945],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr124",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [6.536861640727311, -7.442106290827204],
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
y: [6.97859230261073, -7.996372097458999],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr130",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [7.365329670008347, -8.53035569067743],
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
y: [9.437411141254334, -10.948052126314145],
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
y: [8.250053960880788, -9.568137837860375],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr147",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [7.54445464078504, -8.72042289930617],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr149",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [6.27740528929099, -7.45943965504308],
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
y: [7.19200700520887, -8.565246834249601],
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
y: [7.152494931587926, -8.523582222162567],
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
y: [6.377839565606665, -7.813670456362919],
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
y: [6.272738081176632, -7.835671786668349],
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
y: [5.624614211495654, -7.30475493230674],
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
y: [7.50423820357849, -9.580113231582802],
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
y: [7.480747849596203, -9.4076418167776],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr183",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [8.09783693950515, -9.961133964938949],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr184",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [7.644608106484612, -9.34449380457427],
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
y: [6.4764896504816285, -8.20008553141358],
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
y: [6.2145557428223634, -8.078295014360194],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr191",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [6.015878783995155, -8.028164563504914],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr197",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [5.9425863795388985, -7.8869019388424935],
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
y: [6.061651648370926, -7.999912894947653],
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
y: [7.237267124583128, -9.656685944140593],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr210",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [6.7459897420113295, -9.223291688814573],
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
y: [5.479810382301852, -8.043714433832294],
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
y: [5.958674773464986, -8.35519272107897],
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
y: [5.741766432235509, -7.990407833457555],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr218",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.727980614285654, -6.1014589090095726],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr219",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [4.822092138181313, -7.201985362654401],
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
y: [5.139239004352179, -7.326847948385476],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr222",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [4.598748566767222, -7.166311452066786],
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
y: [5.479576757816146, -8.674532655423802],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr236",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.915355932691464, -6.601659397955754],
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
y: [1.7249191387699905, -4.428658563875534],
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
y: [1.6355403504012764, -4.150958314719501],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr251",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [2.722763703528068, -5.76380975994743],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr265",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.4041600417735367, -7.149794752420784],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr268",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [4.579036574603101, -8.141464151552603],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr271",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.2964077087447223, -7.297561895005151],
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
y: [2.6044441738806863, -5.892810406626031],
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
y: [4.372411573438496, -9.121780677122507],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr281",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [5.968956182050582, -10.372039428872158],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr294",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [8.611313375965961, -16.299953370088154],
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
y: [9.121064235988351, -15.27778386129646],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr309",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [4.586382702032775, -8.805237186232391],
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
y: [11.180267557374922, -19.279083863524505],
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
y: [10.763281474525886, -16.946494830116414],
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
y: [7.689503868107145, -11.796442065905001],
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
y: [4.944164972232619, -7.9100328389068],
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
y: [7.250358269220332, -11.70090654072909],
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
y: [5.143410417022963, -8.66535604672566],
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
y: [8.242091337314251, -15.206007442070966],
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
y: [9.31919620255163, -16.20885763167891],
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
y: [21.087574403153315, -34.10992177258217],
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
y: [10.416468522152405, -15.598282911456192],
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
y: [6.149613020247116, -9.746603084400471],
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
y: [9.299803976791921, -13.705803443144601],
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
y: [8.13058942640778, -11.44762044351796],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr358",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [5.673046649155856, -8.189930685397135],
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
y: [5.049592093407828, -8.303015806231206],
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
y: [4.251213681225883, -6.648597035098793],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr377",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [7.6028000336695705, -11.387336057971973],
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
y: [9.042139560527714, -12.703342999943155],
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
y: [4.168423990686946, -8.304118731833942],
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
y: [7.856786942093343, -12.226274720350812],
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
y: [1.6682638352997334, -7.73863927602657],
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
y: [-1.8119000790237745, -3.4551880798890746],
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
y: [-0.5902753458063575, -3.713272548614068],
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
y: [-1.3497052912058143, -2.4060845698620605],
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
y: [1.0141534944630066, -3.722359975038073],
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
y: [2.211504742711914, -4.655826441753561],
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
y: [9.370731420616323, -15.767465279889478],
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
y: [14.310584223366098, -20.51190185580276],
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
y: [7.395903692876686, -10.41653080628848],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr444",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [10.859433239701236, -13.26845451000843],
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
y: [63.79080697917629, -78.76060888356548],
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
y: [14.701754564803183, -21.089344204785025],
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
y: [4.6045485462314915, -11.69580205229672],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr452",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [83.35393702070331, -179.32362464940363],
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
y: [14.914828782174066, -20.67075920663694],
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
y: [10.036882008319113, -12.741472458780178],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr457",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [22.100674179985095, -25.624148927670376],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr464",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [23.23881291179354, -25.455915595480505],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr466",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [236.29429645222854, -246.3909567437093],
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
y: [18.803161487260077, -18.803161487260077],
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
y: [19.822512011849515, -20.754576322728386],
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
y: [10.845029250595218, -10.845029250595218],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr473",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [3.1062958021786105, -3.552043773993522],
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
y: [2.584405272113004, -4.285787377578173],
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
y: [22.67454665605328, -24.60562216500392],
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
y: [14.024223282158632, -14.024223282158632],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr483",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [-0.3048272499978101, -0.5113129402523817],
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
y: [9.514948677506556, -12.26026879915132],
type: "scatter",
},
],
layout: {
xaxis: { autorange: false },
yaxis: { autorange: false },
},
name: "fr487",
traces: [1],
},
{
data: [
{
mode: "lines",
x: [-8, 8],
y: [5.966831588680798, -6.9914573826649535],
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
Plotly.animate("2170b78f-d199-42ba-a2a3-084da382dcd8", null);
});
}
</script>
</div>

Notice how the Perceptron struggles to find a boundary since the data is
clearly non-separable in 2D space. We'll later learn how this problem can be
fixed. For now, let's look at another simpler example for which such a boundary
exists:

<div style="height: 600px;">
<script type="text/javascript">
window.PlotlyConfig = { MathJaxConfig: "local" };
</script>
<script
charset="utf-8"
src="https://cdn.plot.ly/plotly-2.30.0.min.js"
></script>
<div
id="8a268dda-487a-4e76-b305-4c7905a6c43c"
class="plotly-graph-div"
style="height: 100%; width: 100%"
></div>
<script type="text/javascript">
window.PLOTLYENV = window.PLOTLYENV || {};
if (document.getElementById("8a268dda-487a-4e76-b305-4c7905a6c43c")) {
Plotly.newPlot(
"8a268dda-487a-4e76-b305-4c7905a6c43c",
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
template: { data: { scatter: [{ type: "scatter" }] } },
xaxis: { title: { text: "x1" } },
yaxis: { title: { text: "x2" } },
updatemenus: [
{
buttons: [
{
args: [
null,
{
frame: { duration: 500, redraw: false },
fromcurrent: true,
transition: {
duration: 300,
easing: "quadratic-in-out",
},
},
],
label: "Play",
method: "animate",
},
{
args: [
[null],
{
frame: { duration: 0, redraw: false },
mode: "immediate",
transition: { duration: 0 },
},
],
label: "Pause",
method: "animate",
},
],
direction: "left",
pad: { r: 10, t: 87 },
showactive: false,
type: "buttons",
x: 0.1,
xanchor: "right",
y: 0,
yanchor: "top",
},
],
},
{ responsive: true }
)
.then(function () {
Plotly.addFrames("8a268dda-487a-4e76-b305-4c7905a6c43c", [
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
Plotly.animate("8a268dda-487a-4e76-b305-4c7905a6c43c", null);
});
}
</script>
</div>

# Generalized linear models

You have probably noticed how linear regression and logistic regression work in
a similar fashion: they linearly combine features and then apply a function to
them. They both belong to a larger class of models generalizing this notion
called generalized linear models.

## Exponential families

Before we can dive deeper and take a look at GLMs, we need to introduce
something else first. If a probability distribution $p$ can be written in the
form of $p(y;\eta)=b(y)\exp(\eta^TT(y)-a(\eta))$, it is a member of the
exponential family and has some nice properties. The different terms in this
case mean the following:

- $y$: input
- $\eta$: natural parameter
- $T(y)$: sufficient statistic; usually $T(y)=y$
- $b(y)$: base measure
- $a(\eta)$: log-partition

Let's look at a few examples and identify the parts:

### Bernoulli distribution

A Bernoulli experiment E has a single binary outcome of 0 or 1. If we let
$\phi=\Pr[E=1]$, it is easy to see that $p(y;\phi)=\phi^y(1-\phi)^{1-y}$: if we
let $\phi=0$, then $p(y;\phi)=1-y$ and if $\phi=1$ then $p(y;\phi)=y$. Now let's rewrite
$p(y;\phi)$ in such a way its exponential nature becomes clear:

\begin{gather}
    p(y;\phi)=\phi^y(1-\phi)^{1-y}=\exp(\ln(\phi^y(1-\phi)^{1-y}))=\exp(\ln(\frac{\phi}{1-\phi})+\ln(1-\phi))
\end{gather}

It is a member of the exponential family with

- $b(y)=1$
- $\eta=\ln(\frac{\phi}{1-\phi})\Leftrightarrow\phi=\frac{1}{1+e^{-\eta}}$
- $T(y)=y$
- $a(\eta)=-\ln(1-\frac{1}{1+e^{-\eta}})=\ln(1+e^{-\eta})$

### Gaussian distribution

A Gaussian or normal distribution is perhaps the most important distribution
due to the [central limit
theorem](https://www.youtube.com/watch?v=zeJD6dqJ5lo): the sum of independently
sampled values from any distribution approaches a normal distribution. Lots of
real-world properties assume a normal distribution: age, height, IQ etc. The
Gaussian is defined in terms of its mean $\mu$ and standard deviation $\sigma$:

\begin{gather}
    p(y;\eta=(\mu,\sigma))=\frac{1}{\sigma \sqrt{2\pi}}\exp(-\frac{1}{2}(\frac{x-\mu}{\sigma})^2)=\frac{1}{\sqrt{2\pi}}\exp(-\frac{y^2}{2})\exp(y\mu-\frac{\mu^2}{2})
\end{gather}

We assume a fixed variance of $\sigma=1$ so only the mean is left as a
parameter:

- $b(y)=\frac{1}{\sqrt{2\pi}}$
- $\eta=\mu$
- $T(y)=y$
- $a(\eta)=\frac{\mu^2}{2}$

<div>
<script type="text/javascript">
window.PlotlyConfig = { MathJaxConfig: "local" };
</script>
<script
charset="utf-8"
src="https://cdn.plot.ly/plotly-2.30.0.min.js"
></script>
<div
id="ffafcae9-c4cc-4f0b-8e60-fd020ed98631"
class="plotly-graph-div"
style="height: 100%; width: 70%; margin: 0 auto;"
></div>
<script type="text/javascript">
window.PLOTLYENV = window.PLOTLYENV || {};
if (document.getElementById("ffafcae9-c4cc-4f0b-8e60-fd020ed98631")) {
Plotly.newPlot(
"ffafcae9-c4cc-4f0b-8e60-fd020ed98631",
[
{
fill: "tozeroy",
name: "$\\mu=0,\\sigma=1$",
x: [
-2.0, -1.9595959595959596, -1.9191919191919191,
-1.878787878787879, -1.8383838383838385, -1.797979797979798,
-1.7575757575757576, -1.7171717171717171, -1.6767676767676767,
-1.6363636363636362, -1.595959595959596, -1.5555555555555556,
-1.5151515151515151, -1.4747474747474747, -1.4343434343434343,
-1.393939393939394, -1.3535353535353534, -1.3131313131313131,
-1.2727272727272727, -1.2323232323232323, -1.1919191919191918,
-1.1515151515151514, -1.1111111111111112, -1.0707070707070705,
-1.0303030303030303, -0.9898989898989898, -0.9494949494949494,
-0.909090909090909, -0.8686868686868685, -0.8282828282828283,
-0.7878787878787878, -0.7474747474747474, -0.707070707070707,
-0.6666666666666665, -0.6262626262626261, -0.5858585858585856,
-0.5454545454545454, -0.505050505050505, -0.46464646464646453,
-0.4242424242424241, -0.38383838383838365,
-0.3434343434343432, -0.303030303030303, -0.26262626262626254,
-0.2222222222222221, -0.18181818181818166,
-0.14141414141414121, -0.10101010101010077,
-0.06060606060606055, -0.02020202020202011,
0.020202020202020332, 0.060606060606060996,
0.10101010101010122, 0.14141414141414144, 0.1818181818181821,
0.22222222222222232, 0.262626262626263, 0.3030303030303032,
0.3434343434343434, 0.3838383838383841, 0.4242424242424243,
0.464646464646465, 0.5050505050505052, 0.5454545454545459,
0.5858585858585861, 0.6262626262626263, 0.666666666666667,
0.7070707070707072, 0.7474747474747478, 0.7878787878787881,
0.8282828282828287, 0.868686868686869, 0.9090909090909092,
0.9494949494949498, 0.9898989898989901, 1.0303030303030307,
1.070707070707071, 1.1111111111111112, 1.1515151515151518,
1.191919191919192, 1.2323232323232327, 1.272727272727273,
1.3131313131313136, 1.3535353535353538, 1.393939393939394,
1.4343434343434347, 1.474747474747475, 1.5151515151515156,
1.5555555555555558, 1.5959595959595965, 1.6363636363636367,
1.676767676767677, 1.7171717171717176, 1.7575757575757578,
1.7979797979797985, 1.8383838383838387, 1.878787878787879,
1.9191919191919196, 1.9595959595959598, 2.0,
],
y: [
1.4867195147342987e-6, 2.4510610429423323e-6,
3.999890372416629e-6, 6.461166392648799e-6,
1.0331006581321711e-5, 1.6350958895825323e-5,
2.5616081195138293e-5, 3.9723822381414844e-5,
6.0975903952989717e-5, 9.26476353230142e-5,
0.0001393411231349691, 0.00020744030876792077,
0.0003056862254780556, 0.0004458897248075992,
0.0006437954979268652, 0.000920104769622966,
0.0013016538416489134, 0.0018227310960012774,
0.0025264957781039347, 0.003466437918975763,
0.0047077907631233405, 0.0063287764285827625,
0.008421534484118356, 0.011092554839374867,
0.014462414797634193, 0.01866460993406645,
0.023843274502042877, 0.03014961391680064,
0.037736923140639936, 0.04675414240670547,
0.05733800512481293, 0.06960395839232576, 0.0836361772145172,
0.09947713879274872, 0.11711735953274452, 0.13648600918747045,
0.15744318761884366, 0.17977466512481263, 0.20318983550122394,
0.22732350563136094, 0.2517419469842393, 0.27595337147011534,
0.29942268327109967, 0.3215900234094101, 0.34189229416612926,
0.35978655781262325, 0.3747739794063901, 0.3864228530895687,
0.3943892340049188, 0.39843380169134646, 0.39843380169134646,
0.3943892340049188, 0.3864228530895687, 0.3747739794063901,
0.35978655781262336, 0.34189229416612926, 0.3215900234094101,
0.29942268327109967, 0.27595337147011556, 0.2517419469842393,
0.22732350563136094, 0.20318983550122394, 0.17977466512481274,
0.15744318761884377, 0.13648600918747045, 0.11711735953274452,
0.09947713879274865, 0.08363617721451726, 0.0696039583923258,
0.05733800512481293, 0.04675414240670547, 0.03773692314063997,
0.03014961391680068, 0.023843274502042877,
0.01866460993406645, 0.014462414797634212,
0.011092554839374881, 0.008421534484118356,
0.0063287764285827625, 0.004707790763123345,
0.0034664379189757568, 0.0025264957781039347,
0.0018227310960012824, 0.0013016538416489134,
0.0009201047696229677, 0.0006437954979268629,
0.0004458897248075992, 0.0003056862254780567,
0.00020744030876792036, 0.0001393411231349691,
9.264763532301454e-5, 6.0975903952989717e-5,
3.972382238141498e-5, 2.561608119513816e-5,
1.6350958895825323e-5, 1.0331006581321747e-5,
6.461166392648799e-6, 3.999890372416629e-6,
2.4510610429423238e-6, 1.4867195147342987e-6,
],
type: "scatter",
},
{
fill: "tozeroy",
name: "$\\mu=1,\\sigma=2$",
x: [
-2.0, -1.9595959595959596, -1.9191919191919191,
-1.878787878787879, -1.8383838383838385, -1.797979797979798,
-1.7575757575757576, -1.7171717171717171, -1.6767676767676767,
-1.6363636363636362, -1.595959595959596, -1.5555555555555556,
-1.5151515151515151, -1.4747474747474747, -1.4343434343434343,
-1.393939393939394, -1.3535353535353534, -1.3131313131313131,
-1.2727272727272727, -1.2323232323232323, -1.1919191919191918,
-1.1515151515151514, -1.1111111111111112, -1.0707070707070705,
-1.0303030303030303, -0.9898989898989898, -0.9494949494949494,
-0.909090909090909, -0.8686868686868685, -0.8282828282828283,
-0.7878787878787878, -0.7474747474747474, -0.707070707070707,
-0.6666666666666665, -0.6262626262626261, -0.5858585858585856,
-0.5454545454545454, -0.505050505050505, -0.46464646464646453,
-0.4242424242424241, -0.38383838383838365,
-0.3434343434343432, -0.303030303030303, -0.26262626262626254,
-0.2222222222222221, -0.18181818181818166,
-0.14141414141414121, -0.10101010101010077,
-0.06060606060606055, -0.02020202020202011,
0.020202020202020332, 0.060606060606060996,
0.10101010101010122, 0.14141414141414144, 0.1818181818181821,
0.22222222222222232, 0.262626262626263, 0.3030303030303032,
0.3434343434343434, 0.3838383838383841, 0.4242424242424243,
0.464646464646465, 0.5050505050505052, 0.5454545454545459,
0.5858585858585861, 0.6262626262626263, 0.666666666666667,
0.7070707070707072, 0.7474747474747478, 0.7878787878787881,
0.8282828282828287, 0.868686868686869, 0.9090909090909092,
0.9494949494949498, 0.9898989898989901, 1.0303030303030307,
1.070707070707071, 1.1111111111111112, 1.1515151515151518,
1.191919191919192, 1.2323232323232327, 1.272727272727273,
1.3131313131313136, 1.3535353535353538, 1.393939393939394,
1.4343434343434347, 1.474747474747475, 1.5151515151515156,
1.5555555555555558, 1.5959595959595965, 1.6363636363636367,
1.676767676767677, 1.7171717171717176, 1.7575757575757578,
1.7979797979797985, 1.8383838383838387, 1.878787878787879,
1.9191919191919196, 1.9595959595959598, 2.0,
],
y: [
0.002215924205969004, 0.002575153996180446,
0.0029849958244363346, 0.0034512503966570734,
0.003980168242015007, 0.004578451411595744,
0.005253249261075107, 0.006012147481177914,
0.006863149542032055, 0.007814649738427807,
0.008875397064681833, 0.010054449212293716,
0.011361116072547714, 0.012804892240901141,
0.014395378161138722, 0.016142189714963694,
0.01805485625634431, 0.0201427073081617, 0.02241474837929637,
0.02487952662015703, 0.02754498730981013,
0.030418322453447964, 0.033505813059230116,
0.03681266695207025, 0.04034285426158183,
0.044098942984474344, 0.048081937260190236,
0.05229112120431057, 0.05672391130912995, 0.06137572053700365,
0.06623983729237697, 0.0713073224564121, 0.07656692759852914,
0.08200503733799681, 0.08760563861355154, 0.09335031932923878,
0.09921829848154261, 0.10518648943969244, 0.11122959755318698,
0.117320252705495, 0.1234291768300024, 0.12952538576484962,
0.13557642416023905, 0.14154863147934746, 0.14740743646743862,
0.15311767681905009, 0.15864394016637332, 0.16395092195905192,
0.16900379532180812, 0.17376858757559924, 0.1782125578028573,
0.18230456963817546, 0.18601545338060763, 0.18931835155787455,
0.19218904222862948, 0.19460623458587317, 0.19655183181883817,
0.19801115669531638, 0.19897313593166374, 0.19943044011103903,
0.1993795766768709, 0.19882093435113446, 0.19775877818723295,
0.19620119534555325, 0.1941599925560253, 0.19165054708624257,
0.18869161384649658, 0.1853050920161935, 0.18151575525295313,
0.17735095013175148, 0.17284026794473015, 0.16801519536351928,
0.16290874971884506, 0.1575551047836599, 0.15198921295600418,
0.14624642962929818, 0.1403621453166577, 0.134371430770662,
0.12830869992162205, 0.12220739495859792, 0.11609969731250253,
0.11001626768499537, 0.10398601761649182, 0.09803591441763762,
0.09219082061683077, 0.08647336841693638, 0.08090386902124001,
0.07550025609450743, 0.07027806208069724, 0.06525042561342527,
0.06042812783574951, 0.05581965509718881,
0.051431285220988755, 0.047267194334395995,
0.04332958112808671, 0.03961880535517562, 0.03613353739116636,
0.032870915748228306, 0.029826709563723932,
0.02699548325659403,
],
type: "scatter",
},
{
fill: "tozeroy",
name: "$\\mu=-2,\\sigma=0.7$",
x: [
-2.0, -1.979899497487437, -1.9597989949748744,
-1.9396984924623115, -1.9195979899497488, -1.899497487437186,
-1.879396984924623, -1.8592964824120604, -1.8391959798994975,
-1.8190954773869348, -1.7989949748743719, -1.778894472361809,
-1.7587939698492463, -1.7386934673366834, -1.7185929648241207,
-1.6984924623115578, -1.678391959798995, -1.6582914572864322,
-1.6381909547738693, -1.6180904522613067, -1.5979899497487438,
-1.5778894472361809, -1.557788944723618, -1.5376884422110553,
-1.5175879396984926, -1.4974874371859297, -1.4773869346733668,
-1.457286432160804, -1.4371859296482412, -1.4170854271356785,
-1.3969849246231156, -1.3768844221105527, -1.3567839195979898,
-1.3366834170854272, -1.3165829145728645, -1.2964824120603016,
-1.2763819095477387, -1.2562814070351758, -1.236180904522613,
-1.2160804020100502, -1.1959798994974875, -1.1758793969849246,
-1.1557788944723617, -1.135678391959799, -1.1155778894472361,
-1.0954773869346734, -1.0753768844221105, -1.0552763819095476,
-1.035175879396985, -1.015075376884422, -0.9949748743718594,
-0.9748743718592965, -0.9547738693467336, -0.9346733668341709,
-0.914572864321608, -0.8944723618090451, -0.8743718592964824,
-0.8542713567839195, -0.8341708542713568, -0.8140703517587939,
-0.793969849246231, -0.7738693467336684, -0.7537688442211055,
-0.7336683417085428, -0.7135678391959799, -0.693467336683417,
-0.6733668341708543, -0.6532663316582914, -0.6331658291457287,
-0.6130653266331658, -0.5929648241206029, -0.5728643216080402,
-0.5527638190954773, -0.5326633165829147, -0.5125628140703518,
-0.49246231155778886, -0.4723618090452262,
-0.4522613065326633, -0.4321608040201004, -0.4120603015075377,
-0.3919597989949748, -0.3718592964824121, -0.3517587939698492,
-0.3316582914572863, -0.31155778894472363,
-0.29145728643216073, -0.27135678391959805,
-0.25125628140703515, -0.23115577889447225,
-0.21105527638190957, -0.19095477386934667,
-0.170854271356784, -0.1507537688442211, -0.1306532663316582,
-0.11055276381909551, -0.09045226130653261,
-0.07035175879396993, -0.05025125628140703,
-0.03015075376884413, -0.01005025125628145,
0.010050251256281229, 0.03015075376884413,
0.05025125628140703, 0.07035175879396993, 0.09045226130653283,
0.11055276381909529, 0.1306532663316582, 0.1507537688442211,
0.170854271356784, 0.1909547738693469, 0.2110552763819098,
0.23115577889447225, 0.25125628140703515, 0.27135678391959805,
0.29145728643216096, 0.31155778894472386, 0.3316582914572863,
0.3517587939698492, 0.3718592964824121, 0.391959798994975,
0.4120603015075379, 0.4321608040201004, 0.4522613065326633,
0.4723618090452262, 0.4924623115577891, 0.512562814070352,
0.5326633165829144, 0.5527638190954773, 0.5728643216080402,
0.5929648241206031, 0.613065326633166, 0.6331658291457285,
0.6532663316582914, 0.6733668341708543, 0.6934673366834172,
0.7135678391959801, 0.7336683417085426, 0.7537688442211055,
0.7738693467336684, 0.7939698492462313, 0.8140703517587942,
0.8341708542713566, 0.8542713567839195, 0.8743718592964824,
0.8944723618090453, 0.9145728643216082, 0.9346733668341707,
0.9547738693467336, 0.9748743718592965, 0.9949748743718594,
1.0150753768844223, 1.0351758793969852, 1.0552763819095476,
1.0753768844221105, 1.0954773869346734, 1.1155778894472363,
1.1356783919597992, 1.1557788944723617, 1.1758793969849246,
1.1959798994974875, 1.2160804020100504, 1.2361809045226133,
1.2562814070351758, 1.2763819095477387, 1.2964824120603016,
1.3165829145728645, 1.3366834170854274, 1.3567839195979898,
1.3768844221105527, 1.3969849246231156, 1.4170854271356785,
1.4371859296482414, 1.457286432160804, 1.4773869346733668,
1.4974874371859297, 1.5175879396984926, 1.5376884422110555,
1.557788944723618, 1.5778894472361809, 1.5979899497487438,
1.6180904522613067, 1.6381909547738696, 1.658291457286432,
1.678391959798995, 1.6984924623115578, 1.7185929648241207,
1.7386934673366836, 1.758793969849246, 1.778894472361809,
1.7989949748743719, 1.8190954773869348, 1.8391959798994977,
1.8592964824120601, 1.879396984924623, 1.899497487437186,
1.9195979899497488, 1.9396984924623117, 1.9597989949748746,
1.979899497487437, 2.0,
],
y: [
5.853199333205836e-5, 7.941263835566941e-5,
0.00010718840836943201, 0.00014393549510260207,
0.00019228697906407334, 0.000255560502619376,
0.0003379087807994295, 0.00044449523305945145,
0.0005816967332475655, 0.0007573350008473439,
0.0009809374582874127, 0.0012640274226272637,
0.0016204422555430346, 0.0020666765413038576,
0.002622245487906834, 0.0033100615568750717,
0.0041568148473552415, 0.005193345038828125,
0.006454989810030908, 0.00798189170552735,
0.009819242552769851, 0.012017441908928561,
0.014632143833218212, 0.01772416475398691, 0.0213592245612281,
0.025607493537730383, 0.03054291956763805,
0.03624231342387093, 0.04278417498408651, 0.05024725004279821,
0.05870881597662944, 0.06824270478365298, 0.07891708374828096,
0.09079202685335419, 0.10391692362416032, 0.11832778578517046,
0.1340445252815526, 0.15106828913266596, 0.16937894647173315,
0.1889328302124686, 0.20966083934343857, 0.23146700725392927,
0.2542276362564532, 0.27779108829869287, 0.30197830669227993,
0.3265841237301344, 0.3513793847990654, 0.376113891778664,
0.40052013816947446, 0.4243177767544382, 0.44721872907834637,
0.46893281614149107, 0.48917376297657045, 0.5076654076751194,
0.5241479292577372, 0.5383838996067611, 0.5501639632520694,
0.5593119554997644, 0.5656892841904341, 0.5691984228181826,
0.5697853919753019, 0.5674411408827719, 0.5622017795894416,
0.5541476535142167, 0.5434012934802059, 0.530124314340412,
0.514513371899511, 0.4967953194607186, 0.47722173061172457,
0.45606297279710667, 0.4336020261899318, 0.41012824417487354,
0.3859312456082082, 0.36129511553483346, 0.33649307116689575,
0.3117827248824653, 0.2874020471968229, 0.2635661016076267,
0.24046459144754004, 0.2182602278536452, 0.19708789899864995,
0.17705459493742343, 0.15824002066952528, 0.14069781289716415,
0.12445726378143264, 0.10952544781036541, 0.09588964548707217,
0.08351995951198621, 0.07237202489212798, 0.06238972326828529,
0.05350782294853615, 0.04565447889621758, 0.03875354049028447,
0.03272662856537926, 0.027494956446467222,
0.02298088192526821, 0.019109188007819123,
0.01580809953931991, 0.013010050343901439,
0.010652221267191715, 0.008676873531826845,
0.00703150423530794, 0.005668851813905741,
0.004546779078479542, 0.0036280602286393646,
0.0028800963039339886, 0.00227458105840046,
0.001787136452499389, 0.0013969340231430047,
0.0010863154675846024, 0.0008404229787022933,
0.0006468472854508728, 0.0004952990422707192,
0.00037730720903124783, 0.00028594638121146336,
0.00021559366377349324, 0.00016171461374883775,
0.00012067697891576978, 8.95904003809341e-5,
6.616988993743697e-5, 4.8620702973296285e-5,
3.554217016515652e-5, 2.5848094724641048e-5,
1.8701438695204474e-5, 1.3461187885176118e-5,
9.639480756098829e-6, 6.867296235053033e-6,
4.867206921446671e-6, 3.431908691381208e-6,
2.407429211192353e-6, 1.6800925798883547e-6,
1.1664732620446894e-6, 8.057090643134511e-7,
5.536605724030956e-7, 3.7850431238718165e-7,
2.574304720317084e-7, 1.741850834043695e-7,
1.1725297084106357e-7, 7.852331690168885e-8,
5.231609311293561e-8, 3.4676388378801436e-8,
2.2866215886207904e-8, 1.5000876208092973e-8,
9.790409143543512e-9, 6.356922875071213e-9,
4.106340231083872e-9, 2.6389113696833466e-9,
1.6871611451953524e-9, 1.0731247534466275e-9,
6.7905624393924e-10, 4.2748725150507135e-10,
2.677333519277635e-10, 1.6681829165381413e-10,
1.0340624078893232e-10, 6.376930996218648e-11,
3.912357852581079e-11, 2.3879614548232862e-11,
1.4500331791823505e-11, 8.759724626090905e-12,
5.264593635409478e-12, 3.1477565891192965e-12,
1.872402945504673e-12, 1.1080500463581238e-12,
6.523509865252645e-13, 3.820895476334798e-13,
2.2264393951846827e-13, 1.2906797167332438e-13,
7.44368487036418e-14, 4.270899572551422e-14,
2.4378816953805382e-14, 1.384419730899697e-14,
7.821405828925899e-15, 4.396061381208936e-15,
2.4581283830119686e-15, 1.3674367952678184e-15,
7.567838383047172e-16, 4.166758427217771e-16,
2.282373324108643e-16, 1.243760985511901e-16,
6.742936397935566e-17, 3.636830691876188e-17,
1.9514570190491625e-17, 1.041733868157988e-17,
5.53243683317303e-18, 2.9230620252412212e-18,
1.5364607993883023e-18, 8.03464743521987e-19,
4.1799785310938427e-19, 2.1634315784848534e-19,
1.1139717172682757e-19, 5.706463599529049e-20,
2.908183624452469e-20, 1.4744786369934524e-20,
7.437329647058968e-21, 3.732136029169953e-21,
1.8632013254840775e-21, 9.253882718514496e-22,
4.572461782746183e-22, 2.247698543370579e-22,
1.0992283752437774e-22,
],
type: "scatter",
},
],
{
template: { data: { scatter: [{ type: "scatter" }] } },
title: { text: "Gaussian distribution" },
},
{ responsive: true }
);
}
</script>
</div>

### Poisson distribution

The Poisson distribution can be used to express the probability of a certain
number of events occuring over a certain time if events occur independently and
with a constant rate. If $\lambda$ is the expected value of events observed over
a certain period, then the probability observing $n$ events is defined as follows:

\begin{gather}
    p(n;\lambda)=\frac{\lambda^n e^{-\lambda}}{n!}
\end{gather}

For instance, if we expect to receive exactly one e-mail per hour, the
probability of receiving none is equal to
$p(0;1)=\frac{1^0e^{-1}}{0!}=\frac{1}{e}\approx 0.37$. If we expect to receive
exactly three e-mails per hour, the same probability decreases to
$p(0;3)=\frac{3^0e^{-3}}{0!}=\frac{1}{e^3}\approx 0.05$.

<div>
<script type="text/javascript">
window.PlotlyConfig = { MathJaxConfig: "local" };
</script>
<script
charset="utf-8"
src="https://cdn.plot.ly/plotly-2.30.0.min.js"
></script>
<div
id="05aec9d0-5c7d-453e-8f49-491c93999d52"
class="plotly-graph-div"
style="height: 100%; width: 100%"
></div>
<script type="text/javascript">
window.PLOTLYENV = window.PLOTLYENV || {};
if (document.getElementById("05aec9d0-5c7d-453e-8f49-491c93999d52")) {
Plotly.newPlot(
"05aec9d0-5c7d-453e-8f49-491c93999d52",
[
{
fill: "tozeroy",
mode: "markers+lines",
name: "$\\lambda=1$",
x: [
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
18, 19, 20, 21, 22, 23, 24,
],
y: [
0.36787944117144233, 0.36787944117144233, 0.18393972058572117,
0.06131324019524039, 0.015328310048810098,
0.0030656620097620196, 0.0005109436682936699,
7.299195261338141e-5, 9.123994076672677e-6,
1.0137771196302974e-6, 1.0137771196302975e-7,
9.216155633002704e-9, 7.68012969416892e-10,
5.907792072437631e-11, 4.2198514803125934e-12,
2.8132343202083955e-13, 1.7582714501302472e-14,
1.0342773236060278e-15, 5.745985131144599e-17,
3.0242027006024205e-18, 1.5121013503012103e-19,
7.200482620481953e-21, 3.272946645673615e-22,
1.4230202807276587e-23, 5.929251169698579e-25,
],
type: "scatter",
},
{
fill: "tozeroy",
mode: "markers+lines",
name: "$\\lambda=5$",
x: [
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
18, 19, 20, 21, 22, 23, 24,
],
y: [
0.006737946999085467, 0.03368973499542734,
0.08422433748856833, 0.14037389581428056, 0.1754673697678507,
0.1754673697678507, 0.1462228081398756, 0.104444862957054,
0.06527803934815875, 0.03626557741564375, 0.01813278870782187,
0.00824217668537358, 0.0034342402855723248,
0.0013208616482970478, 0.00047173630296323143,
0.00015724543432107713, 4.91391982253366e-5,
1.4452705360393119e-5, 4.014640377886977e-6,
1.0564843099702573e-6, 2.6412107749256427e-7,
6.288597083156293e-8, 1.4292266098082485e-8,
3.1070143691483657e-9, 6.472946602392429e-10,
],
type: "scatter",
},
{
fill: "tozeroy",
mode: "markers+lines",
name: "$\\lambda=8$",
x: [
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
18, 19, 20, 21, 22, 23, 24,
],
y: [
0.00033546262790251185, 0.002683701023220095,
0.01073480409288038, 0.02862614424768101, 0.05725228849536202,
0.09160366159257924, 0.12213821545677231, 0.13958653195059692,
0.13958653195059692, 0.1240769172894195, 0.09926153383153559,
0.07219020642293499, 0.048126804281956655,
0.029616494942742554, 0.01692371139585289,
0.009025979411121541, 0.004512989705560771,
0.0021237598614403624, 0.0009438932717512722,
0.00039742874600053567, 0.00015897149840021427,
-6.0560570819129245e-5, 0.0, 0.0, 0.0,
],
type: "scatter",
},
],
{
template: { data: { scatter: [{ type: "scatter" }] } },
title: { text: "Poisson distribution" },
},
{ responsive: true }
);
}
</script>
</div>

Expressing the Poisson distribution as a member of the exponential family:

\begin{gather}
    p(k;\lambda)=\frac{\lambda^ne^{-\lambda}}{n!}=\frac{1}{n!}e^{n\ln\lambda}e^{-\lambda}=\frac{1}{n!}\exp(n\ln\lambda-\lambda)
\end{gather}

- $b(y)=\frac{1}{n!}$
- $\eta=\ln\lambda$
- $T(y)=y$
- $a(\eta)=exp(\eta)$

### Useful properties

Exponential families have three particularly useful properties allowing us to
estimate $\eta$ using MLE since it is concave as well as to easily calculate
the expected value and variance:

- $\mathbb{E}[y;\eta]=\frac{\partial}{\partial\eta}a(\eta)$
- $\mathrm{Var}[y;\eta]=\frac{\partial}{\partial\eta}a(\eta)$

## GLMs

Linear regression works well if the target value has a linear relationship with
the features: assuming the price of a house is roughly proportional to its size
is a well-fitting assumption. This breaks down if the target doesn't linearly
respond to the features: we've already seen that using a linear regression
model to predict probabilities doesn't work well since the range of target
values can take on many values outside $[0,1]$. GLMs on the other hand make use
of a link function and appropriate distribution to relate the linear model to
the target value.

Let $(x^{(i)}, y^{(i)})$ be pairs of feature-target values and assume
$p(y^{(i)}|x^{(i)};\eta)$ to be distributed according to some exponential
family distribution, i.e.
$p(y^{(i)}|x^{(i)};\eta)=b(y)\exp(\eta^TT(y)-a(\eta))$. Define the output of
the linear model as $\eta=\theta^Tx$. We now relate this linear prediction to
the mean of the distribution through the *link function* $g$ that expresses
$\eta$ in terms of the mean $\mu$, i.e. $g(\mu)=\eta=\theta^Tx$. For a
Gaussian, we have seen that $\eta=\mu$ so the link function simply is
$g(\mu)=\mu$. In the case of a Bernoulli distribution, the link function
is given by $g(\phi)=\ln(\frac{\phi}{1-\phi})=\eta$. For a Poisson, we have
$g(\lambda)=\ln\lambda=\eta$.

Conceptually, the output of the model $h_\theta(x)$ is the expected value of
the distribution parameterized by the canonical parameters given by applying
the inverse link function to the linear model $\eta=\theta^T x$. Since $g^{-1}(\eta)=\mu$
and $\mathbb{E}[y;\eta]=\frac{\partial}{\partial\eta}a(\eta)$, this simply
becomes:

\begin{gather}
    h_\theta(x)=\mathbb{E}[y;\eta=\theta^Tx]=g^{-1}(\eta)=\frac{\partial}{\partial\eta}a(\eta)
\end{gather}

<img src="/res/machine-learning/linear_model.svg" width=100%/>

Let's take another look at linear and logistic regression as generalized linear
models:

### Linear regression

We have already seen how errors for linear regression are assumed to be
normally distributed with fixed variance. This means we have

- $\mu=\theta^Tx$
- $y\sim\mathcal{N}(\mu)$

Plotting this for the one-dimensional case:

<div>
<div
id="e3a4b4b0-56bd-49ed-a54b-a5b2caf59bd1"
class="plotly-graph-div"
style="height: 100%; width: 100%"
></div>
<script type="text/javascript">
window.PLOTLYENV = window.PLOTLYENV || {};
if (document.getElementById("e3a4b4b0-56bd-49ed-a54b-a5b2caf59bd1")) {
Plotly.newPlot(
"e3a4b4b0-56bd-49ed-a54b-a5b2caf59bd1",
[
{
mode: "markers",
showlegend: false,
x: [
0.0, 0.10101010101010101, 0.20202020202020202,
0.30303030303030304, 0.40404040404040403, 0.5050505050505051,
0.6060606060606061, 0.7070707070707071, 0.8080808080808081,
0.9090909090909091, 1.0101010101010102, 1.1111111111111112,
1.2121212121212122, 1.3131313131313131, 1.4141414141414141,
1.5151515151515151, 1.6161616161616161, 1.7171717171717171,
1.8181818181818181, 1.9191919191919191, 2.0202020202020203,
2.121212121212121, 2.2222222222222223, 2.323232323232323,
2.4242424242424243, 2.525252525252525, 2.6262626262626263,
2.727272727272727, 2.8282828282828283, 2.929292929292929,
3.0303030303030303, 3.131313131313131, 3.2323232323232323,
3.3333333333333335, 3.4343434343434343, 3.5353535353535355,
3.6363636363636362, 3.7373737373737375, 3.8383838383838382,
3.9393939393939394, 4.040404040404041, 4.141414141414141,
4.242424242424242, 4.343434343434343, 4.444444444444445,
4.545454545454545, 4.646464646464646, 4.747474747474747,
4.848484848484849, 4.94949494949495, 5.05050505050505,
5.151515151515151, 5.252525252525253, 5.353535353535354,
5.454545454545454, 5.555555555555555, 5.656565656565657,
5.757575757575758, 5.858585858585858, 5.959595959595959,
6.0606060606060606, 6.161616161616162, 6.262626262626262,
6.363636363636363, 6.4646464646464645, 6.565656565656566,
6.666666666666667, 6.767676767676767, 6.8686868686868685,
6.96969696969697, 7.070707070707071, 7.171717171717171,
7.2727272727272725, 7.373737373737374, 7.474747474747475,
7.575757575757575, 7.6767676767676765, 7.777777777777778,
7.878787878787879, 7.979797979797979, 8.080808080808081,
8.181818181818182, 8.282828282828282, 8.383838383838384,
8.484848484848484, 8.585858585858587, 8.686868686868687,
8.787878787878787, 8.88888888888889, 8.98989898989899,
9.09090909090909, 9.191919191919192, 9.292929292929292,
9.393939393939394, 9.494949494949495, 9.595959595959595,
9.696969696969697, 9.797979797979798, 9.8989898989899, 10.0,
],
y: [
0.017812305219323123, -0.23857770916316318,
0.6313057426552203, -2.124507080581117, -0.7132279943013242,
0.6492106496867909, 1.404876029293948, 0.06098175530953276,
-1.8980988583806297, 0.010948854612150782,
0.32613256829762927, 1.4099329678746728, 1.2721339794349606,
0.7979334301818479, 0.3228624381081924, 1.9925869121986441,
0.09557182387003804, 3.0005713944764416, 1.8109626326539507,
2.597828785613954, 3.757248849866407, 1.0309706220853434,
1.609857301772283, 3.8267274916323704, 5.034834041264794,
4.766670055848289, 0.9035081728457293, 1.9557830146161255,
0.477188406562882, 3.1326168801341, 2.6990775317394347,
3.504521429451342, 3.6714103245064655, 3.09391633946837,
3.145978811248216, 4.9161656727955885, 3.4042090155100126,
3.7692407107642376, 2.4287323684395004, 2.2147982404083653,
6.570272227117275, 3.094985502162718, 3.2776475121500512,
4.448336393927218, 4.970854842006433, 2.819944127441481,
4.196131683015444, 3.5431726029923514, 5.114689537235792,
6.414272200043856, 4.449826444891132, 5.114925919794842,
5.1519035557416935, 5.7209178056752625, 5.519655072087,
5.022192486670331, 4.816087612246493, 4.303572432639594,
6.223171544952829, 6.697851801786418, 7.294154336602904,
6.712013394046925, 6.909734734957113, 6.123103143530976,
6.37830216615678, 4.755937439245109, 5.596042477149844,
7.3452481932468405, 6.865814665133863, 8.308574172689248,
6.62686218580839, 5.953448020515222, 6.680153066325313,
5.395827728475392, 6.33781126824312, 7.3020709799870644,
8.663128316606729, 8.724751774436156, 8.181729789597354,
8.161898661712161, 7.453416248878355, 7.043508806191807,
8.80192195579432, 7.847594513911, 9.540859579911848,
7.990336787940751, 7.699282554595561, 9.168541308685967,
9.072359798701193, 9.414583583554682, 7.257088348931088,
6.46729855118298, 8.120010094184417, 9.400527333718031,
7.660240083682861, 9.682553029061681, 9.826099822726848,
9.072473937799467, 8.741638083108027, 9.18400357818281,
],
type: "scatter",
},
{
showlegend: false,
marker: { color: "red" },
mode: "lines",
x: [0, 10],
y: [0, 10],
type: "scatter",
},
{
showlegend: false,
base: 2,
marker: { color: "black" },
orientation: "h",
x: [
0.05399096651318806, 0.12951759566589174, 0.24197072451914337,
0.3520653267642995, 0.3989422804014327, 0.3520653267642995,
0.24197072451914337, 0.12951759566589174, 0.05399096651318806,
],
y: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
type: "bar",
},
{
showlegend: false,
marker: { color: "black" },
mode: "lines",
x: [
2.053990966513188, 2.1295175956658916, 2.2419707245191436,
2.3520653267642997, 2.3989422804014326, 2.3520653267642997,
2.2419707245191436, 2.1295175956658916, 2.053990966513188,
],
y: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
type: "scatter",
},
{
showlegend: false,
base: 4,
marker: { color: "black" },
orientation: "h",
x: [
0.05399096651318806, 0.12951759566589174, 0.24197072451914337,
0.3520653267642995, 0.3989422804014327, 0.3520653267642995,
0.24197072451914337, 0.12951759566589174, 0.05399096651318806,
],
y: [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
type: "bar",
},
{
showlegend: false,
marker: { color: "black" },
mode: "lines",
x: [
4.053990966513188, 4.129517595665892, 4.241970724519144,
4.3520653267643, 4.398942280401433, 4.3520653267643,
4.241970724519144, 4.129517595665892, 4.053990966513188,
],
y: [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
type: "scatter",
},
{
showlegend: false,
base: 6,
marker: { color: "black" },
orientation: "h",
x: [
0.05399096651318806, 0.12951759566589174, 0.24197072451914337,
0.3520653267642995, 0.3989422804014327, 0.3520653267642995,
0.24197072451914337, 0.12951759566589174, 0.05399096651318806,
],
y: [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
type: "bar",
},
{
showlegend: false,
marker: { color: "black" },
mode: "lines",
x: [
6.053990966513188, 6.129517595665892, 6.241970724519144,
6.3520653267643, 6.398942280401433, 6.3520653267643,
6.241970724519144, 6.129517595665892, 6.053990966513188,
],
y: [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
type: "scatter",
},
{
showlegend: false,
base: 8,
marker: { color: "black" },
orientation: "h",
x: [
0.05399096651318806, 0.12951759566589174, 0.24197072451914337,
0.3520653267642995, 0.3989422804014327, 0.3520653267642995,
0.24197072451914337, 0.12951759566589174, 0.05399096651318806,
],
y: [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
type: "bar",
},
{
showlegend: false,
marker: { color: "black" },
mode: "lines",
x: [
8.053990966513188, 8.129517595665892, 8.241970724519144,
8.352065326764299, 8.398942280401434, 8.352065326764299,
8.241970724519144, 8.129517595665892, 8.053990966513188,
],
y: [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
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
},
{ responsive: true }
);
}
</script>
</div>

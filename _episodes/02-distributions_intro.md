---
title: >-
    Introducing probability distributions
teaching: 30
exercises: 10
questions:
- "How are probability distributions defined and described?"
objectives:
- "Learn how the pdf, cdf, quantiles, ppf are defined and how to plot them using `scipy.stats` distribution functions and methods."
keypoints:
- "Probability distributions show how random variables are distributed. Two common distributions are the uniform and normal distributions."
- "Uniform and normal distributions and many associated functions can be accessed using `scipy.stats.uniform` and `scipy.stats.norm` respectively."
- "The probability density function (pdf) shows the distribution of relative likelihood or frequency of different values of a random variable and can be accessed with the scipy statistical distribution's `pdf` method."
- "The cumulative distribution function (cdf) is the integral of the pdf and shows the cumulative probability for a variable to be equal to or less than a given value. It can be accessed with the scipy statistical distribution's `cdf` method."
- "Quantiles such as percentiles and quartiles give the values of the random variable which correspond to fixed probability intervals (e.g. of 1 per cent and 25 per cent respectively). They can be calculated for a distribution in scipy using the `percentile` or `interval` methods."
- "The percent point function (ppf) (`ppf` method) is the inverse function of the cdf and shows the value of the random variable corresponding to a given quantile in its distribution."
- "Probability distributions are defined by common types of parameter such as the location and scale parameters. Some distributions also include shape parameters."
---

<script src="../code/math-code.js"></script>
<!-- Just one possible MathJax CDN below. You may use others. -->
<script async src="//mathjax.rstudio.com/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

In this episode we will be using numpy, as well as matplotlib's plotting library. Scipy contains an extensive range of distributions in its 'scipy.stats' module, so we will also need to import it. Remember: scipy modules should be installed separately as required - they cannot be called if only scipy is imported.
~~~
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
~~~
{: .language-python}

Michelson's speed-of-light data is a form of [__random variable__]({{ page.root }}/reference/#random-variable). Clearly the measurements are close to the true value of the speed of light in air, to within <0.1 per cent, but they are distributed randomly about some average which may not even be the true value (e.g. due to [systematic error]({{ page.root }}/reference/#systematic-error) in the measurements). 

We can gain further understanding by realising that random variables do not just take on any value - they are drawn from some [__probability distribution__]({{ page.root }}/reference/#probability-distribution). In probability theory, a random measurement (or even a set of measurements) is an [__event__]({{ page.root }}/reference/#event) which occurs (is 'drawn') with a fixed probability, assuming that the experiment is fixed and the underlying distribution being measured does not change over time (statistically we say that the random process is [__stationary__]({{ page.root }}/reference/#stationary-process)).

## The cdf and pdf of a probability distribution

Consider a [__continuous__]({{ page.root }}/reference/#continuous) random variable $$X$$ (for example, a single measurement). For a fixed probability distribution over all possible values $x$, we can define the probability $$P$$ that $$X\leq x$$ as being the [__cumulative distribution function__]({{ page.root }}/reference/#cdf) (or __cdf__), $$F(x)$$:

$$F(x) = P(X\leq x)$$

We can choose the limiting values of our distribution, but since the variable must take on some value (i.e. the definition of an 'event' is that _something_ must happen) it _must_ satisfy: 

$$\lim\limits_{x\rightarrow -\infty} F(x) = 0$$ and $$\lim\limits_{x\rightarrow +\infty} F(x) = 1$$

From these definitions we find that the probability that $$X$$ lies in the closed interval $$[a,b]$$ (note: a _closed_ interval, denoted by square brackets, means that we include the endpoints $$a$$ and $$b$$) is:

$$P(a \leq X \leq b) = F(b) - F(a)$$

We can then take the limit of a very small interval $$[x,x+\delta x]$$ to define the [__probability density function__]({{ page.root }}/reference/#pdf) (or __pdf__), $$p(x)$$:

$$\frac{P(x\leq X \leq x+\delta x)}{\delta x} = \frac{F(x+\delta x)-F(x)}{\delta x}$$

$$p(x) = \lim\limits_{\delta x \rightarrow 0} \frac{P(x\leq X \leq x+\delta x)}{\delta x} = \frac{\mathrm{d}F(x)}{\mathrm{d}x}$$

This means that the cdf is the integral of the pdf, e.g.:

$$P(X \leq x) = F(x) = \int^{x}_{-\infty} p(x^{\prime})\mathrm{d}x^{\prime}$$

where $$x^{\prime}$$ is a dummy variable.  The probability that $$X$$ lies in the interval $$[a,b]$$ is:

$$P(a \leq X \leq b) = F(b) - F(a) = \int_{a}^{b} p(x)\mathrm{d}x$$

and $$\int_{-\infty}^{\infty} p(x)\mathrm{d}x = 1$$.

<p align='center'>
<img alt="pdf vs cdf comparison" src="../fig/ep2_pdfcdfcomparison.png" width="500"/>
</p>

> ## Why use the pdf?
> By definition, the cdf can be used to directly calculate probabilities (which is very useful in statistical assessments of data), while the pdf only gives us the probability density for a specific value of $$X$$. So why use the pdf? One of the main reasons is that it is generally much easier to calculate the pdf for a particular probability distribution, than it is to calculate the cdf, which requires integration (which may be analytically impossible in some cases!). 
>
> Also, the pdf gives the relative probabilities (or [__likelihoods__]({{ page.root }}/reference/#likelihood)) for particular values of $$X$$ and the model parameters, allowing us to compare the relative likelihood of [__hypotheses__]({{ page.root }}/reference/#hypothesis) where the model [parameters]({{ page.root }}/reference/#parameter) are different. This principle is a cornerstone of statistical inference which we will come to later on.
{: .callout}

## Probability distributions: Uniform

Now we'll introduce two common probability distributions, and see how to use them in your Python data analysis. We start with the [__uniform distribution__]({{ page.root }}/reference/#distributions---uniform), which has equal probability values defined over some finite interval $$[a,b]$$ (and zero elsewhere). The pdf is given by:

$$p(x\vert a,b) = 1/(b-a) \quad \mathrm{for} \quad a \leq x \leq b$$

where the notation $$p(x\vert a,b)$$ means _'probability density at x, conditional on model parameters $$a$$ and $$b$$'_. The vertical line $$\vert$$ meaning _'conditional on'_ (i.e. 'given these existing conditions') is notation from probability theory which we will use often in this course.

~~~
## define parameters for our uniform distribution
a = 1
b = 4
print("Uniform distribution with limits",a,"and",b,":")
## freeze the distribution for a given a and b
ud = sps.uniform(loc=a, scale=b-a) # The 2nd parameter is added to a to obtain the upper limit = b
~~~
{: .language-python}

> ## Distribution parameters: location, scale and shape
> As in the above example, it is often useful to _'freeze'_ a distribution by fixing its parameters and defining the frozen distribution as a new function, which saves repeating the parameters each time. The common format for [arguments]({{ page.root }}/reference/#argument) of scipy statistical distributions which represent distribution parameters, corresponds to statistical terminology for the parameters:
> 
> - A [__location parameter__]({{ page.root }}/reference/#parameter) (the `loc` argument in the scipy function) determines the location of the distribution on the $$x$$-axis. Changing the location parameter just shifts the distribution along the $$x$$-axis.
> - A [__scale parameter__]({{ page.root }}/reference/#parameter) (the `scale` argument in the scipy function) determines the width or (more formally) the statistical dispersion of the distribution. Changing the scale parameter just stretches or shrinks the distribution along the $$x$$-axis but does not otherwise alter its shape.
> - There _may_ be one or more [__shape parameters__]({{ page.root }}/reference/#parameter) (scipy function arguments may have different names specific to the distribution). These are parameters which do something other than shifting, or stretching/shrinking the distribution, i.e. they change the shape in some way.
>
> Distributions may have all or just one of these parameters, depending on their form. For example, normal distributions are completely described by their location (the mean) and scale (the standard deviation), while exponential distributions (and the related discrete _Poisson_ distribution) may be defined by a single parameter which sets their location as well as width. Some distributions use a [__rate parameter__]({{ page.root }}/reference/#parameter) which is the reciprocal of the scale parameter (exponential/Poisson distributions are an example of this). 
>
{: .callout}

The uniform distribution has a scale parameter $$\lvert b-a \rvert$$. This statistical distribution's location parameter is formally the centre of the distribution, $$(a+b)/2$$, but for convenience the scipy `uniform` function uses $$a$$ to place a bound on one side of the distribution. We can obtain and plot the pdf and cdf by applying those named [methods]({{ page.root }}/reference/#method) to the scipy function.  Note that we must also use a suitable function (e.g. `numpy.arange`) to create a sufficiently dense range of $$x$$-values to make the plots over. 

~~~
## You can plot the probability density function
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,4))
# change the separation between the sub-plots:
fig.subplots_adjust(wspace=0.3)
x = np.arange(0., 5.0, 0.01)
ax1.plot(x, ud.pdf(x), lw=2)
## or you can plot the cumulative distribution function:
ax2.plot(x, ud.cdf(x), lw=2)
for ax in (ax1,ax2):
    ax.tick_params(labelsize=12)
    ax.set_xlabel("x", fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
ax1.set_ylabel("probability density", fontsize=12)
ax2.set_ylabel("probability", fontsize=12)
plt.show()
~~~
{: .language-python}

<p align='center'>
<img alt="Uniform pdf vs cdf" src="../fig/ep2_uniformpdfcdf.png" width="600"/>
</p>

## Probability distributions: Normal

The [__normal distribution__]({{ page.root }}/reference/#distributions---normal) is one of the most important in statistical data analysis (for reasons which will become clear) and is also known to physicists and engineers as the _Gaussian distribution_. The distribution is defined by location parameter $$\mu$$ (often just called the __mean__, but not to be confused with the mean of a statistical sample) and scale parameter $$\sigma$$ (also called the __standard deviation__, but again not to be confused with the sample standard deviation). The pdf is given by:

$$p(x\vert \mu,\sigma)=\frac{1}{\sigma \sqrt{2\pi}} e^{-(x-\mu)^{2}/(2\sigma^{2})}$$

It is also common to refer to the __standard normal distribution__ which is the normal distribution with $$\mu=0$$ and $$\sigma=1$$:

$$p(z\vert 0,1) = \frac{1}{\sqrt{2\pi}} e^{-z^{2}/2}$$

The standard normal is important for many statistical results, including the approach of defining statistical significance in terms of the number of _'sigmas'_ which refers to the probability contained within a range $$\pm z$$ on the standard normal distribution (we will discuss this in more detail when we discuss statistical significance testing).

> ## Challenge: plotting the normal distribution
> Now that you have seen the example of a uniform distribution, use the appropriate `scipy.stats` function to plot the pdf and cdf of the normal distribution, for a mean and standard deviation of your choice (you can freeze the distribution first if you wish, but it is not essential).
>
>> ## Solution
>> ~~~
>> ## Define mu and sigma:
>> mu = 2.0
>> sigma = 0.7
>> ## Plot the probability density function
>> fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,4))
>> fig.subplots_adjust(wspace=0.3)
>> ## we will plot +/- 3 sigma on either side of the mean
>> x = np.arange(-1.0, 5.0, 0.01)
>> ax1.plot(x, sps.norm.pdf(x,loc=mu,scale=sigma), lw=2)
>> ## and the cumulative distribution function:
>> ax2.plot(x, sps.norm.cdf(x,loc=mu,scale=sigma), lw=2)
>> for ax in (ax1,ax2):
>>     ax.tick_params(labelsize=12)
>>     ax.set_xlabel("x", fontsize=12)
>>     ax.tick_params(axis='x', labelsize=12)
>>     ax.tick_params(axis='y', labelsize=12)
>> ax1.set_ylabel("probability density", fontsize=12)
>> ax2.set_ylabel("probability", fontsize=12)
>> plt.show()
>> ~~~
>> {: .language-python}
>>
>> <p align='center'>
>> <img alt="Uniform pdf vs cdf" src="../fig/ep2_normalpdfcdf.png" width="600"/>
>> </p>
> {: .solution}
{: .challenge}

It's useful to note that the pdf is much more distinctive for different functions than the cdf, which (because of how it is defined) always takes on a similar, slanted 'S'-shape, hence there is some similarity in the form of cdf between the normal and uniform distributions, although their pdfs look radically different.

## Quantiles

It is often useful to be able to calculate the [__quantiles__]({{ page.root }}/reference/#quantile) (such as [__percentiles__]({{ page.root }}/reference/#percentile) or [__quartiles__]({{ page.root }}/reference/#quantile)) of a distribution, that is, what value of $$x$$ corresponds to a fixed interval of integrated probability? We can obtain these from the __inverse function__ of the cdf ($$F(x)$$). E.g. for the quantile $$\alpha$$:

$$F(x_{\alpha}) = \int^{x_{\alpha}}_{-\infty} p(x)\mathrm{d}x = \alpha \Longleftrightarrow x_{\alpha} = F^{-1}(\alpha)$$

Note that $$F^{-1}$$ denotes the inverse function of $$F$$, not $$1/F$$! This is called the [__percent point function__]({{ page.root }}/reference/#ppf)  (or __ppf__). To obtain a given quantile for a distribution we can use the `scipy.stats` method `ppf` applied to the distribution function. For example:

~~~
## Print the 30th percentile of a normal distribution with mu = 3.5 and sigma=0.3
print("30th percentile:",sps.norm.ppf(0.3,loc=3.5,scale=0.3))
## Print the median (50th percentile) of the distribution
print("Median (via ppf):",sps.norm.ppf(0.5,loc=3.5,scale=0.3))
## There is also a median method to quickly return the median for a distribution:
print("Median (via median method):",sps.norm.median(loc=3.5,scale=0.3))
~~~
{: .language-python}
~~~
30th percentile: 3.342679846187588
Median (via ppf): 3.5
Median (via median method): 3.5
~~~
{: .output}

> ## Intervals
> It is sometimes useful to be able to quote an interval, containing some fraction of the probability (and usually centred on the median) as a 'typical' range expected for the random variable $$X$$. We will discuss intervals on probability distributions further when we discuss [confidence intervals]({{ page.root }}/reference/#confidence-interval) on parameters. For now, we note that the `.interval` method can be used to obtain a given interval centred on the median. 
> For example, the [__Interquartile Range__]({{ page.root }}/reference/#interquartile-range) (__IQR__) is often quoted as it marks the interval containing half the probability, between the upper and lower [__quartiles__]({{ page.root }}/reference/#quantile) (i.e. from 0.25 to 0.75):
> ~~~
> ## Print the IQR for a normal distribution with mu = 3.5 and sigma=0.3
> print("IQR:",sps.norm.interval(0.5,loc=3.5,scale=0.3))
> ~~~
> {: .language-python}
> ~~~
> IQR: (3.2976530749411754, 3.7023469250588246)
> ~~~
> {: .output}
> So for the normal distribution, with $$\mu=3.5$$ and $$\sigma=0.3$$, half of the probability is contained in the range $$3.5\pm0.202$$ (to 3 decimal places). 
{: .callout}


---
title: >-
    Maximum likelihood estimation
teaching: 20
exercises: 10
questions:
- "What is the best way to estimate parameters of hypotheses?"
objectives:
- ""
keypoints:
- ""
---

<script src="../code/math-code.js"></script>
<!-- Just one possible MathJax CDN below. You may use others. -->
<script async src="//mathjax.rstudio.com/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

In this episode we will be using numpy, as well as matplotlib's plotting library. Scipy contains an extensive range of distributions in its 'scipy.stats' module, so we will also need to import it and we will also make use of scipy's `scipy.optimize` module. Remember: scipy modules should be installed separately as required - they cannot be called if only scipy is imported.
~~~
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import scipy.optimize as spopt
~~~
{: .language-python}

## Maximum likelihood estimation

Consider a hypothesis which is specified by a single parameter $$\theta$$. We would like to know the probability distribution for that parameter, given the data we have in hand $$D$$. To do so, we can use Bayes' theorem:

$$P(\theta\vert D) = \frac{P(D\vert \theta)P(\theta)}{P(D)},$$

to obtain the [__posterior__]({{ page.root }}/reference/#posterior) probability distribution for $$\theta$$, in terms of the [__likelihood__]({{ page.root }}/reference/#likelihood) $$P(D\vert \theta)$$, which gives the probability distribution of the parameter to produce the observed data $$D$$ and the [__prior__]({{ page.root }}/reference/#prior) $$P(\theta)$$, which shows what values of $$\theta$$ we consider reasonable or unlikely, based on the knowledge we have before collecting the data.  The normalising [__evidence__]({{ page.root }}/reference/#evidence) can be obtained by marginalising over $$\theta$$, i.e. integrating the product of likelihood and the prior over $$\theta$$.

What would be a good estimate for $$\theta$$? We could consider obtaining the mean of $$P(\theta\vert D)$$, but this may often be skewed if the distribution has asymmetric tails, and is also difficult to calculate in many cases. A better estimate would involve finding the value of $$\theta$$ which maximises the posterior probability. I.e. we must find the peak of the probability distribution, $$P=P(\theta\vert D)$$. Therefore we require:

$$\frac{\mathrm{d}P}{\mathrm{d}\theta}\bigg\rvert_{\hat{\theta}} = 0 \quad \mbox{and} \quad \frac{\mathrm{d}^{2}P}{\mathrm{d}\theta^{2}}\bigg\rvert_{\hat{\theta}} < 0$$

where $$\hat{\theta}$$ is the value of the parameter corresponding to the maximum probability.

However, remember that $$P(D)$$ does not explicitly depend on $$\theta$$ so therefore the maximum corresponds only to $$\mathrm{max}[P(D\vert \theta)P(\theta)]$$. This also a form of _likelihood_, since it is a probability distribution in terms of the distribution parameter $$\theta$$, so we refer to this quantity as the [__maximum likelihood__]({{ page.root }}/reference/#maximum-likelihood-estimation). The parameter value $$\hat{\theta}$$ corresponding to the maximum likelihood is the best [__estimator__]({{ page.root }}/reference/#estimator) for $$\theta$$ and is known as the [__maximum likelihood estimate__] of $$\theta$$ or [__MLE__]({{ page.root }}/reference/#mle). The process of maximising the likelihood to obtain MLEs is known as [__maximum likelihood estimation__]({{ page.root }}/reference/#maximum-likelihood-estimation).

## Log-likelihood and MLEs

Many posterior probability distributions $$P(\theta)$$ are quite 'peaky' and it is often easier to work with the smoother transformation $$L(\theta)=\ln[P(\theta)]$$. This is a monotonic function of $$P(\theta)$$ so it must also satisfy the relations for a maximum to occur for the same MLE value, i.e:

$$\frac{\mathrm{d}L}{\mathrm{d}\theta}\bigg\rvert_{\hat{\theta}} = 0 \quad \mbox{and} \quad \frac{\mathrm{d}^{2}L}{\mathrm{d}\theta^{2}}\bigg\rvert_{\hat{\theta}} < 0$$

Furthermore, the log probability also has the advantages that products become sums, and powers become multiplying constants. We will use this property to calculate the MLEs for some well-known distributions (i.e. we assume a uniform prior so we only consider the likelihood function of the distribution) and demonstrate that they are the best estimators of the parameter:

Firstly, consider a [_binomial distribution_]({{ page.root }}/reference/#distributions---binomial):

$$P(\theta\vert D) \propto \theta^{x} (1-\theta)^{n-x}$$

where $$x$$ is now the _observed_ number of successes in $$n$$ trials and $\theta$ is a parameter which is the variable of the function. We can neglect the binomial constant since we will take the logarithm and then differentiate, to obtain the maximum log-likelihood:

$$L(\theta) = x\ln(\theta) + (n-x)\ln(1-\theta) + \mathrm{constant}$$ 

$$\frac{\mathrm{d}L}{\mathrm{d}\theta}\bigg\rvert_{\hat{\theta}} = \frac{x}{\hat{\theta}} - \frac{n-x}{(1-\hat{\theta})} = 0 \quad \rightarrow \quad  \hat{\theta} = \frac{x}{n}$$

Further differentiation will show that the second derivative is negative, i.e. this is indeed the MLE. If we consider repeating our experiment many times, the expectation of our data $$E[x]$$ is equal to that of variates drawn from a binomial distribution with $$\theta$$ fixed at the true value, i.e. $$E[x]=E[X]$$. We therefore obtain $$E[X] = nE[\hat{\theta}]$$ and comparison with the expectation value for binomially distributed variates confirms that $$\hat{\theta}$$ is an unbiased estimator of the true value of $$\theta$$.

> ## Challenge: MLE for a Poisson distribution
> Determine the MLE of the rate parameter $$\lambda$$ for a Poisson distribution and show that it is an unbiased estimator of the true rate parameter.
>
>> ## Solution
>> For fixed observed counts $$x$$ and a uniform prior on $$\lambda$$, the Poisson distribution $$P(\lambda \vert x) \propto \lambda^{x} e^{-\lambda}$$. Therefore the log-likelihood is:
>> $$L(\lambda) = x\ln(\lambda) -\lambda$$
>> $$\quad \rightarrow \quad \frac{\mathrm{d}L}{\mathrm{d}\lambda}\bigg\rvert_{\hat{\lambda}} = \frac{x}{\hat{\lambda}} - 1 = 0 \quad \rightarrow \quad \hat{\lambda} = x$$
>>
>> $$\frac{\mathrm{d}^{2}L}{\mathrm{d}\lambda^{2}}\bigg\rvert_{\hat{\lambda}} = - \frac{x}{\hat{\lambda}^{2}}$$, i.e. negative, so we are considering the MLE. 
>>
>> Therefore, the observed rate $$x$$ is the MLE for $$\lambda$$.
>>
>> For the Poisson distribution, $$E[X]=\lambda$$, therefore since $$E[x]=E[X] = E[\hat{\lambda}]$$, the MLE is an unbiased estimator of the true $$\lambda$$. You might wonder why we get this result when in the challenge in the previous episode we showed that the mean of the prior probability distribution for the Poisson rate parameter and observed rate $$x=4$$ was 5! 
>>
>> The mean of the posterior distribution $$\langle \lambda \rangle$$ is larger than the MLE (which is equivalent to the [_mode_]({{ page.root }}/reference/#mode) of the distribution, because the distribution is positively skewed (i.e. skewed to the right). However, over many repeated experiments with the same rate parameter $$\lambda_{\mathrm{true}}$$, $$E[\langle \lambda \rangle]=\lambda_{\mathrm{true}}+1$$, while $$E[\hat{\lambda}]=\lambda_{\mathrm{true}}$$.  I.e. the mean of the posterior distribution is a biased estimator in this case, while the MLE is not.
> {: .solution}
{: .challenge}


## Errors on MLEs

It's important to remember that the MLE $$\hat{\theta}$$ is only an _estimator_ for the  true parameter value $$\theta_{\mathrm{true}}$$, which is contained somewhere in the posterior probability distribution for $$\theta$$, with probability of it occuring in a certain range, given by integrating the distribution over that range, as is the case for the pdf of a random variable. We will look at this approach of using the distribution to define [__confidence intervals__]({{ page.root }}/reference/#confidence-interval) in a later episode, but for now we will examine a simpler approach to estimating the error on an MLE.

Consider a log-likelihood $$L(\theta)$$ with maximum at the MLE, at $$L(\hat{\theta})$$. We can examine the shape of the probability distribution of $$\theta$$ around $$\hat{\theta}$$ by expanding $$L(\theta)$$ about the maximum:

$$L(\theta) = L(\hat{\theta}) + \frac{1}{2} \frac{\mathrm{d}^{2}L}{\mathrm{d}\theta^{2}}\bigg\rvert_{\hat{\theta}}(\theta-\hat{\theta})^{2} + \cdots$$

where the 1st order term is zero because $$\frac{\mathrm{d}L}{\mathrm{d}\theta}\bigg\rvert_{\hat{\theta}} = 0$$ at $$\theta=\hat{\theta}$$, by definition.

For smooth log-likelihoods, where we can neglect the higher order terms, the distribution around the MLE can be approximated by a parabola with width dependent on the 2nd derivative of the log-likelihood. To see what this means, lets transform back to the probability, $$P(\theta)=\exp\left(L(\theta)\right)$$:

$$L(\theta) = L(\hat{\theta}) + \frac{1}{2} \frac{\mathrm{d}^{2}L}{\mathrm{d}\theta^{2}}\bigg\rvert_{\hat{\theta}}(\theta-\hat{\theta})^{2} \quad \Rightarrow \quad P(\theta) = P(\hat{\theta})\exp\left[\frac{1}{2} \frac{\mathrm{d}^{2}L}{\mathrm{d}\theta^{2}}\bigg\rvert_{\hat{\theta}}(\theta-\hat{\theta})^{2}\right]$$

The equation on the right hand side should be familiar to us: it is the [_normal distribution_]({{ page.root }}/reference/#distributions---binomial)!

$$p(x\vert \mu,\sigma)=\frac{1}{\sigma \sqrt{2\pi}} e^{-(x-\mu)^{2}/(2\sigma^{2})}$$

i.e. for smooth log-likelihood functions, the posterior probability distribution of the parameter $$\theta$$ can be approximated with a normal distribution about the MLE $$\hat{\theta}$$, i.e. with mean $$\mu=\hat{\theta}$$ and variance $$\sigma^{2}=-\left(\frac{\mathrm{d}^{2}L}{\mathrm{d}\theta^{2}}\bigg\rvert_{\hat{\theta}}\right)^{-1}$$. Thus, assuming this _Gaussian_ or _normal approximation_, we can estimate a 1-$$\sigma$$ uncertainty or [__error__]({{ page.root }}/reference/#error) on $$\theta$$ which corresponds to a range about the MLE value where the true value should be $$\simeq$$68.2% of the time:

$$\sigma = \left(-\frac{\mathrm{d}^{2}L}{\mathrm{d}\theta^{2}}\bigg\rvert_{\hat{\theta}}\right)^{-1/2}$$

How accurate this estimate for $$\sigma$$ is will depend on how closely the posterior distribution approximates a normal distribution, at least in the region of parameter values that contains most of the probability. The estimate will become exact in the case where the posterior is normally distributed.

> ## Challenge
> Use the normal approximation to estimate the variance on the MLE for binomial and Poisson distributed likelihood functions, in terms of the observed data ($$x$$ successes in $$n$$ trials, or $$x$$ counts). 
>
>> ## Solution
>> For the binomial distribution we have already shown that:
>>   $$\frac{\mathrm{d}L}{\mathrm{d}\theta} = \frac{x}{\theta} - \frac{n-x}{(1-\theta)} \quad \rightarrow \quad  \frac{\mathrm{d}^{2}L}{\mathrm{d}\theta^{2}}\bigg\rvert_{\hat{\theta}} = -\frac{x}{\hat{\theta}^{2}} - \frac{n-x}{(1-\hat{\theta})^{2}} = -\frac{n}{\hat{\theta}(1-\hat{\theta})}$$
>>
>> So we obtain: $$\sigma = \sqrt{\frac{\hat{\theta}(1-\hat{\theta})}{n}}$$ and since $$\hat{\theta}=x/n$$ our final result is:
>>    $$\sigma = \sqrt{\frac{x(1-x/n)}{n^2}}$$
>> 
>> For the Poisson distributed likelihood we already showed in a previous challenge that $$\frac{\mathrm{d}^{2}L}{\mathrm{d}\lambda^{2}}\bigg\rvert_{\hat{\lambda}} = - \frac{x}{\hat{\lambda}^{2}}$$ and $$\hat{\lambda}=x$$
>>
>> So, $$\sigma = \sqrt{x}$$.
> {: .solution}
{: .challenge}


## Using optimisers to obtain MLEs

For the distributions discussed so far, the MLEs could be obtained analytically from the derivatives of the likelihood function. However, in most practical examples, the data are complex (i.e. multiple measurements) and the model distribution may include multiple parameters, making an analytical solution impossible. In this cases we need to obtain the MLEs numerically, via a numerical approach called _optimisation_.

Optimisation methods use algorithmic approaches to obtain either the minimum or maximum of a function of one or more adjustable parameters. These approaches are implemented in software using _optimisers_. For the case of obtaining MLEs, the function to be optimised is the likelihood function or some variant of it such as the log-likelihood or, as we will see later [_weighted least squares_]({{ page.root }}/reference/#weighted-least-squares), colloquially known as [_chi-squared fitting_]({{ page.root }}/reference/#chi-squared-fitting). 


> ## Optimisation methods and the `scipy.optimize` module.
> There are a variety of optimisation methods which are available in Python's `scipy.optimize` module. Many of these approaches are discussed in some detail in Chapter 10 of the book 'Numerical Recipes', available online [here][numrec_online]. Here we will give a brief summary of some of the main methods and their pros and cons for maximum likelihood estimation. An important aspect of most numerical optimisation methods, including the ones in scipy, is that they are _minimisers_, i.e. they operate to minimise the given function rather than maximise it. This works equally well for maximising the likelihood function, since we can simply multiply the function by -1 and minimise it to achieve our maximisation result.
>
> - __Scalar minimisation__: the function `scipy.optimize.minimize_scalar` has several methods for minimising functions of only one variable. The methods can be specified as arguments, e.g. `method='brent'` uses __Brent's method__ of parabolic interpolation: find a parabola between three points on the function, find the position of its minimum and use the minimum to replace the highest point on the original parabola before evaluating again, repeating until the minimum is found to the required tolerance. The method is fast and robust but can only be used for functions of 1 parameter and as no gradients are used, it does not return the useful 2nd derivative of the function.
> - __Downhill simplex (Nelder-Mead)__: `scipy.optimize.minimize` offers the method `nelder-mead` for rapid and robust minimisation of multi-parameter functions using the 'downhill simplex' approach. The approach assumes a simplex, an object of $$n+1$$ points or vertices in the $$n$$-dimensional parameter space. The function to be minimised is evaluated at all the vertices. Then, depending on where the lowest-valued vertex is and how steep the surrounding 'landscape' mapped by the other vertices is, a set of rules are applied to move one or more points of the simplex to a new location. E.g. via reflection, expansion or contraction of the simplex or some combination of these. In this way the simplex 'explores' the $$n$$-dimensional landscape of function values to find the minimum. Also known as the 'amoeba' method because the simplex 'oozes' through the landscape like an amoeba!
> - __Gradient methods__: a large set of methods calculate the gradients or even second derivatives of the function (hyper)surface in order to quickly converge on the minimum. A commonly used example is the __Broyden-Fletcher-Goldfarb-Shanno__ (__BFGS__) method (`method=BFGS` in `scipy.optimize.minimize` or the legacy function `scipy.optimize.fmin_bfgs`). A more specialised function using a variant of this approach which is optimised for fitting functions to data with normally distributed errors ('_weighted non-linear least squares_') is `scipy.optimize.curve_fit`. The advantage of these functions is that they usually return either a matrix of second derivatives (the 'Hessian') or its inverse, which is the _covariance matrix_ of the fitted parameters. These can be used to obtain estimates of the errors on the MLEs, following the normal approximation approach described in this episode.
>
> An important caveat to bear in mind with all optimisation methods is that for finding minima in complicated hypersurfaces, there is always a risk that the optimiser returns only a local minimum, end hence incorrect MLEs, instead of those at the true minimum for the function. Most optimisers have built-in methods to try and mitigate this problem, e.g. allowing sudden switches to completely different parts of the surface to check that no deeper minimum can be found there. It may be that a hypersurface is too complicated for any of the optimisers available. In this case, you should consider looking at _Markov Chain Monte Carlo_ methods to fit your data.
{: .callout}


[numrec_online]: http://numerical.recipes/book/book.html

{% include links.md %}



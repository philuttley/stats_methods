---
title: >-
    Correlation tests and least-squares fitting
teaching: 40
exercises: 10
questions:
- "How do we determine if two measured variables are significantly correlated?"
- "How do we carry out simple linear fits to data?"
objectives:
- "Learn the basis for and application of correlation tests, including what their assumptions are and how significances are determined."
- "Discover several python approaches to simple linear regression."
- "Learn how to use bootstrapping to estimate errors on model parameters or other quantities measured from data."
keypoints:
- "The sample covariance between two variables is an unbiased estimator for population covariance and shows the part of variance that is produced by linearly related variations in both variables."
- "Normalising the sample covariance by the sample standard deviations in both bands yields Pearson's correlation coefficient, _r_."
- "Spearman's rho correlation coefficient is based on the correlation in the ranking of variables, not their absolute values, so is more robust to outliers than Pearson's coefficient."
- "By assuming that the data are independent (and thus uncorrelated) and identically distributed, significance tests can be carried out on the hypothesis of no correlation, provided the sample is large ($$n>500$$) and/or is normally distributed."
- "By minimising the squared differences between the data and a linear model, linear regression can be used to obtain the model parameters."
- "Bootstrapping uses resampling (with replacement) of the data to estimate the standard deviation of any model parameters or other quantities calculated from the data."
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


## Sample covariance and correlation coefficient

When we are studying multivariate data we can determine sample [statistics]({{ page.root }}/reference/#statistic) for each variable separately, but these do not tell us how the variables are related. For this purpose, we can calculate the __sample covariance__ for two measured variables $$x$$ and $$y$$.:

$$s_{xy}=\frac{1}{n-1}\sum\limits_{i=1}^{n} (x_{i}-\bar{x})(y_{i}-\bar{y})$$

where $$n$$ is the number of pairs of measured data points $$x_{i}$$, $$y_{i}$$. The covariance is closely related to the variance (note the same Bessel's correction), and when covariance is discussed you will sometimes see sample variance denoted as $$s_{xx}$$. In fact, the covariance tells us the variance of the part of the variations which is linearly correlated between the two variables (see the [__Cauchy-Schwarz inequality__]({{ page.root }}/reference/#cauchy-schwarz-inequality) discussed earlier for multivariate distributions).

The sample covariance is an unbiased estimator of the population covariance of the distribution which the measurements are sampled from. Therefore, if the two variables are [__independent__]({{ page.root }}/reference/#independence), the expectation of the sample covariance is zero and the variables are also said to be __uncorrelated__.  Positive and negative covariances, if they are statistically significant, correspond to __correlated__ and __anticorrelated__ data respectively. However, the strength of a correlation is hard to determine from covariance alone, since the amplitude of the covariance depends on the sample variance of the two variables, as well as the degree of linear correlation between them.

Therefore, we can also normalise by the sample standard deviations of each variable to obtain the [__correlation coefficient__]({{ page.root }}/reference/#correlation-coefficient), $$r$$, also known as _Pearson's_ $$r$$, after its developer:

$$r = \frac{s_{xy}}{s_{x}s_{y}} = \frac{1}{n-1} \sum\limits_{i=1}^{n} \frac{(x_{i}-\bar{x})(y_{i}-\bar{y})}{s_{x}s_{y}}$$

The correlation coefficient gives us a way to compare the correlations for variables which have very different magnitudes. It is also an example of a [__test statistic__]({{ page.root }}/reference/#test-statistic), which can be used to test the hypothesis that the variables are uncorrelated, under certain assumptions.


## Correlation tests: Pearson's r and Spearman's rho

Besides Pearson's $$r$$, another commonly used correlation coefficient and test statistic for correlation tests is Spearman's $$\rho$$, which is determined using the following algorithm: 

1. Rank the data values $$x_{i}$$, $$y_{i}$$ separately in numerical order. Equal values in the sequence are assigned a rank equal to their average position, e.g. the 4th and 5th highest positions of the $$x_{i}$$ have equal values and are given a rank 4.5 each. Note that the values are not reordered in $$i$$ by this step, only ranks are assigned based on their numerical ordering.
2. For each pair of $$x_{i}$$ and $$y_{i}$$ a difference $$d_{i}=\mathrm{rank}(x_{i})-\mathrm{rank}(y_{i})$$ is calculated.
3. Spearman's $$\rho$$ is calculated from the resulting rank differences:

    $$\rho = 1-\frac{6\sum^{n}_{i=1} d_{i}^{2}}{n(n^{2}-1)}$$

To assess the statistical significance of a correlation, $$r$$ can be transformed to a new statistic $$t$$:

$$t = r\sqrt{\frac{n-2}{1-r^{2}}}$$

or $$\rho$$ can be transformed in a similar way:

$$t = \rho\sqrt{\frac{n-2}{1-\rho^{2}}}$$

Under the assumption that the data are __i.i.d.__, meaning _independent_ (i.e. no correlation) and _identically distributed_ (i.e. for a given variable, all the data are drawn from the same distribution, although note that both variables do not have to follow this distribution), then provided the data set is large (approximately $$n> 500$$), $$t$$ is distributed following a [$$t$$-distribution]({{ page.root }}/reference/#distributions---t) with $$n-2$$ degrees of freedom. This result follows from the central limit theorem, since the correlation coefficients are calculated from sums of random variates. As one might expect, the same distribution also applies for small ($$n<500$$) samples, if the data are themselves normally distributed, as well as being _i.i.d._.

The concept of being identically distributed means that each data point in one variable is drawn from the same population. This requires that, for example, if there is a bias in the sampling it is the same for all data points, i.e. the data are not made up of multiple samples with different biases.

Measurement of either correlation coefficient enables a comparison with the $$t$$-distribution and a $$p$$-value for the correlation coefficient to be calculated. When used in this way, the correlation coefficient can be used as significance test for whether the data are consistent with following the assumptions (and therefore being uncorrelated) or not. Note that the significance depends on both the measured coefficient and the sample size, so for small samples even large $$r$$ or $$\rho$$ may not be significant, while for very large samples, even $$r$$ or $$\rho$$ which are close to zero could still indicate a significant correlation.

A very low $$p$$-value will imply either that there is a real correlation, or that the other assumptions underlying the test are not valid. The validity of these other assumptions, such as _i.i.d._, and normally distributed data for small sample sizes, can generally be assessed visually from the data distribution. However, sometimes data sets can be so large that even small deviations from these assumptions can produce spuriously significant correlations. In these cases, when the data set is very large, the correlation is (sometimes highly) significant, but $$r$$ or $$\rho$$ are themselves close to zero, great care must be taken to assess whether the assumptions underlying the test are valid or not.

> ## Pearson or Spearman?
> When deciding which correlation coefficient to use, Pearson's $$r$$ is designed to search for linear correlations in the data themselves, while Spearman's $$\rho$$ is suited to monotonically related data, even if the data are not linearly correlated. Spearman's $$\rho$$ is also better able to deal with large outliers in the tails of the data samples, since the contribution of these values to the correlation is limited by their ranks (i.e. irrespective of any large values the outlying data points may have).
{: .callout}

## Correlation tests with scipy

We can compute Pearson's correlation coefficient $$r$$ and Spearman's correlation coefficient, $$\rho$$, for bivariate data using the functions in `scipy.stats`. For both outputs, the first value is the correlation coefficient, the second the p-value. To start with, we will test this approach using randomly generated data (which we also plot on a scatter-plot).

First, we generate a set of $$x$$-values using normally distributed data. Next, we generate corresponding $$y$$-values by taking the $$x$$-values and adding another set of random normal variates of the same size (number of values). You can apply a scaling factor to the new set of random normal variates to change the scatter in the correlation. We will plot the simulated data as a scatter plot.

~~~
x = sps.norm.rvs(size=50)
y = x + 1.0*sps.norm.rvs(size=50)

plt.figure()
plt.scatter(x, y, c="red")
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.show()
~~~
{: .language-python}

<p align='center'>
<img alt="Simulated correlation" src="../fig/ep11_corrscatter.png" width="500"/>
</p>

Finally, use the `scipy.stats` functions `pearsonr` and `spearmanr` to calculate and print the correlation coefficients and corresponding $$p$$-value of the correlation for both tests of the correlation. Since we know the data are normally distributed in both variables, the $$p$$-values should be reliable, even for $$n=50$$. Try changing the relative scaling of the $$x$$ and $$y$$ random variates to make the scatter larger or smaller in your plot and see what happens to the correlation coefficients and $$p$$-values.

~~~
## Calculate Pearson's r and Spearman's rho for the data (x,y) and print them out, also plot the data.
(rcor, rpval) = sps.pearsonr(x,y)
(rhocor, rhopval) = sps.spearmanr(x,y)

print("Pearson's r and p-value:",rcor, rpval)
print("Spearman's rho and p-value:",rhocor, rhopval)
~~~
{: .language-python}

For the example data plotted above, this gives:

~~~
Pearson's r and p-value: 0.5358536492516484 6.062792564158924e-05
Spearman's rho and p-value: 0.5417046818727491 4.851710819096097e-05
~~~
{: .output}

Note that the two methods give different values (including for the $$p$$-value).  _How can this be?_ Surely the data are correlated or they are not, with a certain probability?  It is important to bear in mind that (as in all statistical tests) we are not really asking the question "Are the data correlated?" rather we are asking: assuming that the data are really uncorrelated, independent and identically distributed, what is the probability that we would see such a non-zero absolute value of _this particular test-statistic_ by chance.  $$r$$ and $$\rho$$ are _different_ test-statistics: they are optimised in different ways to spot deviations from random uncorrelated data.

> ## Intuition builder: comparing the effects of outliers on Pearson's $$r$$ and Spearman's $$\rho$$
>
> Let's look at this difference between the two methods in more detail.  What happens when our
data has certain properties, which might favour or disfavour one of the methods? 
>
> Let us consider the case where there is a cloud of data points with no underlying correlation, plus an extreme outlier (as might be expected from some error in the experiment or data recording).  You may remember something like this as one of the four cases from _'Anscombe's quartet'_.
>
> First generate the random data: use the normal distribution to generate 50 data points which are uncorrelated between x and y and then replace one with an outlier which implies a correlation, similar to that seen in Anscombe's quartet. Plot the data, and measure Pearson's $$r$$ and Spearman's $$\rho$$ coefficients and $$p$$-values, and compare them - how are the results of the two methods different in this case? Why do you think this is?
>
>> ## Solution
>> ~~~
>> x = sps.norm.rvs(size=50)
>> y = sps.norm.rvs(size=50)
>> x[49] = 10.0
>> y[49] = 10.0
>>
>> ## Now plot the data and compare Pearson's r and Spearman's rho and the associated p-values
>> 
>> plt.figure()
>> plt.scatter(x, y, c="red",s=10)
>> plt.xlabel("x", fontsize=20)
>> plt.ylabel("y", fontsize=20)
>> plt.tick_params(axis='x', labelsize=20)
>> plt.tick_params(axis='y', labelsize=20)
>> plt.show()
>>
>> (rcor, rpval) = sps.pearsonr(x,y)
>> (rhocor, rhopval) = sps.spearmanr(x,y)
>> 
>> print("Pearson's r and p-value:",rcor, rpval)
>> print("Spearman's rho and p-value:",rhocor, rhopval)
>> ~~~
>> {: .language-python}
> {: .solution}
{: .challenge}


## Simple fits to bivariate data: linear regression

In case our data are linearly correlated, we can try to parameterise the linear function describing the relationship using a simple fitting approach called __linear regression__. The approach is to minimise the scatter (so called __residuals__) around a linear model. For data $$x_{i}$$, $$y_{i}$$, and a linear model with coefficients $$\alpha$$ and $$\beta$$, the residuals $$e_{i}$$ are given by _data$$-$$model_, i.e:

$$e_{i} = y_{i}-(\alpha + \beta x_{i})$$

Since the residuals themselves can be positive or negative, their sum does not tell us how small the residuals are on average, so the best approach to minimising the residuals is to minimise the _sum of squared errors_ ($$SSE$$):

$$SSE = \sum\limits_{i=1}^{n} e_{i}^{2} = \sum\limits_{i=1}^{n} \left[y_{i} - (\alpha + \beta x_{i})\right]^{2}$$

To minimise the $$SSE$$ we can take partial derivatives w.r.t. $$\alpha$$ and $$\beta$$ to find the minimum for each at the corresponding best-fitting values for the fit parameters $$a$$ (the intercept) and $$b$$ (the gradient). These best-fitting parameter values can be expressed as functions of the means or squared-means of the sample:

$$b = \frac{\overline{xy}-\bar{x}\bar{y}}{\overline{x^{2}}-\bar{x}^{2}},\quad a=\bar{y}-b\bar{x} $$

where the bars indicate sample means of the quantity covered by the bar (e.g. $$\overline{xy}$$ is $$\frac{1}{n}\sum_{i=1}^{n} x_{i}y_{i}$$) and the best-fitting model is:

$$y_{i,\mathrm{mod}} = a + b x_{i}$$

It's important to bear in mind some of the limitations and assumptions of linear regression. Specifically it takes no account of uncertainty on the x-axis values and further assumes that the data points are equally-weighted, i.e. the ‘error bars’ on every data point are assumed to be the same. The approach also assumes that experimental errors are uncorrelated. The model which is fitted is necessarily linear – this is often not the case for real physical situations, but many models may be linearised with a suitable mathematical transformation. The same approach of minimising $$SSE$$ can also be applied to non-linear models, but this must often be done numerically via computation.


## Linear regression in numpy, scipy and seaborn

Here we just make and plot some fake data (with no randomisation). First use the following sequences to produce a set of correlated $$x$$, $$y$$ data:

`x = np.array([10.0, 12.2, 14.4, 16.7, 18.9, 21.1, 23.3, 25.6, 27.8, 30.0])`

`y = np.array([12.6, 17.5, 19.8, 17.0, 19.7, 20.6, 23.9, 28.9, 26.0, 30.6])`

There are various methods which you can use to carry out linear regression on your data:
- use `np.polyfit()`
- use `scipy.optimize.curve_fit()`
- If you have it installed, use seaborn's `regplot()` or `lmplot`. Note that this will automatically also draw 68% confidence contours.

Below we use all three methods to fit and then plot the data with the resulting linear regression model. For the `curve_fit` approach we will need to define a linear function (either with a separate function definition or by using a Python __lambda function__, which you can look up online). For the seaborn version you will need to install seaborn if you don't already have it in your Python distribution, and must put `x` and `y` into a Panda's dataframe in order to use the seaborn function.

~~~
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5)) # Put first two plots side by side to save space

## first attempt: numpy.polyfit
r = np.polyfit(x, y, 1)
ax1.plot(x,y, "o");
ax1.plot(x, r[0]*x+r[1], lw=2)
ax1.text(11,27,"polyfit",fontsize=20)
ax1.set_xlabel("x", fontsize=14)
ax1.set_ylabel("y", fontsize=14)
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)

## second attempt: scipy.optimize.curve_fit
func = lambda x, a, b: x*a+b # Here we use a Python lambda function to define our function in a single line.
r2, pcov = spopt.curve_fit(func, x,y, p0=(1,1))
ax2.plot(x,y, "o");
ax2.plot(x, r2[0]*x+r2[1], lw=2)
ax2.text(11,27,"curve_fit",fontsize=20)
ax2.set_xlabel("x", fontsize=14)
ax2.set_ylabel("y", fontsize=14)
ax2.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)

import seaborn as sns

## fancy version with pandas and seaborn
df = pd.DataFrame(np.transpose([x,y]), index=np.arange(len(x)), columns=["x", "y"])
fig = plt.figure()
sns.regplot("x", "y", df)
plt.text(11,27,"seaborn regplot",fontsize=20)
plt.show()
~~~
{: .language-python}

<p align='center'>
<img alt="Linear regression scipy" src="../fig/ep10_linreg.png" width="600"/>
</p>

<p align='center'>
<img alt="Linear regression scipy" src="../fig/ep10_linregseaborn.png" width="400"/>
</p>

## Linear regression using Reynold's fluid flow data

This uses the data in `reynolds.txt`. We can load this into Python very simply using `numpy.genfromtxt` (use the names `["dP", "v"]` for the pressure gradient and velocity columns.  Then change the pressure gradient units to p.p.m. by multiplying the pressure gradient by $$9.80665\times 10^{3}$$.

Now fit the data with a linear model using curve_fit (assume $$v$$ as the explanatory variable, i.e. on the x-axis).

Finally, plot the data and linear model, and also the ___data-model residuals___ as a pair of panels one on top of the other (you can use `plt.subplots` and share the x-axis values using the appropriate function argument). You may need to play with the scaling of the two plot windows, generally it is better to show the residuals with a more compressed vertical size than the data and model, since the former should be a fairly flat function if the fit converges). To set up the subplots with the right ratio of sizes, shared x-axes and no vertical space between them, you can use a sequence of commands like this:

`fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})`
`fig.subplots_adjust(hspace=0)`

and then use `ax1`, `ax2` to add labels or modify tick parameters (note that the commands for these may be different for subplots than for a usual single-panel figure). You can highlight the residuals better by adding a horizontal dotted line at $$y=0$$ in the residual plot, using the `axhline` command.

~~~
reynolds = np.genfromtxt ("reynolds.txt", dtype=np.float, names=["dP", "v"], skip_header=1, autostrip=True)

## change units
ppm = 9.80665e3
dp = reynolds["dP"]*ppm
v = reynolds["v"]

popt, pcov = spopt.curve_fit(func,dp, v)
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
fig.subplots_adjust(hspace=0)
ax1.plot(dp, v, "o")
ax1.plot(dp, popt[0]*dp+popt[1], lw=2)
ax1.set_ylabel("Velocity (m/s)", fontsize=14)
ax1.tick_params(axis="x",direction="in",labelsize=12) # Use this to include visible tick-marks inside the plot
ax2.plot(dp, v-(popt[0]*dp+popt[1]), "o")
ax2.set_xlabel("Pressure gradient (Pa/m)",fontsize=14)
ax2.set_ylabel("Residuals (m/s)", fontsize=14)
# The next two commands can be used to align the y-axis labels
ax1.get_yaxis().set_label_coords(-0.1,0.5)
ax2.get_yaxis().set_label_coords(-0.1,0.5)
ax2.axhline(0.0,ls=':') # plot a horizontal dotted line to better show the deviations from zero
ax2.tick_params(labelsize=12)
plt.show()
~~~
{: .language-python}

The fit doesn't quite work at high values of the pressure gradient. We can exclude those data
points for now. Create new pressure gradient and velocity arrays which only use the first 8 data points. Then repeat the fitting and plotting procedure used above. You should see that the residuals are now more randomly scattered around the model, with no systematic curvature or trend.

~~~
dp_red = dp[:8]
v_red = v[:8]

popt, pcov = spopt.curve_fit(func, dp_red, v_red)
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
fig.subplots_adjust(hspace=0)
ax1.plot(dp_red, v_red, "o")
ax1.plot(dp_red, popt[0]*dp_red+popt[1], lw=2)
ax1.set_ylabel("Velocity (m/s)", fontsize=14)
ax1.tick_params(axis="x",direction="in",labelsize=12) # Use this to include visible tick-marks inside the plot
ax2.plot(dp_red, v_red-(popt[0]*dp_red+popt[1]), "o")
ax2.set_xlabel("Pressure gradient (Pa/m)",fontsize=14)
ax2.set_ylabel("Residuals (m/s)", fontsize=14)
# The next two commands can be used to align the y-axis labels
ax1.get_yaxis().set_label_coords(-0.1,0.5)
ax2.get_yaxis().set_label_coords(-0.1,0.5)
ax2.axhline(0.0,ls=':') # plot a horizontal dotted line to better show the deviations from zero
ax2.tick_params(labelsize=12)
plt.show()
~~~
{: .language-python}


##  Bootstrapping to obtain error estimates

Now let's generate some fake correlated data, and then use the `numpy.random.choice` function to randomly select samples of the data (with replacement) for a bootstrap analysis of the variation in linear fit parameters $$a$$ and $$b$$. First use `numpy.random.normal` to generate fake sets of correlated $$x$$ and $$y$$ values as in the earlier example for exploring the correlation coefficient. Use 100 data points for $$x$$ and $$y$$ to start with and plot your data to check that the correlation is clear.

~~~
x = np.random.normal(size=100)
y = x + 0.5*np.random.normal(size=100)

fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(x, y, "o")
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("y", fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.show()
~~~
{: .language-python}

First use `curve_fit` to obtain the $$a$$ and $$b$$ coefficients for the simulated, 'observed' data set and print your results.

When making our new samples, we need to make sure we sample the same indices of the array for all variables being sampled, otherwise we will destroy any correlations that are present.  Here you can do that by setting up an array of indices matching that of your data (e.g. with `numpy.arange(len(x))`), randomly sampling from that using `numpy.random.choice`, and then using the `numpy.take` function to select the values of `x` and `y` which correspond to those indices of the arrays. Then use `curve_fit` to obtain the coefficients $$a$$ and $$b$$ of the linear correlation and record these values to arrays. Use a loop to repeat the process a large number of times (e.g. 1000 or greater) and finally make a scatter plot of your values of $$a$$ and $$b$$, which shows the bivariate distribution expected for these variables, given the scatter in your data. 

Now find the mean and standard deviations for your bootstrapped distributions of $$a$$ and $$b$$, print them and compare with the expected errors on these values given in the lecture slides. These estimates correspond to the errors of each, ___marginalised over the other variable___. Your distribution could also be used to find the covariance between the two variables. 

__Note that the standard error on the mean of $$a$$ or $$b$$ is not relevant for estimating the errors here__ because you are trying to find the scatter in the values expected from your observed number of data points, not the uncertainty on the many repeated 'bootstrap' versions of the data.

Try repeating for repeated random samples of your original $$x$$ and $$y$$ values to see the change in position of the distribution as your sample changes. Try changing the number of data points in the simulated data set, to see how the scatter in the distributions change. How does the simulated distribution compare to the 'true' model values for the gradient and intercept, that you used to generate the data?

Note that if you want to do bootstrapping using a larger set of variables, you can do this more easily by using a Pandas dataframe and using the `pandas.DataFrame.sample` function.  By setting the number of data points in the sample to be equal to the number of rows in the dataframe, you can make a resampled dataframe of the same size as the original (replacement is the default of the `sample` function, but be sure to check this in case the default changes!).

~~~
nsims = 1000
indices = np.arange(len(x))
func = lambda x, a, b: x*a+b
r2, pcov = spopt.curve_fit(func, x,y, p0=(1,1))
a_obs = r2[0]
b_obs = r2[1]

print("The obtained a and b coefficients are ",a_obs,"and",b_obs,"respectively.")

a_arr = np.zeros(nsims)
b_arr = np.zeros(nsims)
for i in range(nsims):
    new_indices = np.random.choice(indices, size=len(x), replace=True)
    new_x = np.take(x,new_indices)
    new_y = np.take(y,new_indices)
    r2, pcov = spopt.curve_fit(func, new_x,new_y, p0=(1,1))
    a_arr[i] = r2[0]
    b_arr[i] = r2[1]
    
plt.figure()
plt.plot(a_arr, b_arr, "o")
plt.xlabel("a", fontsize=14)
plt.ylabel("b", fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.show()

print("The mean and standard deviations of the bootstrapped samples of $a$ are:",
      np.mean(a_arr),"and",np.std(a_arr,ddof=1),"respectively")
print("The mean and standard deviations of the bootstrapped samples of $b$ are:",
      np.mean(b_arr),"and",np.std(b_arr,ddof=1),"respectively")
~~~
{: .language-python}











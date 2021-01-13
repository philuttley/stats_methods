---
title: >-
    Bayes' Theorem
teaching: 30
exercises: 10
questions:
- "What is Bayes' theorem and how can we use it to answer scientific questions?"
objectives:
- "Learn how Bayes' theorem is derived and how it applies to simple probability problems."
keypoints:
- "For conditional probabilities, Bayes' theorem tells us how to swap the conditionals around."
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


When considering two [__events__]({{ page.root }}/reference/#event) $$A$$ and $$B$$, we have previously seen how the equation for conditional probability gives us the multiplication rule:

$$P(A \mbox{ and } B) = P(A\vert B) P(B)$$

It should be clear that we can invert the ordering of $$A$$ and $$B$$ here and the probability of both happening should still be the same, i.e.:

$$P(A\vert B) P(B) = P(B\vert A) P(A)$$

This simple extension of probability calculus leads us to [__Bayes' theorem__]({{ page.root }}/reference/#bayesian), one of the most important results in statistics and probability theory:

$$P(A\vert B) = \frac{P(B\vert A) P(A)}{P(B)}$$

Bayes' theorem, named after clergyman Rev. Thomas Bayes who proposed it in the middle of the 18th century, is in its simplest form, a method to swap the conditional dependence of two events, i.e. to obtain the probability of $$A$$ conditional on $$B$$, when you only know the probability of $$B$$ conditional on $$A$$, and the probabilities of $$A$$ and $$B$$ (i.e. each marginalised over the conditional term). 

To show the inclusion of marginalisation, we can generalise from two events to a set of mutually exclusive exhaustive events $$\{A_{1},A_{2},...,A_{n}\}$$:

$$P(A_{i}\vert B) = \frac{P(B\vert A_{i}) P(A_{i})}{P(B)} = \frac{P(B\vert A_{i}) P(A_{i})}{\sum^{n}_{i=1} P(B\vert A_{i}) P(A_{i})}$$


> ## Challenge: what kind of binary merger is it?
> Returning to our hypothetical problem of detecting radio counterparts from gravitational wave events corresponding to binary mergers of binary neutron stars (NN), binary black holes (BB) and neutron star-black hole binaries (NB), recall that the probabilities of radio detection (event denoted with $$D$$) are:
> 
> $$P(D\vert NN) = 0.72$$, $$P(D\vert NB) = 0.2$$, $$P(D\vert BB) = 0$$
>
> and for any given merger event the probability of being a particular type of binary is: 
>
> $$P(NN)=0.05$$, $$P(NB) = 0.2$$, $$P(BB)=0.75$$
>
> Calculate the following:
> 1. Assuming that you detect a radio counterpart, what is the probability that the event is a binary neutron star ($$NN$$)?
> 2. Assuming that you $$don't$$ detect a radio counterpart, what is the probability that the merger includes a black hole (either $$BB$$ or $$NB$$)?
>
>> ## Hint
>> Remember that if you need a total probability for an event for which you only have the conditional probabilities, you can use the law of total probability and marginalise over the conditional terms.
> {: .solution}
>> ## Solution
>> 1.  We require $$P(NN\vert D)$$. Using Bayes' theorem:
>> $$P(NN\vert D) = \frac{P(D\vert NN)P(NN)}{P(D)}$$
>> We must marginalise over the conditionals to obtain $$P(D)$$, so that:
>> 
>>     $$P(NN\vert D) = \frac{P(D\vert NN)P(NN)}{(P(D\vert NN)P(NN)+P(D\vert NB)P(NB)} = \frac{0.72\times 0.05}{(0.72\times 0.05)+(0.2\times 0.2)}$$
>>    $$= \frac{0.036}{0.076} = 0.474 \mbox{    (to 3 s.f.)}$$
>>
>> 
>> 2. We require $$P(BB \vert D^{C}) + P(NB \vert D^{C})$$, since radio non-detection is the complement of $$D$$ and the $$BB$$ and $$NB$$ events are [__mutually exclusive__]({{ page.root }}/reference/#mutual-exclusivity). Therefore, using Bayes' theorem we should calculate:
>>
>>    $$P(BB \vert D^{C}) = \frac{P(D^{C} \vert BB)P(BB)}{P(D^{C})} = \frac{1\times 0.75}{0.924} = 0.81169$$
>>
>>    $$P(NB \vert D^{C}) = \frac{P(D^{C} \vert NB)P(NB)}{P(D^{C})} = \frac{0.8\times 0.2}{0.924} = 0.17316$$
>>
>> So our final result is: $$P(BB \vert D^{C}) + P(NB \vert D^{C}) = 0.985 \mbox{    (to 3 s.f.)}$$
>>
>> Here we used the fact that $$P(D^{C})=1-P(D)$$, along with the value of $$P(D)$$ that we already calculated in part 1.
>>
>> There are a few interesting points to note about the calculations:
>> - Firstly, in the absence of any information from the radio data, our _prior expectation_ was that a merger would most likely be a black hole binary (with 75% chance). As soon as we obtained a radio detection, this chance went down to zero. 
>> - Then, although the prior expectation that the merger would be of a binary neutron star system was _4 times smaller_ than that for a neutron star-black hole binary, the fact that a binary neutron star was almost 4 times more likely to be detected in radio almost balanced the difference, so that we had a slightly less than 50/50 chance that the system would be a binary neutron star.
>> - Finally, it's worth noting that the non-detection case weighted the probability that the source would be a black hole binary to slightly more than the prior expectation of $$P(BB)=0.75$$, and correspondingly reduced the expectation that the system would be a neutron star-black hole system, because there is a moderate chance that such a system would produce radio emission, which we did not see.
> {: .solution}
{: .challenge}

## Bayes' theorem for continuous probability distributions

From the multiplication rule for continuous probability distributions, we can obtain the continuous equivalent of Bayes' theorem:

$$p(y\vert x) = \frac{p(x\vert y)p(y)}{p(x)} = \frac{p(x\vert y)p(y)}{\int^{\infty}_{-\infty} p(x\vert y)p(y)\mathrm{d}y}$$

> ## Bayes' billiards game
> This problem is taken from the useful article [Frequentism and Bayesianism: A Python-driven Primer][vdplas_primer] by Jake VanderPlas, and is there adapted from a problem [discussed by Sean J. Eddy][eddy_bayes].
>
> Carol rolls a billiard ball down the table, marking where it stops. Then she starts rolling balls down the table. If the ball lands to the left of the mark, Alice gets a point, to the right and Bob gets a point. First to 6 points wins. After some time, Alice has 5 points and Bob has 3. What is the probability that Bob wins the game ($$P(B)$$)?
>
> Defining a success as a roll for Alice (so that she scores a point) and assuming the probability $$p$$ of success does not change with each roll, the relevant distribution is [_binomial_]({{ page.root }}/reference/#distributions---binomial). For Bob to win, he needs the next three rolls to fail (i.e. the points go to him). A simple approach is to estimate $$p$$ using the number of rolls and successes, since the expectation for $$X\sim \mathrm{Binom}(n,p)$$ is $$E[X]=np$$, so taking the number of successes as an unbiased estimator, our estimate for $$p$$, $$\hat{p}=5/8$$. Then the probability of failing for three successive rolls is:
>
> $$(1-\hat{p})^{3} \simeq 0.053$$
>
> However, this approach does not take into account our uncertainty about Alice's true success rate! 
>
> Let's use Bayes' theorem. We want the probability that Bob wins given the data already in hand ($$D$$), i.e. the $$(5,3)$$ scoring. We don't know the value of $$p$$, so we need to consider the marginal probability of $$B$$ with respect to $$p$$:
>
> $$P(B\vert D) \equiv \int P(B,p \vert D) \mathrm{d}p$$
>
> We can use the multiplication rule $$P(A \mbox{ and } B) = P(A\vert B) P(B)$$, since $$P(B,p \vert D) \equiv P(B \mbox{ and } p \vert D)$$:
>
> $$P(B\vert D) = \int P(B\vert p, D) P(p\vert D) \mathrm{d}p$$
>
> Now we can calculate $$P(D\vert p)$$ from the binomial distribution, so to get there we use Bayes' theorem:
>
> $$P(B\vert D) = \int P(B\vert p, D) \frac{P(D\vert p)P(p)}{P(D)} \mathrm{d}p$$
>
> $$= \frac{\int P(B\vert p, D) P(D\vert p)P(p) \mathrm{d}p}{\int P(D\vert p)P(p)\mathrm{d}p}$$
>
> where we first take $$P(D)$$ outside the integral (since it has no explicit $$p$$ dependence) and then express it as the marginal probability over $$p$$.  Now:
> - The term $$P(B\vert p,D)$$ is just the binomial probability of 3 failures for a given $$p$$, i.e. $$P(B\vert p,D) = (1-p)^{3}$$ (conditionality on $$D$$ is implicit, since we know the number of consecutive failures required).
> - $$P(D\vert p)$$ is just the binomial probability from 5 successes and 3 failures, $$P(D\vert p) \propto p^{5}(1-p)^{3}$$. We ignore the term accounting for permutations and combinations since it is constant for a fixed number of trials and successes, and cancels from the numerator and denominator.
> - Finally we need to consider the distribution of the chance of a success, $$P(p)$$. This is presumably based on Carol's initial roll and success rate, which we have no prior expectation of, so the simplest assumption is to assume a uniform distribution (i.e. a __uniform prior__): $$P(p)= constant$$, which also cancels from the numerator and denominator.
>
> Finally, we solve:
>
> $$P(B\vert D) = \frac{\int_{0}^{1} (1-p)^{6}p^{5}}{\int_{0}^{1} (1-p)^{3}p^{5}} \simeq 0.091$$
>
> The probability of success for Bob is still low, but has increased compared to our initial, simple estimate. The reason for this is that our choice of prior suggests the possibility that $$\hat{p}$$ overestimated the success rate for Alice, since the median $$\hat{p}$$ suggested by the prior is 0.5, which weights the success rate for Alice down, increasing the chances for Bob.
{: .callout}


[vdplas_primer]: https://arxiv.org/abs/1411.5018
[eddy_bayes]: https://www.nature.com/articles/nbt0904-1177

{% include links.md %}



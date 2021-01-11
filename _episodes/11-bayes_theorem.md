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









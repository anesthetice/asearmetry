#set text(font: "Latin Modern Roman 12", size: 12pt)
#set par(spacing: 1.2em, justify: true, first-line-indent: 0em, leading: 0.55em)
#show math.equation: set text(font: "New Computer Modern Math")
#show smallcaps: set text(font: "Latin Modern Roman Caps")
#show raw: set text(font: "ZedMono NF")

== Introduction

Let us first imagine a continuous mono-channel audio signal as a function:
#[
  #show math.equation.where(block: true): set par(leading: 0.25em)
  $
  x: 
  &&RR_+ #h(-.1em) &arrow.long RR \ 
  &&t #h(.6em) &arrow.long.bar x(t) \
  $
]

Where at a given time $t>0$, the amplitude of the audio waveform is given by $x(t)$.

In digital signal processing, a system is any process that produces an output signal given an input signal. These can be characterized as functions whose domain and co-domain are sets of other functions, a sort of meta-function if you will.

In practice, a different notation is used to limit confusion. For example, without using a different notation than normal, assuming $S$ is our system, we can write $y(t) = (S x) (t)$. Now if $x(t)$ is scaled by a constant $a$, we would have to write $y(t) = (S(a x))(t)$ which makes it impossible to tell which of these is a function or a constant. Replacing $x$ with $x(t)$ would not improve the confusion as it would imply that $y(t_1)$ only ever depends on the value $x(t_1)$ instead of the entire input signal.

As such, we use curly braces to indicate that $S{x(t)}$ acts on the entire signal $x(t)$ instead of the scalar value of said signal at time $t$. Formally, one can write: $y(u) = S{x(t)}(u)$, but since the output signal often uses the same underlying parameter as the input signal it stems from (e.g., time, position), one can write $y(t) = S{x(t)}$ for short, just be mindful that the domain and or co-domain of $y(t)$ does not have to be the same as $x(t)$.

== System properties
$
  S "is linear" arrow.l.r.double cases(S{a dot x(t)} = a dot S{x(t)}, S{b + x(t)} = b + S{x(t)}) quad forall a,b in RR
$

$
S "is shift-invariant" arrow.l.r.double S{x(t - a)}(u) = S{x(t)}(u-a) quad forall a in RR
$ 

Note that the term time-invariant is more often encountered than shift-invariant, for instance we say LTI (Linear Time Invariant) systems instead of LSI.


== LTI (LSI) convolution

With these two properties, one can work their way up to the convolution operation for LTI systems. But we do need to use the Dirac delta as well.

The Dirac delta function $delta(x)$, funnily enough cannot be characterized by a typical function (it requires using generalized functions instead). But can be heuristically described as $"if" x = 0 arrow.double delta(x) arrow + infinity$, such that:

$
  forall epsilon > 0, thick
  integral_(x - epsilon)^(x + epsilon) delta(x) dif x = 1
  quad arrow.double quad
  integral_(-infinity)^(+infinity) delta(x) dif x = 1
$

As such, $x(t)$
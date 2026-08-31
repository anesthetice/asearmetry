#set par(spacing: 1.2em, justify: true, first-line-indent: 0em, leading: 0.55em)
#set text(font: "Latin Modern Roman 12", size: 12pt)
#set heading(numbering: "1.")

#show math.equation: set text(font: "New Computer Modern Math")
#show smallcaps: set text(font: "Latin Modern Roman Caps")
#show raw: set text(font: "ZedMono NF")
#import "witharrows.typ": *

#show std.bibliography: set text(10pt)
#show std.bibliography: set block(spacing: 1.0em)
#set std.bibliography(title: [References], style: "ieee")

#let cite_it(label) = {
  cite(label, form: "prose", style: "apa")
}

= Introduction

I've always found it fascinating that one can perceive audio as being three dimensional (i.e., the source of the audio appears to be moving around oneself) while using a pair of headphones which are only capable of outputting only two channels at once. Although with a little perspective, it may not be all that surprising given that your ears also only pick up two different channels at once. When done right, 3D audio is quite an experience, here's a great example produced by the "3D Sound Studio" YouTube channel: #link("https://www.youtube.com/watch?v=6nK0YPaBRK0")["Introspection (3D Sound Experience)"], and another example produced for the "3D Tune-In Toolkit" #link("https://www.youtube.com/watch?v=osJQ0Kxv1P0")["3D Tune-In Toolkit - Binaural spatialisation"].

For the past couple of years I've wanted to create a library and or application mostly from scratch that could create this type of audio, which would differentiate itself by allowing one to use differential equations to describe the trajectory of the source. To be honest, I haven't looked too deeply if this has been done before, as I find the topic interesting regardless, and this is just a hobby of mine.

= Methods

Let's first try to think of what information our ears could be picking up from a sound in order to figure out where it is coming from.

+ The most evident factor would be the delay between the time when the sound reaches one ear and the other, this is called the interaural time difference (ITD).

+ One could think of the difference in amplitude brought about by one ear being more distant than the other but also due to "shadowing" effects caused by the head itself (for instance, bone has an extremely high acoustic impedance compared to air, the sound wave has to at least wrap around your head for your other ear to hear something intelligible, necessitates scattering -- which incurs a loss in intensity)

+ The shape of your ears.

+ Sounds that are distant appear more "muffled", especially higher frequencies.

+ Probably more that I am forgetting.

#pagebreak()

= Pitch shifting

After implementing a resampling algorithm (windowed $sinc$ in our case), it's very trivial to then implement one to modify the time scale of an audio track. For instance, to scale by a factor of 1.5, we can simply set the track's sampling rate to 1.5 times the original (meaning 1.5 times as many samples should be played per second), before resampling to our original rate.

```rust
fn scale_time(self, factor: f64) -> AudioBuffer<C> {
    assert!(factor > 0.0 && factor.is_finite());
    let original_sr = self.sr_or_panic();
    self.into_owned()
        .with_sr(factor * original_sr)
        .resample(original_sr, 21) // M=21 -> Hann window of length 2⋅21+1 used
}
```

With that done, I also wanted a way to counteract the very prominent change in pitch caused by affecting the "speed" of the track; and thought that it wouldn't be all that much more difficult than the algorithm above, as we shall come to see, I was wrong.



=== Short-time Fourier Transform

I am slightly annoyed that there is seemingly no distinction made between the discrete and continuous application of the STFT, unlike with the FT and DFT. But anyhow, unlike the DFT seen earlier, which lets us find what combination of frequencies make up the track over its entire duration, the STFT focuses instead on a specific frame of time.

Note that I will be referencing in part from a notebook by #cite_it(<STFT_AudioLabs>), and SciPy's user guide on signal processing (see @STFT_SciPy, @SciPy_full). Inheriting from the DFT section above, we additionally define the following:

- Let $w: {0, dots.c, W - 1} arrow.long RR$ be our window function, with $W$ being its length, sometimes also referred to as the frame length or frame size among many other names.

- Let $H in {1, dots.c, W - w_("n_edge_zeros")}$ be our hop length. For example, the number of edge zeros for the Hann window is two. If the hop length is not within these bounds, then the reconstruction of the original signal from the STFT becomes impossible.

To make things less frustrating, we will also assume that our window length is odd, with $W = 2 M + 1$, as such $w[M]$ should typically be the peak of the window.

Let $cal(X)[k, i]$ be the short-time Fourier transform of $x[n]$, there is still however something to address before writing down the formula of the STFT. That is, imagine you are interested in the frequencies found at the very start of the signal, looking at $cal(X)[k, 0]$ would not be very useful as window functions taper-off to zero near their edges. To make things more coherent, all we need to do is pad $x[n]$ with $M$ zeros on the left, and $W - (M+N) mod H$ on the right (more complicated as we want to be sure that the whole signal can be reconstructed); we will call the padded version $x_("pad")[n]$

With that settled, we have the following:

$
  cal(X)[k, i_w] = "STFT"{x[n]}[k, i_w] &= "DFT"{x_("pad")[n] dot w[n - i_w H]}[m] \
  &= sum_(n=0)^(W - 1) x_("pad")[n] dot w[n - i_w H] dot e^(-j #h(1pt) 2 pi dot k n slash N)
$

Note that the support of $cal(X)[k, i_w]$ is $(k, i_w) in {0, dots.c, W - 1} times {0, dots.c, ceil((M+N)/H) - 1}$

The pain does not stop here, you might already have noticed one problem. The elements near the boundaries of our discrete signal $x[n]$ are contained within fewer windows than those closer to the middle. As such, summing and normalizing the IDFTs is not sufficient.

To account for this, we want to find the total "weight" that each sample was subject to, and while there does seem to be a way to express this efficiently with a mathematical formula, I've already spent far too much time on this STFT tangent, and thus will just brute-force compute the inverse weight vector at runtime... Let $Omega_"weight"$ be the total weight vector, with a length of $N$.

The equation is very annoying to write down, so we will just describe it. For each frame, obtain its IDFT, merge all of these together while keeping into account the offset of each frame, multiply by $Omega_"weight"^(-1)$, discard the left and right padding.

=== Pitch shifting

Imagine a sinusoidal signal $x(t) = e^(j omega_0 t)$ which oscillates at the angular frequency of $omega_0$. If we sampled this sinusoidal starting at time zero, then performed the DFT, its corresponding frequency bin would have a magnitude of one and a phase of zero. But, if we instead started sampling at a slightly later time, while the magnitude of the bin would still be one, its phase would not be the same as before (unless we accidentally selected a starting time that leads to no apparent phase offset of course).

Let's make this a lot more concrete. For any frame, the bin $cal(X)_(i)[k]$ corresponds to the frequency of $omega_k = (2 pi k)/W$

= Miscellaneous algorithms

== Frequency filters

Suppose we wish to keep only the frequencies found within the range of $[f_"low", f_"high"]$. In the continuous frequency space, this appears at first rather straight-forward, we simply keep the frequencies we want while zeroing those we do not care about.

But even here there are already foot guns to watch out for, notably the #link("https://en.wikipedia.org/wiki/Gibbs_phenomenon")["Gibbs phenomenon"]: we want to avoid sudden "jumps" both in the time domain as well as the frequency domain, given how similar the IFT is to the FT (i.e., the duality property). We also have to be careful of how the filter affects the phases of our signal. Finally, there's also the relation $X(0) = integral_(RR) x(t) dif t$ to keep in mind.

TODO, discrete time pain to discuss, phase shifts as well

=== Low-pass filter

To create a good low-pass filter, let's start with the following transfer function as the basis for our filter: $H(omega) = "rect"(1/2 dot omega/(2pi f_"high")) = "rect"(omega slash 4pi f_"high")$.

From a formula sheet, we know that $sinc(t slash T) arrow.long^(cal(F)) T dot "rect"(omega T slash 2pi)$, where $sinc(t)$ is defined as $sinc(t) = sin(pi t) / (pi t)$, therefore we can write:

$
  h(t)
  &= cal(F)^(-1){H(omega)}(t) \
  &= cal(F)^(-1){"rect"(omega slash 4pi f_"high")}(t) \
  &= 2 f_"high" dot cal(F)^(-1){ 1 / (2 f_"high") dot "rect"((omega dot (2 f_"high")^(-1)) / (2 pi))}(t) \
  &= 2 f_"high" dot sinc(2 f_"high" t) \
  &= 2 f_"high" dot sin(2pi f_"high"  t)/(2pi f_"high" t) \
  &= sin(2pi f_"high" t)/(pi t) \
$

In discrete form (see @DTFT, $t = n T_s = n slash f_s$), we get

$
  h[n] = sin((2 pi f_"high")/f_s dot n)
$

Finally, to avoid the Gibbs phenomenon mentioned above, we apply a window function to our filter,such as the Hann function in our case. Let $M in NN$ such that $n_"taps" = 2M + 1$, we define the discrete filter on the inteval between $-M$ and $+M$ as follows:

$
  h[n] = h_("old")[n] dot w_("hann")[n]
$

=== High-pass filter

Similar to last time, we start with the following transfer function: $H(omega) = 1- "rect"(1/2 dot omega/(2pi f_c)) = 1 - "rect"(omega slash 4pi f_c)$.

Then

$
  h(t)
  &= cal(F)^(-1){H(omega)}(t) \
  &= cal(F)^(-1){1 - "rect"(omega slash 4pi f_c)}(t) \
  &= cal(F)^(-1){1}(t) - cal(F)^(-1){H_"lowpass"(omega)}(t) \
  &= delta(t) - 2 f_c  sinc(2 f_c t)
$

Therefore

$
  h[n] = delta[n] - 2 f_c  sinc(2 f_c t slash f_s)
$

=== Band-pass filter

Let $[f_"low", f_"high"]$ the range of frequencies we want to keep (where we assume $f_"high" > f_"low"$). We define $f_m = 1/2 (f_"low" + f_"high")$ the frequency midpoint and $d_f = f_"high" - f_"low"$ the bandwidth.

$
  H(omega)
  &= "rect"((omega - 2pi f_m)/(2pi d_f)) + "rect"((-omega - 2pi f_m)/(2pi d_f)) \
  &= "rect"((omega - 2pi f_m)/(2pi d_f)) + "rect"((omega + 2pi f_m)/(2pi d_f)) \
$

Notice that since $e^(j omega_0 t) x(t) arrow.long^(cal(F)) X(omega - omega_0)$, then $cos(omega_0t)x(t) = 1/2 (e^(j omega_0 t) + e^(-j omega_0 t))x(t) arrow.long^(cal(F)) 1/2 (X(omega - omega_0) + X(omega + omega_0))$, this allows us to rewrite $H(omega)$ into a form that's easier to invert (which is obviously what we are going to do afterward):

$
  H(omega) = 2 "rect"(omega / (2pi d_f)) convolve 1/2 (delta(omega - 2pi f_m) + delta(omega + 2pi f_m))
$


Then,

$
  h(t)
  &= cal(F)^(-1){H(omega)}(t) \
  &= 2 dot cal(F)^(-1){"rect"((omega d_f^(-1))/(2pi))}(t) dot cal(F)^(-1){(delta(omega - 2pi f_m) + delta(omega + 2pi f_m))/2}(t)\
  &= 2 d_f dot sinc(d_f t) dot cos(2pi f_m t) \
$

Finally,

$
  h_("ideal")[n] = 2 d_f dot sinc(d_f n slash f_s) dot cos(2pi f_m n slash f_s)
$



= Appendix <Appendix>

#context {
  outline(
    target: selector(heading.where(outlined: true)).after(here()).before(<bib>),
    title: none,
    indent: n => (n - 1) * 1.2em
  )
}


== Introduction to systems

=== Notation for systems

A sound, such as a piece of music for instance, can be abstracted as a signal whose amplitude at a certain time relates to the pressure difference (compared to the baseline pressure, in a certain fluid) at that same instant. We can thus reprsent a continuous mono-channel audio signal as a function:

#[
  #show math.equation.where(block: true): set par(leading: 0.25em)
  $
  x:
  &&RR #h(.4em) &arrow.long RR \
  &&t #h(.6em) &arrow.long.bar x(t) \
  $
]

In digital signal processing, a system is any process that produces an output signal given an input signal. These can be characterized as functions whose domain and co-domain are sets of other functions, a sort of meta-function if you will.

In practice, a different notation is used to limit confusion. For example, without using a different notation than normal, assuming $S$ is our system, we can write $y(t) = (S x) (t)$. Now if $x(t)$ is scaled by a constant $a$, we would have to write $y(t) = (S(a x))(t)$ which makes it impossible to tell which of these is a function or a constant. Replacing $x$ with $x(t)$ would not improve the confusion as it would imply that $y(t_1)$ only ever depends on the value $x(t_1)$ instead of the entire input signal.

As such, we use curly braces to indicate that $S{x(t)}$ acts on the entire signal $x(t)$ instead of the scalar value of said signal at time $t$. Formally, one can write: $y(u) = S{x(t)}(u)$, but since the output signal often uses the same underlying parameter as the input signal it stems from (e.g., time, position), one can write $y(t) = S{x(t)}$ for short, just be mindful that the domain and or co-domain of $y(t)$ does is not necessarily the same as $x(t)$.

=== Properties of systems
$
  S "is linear" arrow.l.r.double cases(S{a dot x(t)} = a dot S{x(t)}, S{b + x(t)} = b + S{x(t)}) quad forall a,b in RR
$

$
S "is shift-invariant" arrow.l.r.double S{x(t - a)}(u) = S{x(t)}(u-a) quad forall a in RR
$

Note that the term time-invariant is more often encountered than shift-invariant, for instance we say LTI (Linear Time Invariant) systems instead of LSI.

There exist many other properties that a system can have (e.g., memorylessness, causality, et cetera), but they will not be discussed here.

=== Dirac delta function (abridged)

With the two properties above, one can work their way up to the convolution operation for LTI systems. But we'll also need to use the Dirac delta function, which we will briefly explore here.

The Dirac delta function $delta(x)$, funnily enough, cannot be characterized by a typical function (one would need to use "generalized functions"). But thankfully can be heuristically described as $"if" x = 0 arrow.double delta(x) arrow + infinity$, such that:

$
  forall epsilon > 0, thick
  integral_(-epsilon)^(+epsilon) delta(x) dif x = 1
  quad arrow.double quad
  integral_(-infinity)^(+infinity) delta(x) dif x = 1
$

As such, $x(t)$ can be equivalently written as:

$
  x(t) equiv integral_(-infinity)^(+infinity) x(tau) thin delta(t-tau) dif tau
$

=== LTI (LSI) convolution <LTI>

Finally, with just a little bit more math, we can arrive at the definition of the convolution and its relation with LTI systems. Let $S$ be a LTI system, let $x(t)$ be the input function and let $y(t)$ be its respective output function.

#[
  #show math.equation.where(block: true): set align(left)
  #show math.equation.where(block: true): it => {
    box(inset: (left: 3em))[
      #it
    ]
  }
  #witharrows($
    y(t)

    &= S{x(t)}
    explain("Dirac delta equiv.", place: #true) \

    &= S{integral_(-infinity)^(+infinity) x(tau) thin delta(t-tau) dif tau}
    explain(S "is linear," S{(a+b)} = S{a} + S{b}, place: #true) \

    &= integral_(-infinity)^(+infinity) S{x(tau) thin delta(t-tau) dif tau}
    explain(S "is linear," x(tau) thin \a\n\d thin dif tau "are scalars", place: #true) \

    &= integral_(-infinity)^(+infinity) x(tau) thin S{delta(t-tau)} dif tau
    explain(S "is time-invariant," "let" h(t) = S{delta(t)}, place: #true) \

    &= integral_(-infinity)^(+infinity) x(tau) thin h(t-tau) dif tau
    explain("Definition of the convolution operation", place: #true) \

    &= (x convolve h)(t) \
  $)
]

Where $h(t)$ is usually called the impulse response of the system, which is obvious given that we define it as such: $h(t) = S{delta(t)}$. We have demonstrated that for any input to an LTI system, its resulting output corresponds to the convolution of said input with the system's impulse response.

$
  y(t)
  &= S{x(t)} \
  &= (x convolve h)(t) \
  &eq.def integral_(-infinity)^(+infinity) x(tau) thin h(t-tau) dif tau \
$


=== LTV (LSV) convolution <LTV>

Let $S$ be a linear but time-variant system. Everything from before except the final development can be carried over.

$
  y(t)

  &= S{x(t)}
  explain("Dirac delta equiv.", place: #true) \

  &= S{integral_(-infinity)^(+infinity) x(tau) thin delta(t-tau) dif tau}
  explain(S "is linear, " S{(a+b)} = S{a} + S{b}, place: #true) \

  &= integral_(-infinity)^(+infinity) S{x(tau) thin delta(t-tau) dif tau}
  explain(S "is linear, " x(tau) thin \a\n\d thin dif tau "are scalars", place: #true) \

  &= integral_(-infinity)^(+infinity) x(tau) thin S{delta(t-tau)} dif tau \
$

But this time, the function $h$ must vary not only with respect to $t$ as before, but also with respect to $tau$, as our system $S$ is not time-invariant. As such, $h(t, tau) = S{delta(t-tau)}$, and we can write:

$
  y(t) = integral_(-infinity)^(+infinity) x(tau) thin h(t, tau) dif tau \
$

== Fourier transforms

=== The Fourier transform (FT) <FT>

Let $x: RR arrow.r CC$ denote a function; if $x in L^1 (RR)$, meaning iff. $integral_(RR) abs(x(t)) dif t < infinity$, then the Fourier transform of the function $x$ is given by:

$
  X(omega) = cal(F){x(t)}(omega) = integral_(RR) x(t) e^(-j omega t) dif t
$

Note that this definition is rather restrictive, there exists a less restrained definition on tempered distributions (distributions, specifically related to mathematics and not statistics, are also called generalized functions), but it is outside the scope of this document. Nevertheless, this more complicated definition allows us to show very useful relations such as the following:

#[
  #show math.equation.where(block: true): set align(left)
  #show math.equation.where(block: true): it => {
    box(inset: (left: 1.5em))[
      #it
    ]
  }
  $
    &"‣" thick thick cal(F){delta(t)}(omega) &&= 1 \
    &"‣" thick thick cal(F){1}(omega) &&= 2 pi dot delta(omega) \
    &"‣" thick thick cal(F){u(t)}(omega) &&= pi delta(omega) + 1 slash j omega \
    &"‣" thick thick cal(F){"sign"(t)}(omega) &&= 2 slash j omega \
    &"‣" thick thick cal(F){e^(j omega_0 t)}(omega) &&= 2 pi dot delta(omega - omega_0) \
  $
]

=== The discrete-time Fourier transform (DTFT) <DTFT>

Let $x[n]$ be a discrete signal sampled from its continuous counterpart $x(t)$, with a sampling rate of $f_s$; i.e. $x[n] = x(n slash f_s) = x(n T_s)$.

The discrete-time fourier transform of $x[n]$ is given by:

$
  cal(F)_(d){x[n]}(omega) = sum_(n in ZZ) x[n] e^(-j omega n)
$

We will first show that the DTFT of a discrete signal is $2pi$-periodic, before looking at how to map frequencies correctly.

The discrete signal $x[n]$ mentioned above can be represented in the continuous space as:

$
x[n] stretch(arrow.r, size: #120%)_"cont. repr." x_(T_s)(t) = x(t) dot sum_(n in ZZ) delta(t - n T_s) = sum_(n in ZZ) underbrace(x(n T_s), =x[n]) delta(t - n T_s)
$

With this in mind we can then write the following:

$
cal(F){x_(T_s)(t)}(omega)
&=
cal(F){x_(T_s)(t)}(omega) \

cal(F){sum_(n in ZZ) x[n] delta(t - n T_s)}(omega)
&=
cal(F){x(t) dot sum_(n in ZZ) delta(t - n T_s)}(omega) \

sum_(n in ZZ) x[n] cal(F){delta(t - n T_s)}(omega)
&=
X(omega) convolve cal(F){sum_(n in ZZ) delta(t - n T_s)}(omega) \

sum_(n in ZZ) x[n] e^(-j omega n T_s)
&=
X(omega) convolve omega_s sum_(n in ZZ) delta(omega - n omega_s) \

cal(F)_(d){x[n]}(omega T_s)
&=
underbrace((2pi) / T_s sum_(n in ZZ) X(omega - n (2pi) / T_s), 2pi slash T_s"-periodic")  \
$

If the compressed DTFT of x[n]: $cal(F)_(d){x[n]}(omega T_s)$ is $(2pi)/T_s$-periodic, then it follows that the regular DTFT of a sampled signal: $cal(F)_(d){x[n]}(omega)$ is simply $2pi$-periodic. We can write:

$
arrow.double cal(F)_(d){x[n]}(omega)
&= (2pi) / T_s sum_(n in ZZ) X((omega - n 2pi) / T_s) \
&= underbrace(X(omega f_s) convolve 2pi f_s sum_(n in ZZ) delta(omega - 2pi n), 2pi"-periodic") \
$

Moving on to our second goal, let $x(t) = e^(j 2 pi f t)$ be our continuous signal composed of a single sinusoidal wave oscillating at a frequency of $f$, which is sampled at a certain frequency $f_s$ to create the discrete signal $x[n]$. We can deduce the following:

$
  x[n] = x(n T_s) = x(n slash f_s) = e^(j 2 pi f n slash f_s) = e^(j n omega)
$

Where $omega = 2 pi f slash f_s$, thus the mapping (and its inverse) can be expressed as:

$
  f "[Hz]" arrow.r.long underbrace(2 pi f slash f_s, omega_"DTFT") "[rad/sample]" \
  omega "[rad/sample]" arrow.r.long underbrace(omega f_s slash 2 pi, f_"FT" = omega_"FT" slash 2 pi) "[Hz]"
$

This is coherent, given that the Nyquist frequency ($f_"Nyquist" = f_s slash 2$) maps to $omega=pi$ in the DTFT frequency space: $f_s slash 2 "[Hz]" arrow.l.r.long 2 pi (f_s slash 2)/f_s = pi "[rad/sample]"$. (Note: we expect the Nyquist frequency to map to $pi$ given that it is the largest frequency representable by our sampled signal and the DTFT is $2pi$-periodic).

=== The discrete Fourier transfor (DFT) <DFT>

Let us define a "generic" discrete function as follows:
#[
  #show math.equation.where(block: true): set par(leading: 0.25em)
  $
  x: #h(.5em)
  &&Omega   &arrow.long CC \
  &&n #h(.3em) &arrow.long.bar x[n] \
  $
]

Where $Omega subset ZZ$ and is finite, with $N = |Omega|$. To simplify things, we can reinterpret this discrete function as a sequence, $(x_n)_(n in {0, dots.c, N-1})$ with $x[inf Omega] arrow x_0, dots.c, x[sup Omega] arrow x_(N-1)$. To keep things simple, $x[0]$ will from now on refer to $x_0$, but basically what I am trying to show here is that any generic discrete function like the one above can be "converted" into having a form of ${0, dots.c, abs(Omega)-1} arrow.long CC$. And with that, we can write the discrete Fourier transform as follows:

$
  X[k] = "DFT"{x[n]}[k] = sum_(n=0)^(N-1) x[n] dot e^(-j #h(1pt) 2 pi dot k n slash N)
$

And the inverse transform is given by:

$
  x[n] = "IDFT"{X[k]}[n] = 1/N sum_(m=0)^(N-1) X[k] dot e^(thin j #h(1pt) 2 pi dot k n slash N)
$

Although not proved here, the DFT is simply a sampling of the DTFT:

$
  F[k] = cal(F)_(d){x}(omega=2 pi k slash N)
$

Thus, the distance in actual frequency between each bin, also called the bin width, is given by the following expression:

$
  Delta f

  &= underbrace(2pi dot 1 slash N, Delta omega_"DTFT") dot underbrace(f_s slash 2pi, omega_"DTFT" arrow.r f_"FT") \

  &= f_s slash N \
$

The mappings are as follows this time (trivial to derive from DTFT scenario above):

$
  f "[Hz]" arrow.r.long ^"approx" underbrace(ceil.l f N slash f_s floor.r, k_"DFT") \
  k arrow.r.long underbrace(k f_s slash N, f_"FT") "[Hz]"
$


#pagebreak()



== Binauralization

We are interested in transforming an input audio signal, such that the resulting output audio signal is perceived by a listener to be moving in space around themselves.

Let $arrow(r)(t)$ be the position of the audio source with respect to the head of the listener.

If the source of sound is static (meaning its position is constant with respect to the position of the listener), then our problem becomes trivial, $arrow(r)(t) = arrow(r)_("src") "const."$ And therefore $h(t, tau)$ must be equal to $h_("HRIR," arrow(r)_("src"))(t - tau)$

By extending this logic,

#bibliography("refs.yml", style: "ieee") <bib>

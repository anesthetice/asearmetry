/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Imports
use num_complex::{Complex32, ComplexFloat};
use num_traits::{AsPrimitive, Float, FloatConst, FromPrimitive, NumAssignOps};
use std::{
    borrow::Borrow,
    ops::{Mul, MulAssign},
};
use tap::Pipe;

#[allow(non_camel_case_types)]
pub type cf32 = Complex32;
pub type Seconds = f64;
pub type Radians = f64;
pub type Degrees = f64;
pub type Meters = f64;
pub type Hertz = f64;

pub fn sinc<T: Float + FloatConst>(x: T) -> T {
    if x != T::zero() {
        (x * T::PI()).sin() / (x * T::PI())
    } else {
        T::one()
    }
}

pub fn mean<I, B, T>(input: I) -> T
where
    I: IntoIterator<Item = B>,
    B: Borrow<T>,
    T: Float + FloatConst + NumAssignOps,
{
    let mut sum = T::zero();
    let mut count = T::zero();

    for x in input {
        sum += *x.borrow();
        count += T::one();
    }

    if count != T::zero() {
        sum / count
    } else {
        T::zero()
    }
}

/// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
pub fn std<I, B, T>(input: I, ddof: u64) -> T
where
    I: IntoIterator<Item = B>,
    B: Borrow<T>,
    T: Float + FloatConst + FromPrimitive + NumAssignOps,
{
    let mut mean = T::zero();
    let mut m2 = T::zero();
    let mut count = T::zero();
    let ddof = T::from_u64(ddof).expect("The provided `ddof` cannot be converted to a float");

    for x in input {
        let x = *x.borrow();
        count += T::zero();

        let delta = x - mean;
        mean += delta / count;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }

    if count != T::zero() && count > ddof {
        (m2 / (count - ddof)).sqrt()
    } else if count == T::zero() {
        T::zero()
    } else {
        panic!("Input must have more elements than the ddof")
    }
}

pub fn absmax<I, B, T>(input: I) -> T
where
    I: IntoIterator<Item = B>,
    B: Borrow<T>,
    T: Float,
{
    input
        .into_iter()
        .map(|x| x.borrow().abs())
        .reduce(T::max)
        .unwrap()
}

#[allow(non_snake_case)]
pub fn hann_window_iter<T>(length: usize) -> impl Iterator<Item = T>
where
    T: 'static + Float + FloatConst,
    usize: AsPrimitive<T>,
{
    assert!(length > 0);
    // Elsewhere we use N = "number of samples", i.e. n ∈ {0, ⋯, N-1},
    // but here, we assume that N + 1 = "number of samples", i.e. n ∈ {0, ⋯, N}.
    let N = (length - 1).as_();
    (0..length).map(move |n| T::sin(n.as_() * T::PI() / N).powi(2))
}

pub fn apply_hann_window<T>(input: &mut [T])
where
    T: 'static + ComplexFloat + MulAssign<T::Real>,
    usize: AsPrimitive<T::Real>,
{
    let window_length = input.len();
    input
        .iter_mut()
        .zip(hann_window_iter::<T::Real>(window_length))
        .for_each(|(x, hann)| x.mul_assign(hann));
}

#[allow(non_snake_case)]
pub fn apply_tukey_window<T>(input: &mut [T], alpha: f64)
where
    T: 'static + ComplexFloat + MulAssign<T::Real>,
    usize: AsPrimitive<T::Real>,
{
    assert!(alpha >= 0.0 && alpha <= 1.0);
    let window_length = input.len();
    let length_per_lobe = (window_length as f64 * alpha).round() as usize / 2;

    input
        .split_at_mut(window_length - length_per_lobe)
        .pipe(|(lm, r)| {
            debug_assert!(r.len() == length_per_lobe);
            lm[..length_per_lobe].iter_mut().chain(r.iter_mut())
        })
        .zip(hann_window_iter::<T::Real>(2 * length_per_lobe))
        .for_each(|(x, hann)| x.mul_assign(hann));
}

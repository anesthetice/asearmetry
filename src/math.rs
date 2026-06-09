/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Imports
use num_traits::{Float, FloatConst, FromPrimitive, NumAssignOps};
use std::borrow::Borrow;

pub type Seconds = f64;
pub type Radians = f64;
pub type Meters = f64;
pub type Hertz = f64;

pub fn sinc<T: Float + FloatConst>(x: T) -> T {
    if x != T::zero() {
        (x * T::PI()).sin() / (x * T::PI())
    } else {
        T::one()
    }
}

pub fn mean<I, B, T>(data: I) -> T
where
    I: IntoIterator<Item = B>,
    B: Borrow<T>,
    T: Float + FloatConst + NumAssignOps,
{
    let mut sum = T::zero();
    let mut count = T::zero();

    for x in data {
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
pub fn std<I, B, T>(data: I, ddof: u64) -> T
where
    I: IntoIterator<Item = B>,
    B: Borrow<T>,
    T: Float + FloatConst + FromPrimitive + NumAssignOps,
{
    let mut mean = T::zero();
    let mut m2 = T::zero();
    let mut count = T::zero();
    let ddof = T::from_u64(ddof).expect("The provided `ddof` cannot be converted to a float");

    for x in data {
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

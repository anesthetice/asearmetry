/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

use approx::{AbsDiffEq, RelativeEq};

use super::{clamp_azimuth, clamp_zenith};
use crate::coordinates::{Cart3D, Sphere3D, write_float};
use crate::math::{Degrees, Meters, Radians};
use std::f64::consts::{GOLDEN_RATIO, PI, TAU};

#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Shell2D {
    /// The azimuth angle. θ ∈ (-π, +π]
    pub θ: Radians,
    /// The zenith angle. φ ∈ [0, +π]
    /// Note that φ=0 corresponds to the -z direction, while φ=π corresponds to the +z direction.
    pub φ: Radians,
}

impl std::fmt::Debug for Shell2D {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(θ: ")?;
        write_float(self.θ, f, true)?;

        write!(f, ", φ: ")?;
        write_float(self.φ, f, true)?;
        write!(f, ")")?;

        Ok(())
    }
}

impl std::fmt::Display for Shell2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self, f)
    }
}

impl Shell2D {
    pub fn new(θ: Radians, φ: Radians) -> Self {
        Self { θ, φ }
    }

    pub fn new_from_degrees(θ: Degrees, φ: Degrees) -> Self {
        Self {
            θ: TAU * θ / 360.0,
            φ: TAU * φ / 360.0,
        }
    }

    pub fn clamp_angles_in_place(&mut self) {
        self.θ = clamp_azimuth(self.θ);
        self.φ = clamp_zenith(self.φ)
    }

    pub fn clamp_angles(mut self) -> Self {
        self.clamp_angles_in_place();
        self
    }

    /// See https://observablehq.com/@meetamit/fibonacci-lattices
    #[allow(non_upper_case_globals)]
    pub fn generate_fib_lattice(n_points: usize) -> Vec<Self> {
        // golden ratio
        const Φ: f64 = GOLDEN_RATIO;
        const ε: f64 = 0.4;

        let n = n_points as f64;

        (0..n_points)
            .map(|i| i as f64)
            .map(|i| {
                let (x_i, y_i) = ((i / Φ) % 1.0, (i + ε) / (n - 1.0 + 2.0 * ε));
                Shell2D::new(clamp_azimuth(2.0 * PI * x_i), f64::acos(1.0 - 2.0 * y_i))
            })
            .collect()
    }

    pub fn dist_angular(&self, other: Self) -> Radians {
        self.into_unit_vec_cart()
            .angle_with(other.into_unit_vec_cart())
    }

    /* Still somewhat lazy
    pub fn dist_angular(&self, other: Self) -> Radians {
        let a = self.into_unit_vec_cart();
        let b = other.into_unit_vec_cart();

        f64::acos(a.dot(b).clamp(0.0, 1.0))
    }
    */

    /* The lazy man's angular distance (meaning incorrect)
    pub fn dist_angular(&self, other: Self) -> f64 {
        let θ_diff = f64::min((a.θ - b.θ).abs(), TAU - (a.θ - b.θ).abs());
        debug_assert!(θ_diff < PI + f64::EPSILON);

        let φ_diff = a.φ - b.φ;
        debug_assert!(φ_diff < PI + f64::EPSILON);

        (θ_diff.powi(2) + φ_diff.powi(2)).sqrt()
    }
    */

    pub fn is_nan(&self) -> bool {
        self.θ.is_nan() || self.φ.is_nan()
    }

    pub fn is_finite(&self) -> bool {
        self.θ.is_finite() && self.φ.is_finite()
    }

    pub fn is_null(&self) -> bool {
        // Under IEEE 754, -0.0 is equal to 0.0
        self.θ == 0.0 && self.φ == 0.0
    }

    pub fn into_spherical(self, r: Meters) -> Sphere3D {
        Sphere3D {
            r,
            θ: self.θ,
            φ: self.φ,
        }
    }

    pub fn into_unit_vec_cart(self) -> Cart3D {
        Cart3D {
            x: self.φ.sin() * self.θ.cos(),
            y: self.φ.sin() * self.θ.sin(),
            z: -self.φ.cos(),
        }
    }
}

impl Default for Shell2D {
    fn default() -> Self {
        Self {
            θ: 0.0,
            φ: PI / 2.0,
        }
    }
}

impl From<(Radians, Radians)> for Shell2D {
    fn from(v: (Radians, Radians)) -> Self {
        Self { θ: v.0, φ: v.1 }
    }
}

impl From<Shell2D> for [Radians; 2] {
    fn from(value: Shell2D) -> Self {
        [value.θ, value.φ]
    }
}

impl From<[Radians; 2]> for Shell2D {
    fn from(value: [Radians; 2]) -> Self {
        Self {
            θ: value[0],
            φ: value[1],
        }
    }
}

impl From<&[Radians; 2]> for Shell2D {
    fn from(value: &[Radians; 2]) -> Self {
        Self {
            θ: value[0],
            φ: value[1],
        }
    }
}

impl From<Sphere3D> for Shell2D {
    fn from(v: Sphere3D) -> Self {
        Self { θ: v.θ, φ: v.φ }
    }
}

impl From<Cart3D> for Shell2D {
    fn from(v: Cart3D) -> Self {
        let r = (v.x.powi(2) + v.y.powi(2) + v.z.powi(2)).sqrt();
        Self {
            θ: v.y.atan2(v.x),
            φ: PI - (v.z / r).acos(),
        }
    }
}

impl rstar::RTreeObject for Shell2D {
    type Envelope = rstar::AABB<[Radians; 2]>;
    fn envelope(&self) -> Self::Envelope {
        rstar::AABB::from_point([self.θ, self.φ])
    }
}

impl rstar::PointDistance for Shell2D {
    fn distance_2(
        &self,
        point: &<Self::Envelope as rstar::Envelope>::Point,
    ) -> <<Self::Envelope as rstar::Envelope>::Point as rstar::Point>::Scalar {
        self.dist_angular(point.into())
    }
}

impl AbsDiffEq for Shell2D {
    type Epsilon = f64;
    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f64::abs_diff_eq(&self.θ, &other.θ, epsilon) && f64::abs_diff_eq(&self.φ, &other.φ, epsilon)
    }
}

impl RelativeEq for Shell2D {
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        f64::relative_eq(&self.θ, &other.θ, epsilon, max_relative)
            && f64::relative_eq(&self.φ, &other.φ, epsilon, max_relative)
    }
}

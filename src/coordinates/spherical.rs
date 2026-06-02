use crate::{
    coordinates::Cart3D,
    math::{Meters, Radians},
};
use approx::{AbsDiffEq, RelativeEq};
use std::f64::consts::{PI, TAU};

#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Sphere3D {
    /// The radial distance. r ∈ ℝ₊
    pub r: Meters,
    /// The azimuth angle. θ ∈ (-π, +π]
    pub θ: Radians,
    /// The zenith angle. φ ∈ [0, +π]
    /// Note that φ=0 corresponds to the -z direction, while φ=π corresponds to the +z direction.
    pub φ: Radians,
}

impl std::fmt::Debug for Sphere3D {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let write_float = |float: f64, f: &mut std::fmt::Formatter<'_>| {
            if float.abs() > 1E-3 { write!(f, "{:.3}", float) }
            else if float.abs() < 1E-6 { write!(f, "0.0") }
            else { write!(f, "{:.1E}", float) }
        };

        write!(f, "(r: ")?;
        write_float(self.r, f)?;

        let θ_opi = self.θ / PI;
        write!(f, ", θ: π⋅")?;
        write_float(θ_opi, f)?;

        let φ_opi = self.φ / PI;
        write!(f, ", φ: π⋅")?;
        write_float(φ_opi, f)?;
        write!(f, ")")?;

        Ok(())
    }
}

impl std::fmt::Display for Sphere3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self, f)
    }
}

impl Sphere3D {
    pub fn new(r: Meters, θ: Radians, φ: Radians) -> Self {
        Self { r, θ, φ }
    }

    pub fn to_shell_point(&self) -> [Radians; 2] {
        [self.θ, self.φ]
    }

    pub fn theta(&self) -> Radians {
        self.θ
    }

    pub fn theta_mut(&mut self) -> &mut Radians {
        &mut self.θ
    }

    pub fn azimuth(&self) -> Radians {
        self.θ
    }

    pub fn azimuth_mut(&mut self) -> &mut Radians {
        &mut self.θ
    }

    pub fn phi(&self) -> Radians {
        self.φ
    }

    pub fn phi_mut(&mut self) -> &mut Radians {
        &mut self.φ
    }

    pub fn zenith(&self) -> Radians {
        self.φ
    }

    pub fn zenith_mut(&mut self) -> &mut Radians {
        &mut self.φ
    }

    pub fn are_angles_clamped(&self) -> bool {
        self.θ <= -PI || self.θ > PI || self.φ < 0.0 || self.φ > PI
    }

    pub fn clamp_angles_in_place(&mut self) {
        self.θ = clamp_azimuth(self.θ);
        self.φ = clamp_zenith(self.φ)
    }

    pub fn clamp_angles(mut self) -> Self {
        self.clamp_angles_in_place();
        self
    }

    pub fn is_nan(&self) -> bool {
        self.r.is_nan() || self.θ.is_nan() || self.φ.is_nan()
    }

    pub fn is_finite(&self) -> bool {
        self.r.is_finite() && self.θ.is_finite() && self.φ.is_finite()
    }

    pub fn is_null(&self) -> bool {
        // Under IEEE 754, -0.0 is equal to 0.0
        self.r == 0.0 && self.θ == 0.0 && self.φ == 0.0
    }
}

impl From<Cart3D> for Sphere3D {
    fn from(v: Cart3D) -> Self {
        let r = (v.x.powi(2) + v.y.powi(2) + v.z.powi(2)).sqrt();
        Self {
            r,
            θ: v.y.atan2(v.x),
            φ: PI - (v.z / r).acos(),
        }
    }
}

impl From<(f64, f64, f64)> for Sphere3D {
    fn from(v: (f64, f64, f64)) -> Self {
        Self {
            r: v.0,
            θ: v.1,
            φ: v.2,
        }
    }
}

impl From<Sphere3D> for (f64, f64, f64) {
    fn from(v: Sphere3D) -> Self {
        (v.r, v.θ, v.φ)
    }
}

impl<T: Into<Self>> std::ops::AddAssign<T> for Sphere3D {
    fn add_assign(&mut self, rhs: T) {
        let rhs = rhs.into();
        self.r += rhs.r;
        self.θ += rhs.θ;
        self.φ += rhs.φ;
    }
}

impl<T: Into<Self>> std::ops::Add<T> for Sphere3D {
    type Output = Self;
    fn add(mut self, rhs: T) -> Self::Output {
        self += rhs.into();
        self
    }
}

impl std::ops::MulAssign<f64> for Sphere3D {
    fn mul_assign(&mut self, rhs: f64) {
        self.r *= rhs;
        self.θ *= rhs;
        self.φ *= rhs;
    }
}

impl std::ops::Mul<f64> for Sphere3D {
    type Output = Self;
    fn mul(mut self, rhs: f64) -> Self::Output {
        self *= rhs;
        self
    }
}

impl AbsDiffEq for Sphere3D {
    type Epsilon = f64;
    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f64::abs_diff_eq(&self.r, &other.r, epsilon)
            && f64::abs_diff_eq(&self.θ, &other.θ, epsilon)
            && f64::abs_diff_eq(&self.φ, &other.φ, epsilon)
    }
}

impl RelativeEq for Sphere3D {
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        f64::relative_eq(&self.r, &other.r, epsilon, max_relative)
            && f64::relative_eq(&self.θ, &other.θ, epsilon, max_relative)
            && f64::relative_eq(&self.φ, &other.φ, epsilon, max_relative)
    }
}

pub fn clamp_azimuth(mut θ: Radians) -> Radians {
    if θ <= -PI || θ > PI {
        θ -= θ.signum() * TAU * ((θ.abs() - PI) / TAU).ceil()
    }
    θ
}

pub fn clamp_zenith(mut φ: Radians) -> Radians {
    #[allow(clippy::manual_range_contains)]
    if φ < 0.0 || φ > PI {
        φ = φ.abs() - TAU * (φ.abs() / TAU).floor();
        if φ > PI {
            φ = TAU - φ;
        }
    }
    φ
}

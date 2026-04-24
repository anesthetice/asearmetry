use crate::{Meters, Radians, coordinates::Cart3D};
use approx::{AbsDiffEq, RelativeEq};
use std::f32::consts::{PI, TAU};

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
        write!(f, "")?;
        if self.r.abs() == 0.0 { write!(f, "(r: 0.0")? }
        else if self.r.abs() > 0.001 { write!(f, "(r: {:.3}", self.r)? }
        else { write!(f, "(r: {:.1E}", self.r)? };

        let θ_opi = self.θ / PI;
        if θ_opi.abs() == 0.0 { write!(f, ", θ: 0.0")? }
        else if θ_opi.abs() > 0.001 { write!(f, ", θ: π⋅{θ_opi:.3}")? }
        else { write!(f, ", θ: π⋅{θ_opi:.1E}")? };

        let φ_opi = self.φ / PI;
        if φ_opi.abs() == 0.0 { write!(f, ", φ: 0.0)")? }
        if φ_opi.abs() > 0.001 { write!(f, ", φ: π⋅{φ_opi:.3})")? }
        else { write!(f, ", φ: π⋅{φ_opi:.1E})")? };

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

    pub fn to_shell_point(&self) -> [f32; 2] {
        [self.θ, self.φ]
    }

    pub fn theta(&self) -> f32 {
        self.θ
    }

    pub fn theta_mut(&mut self) -> &mut f32 {
        &mut self.θ
    }

    pub fn azimuth(&self) -> f32 {
        self.θ
    }

    pub fn azimuth_mut(&mut self) -> &mut f32 {
        &mut self.θ
    }

    pub fn phi(&self) -> f32 {
        self.φ
    }

    pub fn phi_mut(&mut self) -> &mut f32 {
        &mut self.φ
    }

    pub fn zenith(&self) -> f32 {
        self.φ
    }

    pub fn zenith_mut(&mut self) -> &mut f32 {
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
}

impl From<Cart3D> for Sphere3D {
    fn from(v: Cart3D) -> Self {
        let r = (v.x.powi(2) + v.y.powi(2) + v.z.powi(2)).sqrt();
        if r < 1E-6 {
            println!("PANICAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        }
        Self {
            r,
            θ: v.y.atan2(v.x),
            φ: PI - (v.z / r).acos(),
        }
    }
}

impl From<(f32, f32, f32)> for Sphere3D {
    fn from(v: (f32, f32, f32)) -> Self {
        Self {
            r: v.0,
            θ: v.1,
            φ: v.2,
        }
    }
}

impl From<Sphere3D> for (f32, f32, f32) {
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

impl std::ops::MulAssign<f32> for Sphere3D {
    fn mul_assign(&mut self, rhs: f32) {
        self.r *= rhs;
        self.θ *= rhs;
        self.φ *= rhs;
    }
}

impl std::ops::Mul<f32> for Sphere3D {
    type Output = Self;
    fn mul(mut self, rhs: f32) -> Self::Output {
        self *= rhs;
        self
    }
}

impl AbsDiffEq for Sphere3D {
    type Epsilon = f32;
    fn default_epsilon() -> Self::Epsilon {
        f32::EPSILON
    }
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f32::abs_diff_eq(&self.r, &other.r, epsilon)
            && f32::abs_diff_eq(&self.θ, &other.θ, epsilon)
            && f32::abs_diff_eq(&self.φ, &other.φ, epsilon)
    }
}

impl RelativeEq for Sphere3D {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        f32::relative_eq(&self.r, &other.r, epsilon, max_relative)
            && f32::relative_eq(&self.θ, &other.θ, epsilon, max_relative)
            && f32::relative_eq(&self.φ, &other.φ, epsilon, max_relative)
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

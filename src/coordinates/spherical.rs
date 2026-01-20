use approx::{AbsDiffEq, RelativeEq};
use std::f32::consts::{PI, TAU};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere3D {
    /// The radial distance. r ∈ ℝ
    pub r: f32,
    /// The azimuth angle. θ ∈ (-π, +π]
    pub θ: f32,
    /// The zenith angle. φ ∈ [0, +π]
    /// Note that φ=0 corresponds to the -z direction, while φ=π corresponds to the +z direction.
    pub φ: f32,
}

impl Sphere3D {
    pub fn new(r: f32, θ: f32, φ: f32) -> Self {
        Self { r, θ, φ }
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

    pub fn adjust_angles(&mut self) {
        if self.θ <= -PI || self.θ > PI {
            self.θ -= self.θ.signum() * TAU * ((self.θ.abs() - PI) / TAU).ceil()
        }
        if self.φ < 0.0 || self.φ > PI {
            self.φ = self.φ.abs() - TAU * (self.φ.abs() / TAU).floor();
            if self.φ > PI {
                self.φ = TAU - self.φ;
            }
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

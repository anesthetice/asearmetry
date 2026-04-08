use crate::{Meters, coordinates::Sphere3D};
use approx::{AbsDiffEq, RelativeEq};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cart3D {
    pub x: Meters,
    pub y: Meters,
    pub z: Meters,
}

impl Cart3D {
    pub fn new(x: Meters, y: Meters, z: Meters) -> Self {
        Self { x, y, z }
    }

    pub fn dist(&self, other: Cart3D) -> Meters {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2))
            .sqrt()
    }

    pub fn is_nan(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.y.is_nan()
    }
}

impl From<Sphere3D> for Cart3D {
    fn from(v: Sphere3D) -> Self {
        let r_xy = v.r * v.φ.sin();
        Self {
            x: r_xy * v.θ.cos(),
            y: r_xy * v.θ.sin(),
            z: -v.r * v.φ.cos(),
        }
    }
}

impl From<(Meters, Meters, Meters)> for Cart3D {
    fn from(v: (Meters, Meters, Meters)) -> Self {
        Self {
            x: v.0,
            y: v.1,
            z: v.2,
        }
    }
}

impl From<Cart3D> for (Meters, Meters, Meters) {
    fn from(v: Cart3D) -> Self {
        (v.x, v.y, v.z)
    }
}

impl<T: Into<Self>> std::ops::AddAssign<T> for Cart3D {
    fn add_assign(&mut self, rhs: T) {
        let rhs = rhs.into();
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl<T: Into<Self>> std::ops::Add<T> for Cart3D {
    type Output = Self;
    fn add(mut self, rhs: T) -> Self::Output {
        self += rhs.into();
        self
    }
}

impl std::ops::MulAssign<f32> for Cart3D {
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl std::ops::Mul<f32> for Cart3D {
    type Output = Self;
    fn mul(mut self, rhs: f32) -> Self::Output {
        self *= rhs;
        self
    }
}

impl AbsDiffEq for Cart3D {
    type Epsilon = f32;
    fn default_epsilon() -> Self::Epsilon {
        f32::EPSILON
    }
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f32::abs_diff_eq(&self.x, &other.x, epsilon)
            && f32::abs_diff_eq(&self.y, &other.y, epsilon)
            && f32::abs_diff_eq(&self.z, &other.z, epsilon)
    }
}

impl RelativeEq for Cart3D {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        f32::relative_eq(&self.x, &other.x, epsilon, max_relative)
            && f32::relative_eq(&self.y, &other.y, epsilon, max_relative)
            && f32::relative_eq(&self.z, &other.z, epsilon, max_relative)
    }
}

use crate::{coordinates::Sphere3D, math::Meters};
use approx::{AbsDiffEq, RelativeEq};

#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Cart3D {
    pub x: Meters,
    pub y: Meters,
    pub z: Meters,
}

impl std::fmt::Debug for Cart3D {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let write_float = |float: f64, f: &mut std::fmt::Formatter<'_>| {
            if float.abs() > 1E-3 { write!(f, "{:.3}", float) }
            else if float.abs() < 1E-6 { write!(f, "0.0") }
            else { write!(f, "{:.1E}", float) }
        };

        write!(f, "(x: ")?;
        write_float(self.x, f)?;

        write!(f, ", y: ")?;
        write_float(self.y, f)?;

        write!(f, ", z: ")?;
        write_float(self.z, f)?;
        write!(f, ")")?;

        Ok(())
    }
}

impl std::fmt::Display for Cart3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self, f)
    }
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
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }

    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    pub fn is_null(&self) -> bool {
        // Under IEEE 754, -0.0 is equal to 0.0
        self.x == 0.0 && self.y == 0.0 && self.z == 0.0
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

impl std::ops::MulAssign<f64> for Cart3D {
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl std::ops::Mul<f64> for Cart3D {
    type Output = Self;
    fn mul(mut self, rhs: f64) -> Self::Output {
        self *= rhs;
        self
    }
}

impl AbsDiffEq for Cart3D {
    type Epsilon = f64;
    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f64::abs_diff_eq(&self.x, &other.x, epsilon)
            && f64::abs_diff_eq(&self.y, &other.y, epsilon)
            && f64::abs_diff_eq(&self.z, &other.z, epsilon)
    }
}

impl RelativeEq for Cart3D {
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        f64::relative_eq(&self.x, &other.x, epsilon, max_relative)
            && f64::relative_eq(&self.y, &other.y, epsilon, max_relative)
            && f64::relative_eq(&self.z, &other.z, epsilon, max_relative)
    }
}

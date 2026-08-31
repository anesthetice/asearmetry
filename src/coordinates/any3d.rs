use crate::{
    coordinates::{Cart3D, Sphere3D},
    math::{Meters, Radians},
};

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Coord3D {
    Cart(Cart3D),
    Sphere(Sphere3D),
}

impl From<Cart3D> for Coord3D {
    fn from(value: Cart3D) -> Self {
        Self::Cart(value)
    }
}

impl From<Sphere3D> for Coord3D {
    fn from(value: Sphere3D) -> Self {
        Self::Sphere(value)
    }
}

impl Coord3D {
    pub fn new_cart(x: Meters, y: Meters, z: Meters) -> Self {
        Self::Cart(Cart3D::new(x, y, z))
    }

    pub fn new_sphere(r: Meters, θ: Radians, φ: Radians) -> Self {
        Self::Sphere(Sphere3D::new(r, θ, φ))
    }

    pub fn as_cart(&self) -> Cart3D {
        (*self).into()
    }

    pub fn as_sphere(&self) -> Sphere3D {
        (*self).into()
    }
}

impl From<Coord3D> for Cart3D {
    fn from(value: Coord3D) -> Self {
        match value {
            Coord3D::Cart(cart_3d) => cart_3d,
            Coord3D::Sphere(sphere_3d) => sphere_3d.into(),
        }
    }
}

impl From<Coord3D> for Sphere3D {
    fn from(value: Coord3D) -> Self {
        match value {
            Coord3D::Cart(cart_3d) => cart_3d.into(),
            Coord3D::Sphere(sphere_3d) => sphere_3d,
        }
    }
}

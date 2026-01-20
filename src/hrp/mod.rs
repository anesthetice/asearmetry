use crate::{coordinates::Sphere3D, signal::Signal};

pub struct HRIR {
    pub position: Sphere3D,
    pub ir: Signal<2>,
}

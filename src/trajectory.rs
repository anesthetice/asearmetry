use crate::{
    IntoSeconds, Seconds,
    coordinates::{Cart3D, Sphere3D},
};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Trajectory<T> {
    pub path: Vec<T>,
    pub δt: Seconds,
}

impl<T> Trajectory<T> {
    pub fn delta_t(&self) -> Seconds {
        self.δt
    }

    pub fn from_equations<T1: IntoSeconds, T2: IntoSeconds>(
        mut eq: impl FnMut(Seconds) -> T,
        t_tot: T1,
        δt: T2,
    ) -> Self {
        let δt = δt.to_seconds();
        let t_tot = t_tot.to_seconds();

        let path_len = (t_tot / δt).ceil() as usize;
        let mut path = Vec::with_capacity(path_len);

        for i in 0..(path_len as u32) {
            path.push(eq(δt * i as f32));
        }

        Self { path, δt }
    }
}

impl Trajectory<Cart3D> {
    pub fn from_diff_equations<T1: IntoSeconds, T2: IntoSeconds>(
        eq: impl Fn(Cart3D) -> Cart3D,
        init: Cart3D,
        t_tot: T1,
        δt: T2,
    ) -> Self {
        let δt = δt.to_seconds();
        let t_tot = t_tot.to_seconds();

        let path_len = (t_tot / δt).ceil() as usize;
        let mut path = Vec::with_capacity(path_len);

        let mut position = init;
        path.push(position);

        for _ in 1..(path_len as u32) {
            position += eq(position) * δt;
            path.push(position);
        }

        Self { path, δt }
    }
}

impl Trajectory<Sphere3D> {
    pub fn from_diff_equations<T1: IntoSeconds, T2: IntoSeconds>(
        eq: impl Fn(Sphere3D) -> Sphere3D,
        init: Sphere3D,
        t_tot: T1,
        δt: T2,
    ) -> Self {
        let δt = δt.to_seconds();
        let t_tot = t_tot.to_seconds();

        let path_len = (t_tot / δt).ceil() as usize;
        let mut path = Vec::with_capacity(path_len);

        let mut position = init;
        path.push(position);

        for _ in 1..(path_len as u32) {
            position += eq(position) * δt;
            // TODO: maybe add constraints here?
            path.push(position);
        }

        Self { path, δt }
    }
}

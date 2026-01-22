#![allow(mixed_script_confusables)]

// Modules
pub mod brp;
pub mod coordinates;
pub mod io;
pub mod signal;
pub mod trajectory;

pub type Seconds = f32;
pub type Radians = f32;

pub trait IntoSeconds {
    fn to_seconds(self) -> Seconds;
}

impl IntoSeconds for f32 {
    fn to_seconds(self) -> Seconds {
        self
    }
}

impl IntoSeconds for f64 {
    fn to_seconds(self) -> Seconds {
        self as f32
    }
}

impl IntoSeconds for std::time::Duration {
    fn to_seconds(self) -> Seconds {
        self.as_secs_f32()
    }
}

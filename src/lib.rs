#![allow(mixed_script_confusables)]

// Modules
pub mod audio;
pub mod brp;
pub mod coordinates;
pub mod stat;
pub mod trajectory;

pub type Seconds = f32;
pub type Radians = f32;
pub type Meters = f32;
pub type Hertz = f32;

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

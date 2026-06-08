use crate::{
    audio::{AudioBuffer, AudioBufferSlice, AudioSignal},
    coordinates::Cart3D,
    math::Seconds,
    trajectory::Trajectory,
};

pub struct Scene<'a> {
    pub(crate) elements: Vec<SceneElement<'a>>,
    pub(crate) duration: Seconds,
}

pub struct SceneElement<'a> {
    /// The associated mono audio source of the element to be binauralized and processed.
    pub(crate) source: AudioBufferSlice<'a, 1>,
    /// The trajectory of the element, its duration should be lower or equal to that of the source.
    pub(crate) trajectory: Trajectory<Cart3D>,
    /// The absolute time delay between the start of the scene and when this element should play.
    pub(crate) time_delay: Seconds,
    pub(crate) post_binauralization_modifiers: Vec<Modifiers>,

    /// The name of the element, should be kept short, for debugging or visualization purposes.
    pub(crate) name: Option<String>,
    /// For visualization only, what shape represents the element.
    #[cfg(feature = "viewer")]
    pub(crate) shape: Option<ActorShape>,
    /// For visualization only, what color is the shape that represents the element.
    #[cfg(feature = "viewer")]
    pub(crate) color: Option<String>,
}

pub enum ActorShape {
    Square,
    Circle,
}

pub enum Modifiers {}

impl Modifiers {
    fn apply_to<A: AudioSignal<1>>(&self, audio: A) -> AudioBuffer<1> {
        match self {
            _ => audio.into_owned(),
        }
    }
}

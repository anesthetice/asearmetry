/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

#[cfg(feature = "viewer")]
use crate::scene::core::ActorShape;
use crate::{
    coordinates::{Cart3D, Coord3D},
    math::Seconds,
    scene::core::Modifiers,
    signal::audio::AudioBufferSlice,
    trajectory::Trajectory,
};
use petgraph::Directed;
use tap::Tap;

#[derive(Debug, Clone)]
pub enum Action {
    /// Delay (pause) the audio source for `duration` (index is not advanced)
    Delay { duration: Seconds },

    /// Reset audio source index to the beginning
    Reset,

    /// Adjust the audio source index, e.g., AdjustTime { by: -3.0 } will go back 3 seconds into the past
    AdjustTime { by: Seconds },

    Fixed {
        /// If set to `None`, will try to inherit from last previous position
        position: Option<Coord3D>,
        duration: Seconds,
        modifiers: Option<Vec<Modifiers>>,
        name: Option<String>,
        #[cfg(feature = "viewer")]
        shape: Option<ActorShape>,
        #[cfg(feature = "viewer")]
        color: Option<String>,
    },

    Move {
        /// If set to `None`, will try to inherit from last previous position
        position_1: Option<Coord3D>,
        position_2: Coord3D,
        duration: Seconds,
        modifiers: Option<Vec<Modifiers>>,
        name: Option<String>,
        #[cfg(feature = "viewer")]
        shape: Option<ActorShape>,
        #[cfg(feature = "viewer")]
        color: Option<String>,
    },

    Trajectory {
        inner: Trajectory<Cart3D>,
        modifiers: Option<Vec<Modifiers>>,
        name: Option<String>,
        #[cfg(feature = "viewer")]
        shape: Option<ActorShape>,
        #[cfg(feature = "viewer")]
        color: Option<String>,
    },

    Sequential {
        elements: Vec<Self>,
        modifiers: Option<Vec<Modifiers>>,
        name: Option<String>,
        #[cfg(feature = "viewer")]
        shape: Option<ActorShape>,
        #[cfg(feature = "viewer")]
        color: Option<String>,
    },

    Parallel {
        elements: Vec<Self>,
        modifiers: Option<Vec<Modifiers>>,
        name: Option<String>,
        #[cfg(feature = "viewer")]
        shape: Option<ActorShape>,
        #[cfg(feature = "viewer")]
        color: Option<String>,
    },
}

impl Action {
    pub fn set_name(&mut self, new_name: String) -> &mut Self {
        match self {
            Self::Fixed { name, .. } => name.replace(new_name),
            Self::Move { name, .. } => name.replace(new_name),
            Self::Trajectory { name, .. } => name.replace(new_name),
            Self::Sequential { name, .. } => name.replace(new_name),
            Self::Parallel { name, .. } => name.replace(new_name),
            Self::Delay { .. } | Self::Reset | Self::AdjustTime { .. } => None,
        };
        self
    }

    pub fn with_name(mut self, new_name: String) -> Self {
        self.set_name(new_name);
        self
    }

    #[cfg(feature = "viewer")]
    pub fn set_shape(&mut self, new_shape: ActorShape) -> &mut Self {
        match self {
            Self::Fixed { shape, .. } => shape.replace(new_shape),
            Self::Move { shape, .. } => shape.replace(new_shape),
            Self::Trajectory { shape, .. } => shape.replace(new_shape),
            Self::Sequential { shape, .. } => shape.replace(new_shape),
            Self::Parallel { shape, .. } => shape.replace(new_shape),
            Self::Delay { .. } | Self::Reset | Self::AdjustTime { .. } => None,
        };
        self
    }

    #[cfg(feature = "viewer")]
    pub fn with_shape(mut self, new_shape: ActorShape) -> Self {
        self.set_shape(new_shape);
        self
    }

    #[cfg(feature = "viewer")]
    pub fn set_color(&mut self, new_color: String) -> &mut Self {
        match self {
            Self::Fixed { color, .. } => color.replace(new_color),
            Self::Move { color, .. } => color.replace(new_color),
            Self::Trajectory { color, .. } => color.replace(new_color),
            Self::Sequential { color, .. } => color.replace(new_color),
            Self::Parallel { color, .. } => color.replace(new_color),
            Self::Delay { .. } | Self::Reset | Self::AdjustTime { .. } => None,
        };
        self
    }

    #[cfg(feature = "viewer")]
    pub fn with_color(mut self, new_color: String) -> Self {
        self.set_color(new_color);
        self
    }

    pub fn delay<T: num_traits::AsPrimitive<Seconds>>(duration: T) -> Self {
        Self::Delay {
            duration: seconds_parse_positive(duration),
        }
    }

    /// Alias for `Action::delay`
    pub fn pause<T: num_traits::AsPrimitive<Seconds>>(duration: T) -> Self {
        Self::delay(duration)
    }

    pub fn reset() -> Self {
        Self::Reset
    }

    pub fn adjust_time<T: num_traits::AsPrimitive<Seconds>>(by: T) -> Self {
        Self::AdjustTime {
            by: seconds_parse(by),
        }
    }

    pub fn fixed_at_new_position<T: num_traits::AsPrimitive<Seconds>>(
        position: impl Into<Coord3D>,
        duration: T,
    ) -> Self {
        Self::Fixed {
            position: Some(position.into()),
            duration: seconds_parse_positive(duration),
            modifiers: None,
            name: None,
            #[cfg(feature = "viewer")]
            shape: None,
            #[cfg(feature = "viewer")]
            color: None,
        }
    }

    pub fn fixed_at_prev_position<T: num_traits::AsPrimitive<Seconds>>(duration: T) -> Self {
        Self::Fixed {
            position: None,
            duration: seconds_parse_positive(duration),
            modifiers: None,
            name: None,
            #[cfg(feature = "viewer")]
            shape: None,
            #[cfg(feature = "viewer")]
            color: None,
        }
    }

    pub fn move_from_to<T: num_traits::AsPrimitive<Seconds>>(
        pos_1: impl Into<Coord3D>,
        pos_2: impl Into<Coord3D>,
        duration: T,
    ) -> Self {
        Self::Move {
            position_1: Some(pos_1.into()),
            position_2: pos_2.into(),
            duration: seconds_parse_positive(duration),
            modifiers: None,
            name: None,
            #[cfg(feature = "viewer")]
            shape: None,
            #[cfg(feature = "viewer")]
            color: None,
        }
    }

    pub fn move_to<T: num_traits::AsPrimitive<Seconds>>(
        pos: impl Into<Coord3D>,
        duration: T,
    ) -> Self {
        Self::Move {
            position_1: None,
            position_2: pos.into(),
            duration: seconds_parse_positive(duration),
            modifiers: None,
            name: None,
            #[cfg(feature = "viewer")]
            shape: None,
            #[cfg(feature = "viewer")]
            color: None,
        }
    }

    pub fn trajectory(trajectory: Trajectory<Cart3D>) -> Self {
        Self::Trajectory {
            inner: trajectory,
            modifiers: None,
            name: None,
            #[cfg(feature = "viewer")]
            shape: None,
            #[cfg(feature = "viewer")]
            color: None,
        }
    }

    pub fn sequential(elements: Vec<Self>) -> Self {
        Self::Sequential {
            elements,
            modifiers: None,
            name: None,
            #[cfg(feature = "viewer")]
            shape: None,
            #[cfg(feature = "viewer")]
            color: None,
        }
    }

    pub fn parallel(elements: Vec<Self>) -> Self {
        Self::Parallel {
            elements,
            modifiers: None,
            name: None,
            #[cfg(feature = "viewer")]
            shape: None,
            #[cfg(feature = "viewer")]
            color: None,
        }
    }
}

fn seconds_parse<T: num_traits::AsPrimitive<Seconds>>(value: T) -> Seconds {
    value.as_().tap(|x| assert!(x.is_finite()))
}

fn seconds_parse_positive<T: num_traits::AsPrimitive<Seconds>>(value: T) -> Seconds {
    value.as_().tap(|x| {
        assert!(x.is_finite());
        assert!(*x > 0.0)
    })
}

/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

use petgraph::Directed;

#[cfg(feature = "viewer")]
use crate::scene::core::ActorShape;
use crate::{audio::AudioBufferSlice, coordinates::Cart3D, math::Seconds, trajectory::Trajectory};

pub struct SceneBuilder<'a> {
    inner: petgraph::Graph<Actor<'a>, (), Directed>,
}

pub enum SceneBuilderElement<'a> {
    Delay(Seconds),
    Actor(Actor<'a>),
}

pub struct Actor<'a> {
    pub(crate) actions: Vec<(AudioBufferSlice<'a, 1>, Trajectory<Cart3D>)>,

    /// The name of the actor, should be kept short, for debugging or visualization purposes.
    pub(crate) name: Option<String>,
    /// For visualization only, what shape represents the element.
    #[cfg(feature = "viewer")]
    pub(crate) shape: Option<ActorShape>,
    /// For visualization only, what color is the shape that represents the element.
    #[cfg(feature = "viewer")]
    pub(crate) color: Option<String>,
}

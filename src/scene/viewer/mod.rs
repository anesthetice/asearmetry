/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Imports
use crate::coordinates::Cart3D;
use crate::trajectory::Trajectory;
use pixels::{Error, Pixels, SurfaceTexture};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tap::{Pipe, Tap};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::KeyCode;
use winit::window::Window;

use tiny_skia::{Color, FillRule, Paint, PathBuilder, Pixmap, Stroke, Transform};

#[bon::builder]
pub fn build(width: Option<u32>, height: Option<u32>, trajectory: Trajectory<Cart3D>) {
    let (width, height) = (width.unwrap_or(512), height.unwrap_or(512));
    assert!(width != 0 && height != 0);

    let pixmap = Pixmap::new(width, height)
        .unwrap()
        .tap_mut(|pm| pm.fill(Color::WHITE));

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App {
        width,
        height,
        pixmap,
        window: None,
        pixels: None,
        trajectory,
        frame_index: 0,
        last_frame: Instant::now(),
    };

    event_loop.run_app(&mut app).unwrap();
}

struct App {
    width: u32,
    height: u32,

    pixmap: Pixmap,

    window: Option<Arc<Window>>,
    pixels: Option<Pixels<'static>>,

    trajectory: Trajectory<Cart3D>,

    frame_index: usize,
    last_frame: Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("asearmetry viewer")
                        .with_inner_size(LogicalSize::new(self.width, self.height))
                        .with_resizable(false),
                )
                .unwrap(),
        );

        let surface_texture = SurfaceTexture::new(self.width, self.height, window.clone());

        let pixels = Pixels::new(self.width, self.height, surface_texture).unwrap();

        self.window = Some(window);
        self.pixels = Some(pixels);
        self.last_frame = Instant::now();
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let frame_time = Duration::from_secs_f64(self.trajectory.δt);

        let now = Instant::now();

        if now.duration_since(self.last_frame) >= frame_time {
            self.last_frame += frame_time;

            self.frame_index = (self.frame_index + 1) % self.trajectory.path.len();

            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::RedrawRequested => {
                self.pixmap.fill(Color::WHITE);

                let point = &self.trajectory.path[self.frame_index];
                let cx = (point.x + 1.0) * self.width as f64 * 0.5;
                let cy = (1.0 - (point.y + 1.0) * 0.5) * self.height as f64;

                draw_circle(
                    &mut self.pixmap,
                    self.width as f32 * 0.5,
                    self.height as f32 * 0.5,
                    20.0,
                );

                draw_circle(&mut self.pixmap, cx as f32, cy as f32, 8.0);

                self.pixels.as_mut().unwrap().pipe(|pxs| {
                    pxs.frame_mut().copy_from_slice(self.pixmap.data());
                    pxs.render().unwrap();
                })
            }

            other => println!("other window event: {other:?}"),
        }
    }
}

fn draw_circle(pixmap: &mut Pixmap, cx: f32, cy: f32, radius: f32) {
    let path = PathBuilder::from_circle(cx, cy, radius).unwrap();

    let mut paint = Paint::default();
    paint.set_color(Color::BLACK);

    pixmap.fill_path(
        &path,
        &paint,
        FillRule::Winding,
        Transform::identity(),
        None,
    );
}

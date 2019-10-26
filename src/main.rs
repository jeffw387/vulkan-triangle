use cgmath;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::framebuffer::{
    Framebuffer, FramebufferAbstract, RenderPassAbstract,
};
use vulkano::image::SwapchainImage;
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain;
use vulkano::swapchain::{
    AcquireError, PresentMode, SurfaceTransform, Swapchain,
    SwapchainCreationError,
};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use vulkano_win::VkSurfaceBuild;

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use std::sync::Arc;
use vulkano_triangle::dbgpipe;

fn main() {
    let instance = {
        let extensions = vulkano_win::required_extensions();

        Instance::new(None, &extensions, None).unwrap()
    };

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!(
        "Using device: {} (type: {:?})",
        physical.name(),
        physical.ty()
    );

    let events_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&events_loop, instance.clone())
        .unwrap();
    let window = surface.window();

    let queue_family = physical
        .queue_families()
        .find(|&q| {
            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
        })
        .unwrap();

    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let initial_dimensions = {
            let dimensions = window.inner_size();
            let dimensions: (u32, u32) =
                dimensions.to_physical(window.hidpi_factor()).into();
            [dimensions.0, dimensions.1]
        };

        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            initial_dimensions,
            1,
            usage,
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
            true,
            None,
        )
        .unwrap()
    };

    let vertex_buffer = {
        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            [
                dbgpipe::Vertex {
                    position: [-0.5, -0.25, 0.0, 1.0],
                },
                dbgpipe::Vertex {
                    position: [0.0, 0.5, 0.0, 1.0],
                },
                dbgpipe::Vertex {
                    position: [0.25, -0.1, 0.0, 1.0],
                },
            ]
            .iter()
            .cloned(),
        )
        .unwrap()
    };

    let vp_data = dbgpipe::vs::ty::VP_BLOCK {
        vp: cgmath::ortho(-5.0, 5.0, 5.0, -5.0, -1.0, 1.0).into(),
    };

    let vp_buffer = CpuBufferPool::<dbgpipe::vs::ty::VP_BLOCK>::new(
        device.clone(),
        BufferUsage::all(),
    );

    let vp_subbuffer = vp_buffer.next(vp_data).unwrap();

    let debug_pipeline = dbgpipe::build(device.clone(), swapchain.clone());

    let set = Arc::new(
        PersistentDescriptorSet::start(debug_pipeline.pipeline.clone(), 0)
            .add_buffer(vp_subbuffer)
            .unwrap()
            .build()
            .unwrap(),
    );

    let mut dynamic_state = DynamicState {
        line_width: None,
        viewports: None,
        scissors: None,
    };

    let mut framebuffers = window_size_dependent_setup(
        &images,
        debug_pipeline.render_pass.clone(),
        &mut dynamic_state,
    );

    let mut recreate_swapchain = false;

    let mut previous_frame_end =
        Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

    events_loop.run(move |ev, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        let window = surface.window();

        previous_frame_end.as_mut().unwrap().cleanup_finished();
        match ev {
    Event::EventsCleared => {
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
        if recreate_swapchain {
            let dimensions = window.inner_size();
            let dimensions: (u32, u32) =
                dimensions.to_physical(window.hidpi_factor()).into();
            let dimensions = [dimensions.0, dimensions.1];

            let (new_swapchain, new_images) = match swapchain
                .recreate_with_dimension(dimensions)
            {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => return,
                Err(err) => panic!("{:?}", err),
            };

            swapchain = new_swapchain;
            framebuffers = window_size_dependent_setup(
                &new_images,
                debug_pipeline.render_pass.clone(),
                &mut dynamic_state,
            );

            recreate_swapchain = false;
        }

        let (image_num, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    return;
                }
                Err(err) => panic!("{:?}", err),
            };

        let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
            device.clone(),
            queue.family(),
        )
        .unwrap()
        .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
        .unwrap()
        .draw(
            debug_pipeline.pipeline.clone(),
            &dynamic_state,
            vec![vertex_buffer.clone()],
            vec![set.clone()],
            ()
        )
        .unwrap()
        .end_render_pass()
        .unwrap()
        .build()
        .unwrap();

        let prev = previous_frame_end.take();

        let future = prev.unwrap()
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                future.wait(None).unwrap();
                previous_frame_end = Some(Box::new(future) as Box<_>);
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end =
                    Some(Box::new(sync::now(device.clone())) as Box<_>);
            }
            Err(e) => {
                eprintln!("{:?}", e);
                previous_frame_end =
                    Some(Box::new(sync::now(device.clone())) as Box<_>);
            }
        }

            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => recreate_swapchain = true,
            _ => (),
        }
    });
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}

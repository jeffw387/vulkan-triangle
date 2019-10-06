use std::sync::Arc;
use vulkano::device::Device;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::framebuffer::Subpass;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::swapchain::Swapchain;
use winit::Window;

#[derive(Debug, Clone, Default)]
pub struct Vertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

vulkano::impl_vertex!(Vertex, position, uv);

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450

layout (location = 0) in vec2 position;
layout (location = 1) in vec2 uv;

layout (set = 0, binding = 0) uniform MVP_BLOCK {
    mat4 mvp;
} mvp_inst;

layout (location = 0) out vec2 out_uv;

void main() {
    gl_Position = mvp_inst.mvp * vec4(position, 0, 1);
    out_uv = uv;
}"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450
layout (location = 0) in vec2 uv;

layout (set = 1, binding = 1) uniform sampler2D bitmap;

layout (location = 0) out vec4 f_color;

void main() {
    f_color = texture(bitmap, uv); 
}
"
    }
}

pub struct Pipeline {
    pub render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    pub pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
}

pub fn build(
    device: Arc<Device>,
    swapchain: Arc<Swapchain<Window>>,
) -> Pipeline {
    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )
        .unwrap(),
    );

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    Pipeline {
        render_pass,
        pipeline,
    }
}

// use crate::{error_window, print_deps, ENCODER_PATH, TEST_IMG_PATH};
use anyhow::{anyhow, Result};
use ndarray::{concatenate, prelude::*, ArrayD, Axis};
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, DirectMLExecutionProvider, ExecutionProvider,
    OneDNNExecutionProvider, OpenVINOExecutionProvider,
};
use ort::inputs;
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::value::Tensor;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::process::exit;

pub struct GoTEncoder {
    session: Session,
}

impl Deref for GoTEncoder {
    type Target = Session;

    fn deref(&self) -> &Self::Target {
        &self.session
    }
}

impl DerefMut for GoTEncoder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.session
    }
}

impl GoTEncoder {
    fn get_builder() -> Result<SessionBuilder> {
        println!(
            "cuda: {:?} dml: {:?} openvino: {:?} onednn:{:?}",
            CUDAExecutionProvider::default().is_available()?,
            DirectMLExecutionProvider::default().is_available()?,
            OpenVINOExecutionProvider::default().is_available()?,
            OneDNNExecutionProvider::default().is_available()?,
        );

        let session_builder = Session::builder()?;
        Ok(session_builder
            .with_optimization_level(GraphOptimizationLevel::Level2)?
            // .with_memory_pattern(true)?
            // .with_parallel_execution(true)?
            // .with_intra_threads(16)?
            // .with_device_allocator_for_initializers()?
            // .with_optimized_model_path(r#"D:\whl\Desktop\got.cpp\encoder_opt2.onnx"#)?
            .with_profiling(r#"C:\Users\whl\WorkSpace\RustProjects\GotOnnx\profile"#)?
            .with_execution_providers([
                #[cfg(target_os = "windows")]
                DirectMLExecutionProvider::default().build(),
                // OpenVINOExecutionProvider::default()
                //     .with_dynamic_shapes()
                //     .with_num_threads(16)
                //     .build(),
                OneDNNExecutionProvider::default().with_use_arena(true).build(),
                CPUExecutionProvider::default().build(),
            ])?)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let session = Self::get_builder()?.commit_from_file(path)?;
        Ok(GoTEncoder { session })
    }

    pub fn from_memory(data: &[u8]) -> Result<Self> {
        let session = Self::get_builder()?.commit_from_memory(data)?;
        Ok(GoTEncoder { session })
    }

    fn get_input_allocator(&self) -> Result<Allocator> {

        Ok(Allocator::new(
            &self.session,
            MemoryInfo::new(
                AllocationDevice::DIRECTML,
                0,
                AllocatorType::Device,
                MemoryType::CPUInput,
            )?,
        )?)
    }

    fn get_output_allocator(&self) -> Result<&Allocator> {
        Ok(self.session.allocator())
    }

    pub fn call(&self, input: Vec<f32>) -> Result<ArrayD<f32>> {
        if input.len() != 1 * 3 * 1024 * 1024 {
            return Err(anyhow!(
                "expect shape [1 * 3 * 1024 * 1024] but got vec {}",
                input.len()
            ));
        }

        let mut binding = self.session.create_binding()?;
        let allocator = self.get_output_allocator()?;

        let input: Tensor<f32> =
            Tensor::from_array(([1, 3, 1024, 1024], input.into_boxed_slice()))?;

        binding.bind_input("input", &input)?;
        binding.bind_output("output", Tensor::<f32>::new(allocator, [256, 1024])?)?;

        // println!("{:?}", s.elapsed());

        // let outputs = self
        //     .session
        //     .run(inputs! {
        //         "input" => input.into_dyn()
        //     }?)?
        //     .remove("output")
        //     .unwrap();
        let outputs: Tensor<f32> = binding.run()?.remove("output").unwrap().downcast()?;
        // outputs.
        // return Err(anyhow!("{:?}",outputs.shape()));
        let result_tensor = outputs.try_extract_tensor::<f32>()?;
        Ok(result_tensor.to_owned())
    }

    pub fn call_batch(&self, input: Vec<Vec<f32>>) -> Result<ArrayD<f32>> {
        let b = input.len();
        if input[0].len() != 1 * 3 * 1024 * 1024 {
            return Err(anyhow!(
                "expect shape [batch_size * 3 * 1024 * 1024] but got vec with len: {}",
                input[0].len()
            ));
        }

        let mut binding = self.session.create_binding()?;
        let allocator = self.get_output_allocator()?;

        let input = input
            .into_iter()
            .map(|i| Array::from_shape_vec((1, 3, 1024, 1024), i).unwrap())
            .collect::<Vec<_>>();

        let input: Tensor<f32> = concatenate(
            Axis(0),
            &(input.iter().map(|i| i.view()).collect::<Vec<_>>()),
        )?
            .try_into()?;

        binding.bind_input("input", &input)?;
        binding.bind_output("output", Tensor::<f32>::new(allocator, [1, 256 * b, 1024])?)?;

        // println!("{:?}", s.elapsed());

        // let outputs = session
        //     .run(inputs! {
        //         "input" => input.into_dyn()
        //     }?)?
        //     .remove("output")
        //     .unwrap();
        let outputs: Tensor<f32> = binding
            .run()?
            .remove("output")
            .expect("Fail to ext")
            .downcast()?;
        // return Err(anyhow!("{:?}",outputs.shape()));
        let result_tensor = outputs.try_extract_tensor::<f32>()?;

        Ok(result_tensor.to_owned())
    }
}

#![allow(unused)]
use crate::vit_image_processor::{rescale_and_normalize, resize};
use anyhow::{anyhow, Result};
use image::imageops::FilterType;
use image::{load_from_memory, open, DynamicImage, GenericImageView};
use std::path::Path;

pub struct PreProcessConfig {
    pub min_num: usize,
    pub max_num: usize,
    pub image_size: usize,
    pub use_thumbnail: bool,
}

impl Default for PreProcessConfig {
    fn default() -> Self {
        PreProcessConfig {
            min_num: 1,
            max_num: 6,
            image_size: 1024,
            use_thumbnail: true,
        }
    }
}

pub fn dynamic_preprocess(
    img: DynamicImage,
    pre_process_config: PreProcessConfig,
) -> Vec<DynamicImage> {
    let (orig_width, orig_height) = img.dimensions();
    let aspect_ratio = orig_width as f64 / orig_height as f64;
    let mut target_ratios: Vec<_> = (1..=pre_process_config.max_num)
        .flat_map(|i| (1..=pre_process_config.max_num).map(move |j| (i, j)))
        .filter(|(i, j)| i * j >= pre_process_config.min_num && i * j <= pre_process_config.max_num)
        .collect();
    target_ratios.sort_by_key(|&(i, j)| i * j);


    let best_ratio = target_ratios
        .into_iter()
        .max_by(|&(i1, j1), &(i2, j2)| {
            (i1 as f64 / j1 as f64 - aspect_ratio)
                .abs()
                .partial_cmp(&(i2 as f64 / j2 as f64 - aspect_ratio).abs())
                .unwrap()
                .reverse()
        })
        .unwrap();


    let target_width = (pre_process_config.image_size * best_ratio.0) as u32;
    let target_height = (pre_process_config.image_size * best_ratio.1) as u32;
    let blocks = best_ratio.0 * best_ratio.1;

    let resize_img = img.resize_exact(target_width, target_height, FilterType::CatmullRom);

    let block_width = best_ratio.0;
    let image_size = pre_process_config.image_size as u32;

    let mut precessed_img: Vec<_> = (0..blocks)
        .map(|i| {
            let left = (i % block_width) as u32;
            let upper = (i / block_width) as u32;
            DynamicImage::from(
                resize_img
                    .view(left, upper, image_size, image_size)
                    .to_image(),
            )
        })
        .collect();

    if pre_process_config.use_thumbnail && precessed_img.len() != 1 {
        precessed_img.push(img.resize_exact(image_size, image_size, FilterType::CatmullRom))
    }

    precessed_img
}

pub fn read_from_image_crop(img: DynamicImage) -> Result<Vec<Vec<f32>>> {
    let images: Vec<_> = dynamic_preprocess(img, Default::default())
        .into_iter()
        .map(|img| crate::vit_image_processor::preprocess_from_image(img).unwrap())
        .collect();
    Ok(images)
}
pub fn read_from_image(img: DynamicImage) -> Result<Vec<f32>> {
    Ok(crate::vit_image_processor::preprocess_from_image(img)
        .map_err(|e| anyhow::anyhow!("{e}"))?)
}

pub fn read_from_file_crop<P>(path: P) -> Result<Vec<Vec<f32>>>
where
    P: AsRef<Path>,
{
    let images = read_from_image_crop(open(path)?)?;
    Ok(images)
}

pub fn read_from_file<P>(path: P) -> Result<Vec<f32>>
where
    P: AsRef<Path>,
{
    let images = read_from_image(open(path)?)?;
    Ok(images)
}

pub fn read_from_memory_crop(data: &[u8]) -> Result<Vec<Vec<f32>>> {
    let img = rescale_and_normalize(resize(load_from_memory(data)?));

    let images = read_from_image_crop(img)?;
    Ok(images)
}

pub fn read_from_memory(data: &[u8]) -> Result<Vec<f32>> {
    let img = rescale_and_normalize(resize(load_from_memory(data)?));

    let images = read_from_image(img)?;
    Ok(images)
}

#[test]
fn test_preprocess() {
    dynamic_preprocess(
        open(r#"E:\WorkSpace\RustProjects\MixTex-rs-GUI\tests\test.png"#).unwrap(),
        // .resize_exact(1024, 1024, FilterType::CatmullRom),
        Default::default(),
    );
}

#[test]
fn test_vit_preprocess() {
    let images =
        read_from_file_crop(r#"E:\WorkSpace\RustProjects\MixTex-rs-GUI\tests\test.png"#).unwrap();
    println!("{}", images.len())
}

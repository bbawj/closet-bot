use image::{imageops::FilterType, ImageBuffer, Rgb};
use image::{ImageReader, Pixel};
use ndarray::{Array, ArrayBase, Dim, OwnedRepr};

use crate::BoundingBox;

#[allow(dead_code)]
pub struct TensorImage {
    pub array: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>,
    pub orig_size: (u32, u32),
    pub processed_size: (u32, u32),
    pub padding: (u32, u32),
}

const IMAGE_MEAN: &'static [f32] = &[0.48145466, 0.4578275, 0.40821073];
const IMAGE_STD: &'static [f32] = &[0.26862954, 0.26130258, 0.27577711];
const DEFAULT_MEAN: [f32; 3] = [0.0, 0.0, 0.0];
const DEFAULT_STD: [f32; 3] = [1.0, 1.0, 1.0];

pub fn preprocess(
    image_bytes: Vec<u8>,
    target_width: u32,
    target_height: u32,
    bounding_box: Option<BoundingBox>,
    save_path: Option<&str>,
) -> TensorImage {
    let mut image = ImageReader::new(std::io::Cursor::new(image_bytes))
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap();

    let orig_width = image.width();
    let orig_height = image.height();
    log::info!("Original size {:?}", (orig_width, orig_height));
    if let Some(bb) = bounding_box {
        image = image.crop_imm(bb.x, bb.y, bb.w, bb.h)
    }
    image = image.resize(target_height, target_width, FilterType::Nearest);

    let processed_image = image.into_rgb8();

    let processed_width = processed_image.width();
    let processed_height = processed_image.height();
    let pad_left = (target_width - processed_width) / 2;
    let pad_top = (target_height - processed_height) / 2;

    let mut padded_image =
        ImageBuffer::from_pixel(target_width, target_height, Rgb([112, 112, 112]));
    for (x, y, pixel) in processed_image.enumerate_pixels() {
        padded_image.put_pixel(x + pad_left, y + pad_top, *pixel);
    }
    if let Some(path) = save_path {
        padded_image
            .save(path)
            .expect("Failed to save processed image");
    }
    let array = Array::from_shape_fn(
        (1, 3, target_height as usize, target_width as usize),
        |(_, c, j, i)| {
            let pixel = padded_image.get_pixel(i as u32, j as u32);
            pixel.channels()[c] as f32 / 255.0
        },
    );

    TensorImage {
        array,
        orig_size: (orig_height, orig_width),
        processed_size: (processed_height, processed_width),
        padding: (pad_top, pad_left),
    }
}

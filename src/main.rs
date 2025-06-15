use image::EncodableLayout;
use image_processor::preprocess;
use itertools::Itertools;
use libsql::Builder;
use ndarray::{Array2, ArrayBase, CowArray, CowRepr, Dim, IxDynImpl};
use ort::Environment;
use ort::{session::Session, GraphOptimizationLevel, SessionBuilder, Value};
use std::env;
use std::error::Error;
use std::sync::Arc;
use teloxide::net::Download;
use teloxide::prelude::*;
use teloxide::sugar::request::RequestReplyExt;
use teloxide::types::{InputFile, InputMedia, InputMediaPhoto, PhotoSize};
use tokenizers::tokenizer::Tokenizer;
use tokenizers::Encoding;
use tokio::fs;

pub mod image_processor;

pub type HandlerResult = Result<(), Box<dyn Error + Send + Sync>>;

#[derive(Clone, Debug)]
pub struct Models {
    image: Arc<Session>,
    text: Arc<Session>,
    det: Arc<Session>,
}

#[tokio::main]
async fn main() {
    pretty_env_logger::init();

    let bot = Bot::from_env();

    fs::create_dir_all("data")
        .await
        .expect("Could not create data dir");
    let db = Builder::new_local("data/data.db").build().await.unwrap();
    let db_conn = db.connect().unwrap();
    db_conn
        .execute(
            "CREATE TABLE IF NOT EXISTS photos (file_id varchar(255), embeddings F32_BLOB(512))",
            (),
        )
        .await
        .unwrap();
    db_conn
        .execute(
            "CREATE INDEX IF NOT EXISTS photos_vector_idx ON photos(libsql_vector_idx(embeddings))",
            (),
        )
        .await
        .unwrap();

    let env = Environment::builder().build().unwrap().into_arc();
    let det_model = Arc::new(
        SessionBuilder::new(&env)
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .with_model_from_file("yolo11n.onnx")
            .unwrap(),
    );
    log::info!("Detection Model loaded successfully!");
    log::info!("Inputs: {:?}", det_model.inputs);
    log::info!("Outputs: {:?}", det_model.outputs);
    let image_model = Arc::new(
        SessionBuilder::new(&env)
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .with_model_from_file("clip-image-vit-32.onnx")
            .unwrap(),
    );
    log::info!("Image Model loaded successfully!");
    log::info!("Inputs: {:?}", image_model.inputs);
    log::info!("Outputs: {:?}", image_model.outputs);
    let text_model = Arc::new(
        SessionBuilder::new(&env)
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .with_model_from_file("clip-text-vit-32.onnx")
            .unwrap(),
    );
    log::info!("Text Model loaded successfully!");
    log::info!("Inputs: {:?}", text_model.inputs);
    log::info!("Outputs: {:?}", text_model.outputs);

    let mut tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::Fixed(77),
        direction: tokenizers::PaddingDirection::Right,
        pad_to_multiple_of: None,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: "[PAD]".to_string(),
    }));
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            direction: tokenizers::TruncationDirection::Right,
            max_length: 77,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            stride: 0,
        }))
        .unwrap();

    let owners: Vec<u64> = env::var("CLOSET_OWNERS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    log::info!("Owners: {:?}", owners);
    let handler = Update::filter_message().branch(
        dptree::filter(move |msg: Message| {
            let from = msg.from.unwrap().id.0;
            owners.contains(&from)
        })
        .branch(Message::filter_photo().endpoint(photo_upload))
        .endpoint(photo_find),
    );

    let models = Models {
        image: image_model,
        text: text_model,
        det: det_model,
    };
    Dispatcher::builder(bot, handler)
        .dependencies(dptree::deps![models, db_conn, tokenizer])
        // If no handler succeeded to handle an update, this closure will be called.
        .default_handler(|upd| async move {
            log::warn!("Unhandled update: {:?}", upd);
        })
        // If the dispatcher fails for some reason, execute this handler.
        .error_handler(LoggingErrorHandler::with_custom_text(
            "An error has occurred in the dispatcher",
        ))
        .enable_ctrlc_handler()
        .build()
        .dispatch()
        .await;
}

async fn photo_upload(
    bot: Bot,
    models: Models,
    db: libsql::Connection,
    msg: Message,
    photos: Vec<PhotoSize>,
) -> HandlerResult {
    let chat_id = msg.chat.id;

    let photo = match photos.last() {
        Some(p) => p,
        None => return Err("No photos in message".into()),
    };
    let file = bot.get_file(&photo.file.id).await?;
    let path = format!("data/{}", &file.id);
    let mut dst = fs::File::create(&path).await?;
    bot.download_file(&file.path, &mut dst).await?;

    let embeddings = match encode(
        models.image.clone(),
        models.det.clone(),
        fs::read(&path).await?,
    ) {
        Ok(embeddings) => embeddings,
        Err(e) => {
            log::error!("Failed to get embeddings");
            return Err(e);
        }
    };

    match db
        .execute(
            "INSERT INTO photos (file_id, embeddings) VALUES (?1, vector32(?2))",
            libsql::params![file.id.clone(), embeddings.as_bytes()],
        )
        .await
    {
        Ok(_) => (),
        Err(e) => log::error!("Failed to insert photo into db because {}", e),
    };
    bot.send_message(chat_id, "done!").reply_to(&msg).await?;

    Ok(())
}

async fn photo_find(
    bot: Bot,
    models: Models,
    db: libsql::Connection,
    tokenizer: Tokenizer,
    msg: Message,
) -> HandlerResult {
    let query = match msg.text() {
        Some(e) => e,
        None => {
            return Err("Query is empty".into());
        }
    };

    let embedding = match encode_text(models.text, &query, tokenizer) {
        Ok(e) => e,
        Err(e) => {
            return Err(format!("Failed to encode query {} because: {}", query, e).into());
        }
    };

    let top = 3;
    let mut similar = match db
        .query(
            "SELECT p.file_id, vector_distance_cos(p.embeddings, vector32(?1)) AS distance, vector_extract(p.embeddings)
             FROM vector_top_k('photos_vector_idx', vector32(?1), ?2) AS i
             JOIN photos p ON p.rowid = i.id
             WHERE distance > 0.5
             ORDER BY distance ASC
             LIMIT ?2",
            libsql::params!(embedding.as_bytes(), top),
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return Err(format!("Failed vector index search because: {}", e).into());
        }
    };

    let mut file_ids: Vec<InputMedia> = Vec::new();
    while let Some(row) = similar.next().await.unwrap_or(None) {
        log::info!(
            "File: {:?} matched with similarity: {:?}, embedding: {:?}",
            row.get::<String>(0),
            row.get::<f64>(1),
            row.get::<String>(2),
        );
        match row.get::<String>(0) {
            Ok(id) => file_ids.push(InputMedia::Photo(
                InputMediaPhoto::new(InputFile::file_id(id)).show_caption_above_media(true),
            )),
            Err(e) => {
                log::error!("Failed to get field file_id because: {}", e);
                continue;
            }
        };
    }

    if let Some(InputMedia::Photo(p)) = file_ids.first_mut() {
        p.caption = Some("How about these?".to_owned());
    } else {
        bot.send_message(msg.chat.id, "Did not find matching clothes")
            .reply_to(msg)
            .await?;
        return Ok(());
    };

    bot.send_media_group(msg.chat.id, file_ids)
        .reply_to(msg)
        .await?;

    Ok(())
}

pub fn encode(
    image: Arc<Session>,
    det: Arc<Session>,
    images_bytes: Vec<u8>,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    let bb = find_bounding_box(det, &images_bytes)?;
    let pixels = preprocess(
        images_bytes.to_vec(),
        224,
        224,
        bb,
        if cfg!(debug_assertions) {
            Some("data/processed.jpg")
        } else {
            None
        },
    );

    let outputs = image.run(vec![Value::from_array(
        image.allocator(),
        &pixels.array.as_standard_layout().into_dyn(),
    )?])?;

    let binding = outputs[0].try_extract()?;
    log::debug!("CLIP image output: {:?}", binding);
    let embeddings = binding.view();

    let seq_len = embeddings.shape().get(1).unwrap();

    let embeddings: Vec<Vec<f32>> = embeddings
        .iter()
        .copied()
        .chunks(*seq_len)
        .into_iter()
        .map(|b| b.collect())
        .collect();
    let embedding = embeddings[0].clone();
    Ok(embedding)
}

pub fn encode_text(
    session: Arc<Session>,
    text: &str,
    tokenizer: Tokenizer,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    if text.is_empty() {
        return Err("No text provided".into());
    }

    let tokens = tokenizer.encode(text, true)?;
    log::debug!("Encoded text tokens: {:?}", tokens);

    let outputs = session.run(vec![Value::from_array(
        session.allocator(),
        &get_input_ids_vector(tokens, &vec![text.to_string()])?,
    )?])?;

    let binding = outputs[0].try_extract()?;
    log::debug!("CLIP text output: {:?}", binding);
    let embeddings = binding.view();
    let seq_len = embeddings.shape().get(1).unwrap();

    let embeddings: Vec<Vec<f32>> = embeddings
        .iter()
        .copied()
        .chunks(*seq_len)
        .into_iter()
        .map(|b| b.collect())
        .collect();
    let embedding = embeddings[0].clone();
    Ok(embedding)
}

fn get_input_ids_vector(
    preprocessed: Encoding,
    text: &Vec<String>,
) -> Result<ArrayBase<CowRepr<'_, i32>, Dim<IxDynImpl>>, Box<dyn Error + Send + Sync>> {
    let input_ids_vector: Vec<i32> = preprocessed
        .get_ids()
        .iter()
        .map(|b| *b as i32)
        .collect::<Vec<i32>>();
    let ids_shape = (text.len(), input_ids_vector.len() / text.len());
    let input_ids_vector =
        CowArray::from(Array2::from_shape_vec(ids_shape, input_ids_vector)?).into_dyn();
    Ok(input_ids_vector)
}

#[derive(Default, Debug)]
pub struct BoundingBox {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    prob: f32,
}

fn find_bounding_box(
    session: Arc<Session>,
    images_bytes: &Vec<u8>,
) -> Result<Option<BoundingBox>, Box<dyn Error + Send + Sync>> {
    let yolo_input_height = session.inputs[0].dimensions[2].unwrap();
    let yolo_input_width = session.inputs[0].dimensions[3].unwrap();

    log::info!(
        "Processing image for YOLO model. Target width: {}. Target height: {}",
        yolo_input_width,
        yolo_input_height,
    );
    let tensor = preprocess(
        images_bytes.to_vec(),
        yolo_input_width,
        yolo_input_height,
        None,
        if cfg!(debug_assertions) {
            Some("data/bb.jpg")
        } else {
            None
        },
    );

    let outputs = session.run(vec![Value::from_array(
        session.allocator(),
        &tensor.array.as_standard_layout().into_dyn(),
    )?])?;
    let binding = outputs[0].try_extract::<f32>()?;
    let output = binding.view();

    let channels = output.shape()[1];
    let number_of_prediction_points = output.shape()[2];
    let reshaped = output
        .to_shape((channels, number_of_prediction_points))
        .unwrap();

    let mut best = BoundingBox::default();
    let horz_scale = tensor.orig_size.1 as f32 / tensor.processed_size.1 as f32;
    let vert_scale = tensor.orig_size.0 as f32 / tensor.processed_size.0 as f32;
    for pred in reshaped.axis_iter(ndarray::Axis(1)) {
        let id = pred[5] as u32;
        if id != 0 {
            continue;
        }
        let person_object_confidence = pred[4];
        if person_object_confidence > 0.25 && person_object_confidence > best.prob {
            // bb output is in terms of the input scale, fix up to the original image size
            // bb output bounding box is the centre point, fix up to represent top left
            best.x =
                ((pred[0] - tensor.padding.1 as f32 - pred[2] / 2.0) * horz_scale).round() as u32;
            best.y =
                ((pred[1] - tensor.padding.0 as f32 - pred[3] / 2.0) * vert_scale).round() as u32;
            best.w = (pred[2] * horz_scale).round() as u32;
            best.h = (pred[3] * vert_scale).round() as u32;
            best.prob = person_object_confidence;
            log::debug!(
                "Found bounding box. x: {:?} y: {:?} h_scale: {} v_scale: {}",
                pred[0],
                pred[1],
                horz_scale,
                vert_scale
            );
        }
    }

    if best.prob > 0.25 {
        log::info!("Final bounding box: {:?} for person", best,);
        return Ok(Some(best));
    } else {
        return Ok(None);
    }
}

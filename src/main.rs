use image::EncodableLayout;
use std::env;
use ndarray::{Array2, ArrayBase, CowArray, CowRepr, Dim, IxDynImpl};
use ort::Environment;
use ort::{session::Session, GraphOptimizationLevel, SessionBuilder, Value};
use std::error::Error;
use std::sync::Arc;
use teloxide::net::Download;
use teloxide::prelude::*;
use teloxide::sugar::request::RequestReplyExt;
use teloxide::types::{InputFile, InputMedia, InputMediaPhoto, PhotoSize};
use tokenizers::tokenizer::Tokenizer;
use tokenizers::Encoding;
pub mod clip_image_processor;
use crate::clip_image_processor::CLIPImageProcessor;
use itertools::Itertools;
use libsql::Builder;

pub type HandlerResult = Result<(), Box<dyn Error + Send + Sync>>;

#[derive(Clone, Debug)]
pub struct Models {
    image: Arc<Session>,
    text: Arc<Session>,
}

#[tokio::main]
async fn main() {
    pretty_env_logger::init();

    let bot = Bot::from_env();

    let db = Builder::new_local("data.db").build().await.unwrap();
    let db_conn = db.connect().unwrap();
    db_conn.execute("CREATE TABLE IF NOT EXISTS photos (msg_id INT PRIMARY KEY, file_id varchar(255), embeddings F32_BLOB(512))", ()).await.unwrap();
    db_conn
        .execute(
            "CREATE INDEX IF NOT EXISTS photos_vector_idx ON photos(libsql_vector_idx(embeddings))",
            (),
        )
        .await
        .unwrap();

    let env = Environment::builder().build().unwrap().into_arc();
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

    let mut tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::Fixed(77),
        direction: tokenizers::PaddingDirection::Right,
        pad_to_multiple_of: None,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: "[PAD]".to_string(),
    }));
    tokenizer.with_truncation(Some(tokenizers::TruncationParams {
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
    let handler = Update::filter_message()
        .branch(
            dptree::filter(move |msg: Message| {
                let from = msg.from.unwrap().id.0;
                owners.contains(&from)
            })
            .branch(Message::filter_photo().endpoint(photo_upload))
            .endpoint(photo_find)
        );

    let models = Models {
        image: image_model,
        text: text_model,
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
    use tokio::fs;

    let chat_id = msg.chat.id;

    let photo = &photos[0];
    let file = bot.get_file(&photo.file.id).await?;
    fs::create_dir_all("data").await?;
    let path = format!("data/{}", &file.id);
    let mut dst = fs::File::create(&path).await?;
    bot.download_file(&file.path, &mut dst).await?;

    let embeddings = match encode(models.image.clone(), fs::read(&path).await?) {
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

    let mut similar = match db
        .query(
            "SELECT file_id
             FROM vector_top_k('photos_vector_idx', vector32(?1), 3)
             JOIN photos ON photos.rowid = id",
            libsql::params!(embedding.as_bytes()),
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
        match row.get::<String>(0) {
            Ok(id) => file_ids.push(InputMedia::Photo(InputMediaPhoto::new(InputFile::file_id(
                id,
            )).show_caption_above_media(true))),
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
    session: Arc<Session>,
    images_bytes: Vec<u8>,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    let processor = CLIPImageProcessor::default();
    let pixels = processor.preprocess(images_bytes.to_vec());

    let binding = pixels.into_dyn();
    let outputs = session.run(vec![Value::from_array(session.allocator(), &binding)?])?;

    let binding = outputs[0].try_extract()?;
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
    log::debug!("{:?}", tokens);

    let outputs = session.run(vec![Value::from_array(
        session.allocator(),
        &get_input_ids_vector(tokens, &vec![text.to_string()])?,
    )?])?;

    let binding = outputs[0].try_extract()?;
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

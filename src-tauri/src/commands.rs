//! Tauri commands module

use std::path::PathBuf;
use std::time::Instant;

use parakeet_rs::{ExecutionConfig, ExecutionProvider, TimestampMode, Transcriber};
use tauri::State;
use tracing::{error, info, warn};

use crate::{AppState, TranscriptionResult};

/// Max chunk length fed to the encoder — 60 s at 16 kHz.
const CHUNK_SAMPLES: usize = 60 * 16_000;

/// Overlap between chunks to avoid cutting words — 1 s.
const OVERLAP_SAMPLES: usize = 1 * 16_000;

/// Initialize the model from the tdtv2.int8 folder
#[tauri::command]
pub async fn init_model(state: State<'_, AppState>) -> Result<TranscriptionResult, String> {
    info!("Initializing Parakeet TDT model from: {:?}", state.model_dir);

    let load_start = Instant::now();

    if !state.model_dir.exists() {
        return Err(format!("Model directory not found: {:?}", state.model_dir));
    }

    let required_files = [
        "decoder_joint-model.int8.onnx",
        "encoder-model.int8.onnx",
        "tokenizer.json",
        "vocab.txt",
    ];

    for file in required_files {
        let file_path = state.model_dir.join(file);
        if !file_path.exists() {
            warn!("Model file not found: {:?}", file_path);
        }
    }

    let exec_config = ExecutionConfig::new()
        .with_execution_provider(ExecutionProvider::Cpu);

    let model = match parakeet_rs::ParakeetTDT::from_pretrained(
        state.model_dir.to_string_lossy().as_ref(),
        Some(exec_config),
    ) {
        Ok(m) => m,
        Err(e) => {
            error!("Failed to load model: {}", e);
            return Err(format!("Failed to load model: {}", e));
        }
    };

    let mut model_guard = state.model.lock().map_err(|e| e.to_string())?;
    *model_guard = Some(model);

    let load_duration = load_start.elapsed().as_secs_f64();
    info!("Model loaded in {:.2}s", load_duration);

    Ok(TranscriptionResult {
        text: format!("Model loaded in {:.2}s", load_duration),
        duration_secs: load_duration,
        success: true,
        error: None,
    })
}

/// Transcribe an audio file, chunking into 60 s segments.
#[tauri::command]
pub async fn transcribe_audio(
    audio_path: String,
    state: State<'_, AppState>,
) -> Result<TranscriptionResult, String> {
    info!("Transcribing: {}", audio_path);

    let path = PathBuf::from(&audio_path);
    if !path.exists() {
        return Err(format!("Audio file not found: {}", audio_path));
    }

    let mut model_guard = state.model.lock().map_err(|e| e.to_string())?;
    let model = model_guard
        .as_mut()
        .ok_or_else(|| "Model not initialized. Call init_model first.".to_string())?;

    // Decode and convert to 16 kHz mono
    let decode_start = Instant::now();
    let audio_data = crate::audio::decode_audio(&path)
        .map_err(|e| format!("Failed to decode audio: {}", e))?;

    let samples = audio_data.to_16k_mono();
    let total_samples = samples.len();
    let decode_duration = decode_start.elapsed().as_secs_f64();

    info!(
        "Decoded: {} samples ({:.1}s) in {:.2}s",
        total_samples,
        total_samples as f64 / 16_000.0,
        decode_duration
    );

    // Chunk and transcribe
    let transcribe_start = Instant::now();
    let mut parts: Vec<String> = Vec::new();
    let step = CHUNK_SAMPLES.saturating_sub(OVERLAP_SAMPLES);
    let mut offset = 0;

    while offset < total_samples {
        let end = (offset + CHUNK_SAMPLES).min(total_samples);
        let chunk = samples[offset..end].to_vec();
        let chunk_secs = chunk.len() as f64 / 16_000.0;

        info!(
            "Chunk {:.1}s–{:.1}s ({} samples)",
            offset as f64 / 16_000.0,
            end as f64 / 16_000.0,
            chunk.len()
        );

        match model.transcribe_samples(chunk, 16_000, 1, Some(TimestampMode::Sentences)) {
            Ok(r) => {
                let text = r.text.trim().to_string();
                if !text.is_empty() {
                    parts.push(text);
                }
            }
            Err(e) => {
                error!("Chunk failed at offset {}: {}", offset, e);
                return Err(format!("Transcription failed at {:.1}s: {}", offset as f64 / 16_000.0, e));
            }
        }

        if end == total_samples {
            break;
        }
        offset += step;
    }

    let transcribe_duration = transcribe_start.elapsed().as_secs_f64();
    let total_duration = decode_duration + transcribe_duration;
    let full_text = parts.join(" ");

    info!(
        "Done: {} chars from {} chunks in {:.2}s",
        full_text.len(),
        parts.len(),
        transcribe_duration
    );

    let mut transcript_guard = state.last_transcript.lock().map_err(|e| e.to_string())?;
    *transcript_guard = Some(full_text.clone());

    Ok(TranscriptionResult {
        text: full_text,
        duration_secs: total_duration,
        success: true,
        error: None,
    })
}

/// Get the last transcription result
#[tauri::command]
pub fn get_last_transcript(state: State<'_, AppState>) -> Result<Option<String>, String> {
    let guard = state.last_transcript.lock().map_err(|e| e.to_string())?;
    Ok(guard.clone())
}

from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.middleware.cors import CORSMiddleware
import io
import hashlib
import json
from datetime import datetime
from typing import Optional, List
import asyncio
from collections import Counter
import os

# initialize FastAPI app
app = FastAPI(title='NovaSumm AI', description='Advanced AI Text Summarization Suite', version='3.0')

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for frequent summaries
summary_cache = {}
cache_size = 100

# model and tokenizer initialization
try:
    model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
    tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")
    print("✅ Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading saved model: {e}")
    print("🔄 Loading t5-base model...")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    print("✅ Default model loaded successfully!")

# device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using NVIDIA CUDA")
else:
    device = torch.device("cpu")
    print("✅ Using CPU")

model.to(device)
model.eval()

# Summary history storage
summary_history = []
MAX_HISTORY = 50

class DialogueInput(BaseModel):
    dialogue: str = Field(..., min_length=10, max_length=5000)
    summary_type: Optional[str] = Field("general")
    max_length: Optional[int] = Field(150, ge=50, le=500)
    min_length: Optional[int] = Field(30, ge=10, le=200)

def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^\w\s\.\,\!\?\'\-\:\(\)]", "", text)
    text = text.strip()
    return text

def summarize_dialogue(dialogue: str, summary_type: str = "general", max_length: int = 150, min_length: int = 30) -> str:
    cache_key = hashlib.md5(f"{dialogue}_{summary_type}_{max_length}".encode()).hexdigest()
    if cache_key in summary_cache:
        return summary_cache[cache_key]
    
    cleaned_text = clean_data(dialogue)
    
    if summary_type == "concise":
        max_length = 80
        min_length = 20
    elif summary_type == "detailed":
        max_length = 300
        min_length = 100
    
    input_text = "summarize: " + cleaned_text
    
    inputs = tokenizer(
        input_text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.5
        )
    
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    
    if len(summary_cache) >= cache_size:
        summary_cache.pop(next(iter(summary_cache)))
    summary_cache[cache_key] = summary
    
    return summary

@app.post("/summarize")
async def summarize_endpoint(dialogue_input: DialogueInput):
    try:
        if len(dialogue_input.dialogue.strip()) < 20:
            return {"summary": "⚠️ Please provide more text (at least 20 characters)", "status": "error"}
        
        start_time = datetime.now()
        summary = summarize_dialogue(
            dialogue_input.dialogue,
            summary_type=dialogue_input.summary_type,
            max_length=dialogue_input.max_length,
            min_length=dialogue_input.min_length
        )
        processing_time = (datetime.now() - start_time).total_seconds()
        
        history_entry = {
            "id": hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:8],
            "timestamp": datetime.now().isoformat(),
            "input_length": len(dialogue_input.dialogue),
            "output_length": len(summary),
            "summary_type": dialogue_input.summary_type,
            "processing_time": processing_time,
            "compression_ratio": (1 - len(summary) / len(dialogue_input.dialogue)) * 100
        }
        summary_history.append(history_entry)
        if len(summary_history) > MAX_HISTORY:
            summary_history.pop(0)
        
        compression_ratio = (1 - len(summary) / len(dialogue_input.dialogue)) * 100
        
        return {
            "summary": summary,
            "status": "success",
            "metrics": {
                "input_length": len(dialogue_input.dialogue),
                "output_length": len(summary),
                "compression_ratio": f"{compression_ratio:.1f}%",
                "processing_time": f"{processing_time:.2f}s",
                "total_summaries": len(summary_history)
            }
        }
    except Exception as e:
        return {"summary": f"❌ Error: {str(e)}", "status": "error"}

@app.get("/stats")
async def get_stats():
    if not summary_history:
        return {
            "total_summaries": 0,
            "total_saved_chars": 0,
            "avg_processing_time": "0s",
            "avg_compression": "0%",
            "summary_type_distribution": {},
            "total_input_chars": 0,
            "total_output_chars": 0
        }
    
    total_input = sum(h["input_length"] for h in summary_history)
    total_output = sum(h["output_length"] for h in summary_history)
    avg_compression = sum(h["compression_ratio"] for h in summary_history) / len(summary_history)
    
    # Calculate summary type distribution
    type_dist = {}
    for h in summary_history:
        stype = h["summary_type"]
        type_dist[stype] = type_dist.get(stype, 0) + 1
    
    return {
        "total_summaries": len(summary_history),
        "total_saved_chars": total_input - total_output,
        "avg_processing_time": f"{sum(h['processing_time'] for h in summary_history) / len(summary_history):.2f}s",
        "avg_compression": f"{avg_compression:.1f}%",
        "summary_type_distribution": type_dist,
        "total_input_chars": total_input,
        "total_output_chars": total_output,
        "history": summary_history[-10:]  # Last 10 entries
    }

@app.get("/history")
async def get_history():
    return {"history": summary_history}

@app.delete("/clear-history")
async def clear_history():
    global summary_history
    summary_history = []
    return {"status": "success", "message": "History cleared"}

@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NovaSumm | AI Neural Text Synthesizer</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Space Grotesk', sans-serif;
            transition: all 0.3s ease;
            overflow-x: hidden;
        }

        /* Dark Theme (Default) */
        body.dark {
            background: #0a0a0f;
        }

        body.light {
            background: #f5f7fa;
        }

        /* Animated Gradient Background */
        .gradient-bg {
            position: fixed;
            width: 100%;
            height: 100%;
            z-index: -2;
            transition: all 0.5s ease;
        }

        body.dark .gradient-bg {
            background: radial-gradient(circle at 20% 50%, rgba(30, 30, 60, 0.8), rgba(10, 10, 15, 0.95));
        }

        body.light .gradient-bg {
            background: radial-gradient(circle at 20% 50%, rgba(200, 200, 240, 0.8), rgba(245, 247, 250, 0.95));
        }

        /* Floating Particles */
        .particles {
            position: fixed;
            width: 100%;
            height: 100%;
            overflow: hidden;
            z-index: -1;
        }

        .particle {
            position: absolute;
            border-radius: 50%;
            animation: float 20s infinite ease-in-out;
        }

        body.dark .particle {
            background: linear-gradient(135deg, rgba(100, 108, 255, 0.4), rgba(139, 92, 246, 0.4));
        }

        body.light .particle {
            background: linear-gradient(135deg, rgba(100, 108, 255, 0.2), rgba(139, 92, 246, 0.2));
        }

        @keyframes float {
            0%, 100% {
                transform: translateY(0) translateX(0);
                opacity: 0;
            }
            10% {
                opacity: 0.5;
            }
            90% {
                opacity: 0.5;
            }
            100% {
                transform: translateY(-100vh) translateX(100px);
                opacity: 0;
            }
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            position: relative;
            z-index: 1;
        }

        /* Glass Morphism Navbar */
        .navbar {
            border-radius: 30px;
            padding: 20px 30px;
            margin-bottom: 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s ease;
        }

        body.dark .navbar {
            background: rgba(20, 20, 30, 0.7);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(100, 108, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        body.light .navbar {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(100, 108, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .logo-icon {
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, #646cff, #8b5cf6);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }

        .logo-icon::before {
            content: '';
            position: absolute;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.2), transparent);
            animation: shine 3s infinite;
        }

        @keyframes shine {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        .logo-icon i {
            font-size: 28px;
            color: white;
            position: relative;
            z-index: 1;
        }

        .logo-text {
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg, #fff, #a78bfa);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        body.light .logo-text {
            background: linear-gradient(135deg, #646cff, #8b5cf6);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .logo-badge {
            background: rgba(100, 108, 255, 0.2);
            border: 1px solid rgba(100, 108, 255, 0.3);
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            color: #a78bfa;
            margin-left: 10px;
        }

        /* Theme Toggle Button */
        .theme-toggle {
            width: 60px;
            height: 30px;
            background: rgba(100, 108, 255, 0.2);
            border-radius: 50px;
            position: relative;
            cursor: pointer;
            transition: all 0.3s;
            border: 1px solid rgba(100, 108, 255, 0.3);
        }

        .theme-toggle-slider {
            width: 26px;
            height: 26px;
            background: linear-gradient(135deg, #646cff, #8b5cf6);
            border-radius: 50%;
            position: absolute;
            top: 1px;
            left: 2px;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .theme-toggle-slider i {
            font-size: 14px;
            color: white;
        }

        body.light .theme-toggle-slider {
            left: 30px;
        }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }

        .stat-card {
            border-radius: 20px;
            padding: 20px;
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
            cursor: pointer;
        }

        body.dark .stat-card {
            background: rgba(20, 20, 30, 0.6);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(100, 108, 255, 0.2);
        }

        body.light .stat-card {
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(100, 108, 255, 0.2);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(100,108,255,0.1), transparent);
            transition: left 0.5s;
        }

        .stat-card:hover::before {
            left: 100%;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            border-color: rgba(100, 108, 255, 0.5);
        }

        body.dark .stat-card:hover {
            box-shadow: 0 10px 30px rgba(100, 108, 255, 0.2);
        }

        .stat-icon {
            font-size: 32px;
            margin-bottom: 10px;
        }

        .stat-value {
            font-size: 36px;
            font-weight: 700;
            background: linear-gradient(135deg, #fff, #a78bfa);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        body.light .stat-value {
            background: linear-gradient(135deg, #646cff, #8b5cf6);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .stat-label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 5px;
        }

        body.dark .stat-label {
            color: #888;
        }

        body.light .stat-label {
            color: #666;
        }

        /* Analytics Modal */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            z-index: 2000;
            justify-content: center;
            align-items: center;
            animation: fadeIn 0.3s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .modal-content {
            max-width: 800px;
            width: 90%;
            max-height: 80vh;
            border-radius: 30px;
            overflow-y: auto;
            animation: slideUp 0.3s ease-out;
        }

        @keyframes slideUp {
            from { transform: translateY(50px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        body.dark .modal-content {
            background: rgba(20, 20, 30, 0.95);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(100, 108, 255, 0.3);
        }

        body.light .modal-content {
            background: white;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(100, 108, 255, 0.3);
        }

        .modal-header {
            padding: 25px;
            border-bottom: 1px solid rgba(100, 108, 255, 0.2);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .modal-header h2 {
            font-size: 24px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        body.dark .modal-header h2 {
            color: white;
        }

        .close-modal {
            background: rgba(100, 108, 255, 0.1);
            border: none;
            width: 40px;
            height: 40px;
            border-radius: 12px;
            cursor: pointer;
            color: #a78bfa;
            font-size: 20px;
            transition: all 0.3s;
        }

        .close-modal:hover {
            background: rgba(100, 108, 255, 0.3);
            transform: scale(1.05);
        }

        .modal-body {
            padding: 25px;
        }

        .analytics-section {
            margin-bottom: 30px;
        }

        .analytics-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        body.dark .analytics-title {
            color: #a78bfa;
        }

        body.light .analytics-title {
            color: #646cff;
        }

        .analytics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }

        .analytics-card {
            padding: 15px;
            border-radius: 15px;
        }

        body.dark .analytics-card {
            background: rgba(30, 30, 40, 0.5);
            border: 1px solid rgba(100, 108, 255, 0.1);
        }

        body.light .analytics-card {
            background: rgba(240, 240, 250, 0.5);
            border: 1px solid rgba(100, 108, 255, 0.1);
        }

        .analytics-value {
            font-size: 28px;
            font-weight: 700;
        }

        body.dark .analytics-value {
            color: white;
        }

        body.light .analytics-value {
            color: #333;
        }

        .analytics-label {
            font-size: 12px;
            margin-top: 5px;
        }

        body.dark .analytics-label {
            color: #888;
        }

        body.light .analytics-label {
            color: #999;
        }

        .history-list {
            max-height: 300px;
            overflow-y: auto;
        }

        .history-item {
            padding: 12px;
            border-radius: 12px;
            margin-bottom: 10px;
        }

        body.dark .history-item {
            background: rgba(30, 30, 40, 0.5);
            border: 1px solid rgba(100, 108, 255, 0.1);
        }

        body.light .history-item {
            background: rgba(240, 240, 250, 0.5);
            border: 1px solid rgba(100, 108, 255, 0.1);
        }

        .history-time {
            font-size: 11px;
            margin-bottom: 5px;
        }

        body.dark .history-time {
            color: #888;
        }

        body.light .history-time {
            color: #999;
        }

        .history-stats {
            font-size: 13px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }

        body.dark .history-stats {
            color: #ccc;
        }

        body.light .history-stats {
            color: #666;
        }

        .clear-history-btn {
            background: linear-gradient(135deg, #dc3545, #c82333);
            border: none;
            padding: 10px 20px;
            border-radius: 12px;
            color: white;
            cursor: pointer;
            font-weight: 600;
            margin-top: 15px;
            width: 100%;
            transition: all 0.3s;
        }

        .clear-history-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(220, 53, 69, 0.3);
        }

        /* Main Split Layout */
        .split-layout {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }

        .card {
            border-radius: 30px;
            overflow: hidden;
            transition: all 0.3s;
        }

        body.dark .card {
            background: rgba(20, 20, 30, 0.6);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(100, 108, 255, 0.2);
        }

        body.light .card {
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(100, 108, 255, 0.2);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
        }

        .card:hover {
            transform: translateY(-5px);
        }

        body.dark .card:hover {
            border-color: rgba(100, 108, 255, 0.5);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }

        body.light .card:hover {
            border-color: rgba(100, 108, 255, 0.5);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }

        .card-header {
            padding: 20px 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        body.dark .card-header {
            background: rgba(30, 30, 40, 0.5);
            border-bottom: 1px solid rgba(100, 108, 255, 0.2);
        }

        body.light .card-header {
            background: rgba(240, 240, 250, 0.5);
            border-bottom: 1px solid rgba(100, 108, 255, 0.2);
        }

        .card-header h3 {
            font-size: 18px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        body.dark .card-header h3 {
            color: white;
        }

        body.light .card-header h3 {
            color: #333;
        }

        .card-actions {
            display: flex;
            gap: 8px;
        }

        .action-btn {
            background: rgba(100, 108, 255, 0.1);
            border: none;
            padding: 8px 12px;
            border-radius: 12px;
            color: #a78bfa;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 14px;
        }

        .action-btn:hover {
            background: rgba(100, 108, 255, 0.3);
            transform: scale(1.05);
        }

        .card-content {
            padding: 25px;
            min-height: 500px;
        }

        textarea {
            width: 100%;
            height: 100%;
            min-height: 450px;
            border: none;
            font-family: 'Space Grotesk', monospace;
            font-size: 15px;
            line-height: 1.6;
            resize: none;
            outline: none;
        }

        body.dark textarea {
            background: rgba(10, 10, 15, 0.5);
            color: white;
        }

        body.light textarea {
            background: rgba(245, 247, 250, 0.5);
            color: #333;
        }

        body.dark textarea::placeholder {
            color: #555;
        }

        body.light textarea::placeholder {
            color: #999;
        }

        .summary-content {
            line-height: 1.8;
            font-size: 15px;
            white-space: pre-wrap;
        }

        body.dark .summary-content {
            color: #e0e0e0;
        }

        body.light .summary-content {
            color: #444;
        }

        /* Controls Bar */
        .controls-bar {
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 30px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
        }

        body.dark .controls-bar {
            background: rgba(20, 20, 30, 0.6);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(100, 108, 255, 0.2);
        }

        body.light .controls-bar {
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(100, 108, 255, 0.2);
        }

        .control-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px 16px;
            border-radius: 15px;
        }

        body.dark .control-item {
            background: rgba(30, 30, 40, 0.5);
        }

        body.light .control-item {
            background: rgba(240, 240, 250, 0.5);
        }

        .control-item label {
            font-size: 13px;
            font-weight: 500;
            letter-spacing: 0.5px;
        }

        body.dark .control-item label {
            color: #a78bfa;
        }

        body.light .control-item label {
            color: #646cff;
        }

        select, input {
            background: rgba(10, 10, 15, 0.8);
            border: 1px solid rgba(100, 108, 255, 0.3);
            padding: 8px 15px;
            border-radius: 12px;
            font-family: 'Space Grotesk', sans-serif;
            cursor: pointer;
            transition: all 0.3s;
        }

        body.dark select, body.dark input {
            background: rgba(10, 10, 15, 0.8);
            color: white;
        }

        body.light select, body.light input {
            background: white;
            color: #333;
        }

        select:hover, input:hover {
            border-color: #646cff;
        }

        /* Action Buttons */
        .action-group {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 20px;
        }

        .btn-primary {
            background: linear-gradient(135deg, #646cff, #8b5cf6);
            border: none;
            padding: 14px 35px;
            border-radius: 50px;
            color: white;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 10px;
            position: relative;
            overflow: hidden;
        }

        .btn-primary::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            background: rgba(255,255,255,0.2);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }

        .btn-primary:hover::before {
            width: 300px;
            height: 300px;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(100, 108, 255, 0.4);
        }

        .btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .btn-secondary {
            padding: 14px 35px;
            border-radius: 50px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        body.dark .btn-secondary {
            background: rgba(30, 30, 40, 0.8);
            border: 1px solid rgba(100, 108, 255, 0.3);
            color: #a78bfa;
        }

        body.light .btn-secondary {
            background: white;
            border: 1px solid rgba(100, 108, 255, 0.3);
            color: #646cff;
        }

        .btn-secondary:hover {
            transform: translateY(-2px);
        }

        body.dark .btn-secondary:hover {
            background: rgba(100, 108, 255, 0.2);
        }

        body.light .btn-secondary:hover {
            background: rgba(100, 108, 255, 0.1);
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 30px;
            font-size: 12px;
        }

        body.dark .footer {
            color: #555;
        }

        body.light .footer {
            color: #999;
        }

        /* Loading Animation */
        .loader {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Toast */
        .toast {
            position: fixed;
            bottom: 30px;
            right: 30px;
            padding: 12px 24px;
            border-radius: 15px;
            z-index: 1000;
            animation: slideIn 0.3s ease-out;
            border-left: 3px solid #646cff;
        }

        body.dark .toast {
            background: rgba(20, 20, 30, 0.95);
            backdrop-filter: blur(20px);
            color: white;
        }

        body.light .toast {
            background: white;
            backdrop-filter: blur(20px);
            color: #333;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }

        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        /* Status Badge */
        .status-badge {
            position: fixed;
            bottom: 20px;
            left: 20px;
            padding: 8px 16px;
            border-radius: 12px;
            font-size: 11px;
            font-family: monospace;
            z-index: 100;
        }

        body.dark .status-badge {
            background: rgba(20, 20, 30, 0.8);
            backdrop-filter: blur(10px);
            color: #a78bfa;
        }

        body.light .status-badge {
            background: white;
            backdrop-filter: blur(10px);
            color: #646cff;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }

        @media (max-width: 968px) {
            .split-layout {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .controls-bar {
                flex-direction: column;
            }
            
            .control-item {
                width: 100%;
                justify-content: space-between;
            }
        }
    </style>
</head>
<body class="dark">
    <div class="gradient-bg"></div>
    <div class="particles" id="particles"></div>
    <div class="status-badge" id="status-badge">
        <i class="fas fa-circle" style="color: #4caf50; font-size: 8px;"></i> NEURAL ENGINE ACTIVE
    </div>

    <!-- Analytics Modal -->
    <div id="analytics-modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2><i class="fas fa-chart-line"></i> Quantum Analytics Dashboard</h2>
                <button class="close-modal" onclick="closeAnalytics()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="modal-body">
                <div id="analytics-data">
                    <div style="text-align: center; padding: 20px;">
                        <div class="loader"></div>
                        <p style="margin-top: 10px;">Loading quantum data...</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="container">
        <nav class="navbar">
            <div class="logo">
                <div class="logo-icon">
                    <i class="fas fa-brain"></i>
                </div>
                <div>
                    <span class="logo-text">NovaSumm</span>
                    <span class="logo-badge">QUANTUM v3.0</span>
                </div>
            </div>
            <div style="display: flex; gap: 15px; align-items: center;">
                <div class="theme-toggle" onclick="toggleTheme()">
                    <div class="theme-toggle-slider">
                        <i class="fas fa-moon" id="theme-icon"></i>
                    </div>
                </div>
                <div class="card-actions">
                    <button class="action-btn" onclick="openAnalytics()">
                        <i class="fas fa-chart-line"></i> ANALYTICS
                    </button>
                    <button class="action-btn" onclick="window.open('/docs', '_blank')">
                        <i class="fas fa-code"></i> API
                    </button>
                </div>
            </div>
        </nav>

        <div class="stats-grid" id="stats-grid" style="display: none;">
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-file-alt"></i></div>
                <div class="stat-value" id="total-summaries">0</div>
                <div class="stat-label">Total Processed</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-save"></i></div>
                <div class="stat-value" id="saved-chars">0</div>
                <div class="stat-label">Characters Saved</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-clock"></i></div>
                <div class="stat-value" id="avg-time">0s</div>
                <div class="stat-label">Avg Response</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-chart-line"></i></div>
                <div class="stat-value" id="avg-compression">0%</div>
                <div class="stat-label">Avg Compression</div>
            </div>
        </div>

        <div class="split-layout">
            <div class="card">
                <div class="card-header">
                    <h3><i class="fas fa-pen-fancy"></i> INPUT MATRIX</h3>
                    <div class="card-actions">
                        <button class="action-btn" onclick="clearInput()"><i class="fas fa-trash-alt"></i></button>
                        <button class="action-btn" onclick="loadExample()"><i class="fas fa-database"></i></button>
                        <button class="action-btn" onclick="copyInput()"><i class="fas fa-copy"></i></button>
                    </div>
                </div>
                <div class="card-content">
                    <textarea id="input-text" placeholder="// Initialize neural input stream...&#10;// Minimum 20 characters required for quantum synthesis"></textarea>
                </div>
                <div style="padding: 15px 25px; background: rgba(30,30,40,0.3); border-top: 1px solid rgba(100,108,255,0.1);">
                    <span id="char-counter" style="color: #888; font-size: 12px;"><i class="fas fa-info-circle"></i> 0 characters</span>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h3><i class="fas fa-star-of-life"></i> QUANTUM OUTPUT</h3>
                    <div class="card-actions">
                        <button class="action-btn" onclick="copySummary()"><i class="fas fa-copy"></i></button>
                        <button class="action-btn" onclick="downloadSummary()"><i class="fas fa-download"></i></button>
                        <button class="action-btn" onclick="speakSummary()"><i class="fas fa-waveform"></i></button>
                    </div>
                </div>
                <div class="card-content">
                    <div id="summary-output" class="summary-content">
                        <i class="fas fa-brain" style="margin-right: 10px; color: #646cff;"></i>
                        Awaiting quantum synthesis...
                    </div>
                </div>
                <div style="padding: 15px 25px; background: rgba(30,30,40,0.3); border-top: 1px solid rgba(100,108,255,0.1);">
                    <span id="summary-stats" style="color: #888; font-size: 12px;"><i class="fas fa-chart-simple"></i> Ready for processing</span>
                </div>
            </div>
        </div>

        <div class="controls-bar">
            <div class="control-item">
                <label><i class="fas fa-sliders-h"></i> SYNTHESIS MODE</label>
                <select id="summary-type">
                    <option value="general">GENERAL</option>
                    <option value="concise">CONCISE</option>
                    <option value="detailed">DETAILED</option>
                </select>
            </div>
            <div class="control-item">
                <label><i class="fas fa-ruler"></i> TOKEN LIMIT</label>
                <input type="range" id="max-length" min="50" max="500" value="150" style="width: 150px;">
                <span id="max-length-value" style="color: #a78bfa;">150</span>
            </div>
            <div class="control-item">
                <label><i class="fas fa-chart-line"></i> COMPRESSION</label>
                <span id="compression-rate" style="color: #4caf50;">0%</span>
            </div>
        </div>

        <div class="action-group">
            <button class="btn-primary" id="summarize-btn" onclick="summarize()">
                <i class="fas fa-play"></i> EXECUTE QUANTUM SYNTHESIS
            </button>
            <button class="btn-secondary" onclick="clearAll()">
                <i class="fas fa-undo-alt"></i> RESET SYSTEM
            </button>
        </div>

        <div class="footer">
            <p>⚡ QUANTUM NEURAL ENGINE v3.0 | POWERED BY T5 TRANSFORMER ARCHITECTURE</p>
        </div>
    </div>

    <script>
        let currentSummary = '';
        
        // Theme Toggle Function
        function toggleTheme() {
            const body = document.body;
            const themeIcon = document.getElementById('theme-icon');
            
            if (body.classList.contains('dark')) {
                body.classList.remove('dark');
                body.classList.add('light');
                themeIcon.classList.remove('fa-moon');
                themeIcon.classList.add('fa-sun');
                localStorage.setItem('theme', 'light');
                showToast('🌞 LIGHT MODE ACTIVATED');
            } else {
                body.classList.remove('light');
                body.classList.add('dark');
                themeIcon.classList.remove('fa-sun');
                themeIcon.classList.add('fa-moon');
                localStorage.setItem('theme', 'dark');
                showToast('🌙 DARK MODE ACTIVATED');
            }
        }
        
        // Load saved theme
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {
            document.body.classList.remove('dark');
            document.body.classList.add('light');
            document.getElementById('theme-icon').classList.remove('fa-moon');
            document.getElementById('theme-icon').classList.add('fa-sun');
        }
        
        // Analytics Modal Functions
        async function openAnalytics() {
            const modal = document.getElementById('analytics-modal');
            modal.style.display = 'flex';
            await loadAnalyticsData();
        }
        
        function closeAnalytics() {
            const modal = document.getElementById('analytics-modal');
            modal.style.display = 'none';
        }
        
        async function loadAnalyticsData() {
            try {
                const response = await fetch('/stats');
                const data = await response.json();
                
                const historyHtml = data.history && data.history.length > 0 ? 
                    data.history.map(item => `
                        <div class="history-item">
                            <div class="history-time">
                                <i class="far fa-clock"></i> ${new Date(item.timestamp).toLocaleString()}
                            </div>
                            <div class="history-stats">
                                <span><i class="fas fa-file-alt"></i> ${item.summary_type}</span>
                                <span><i class="fas fa-arrows-alt-h"></i> ${item.input_length} → ${item.output_length}</span>
                                <span><i class="fas fa-chart-line"></i> ${item.compression_ratio.toFixed(1)}%</span>
                                <span><i class="fas fa-tachometer-alt"></i> ${item.processing_time.toFixed(2)}s</span>
                            </div>
                        </div>
                    `).join('') : '<p style="text-align: center; padding: 20px;">No history available</p>';
                
                const analyticsHtml = `
                    <div class="analytics-section">
                        <div class="analytics-title">
                            <i class="fas fa-chart-pie"></i> Performance Metrics
                        </div>
                        <div class="analytics-grid">
                            <div class="analytics-card">
                                <div class="analytics-value">${data.total_summaries}</div>
                                <div class="analytics-label">Total Summaries</div>
                            </div>
                            <div class="analytics-card">
                                <div class="analytics-value">${data.total_saved_chars.toLocaleString()}</div>
                                <div class="analytics-label">Characters Saved</div>
                            </div>
                            <div class="analytics-card">
                                <div class="analytics-value">${data.avg_processing_time}</div>
                                <div class="analytics-label">Avg Processing Time</div>
                            </div>
                            <div class="analytics-card">
                                <div class="analytics-value">${data.avg_compression}</div>
                                <div class="analytics-label">Avg Compression Rate</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="analytics-section">
                        <div class="analytics-title">
                            <i class="fas fa-chart-bar"></i> Summary Type Distribution
                        </div>
                        <div class="analytics-grid">
                            ${Object.entries(data.summary_type_distribution || {}).map(([type, count]) => `
                                <div class="analytics-card">
                                    <div class="analytics-value">${count}</div>
                                    <div class="analytics-label">${type.toUpperCase()}</div>
                                </div>
                            `).join('')}
                            ${Object.keys(data.summary_type_distribution || {}).length === 0 ? '<p style="grid-column: span 2; text-align: center;">No data yet</p>' : ''}
                        </div>
                    </div>
                    
                    <div class="analytics-section">
                        <div class="analytics-title">
                            <i class="fas fa-history"></i> Recent Activity
                        </div>
                        <div class="history-list">
                            ${historyHtml}
                        </div>
                        <button class="clear-history-btn" onclick="clearHistoryData()">
                            <i class="fas fa-trash-alt"></i> Clear All History
                        </button>
                    </div>
                `;
                
                document.getElementById('analytics-data').innerHTML = analyticsHtml;
            } catch (error) {
                document.getElementById('analytics-data').innerHTML = `
                    <div style="text-align: center; padding: 20px; color: #dc3545;">
                        <i class="fas fa-exclamation-triangle"></i>
                        <p>Error loading analytics data</p>
                    </div>
                `;
            }
        }
        
        async function clearHistoryData() {
            if (confirm('⚠️ WARNING: This will permanently delete all summary history. Continue?')) {
                try {
                    await fetch('/clear-history', { method: 'DELETE' });
                    showToast('History cleared successfully');
                    await loadAnalyticsData();
                    loadStats();
                } catch (error) {
                    showToast('Error clearing history', true);
                }
            }
        }
        
        function createParticles() {
            const container = document.getElementById('particles');
            for (let i = 0; i < 50; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                const size = Math.random() * 4 + 2;
                particle.style.width = size + 'px';
                particle.style.height = size + 'px';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 20 + 's';
                particle.style.animationDuration = (Math.random() * 15 + 10) + 's';
                container.appendChild(particle);
            }
        }
        
        createParticles();
        
        function showToast(message, isError = false) {
            const toast = document.createElement('div');
            toast.className = 'toast';
            toast.innerHTML = message;
            document.body.appendChild(toast);
            setTimeout(() => toast.remove(), 3000);
        }
        
        async function summarize() {
            const text = document.getElementById('input-text').value;
            if (!text.trim()) {
                showToast('⚠️ INPUT REQUIRED: Please provide text for analysis', true);
                return;
            }
            
            if (text.length < 20) {
                showToast('⚠️ INSUFFICIENT DATA: Minimum 20 characters required', true);
                return;
            }
            
            const summaryType = document.getElementById('summary-type').value.toLowerCase();
            const maxLength = document.getElementById('max-length').value;
            
            const btn = document.getElementById('summarize-btn');
            const originalText = btn.innerHTML;
            btn.innerHTML = '<div class="loader"></div> QUANTUM PROCESSING...';
            btn.disabled = true;
            
            try {
                const startTime = Date.now();
                const response = await fetch('/summarize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        dialogue: text,
                        summary_type: summaryType,
                        max_length: parseInt(maxLength)
                    })
                });
                
                const data = await response.json();
                const processingTime = ((Date.now() - startTime) / 1000).toFixed(2);
                
                if (data.status === 'success') {
                    currentSummary = data.summary;
                    document.getElementById('summary-output').innerHTML = data.summary;
                    document.getElementById('summary-stats').innerHTML = `<i class="fas fa-chart-simple"></i> ${data.metrics.output_length} chars | ${data.metrics.compression_ratio} compression`;
                    
                    const compression = parseFloat(data.metrics.compression_ratio);
                    const compressionElem = document.getElementById('compression-rate');
                    compressionElem.innerHTML = data.metrics.compression_ratio;
                    compressionElem.style.color = compression > 50 ? '#4caf50' : '#ff9800';
                    
                    showToast(`✅ QUANTUM SYNTHESIS COMPLETE: ${processingTime}s`);
                    loadStats();
                } else {
                    document.getElementById('summary-output').innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${data.summary}`;
                    showToast('❌ SYNTHESIS ERROR', true);
                }
            } catch (error) {
                showToast('⚠️ NETWORK ERROR: Check connection', true);
            } finally {
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        }
        
        async function loadStats() {
            try {
                const response = await fetch('/stats');
                const data = await response.json();
                if (data.total_summaries > 0) {
                    document.getElementById('stats-grid').style.display = 'grid';
                    document.getElementById('total-summaries').textContent = data.total_summaries;
                    document.getElementById('saved-chars').textContent = data.total_saved_chars.toLocaleString();
                    document.getElementById('avg-time').textContent = data.avg_processing_time;
                    document.getElementById('avg-compression').textContent = data.avg_compression;
                }
            } catch (error) {
                console.error('Stats error:', error);
            }
        }
        
        function updateCharCounter() {
            const text = document.getElementById('input-text').value;
            const length = text.length;
            const counter = document.getElementById('char-counter');
            counter.innerHTML = `<i class="fas fa-info-circle"></i> ${length} characters`;
            if (length > 0 && length < 20) {
                counter.style.color = '#ff9800';
            } else if (length >= 20) {
                counter.style.color = '#4caf50';
            } else {
                counter.style.color = '#888';
            }
        }
        
        function loadExample() {
            const example = `Quantum computing represents a paradigm shift in computational capabilities, leveraging the principles of superposition and entanglement to solve problems that are intractable for classical computers. Unlike classical bits that exist in states of either 0 or 1, quantum bits or qubits can exist in multiple states simultaneously, enabling parallel computation at unprecedented scales. This technology holds immense promise for fields such as cryptography, drug discovery, materials science, and artificial intelligence. Recent breakthroughs in quantum supremacy demonstrations have shown that quantum processors can perform specific calculations in seconds that would take classical supercomputers thousands of years. However, significant challenges remain in error correction, qubit coherence times, and scaling quantum systems to practical sizes. The race to develop fault-tolerant quantum computers has attracted investments from major technology companies and governments worldwide, signaling a new era of computational innovation.`;
            document.getElementById('input-text').value = example;
            updateCharCounter();
            showToast('📀 QUANTUM BENCHMARK DATA LOADED');
        }
        
        function clearInput() {
            document.getElementById('input-text').value = '';
            updateCharCounter();
            showToast('🔄 INPUT MATRIX CLEARED');
        }
        
        function clearAll() {
            document.getElementById('input-text').value = '';
            document.getElementById('summary-output').innerHTML = '<i class="fas fa-brain" style="margin-right: 10px; color: #646cff;"></i> Awaiting quantum synthesis...';
            document.getElementById('summary-stats').innerHTML = '<i class="fas fa-chart-simple"></i> Ready for processing';
            document.getElementById('compression-rate').innerHTML = '0%';
            currentSummary = '';
            updateCharCounter();
            showToast('🔄 SYSTEM RESET COMPLETE');
        }
        
        function copyInput() {
            const text = document.getElementById('input-text').value;
            if (text) {
                navigator.clipboard.writeText(text);
                showToast('📋 INPUT COPIED TO CLIPBOARD');
            }
        }
        
        function copySummary() {
            if (currentSummary) {
                navigator.clipboard.writeText(currentSummary);
                showToast('📋 QUANTUM OUTPUT COPIED');
            } else {
                showToast('⚠️ NO OUTPUT TO COPY', true);
            }
        }
        
        function downloadSummary() {
            if (currentSummary) {
                const blob = new Blob([currentSummary], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `quantum_synthesis_${Date.now()}.txt`;
                a.click();
                URL.revokeObjectURL(url);
                showToast('💾 OUTPUT ARCHIVED');
            } else {
                showToast('⚠️ NO OUTPUT TO DOWNLOAD', true);
            }
        }
        
        function speakSummary() {
            if (currentSummary && 'speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(currentSummary);
                utterance.rate = 0.9;
                utterance.pitch = 1;
                window.speechSynthesis.speak(utterance);
                showToast('🔊 AUDIO WAVEFORM ACTIVE');
            } else {
                showToast('⚠️ AUDIO UNAVAILABLE', true);
            }
        }
        
        document.getElementById('max-length').addEventListener('input', function() {
            document.getElementById('max-length-value').textContent = this.value;
        });
        
        document.getElementById('input-text').addEventListener('input', function() {
            localStorage.setItem('novaSummText', this.value);
            updateCharCounter();
        });
        
        const savedText = localStorage.getItem('novaSummText');
        if (savedText) {
            document.getElementById('input-text').value = savedText;
            updateCharCounter();
        }
        
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                summarize();
            }
        });
        
        loadStats();
        
        // Animated status indicator
        setInterval(() => {
            const statusDot = document.querySelector('.status-badge i');
            if (statusDot) {
                statusDot.style.opacity = statusDot.style.opacity === '0.3' ? '1' : '0.3';
            }
        }, 1000);
        
        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('analytics-modal');
            if (event.target === modal) {
                closeAnalytics();
            }
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "cache_size": len(summary_cache),
        "history_count": len(summary_history),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║     🚀 NovaSumm QUANTUM v3.0 - Neural Text Synthesis Engine        ║
    ║                                                                      ║
    ║     📍 Quantum Core: http://127.0.0.1:8000                         ║
    ║     📖 API Gateway: http://127.0.0.1:8000/docs                     ║
    ║     🔍 System Health: http://127.0.0.1:8000/health                 ║
    ║                                                                      ║
    ║     ✨ NEW FEATURES:                                               ║
    ║     • Fully functional Analytics Dashboard                         ║
    ║     • Real-time performance metrics                                ║
    ║     • Summary type distribution charts                             ║
    ║     • Recent activity history                                      ║
    ║     • Clear history functionality                                  ║
    ║     • Modal popup with detailed stats                              ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
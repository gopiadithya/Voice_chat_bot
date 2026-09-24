"""
VoiceBot AI — Intelligent Conversational Agent
==============================================
YouTube-style voice conversational assistant:
  - Click to speak -> streams speech live on screen in real time.
  - Automatic silence detection (stops when you finish speaking, just like YouTube).
  - Directly delivers response without requiring a submit button.
  - Speaks answer aloud automatically via Web Speech Synthesis.
  - Seamless automatic rollback to trained BiLSTM model if cloud API fails.
"""

import os
import re
import time
import json
import pickle
import random
import requests
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load local .env if available
load_dotenv()

# ─────────────────────────────────────────────────────────
# PAGE CONFIGURATION & STYLING
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VoiceBot AI — Conversational Agent",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom styling with vibrant animated background, cosmic meteors, and glowing cyber orbs
st.markdown("""
<style>
    /* Full App Deep Canvas */
    .stApp {
        background-color: #030712 !important;
        overflow-x: hidden;
    }

    /* Ambient animated container */
    .animated-bg-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
        z-index: 0;
        pointer-events: none;
    }

    /* Floating glowing neon orbs */
    .orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(85px);
        opacity: 0.65;
        animation-timing-function: ease-in-out;
        animation-iteration-count: infinite;
        animation-direction: alternate;
    }
    .orb-1 {
        width: 540px;
        height: 540px;
        background: radial-gradient(circle, #2563eb 0%, #1e40af 60%, transparent 100%);
        top: -12%;
        left: -10%;
        animation: floatOrb1 14s infinite alternate;
    }
    .orb-2 {
        width: 580px;
        height: 580px;
        background: radial-gradient(circle, #8b5cf6 0%, #6d28d9 60%, transparent 100%);
        top: 20%;
        right: -15%;
        animation: floatOrb2 17s infinite alternate;
    }
    .orb-3 {
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, #06b6d4 0%, #0e7490 60%, transparent 100%);
        bottom: -10%;
        left: 10%;
        animation: floatOrb3 15s infinite alternate;
    }
    .orb-4 {
        width: 440px;
        height: 440px;
        background: radial-gradient(circle, #ec4899 0%, #a21caf 60%, transparent 100%);
        top: 48%;
        left: 38%;
        opacity: 0.45;
        animation: floatOrb4 20s infinite alternate;
    }
    .orb-5 {
        width: 470px;
        height: 470px;
        background: radial-gradient(circle, #10b981 0%, #047857 60%, transparent 100%);
        bottom: 5%;
        right: 12%;
        opacity: 0.4;
        animation: floatOrb5 16s infinite alternate;
    }

    @keyframes floatOrb1 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(160px, 90px) scale(1.18); }
        100% { transform: translate(80px, 170px) scale(0.92); }
    }
    @keyframes floatOrb2 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(-140px, 90px) scale(1.22); }
        100% { transform: translate(-90px, -130px) scale(0.88); }
    }
    @keyframes floatOrb3 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(120px, -100px) scale(1.15); }
        100% { transform: translate(-70px, -60px) scale(1.05); }
    }
    @keyframes floatOrb4 {
        0% { transform: translate(0, 0) scale(0.9); }
        50% { transform: translate(-100px, 120px) scale(1.2); }
        100% { transform: translate(80px, -90px) scale(0.95); }
    }
    @keyframes floatOrb5 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(-120px, -110px) scale(1.18); }
        100% { transform: translate(60px, -70px) scale(0.88); }
    }

    /* Aurora Bioluminescent Ribbons */
    .aurora-ribbon {
        position: absolute;
        width: 130vw;
        height: 340px;
        left: -15vw;
        filter: blur(85px);
        opacity: 0.28;
        pointer-events: none;
        border-radius: 50%;
    }
    .aurora-1 {
        top: -80px;
        background: radial-gradient(ellipse at 50% 50%, #38bdf8 0%, #06b6d4 40%, transparent 70%);
        animation: auroraWave1 22s ease-in-out infinite alternate;
    }
    .aurora-2 {
        top: 32%;
        background: radial-gradient(ellipse at 50% 50%, #8b5cf6 0%, #ec4899 40%, transparent 70%);
        animation: auroraWave2 26s ease-in-out infinite alternate;
        opacity: 0.20;
    }
    @keyframes auroraWave1 {
        0% { transform: translateY(0px) rotate(0deg) scaleY(1); }
        50% { transform: translateY(50px) rotate(3deg) scaleY(1.2); }
        100% { transform: translateY(-30px) rotate(-2deg) scaleY(0.92); }
    }
    @keyframes auroraWave2 {
        0% { transform: translateY(0px) rotate(0deg) scaleX(1); }
        50% { transform: translateY(-60px) rotate(-4deg) scaleX(1.15); }
        100% { transform: translateY(40px) rotate(2deg) scaleX(0.95); }
    }

    /* Rotating Cyber Constellation Rings */
    .cyber-ring {
        position: absolute;
        border-radius: 50%;
        border: 1px dashed rgba(96, 165, 250, 0.14);
        pointer-events: none;
        animation: spinRing 45s linear infinite;
    }
    .ring-1 {
        width: 700px;
        height: 700px;
        top: -180px;
        right: -160px;
    }
    .ring-2 {
        width: 480px;
        height: 480px;
        bottom: 8%;
        left: -120px;
        border-color: rgba(192, 132, 252, 0.12);
        animation-duration: 55s;
        animation-direction: reverse;
    }
    @keyframes spinRing {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    /* Diagonal Cosmic Shooting Stars */
    .shooting-star {
        position: absolute;
        height: 2px;
        background: linear-gradient(-45deg, #38bdf8, rgba(56, 189, 248, 0));
        filter: drop-shadow(0 0 6px #38bdf8);
        animation: meteorAnimation 11s ease-in-out infinite;
        opacity: 0;
    }
    .star-1 {
        top: 14%;
        right: 12%;
        width: 140px;
        animation-delay: 2s;
    }
    .star-2 {
        top: 36%;
        right: 26%;
        width: 180px;
        animation-delay: 6.5s;
        background: linear-gradient(-45deg, #c084fc, rgba(192, 132, 252, 0));
        filter: drop-shadow(0 0 6px #c084fc);
    }
    .star-3 {
        top: 60%;
        right: 18%;
        width: 120px;
        animation-delay: 10s;
        background: linear-gradient(-45deg, #34d399, rgba(52, 211, 153, 0));
        filter: drop-shadow(0 0 6px #34d399);
    }
    @keyframes meteorAnimation {
        0% {
            transform: rotate(-35deg) translateX(0);
            opacity: 0;
        }
        5% {
            opacity: 1;
        }
        14% {
            transform: rotate(-35deg) translateX(-680px);
            opacity: 0;
        }
        100% {
            transform: rotate(-35deg) translateX(-680px);
            opacity: 0;
        }
    }

    /* Ambient Holographic Vertical Light Scan */
    .holographic-beam {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 140px;
        background: linear-gradient(180deg, transparent 0%, rgba(56, 189, 248, 0.035) 50%, transparent 100%);
        animation: beamScan 14s ease-in-out infinite;
        pointer-events: none;
    }
    @keyframes beamScan {
        0% { transform: translateY(-150px); opacity: 0; }
        20% { opacity: 1; }
        80% { opacity: 1; }
        100% { transform: translateY(110vh); opacity: 0; }
    }

    /* Cyber grid overlay */
    .cyber-grid {
        position: absolute;
        inset: 0;
        background-image: 
            linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px);
        background-size: 55px 55px;
        mask-image: radial-gradient(circle at 50% 50%, black 45%, transparent 88%);
        pointer-events: none;
    }

    /* Floating glowing star particles */
    .particle {
        position: absolute;
        border-radius: 50%;
        animation: floatParticle 9s infinite ease-in-out alternate;
    }
    .p1 { width: 4px; height: 4px; top: 16%; left: 12%; background: #38bdf8; box-shadow: 0 0 14px 3px #38bdf8; animation-duration: 9s; }
    .p2 { width: 5px; height: 5px; top: 32%; right: 16%; background: #c084fc; box-shadow: 0 0 15px 3px #c084fc; animation-duration: 11s; }
    .p3 { width: 4px; height: 4px; top: 62%; left: 20%; background: #34d399; box-shadow: 0 0 14px 3px #34d399; animation-duration: 13s; }
    .p4 { width: 3px; height: 3px; top: 78%; right: 24%; background: #38bdf8; box-shadow: 0 0 12px 2px #38bdf8; animation-duration: 10s; }
    .p5 { width: 5px; height: 5px; top: 12%; right: 30%; background: #f472b6; box-shadow: 0 0 16px 3px #f472b6; animation-duration: 14s; }
    .p6 { width: 3px; height: 3px; top: 48%; left: 8%; background: #38bdf8; box-shadow: 0 0 12px 2px #38bdf8; animation-duration: 12s; }
    .p7 { width: 4px; height: 4px; top: 84%; left: 45%; background: #facc15; box-shadow: 0 0 14px 3px #facc15; animation-duration: 15s; }
    .p8 { width: 3px; height: 3px; top: 25%; left: 55%; background: #60a5fa; box-shadow: 0 0 12px 2px #60a5fa; animation-duration: 10s; }
    .p9 { width: 4px; height: 4px; top: 70%; right: 40%; background: #c084fc; box-shadow: 0 0 14px 3px #c084fc; animation-duration: 12s; }
    .p10 { width: 3px; height: 3px; top: 92%; left: 16%; background: #34d399; box-shadow: 0 0 12px 2px #34d399; animation-duration: 13s; }

    @keyframes floatParticle {
        0% { transform: translateY(0px) translateX(0px); opacity: 0.3; }
        50% { transform: translateY(-40px) translateX(25px); opacity: 0.95; }
        100% { transform: translateY(15px) translateX(-20px); opacity: 0.4; }
    }

    /* Main container bounds - Instagram-style scrollable chat feed */
    .main .block-container {
        max-width: 820px !important;
        padding-top: 86px !important;    /* Clears fixed Instagram header */
        padding-bottom: 118px !important; /* Clears fixed bottom input dock */
        position: relative;
        z-index: 1;
    }

    /* ─── FIXED TOP-CENTER INSTAGRAM-STYLE HEADER ─── */
    .instagram-top-bar {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: 60px !important;
        z-index: 9998 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: rgba(3, 7, 18, 0.88) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding: 0 16px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        pointer-events: none !important;
    }

    .instagram-top-bar * {
        pointer-events: auto !important;
    }

    /* Shift header centering when sidebar is expanded */
    body:has([data-testid="stSidebar"][aria-expanded="true"]) .instagram-top-bar {
        padding-left: 336px !important;
    }

    .theme-light-active .instagram-top-bar,
    [data-theme="light"] .instagram-top-bar {
        background: rgba(255, 255, 255, 0.9) !important;
        border-bottom: 1px solid rgba(0, 0, 0, 0.08) !important;
    }

    .instagram-header-center {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 12px !important;
        text-align: left !important;
    }

    .insta-avatar-ring {
        position: relative !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 38px !important;
        height: 38px !important;
        border-radius: 50% !important;
        background: rgba(37, 99, 235, 0.2) !important;
        border: 1px solid rgba(59, 130, 246, 0.4) !important;
        box-shadow: 0 0 12px rgba(37, 99, 235, 0.3) !important;
        flex-shrink: 0 !important;
    }

    .insta-active-dot {
        position: absolute !important;
        bottom: 0px !important;
        right: 0px !important;
        width: 9px !important;
        height: 9px !important;
        border-radius: 50% !important;
        background: #22c55e !important;
        box-shadow: 0 0 8px #22c55e !important;
        border: 2px solid #030712 !important;
    }

    .insta-title-wrap {
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-start !important;
        justify-content: center !important;
    }

    .insta-main-title {
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.01em !important;
        background: linear-gradient(90deg, #60a5fa, #c084fc, #34d399, #38bdf8, #60a5fa) !important;
        background-size: 200% auto !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        line-height: 1.25 !important;
    }

    .insta-sub-title {
        font-size: 0.72rem !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em !important;
        line-height: 1.2 !important;
    }

    .theme-light-active .insta-sub-title,
    [data-theme="light"] .insta-sub-title {
        color: #64748b !important;
    }

    /* ─── FIXED BOTTOM-CENTER INPUT DOCK (NEVER MOVES) ─── */
    .fixed-bottom-input-dock,
    div.element-container[class*="st-key-unified_input_bar"],
    div.stElementContainer[class*="st-key-unified_input_bar"],
    div.element-container:has(iframe[title*="voice_input_widget"]) {
        position: fixed !important;
        bottom: 18px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: calc(100% - 32px) !important;
        max-width: 620px !important;
        z-index: 9998 !important;
        margin: 0 !important;
        pointer-events: auto !important;
        transition: left 0.3s cubic-bezier(0.4, 0, 0.2, 1), max-width 0.3s ease !important;
    }

    body:has([data-testid="stSidebar"][aria-expanded="true"]) .fixed-bottom-input-dock,
    body:has([data-testid="stSidebar"][aria-expanded="true"]) div.element-container[class*="st-key-unified_input_bar"],
    body:has([data-testid="stSidebar"][aria-expanded="true"]) div.stElementContainer[class*="st-key-unified_input_bar"],
    body:has([data-testid="stSidebar"][aria-expanded="true"]) div.element-container:has(iframe[title*="voice_input_widget"]) {
        left: calc(50% + 168px) !important;
        max-width: min(620px, calc(100vw - 336px - 150px)) !important;
    }

    /* Shimmering Holographic Title */
    @keyframes titleShine {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .gradient-title {
        background: linear-gradient(90deg, #60a5fa, #c084fc, #34d399, #38bdf8, #60a5fa);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titleShine 5s ease infinite;
    }

    /* Modern Glassmorphism Chat Bubbles */
    div[data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.72) !important;
        backdrop-filter: blur(18px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 18px !important;
        padding: 14px 20px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    div[data-testid="stChatMessage"]:hover {
        border-color: rgba(96, 165, 250, 0.45) !important;
        transform: translateY(-1px);
    }

    .badge-bilstm {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.4);
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-fallback {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }

    /* ─── SIDEBAR MODERN GLASSMORPHISM ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8, 14, 28, 0.96) 0%, rgba(3, 7, 18, 0.98) 100%) !important;
        backdrop-filter: blur(24px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        display: flex !important;
        flex-direction: column !important;
        min-height: calc(100vh - 4.5rem) !important;
        padding-bottom: 1.2rem !important;
    }

    /* Expander card in sidebar */
    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.55) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"]:hover {
        border-color: rgba(96, 165, 250, 0.35) !important;
    }

    /* ─── FLOATING CLEAR CHAT BUTTON (OUTSIDE PANEL, BOTTOM-LEFT CORNER) ─── */
    .floating-corner-clear-wrap,
    div.st-key-main_corner_clear_btn,
    div.element-container.st-key-main_corner_clear_btn,
    div.stElementContainer.st-key-main_corner_clear_btn {
        position: fixed !important;
        bottom: 22px !important;
        left: 20px !important;
        z-index: 99999 !important;
        width: auto !important;
        margin: 0 !important;
        transition: left 0.3s cubic-bezier(0.4, 0, 0.2, 1), transform 0.2s ease !important;
    }

    /* When sidebar is expanded, smoothly shift button outside the sidebar panel */
    body:has([data-testid="stSidebar"][aria-expanded="true"]) .floating-corner-clear-wrap,
    body:has([data-testid="stSidebar"][aria-expanded="true"]) div.st-key-main_corner_clear_btn,
    body:has([data-testid="stSidebar"][aria-expanded="true"]) div.element-container.st-key-main_corner_clear_btn,
    body:has([data-testid="stSidebar"][aria-expanded="true"]) div.stElementContainer.st-key-main_corner_clear_btn {
        left: calc(336px + 20px) !important;
    }

    .floating-corner-clear-wrap button,
    div.st-key-main_corner_clear_btn button {
        border-radius: 9999px !important;
        padding: 8px 14px !important;
        background: rgba(15, 23, 42, 0.92) !important;
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
        border: 1px solid rgba(239, 68, 68, 0.45) !important;
        color: #fca5a5 !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        box-shadow: 0 4px 22px rgba(0, 0, 0, 0.5) !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        white-space: nowrap !important;
        transition: all 0.25s ease !important;
        cursor: pointer !important;
    }

    .floating-corner-clear-wrap button:hover,
    div.st-key-main_corner_clear_btn button:hover {
        background: rgba(239, 68, 68, 0.22) !important;
        border-color: rgba(239, 68, 68, 0.8) !important;
        color: #ffffff !important;
        box-shadow: 0 6px 26px rgba(239, 68, 68, 0.4) !important;
        transform: translateY(-2px) !important;
    }

    /* ─── TOP HEADER & TOOLBAR ACCESSIBILITY (THEMES, MENU, SHARE, GITHUB) ─── */
    header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 1000000 !important;
        pointer-events: auto !important;
    }
    [data-testid="stToolbar"],
    [data-testid="stToolbarActions"],
    [data-testid="stMainMenu"],
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] a {
        pointer-events: auto !important;
        z-index: 1000001 !important;
    }

    /* Ensure animated canvas elements NEVER capture pointer events */
    .animated-bg-container,
    .animated-bg-container * {
        pointer-events: none !important;
        user-select: none !important;
    }

    /* Remove Streamlit default black bottom container & footer */
    [data-testid="stBottom"], [data-testid="stBottom"] > div, footer {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    footer {
        display: none !important;
        visibility: hidden !important;
    }

    /* ─── STREAMLIT THEME ADAPTATION (SYSTEM, LIGHT, DARK) ─── */
    html.theme-light-active .stApp,
    body.theme-light-active .stApp,
    [data-theme="light"] .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    html.theme-light-active .animated-bg-container .orb,
    [data-theme="light"] .animated-bg-container .orb {
        opacity: 0.18 !important;
        filter: blur(100px) !important;
    }

    html.theme-light-active .cyber-grid,
    [data-theme="light"] .cyber-grid {
        background-image: 
            linear-gradient(rgba(0, 0, 0, 0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 0, 0, 0.04) 1px, transparent 1px) !important;
        opacity: 0.55 !important;
    }

    html.theme-light-active div[data-testid="stChatMessage"],
    [data-theme="light"] div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(0, 0, 0, 0.09) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06) !important;
        color: #0f172a !important;
    }

    html.theme-light-active div[data-testid="stChatMessage"] *,
    [data-theme="light"] div[data-testid="stChatMessage"] * {
        color: #0f172a !important;
    }

    html.theme-light-active section[data-testid="stSidebar"],
    [data-theme="light"] section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(248, 250, 252, 0.98) 0%, rgba(241, 245, 249, 0.98) 100%) !important;
        border-right: 1px solid rgba(0, 0, 0, 0.08) !important;
    }

    html.theme-light-active section[data-testid="stSidebar"] *,
    [data-theme="light"] section[data-testid="stSidebar"] * {
        color: #0f172a !important;
    }

    html.theme-light-active .floating-corner-clear-wrap button,
    [data-theme="light"] .floating-corner-clear-wrap button {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(239, 68, 68, 0.45) !important;
        color: #dc2626 !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.08) !important;
    }
</style>

<!-- Animated Background Canvas Elements (Orbs, Aurora, Meteors, Rings, Particles) -->
<div class="animated-bg-container">
    <div class="aurora-ribbon aurora-1"></div>
    <div class="aurora-ribbon aurora-2"></div>
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
    <div class="orb orb-4"></div>
    <div class="orb orb-5"></div>
    <div class="cyber-ring ring-1"></div>
    <div class="cyber-ring ring-2"></div>
    <div class="shooting-star star-1"></div>
    <div class="shooting-star star-2"></div>
    <div class="shooting-star star-3"></div>
    <div class="holographic-beam"></div>
    <div class="cyber-grid"></div>
    <div class="particle p1"></div>
    <div class="particle p2"></div>
    <div class="particle p3"></div>
    <div class="particle p4"></div>
    <div class="particle p5"></div>
    <div class="particle p6"></div>
    <div class="particle p7"></div>
    <div class="particle p8"></div>
    <div class="particle p9"></div>
    <div class="particle p10"></div>
</div>

<!-- Dynamic Theme & Corner Button Synchronizer -->
<script>
(function() {
    function pinBottomDock() {
        var el = document.querySelector('[class*="st-key-unified_input_bar"]');
        if (el && !el.classList.contains('fixed-bottom-input-dock')) {
            el.classList.add('fixed-bottom-input-dock');
        }
        var iframes = document.querySelectorAll('iframe');
        iframes.forEach(function(f) {
            if ((f.title && f.title.includes('voice_input_widget')) || (f.src && f.src.includes('voice_input_widget'))) {
                var container = f.closest('.element-container') || f.parentElement;
                if (container && !container.classList.contains('fixed-bottom-input-dock')) {
                    container.classList.add('fixed-bottom-input-dock');
                }
            }
        });
    }

    function positionClearBtn() {
        var el = document.querySelector('.st-key-main_corner_clear_btn');
        if (el && !el.classList.contains('floating-corner-clear-wrap')) {
            el.classList.add('floating-corner-clear-wrap');
        }
    }

    function syncTheme() {
        var isLight = false;
        var themeAttr = document.documentElement.getAttribute('data-theme') || 
                        document.body.getAttribute('data-theme') ||
                        document.body.getAttribute('data-base-theme');
        if (themeAttr === 'light') {
            isLight = true;
        } else {
            var stApp = document.querySelector('.stApp');
            var bg = stApp ? window.getComputedStyle(stApp).backgroundColor : '';
            if (!bg || bg.indexOf('rgba(0, 0, 0, 0)') !== -1 || bg === 'transparent') {
                bg = window.getComputedStyle(document.body).backgroundColor;
            }
            var rgb = (bg || '').match(/\d+/g);
            if (rgb && rgb.length >= 3) {
                var lum = (0.299 * parseInt(rgb[0]) + 0.587 * parseInt(rgb[1]) + 0.114 * parseInt(rgb[2]));
                if (lum > 140) isLight = true;
            }
        }

        if (isLight) {
            document.documentElement.classList.add('theme-light-active');
            document.body.classList.add('theme-light-active');
            document.documentElement.classList.remove('theme-dark-active');
            document.body.classList.remove('theme-dark-active');
        } else {
            document.documentElement.classList.add('theme-dark-active');
            document.body.classList.add('theme-dark-active');
            document.documentElement.classList.remove('theme-light-active');
            document.body.classList.remove('theme-light-active');
        }
    }

    var lastScrollTime = 0;
    function scrollChatToBottom(force) {
        var now = Date.now();
        if (!force && (now - lastScrollTime < 50)) return;
        lastScrollTime = now;

        var scrollingEl = document.scrollingElement || document.documentElement || document.body;
        var mainSec = document.querySelector('section.main') || document.querySelector('.stMain');

        window.scrollTo({
            top: scrollingEl.scrollHeight,
            behavior: 'smooth'
        });
        if (mainSec) {
            mainSec.scrollTo({
                top: mainSec.scrollHeight,
                behavior: 'smooth'
            });
        }
    }

    pinBottomDock();
    positionClearBtn();
    syncTheme();

    var observer = new MutationObserver(function(mutations) {
        pinBottomDock();
        positionClearBtn();
        syncTheme();

        var hasTextStream = mutations.some(function(m) {
            return m.type === 'characterData' ||
                   (m.target && m.target.closest && m.target.closest('[data-testid="stChatMessage"]'));
        });
        if (hasTextStream) {
            scrollChatToBottom(false);
        }
    });

    observer.observe(document.documentElement, {
        attributes: true,
        subtree: true,
        childList: true,
        characterData: true
    });

    window.addEventListener('resize', function() {
        pinBottomDock();
        syncTheme();
    });

    setInterval(function() {
        pinBottomDock();
        positionClearBtn();
        syncTheme();
    }, 500);
})();
</script>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# SECRET & ENVIRONMENT HELPER
# ─────────────────────────────────────────────────────────

def get_secret(key_name: str) -> str:
    """Retrieve secret safely from .env, OS environment, or Streamlit secrets."""
    val = os.getenv(key_name, "")
    if not val:
        try:
            if hasattr(st, "secrets") and key_name in st.secrets:
                val = st.secrets[key_name]
        except Exception:
            pass
    return str(val).strip() if val else ""


# ─────────────────────────────────────────────────────────
# LOAD LOCAL DEEP LEARNING MODEL & ARTIFACTS
# ─────────────────────────────────────────────────────────

@st.cache_resource
def load_bilstm_pipeline():
    """Load the trained BiLSTM model and tokenization objects."""
    model = load_model("chatbot_model.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    with open("intents.json", "r", encoding="utf-8") as f:
        intents_data = json.load(f)

    responses_lookup = {}
    for intent in intents_data["intents"]:
        responses_lookup[intent["tag"]] = intent["responses"]

    return model, tokenizer, label_encoder, metadata, responses_lookup


bilstm_model, tokenizer, label_encoder, metadata, intent_responses = load_bilstm_pipeline()
max_len = metadata["max_len"]


# ─────────────────────────────────────────────────────────
# YOUTUBE-STYLE CUSTOM VOICE COMPONENT DECLARATION
# ─────────────────────────────────────────────────────────

voice_component_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_input")
voice_input_widget = components.declare_component("voice_input_widget", path=voice_component_dir)


# ─────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "auto_tts" not in st.session_state:
    st.session_state.auto_tts = True

if "speech_to_speak" not in st.session_state:
    st.session_state.speech_to_speak = ""

if "last_processed_speech" not in st.session_state:
    st.session_state.last_processed_speech = ""

if "last_processed_id" not in st.session_state:
    st.session_state.last_processed_id = ""

if "widget_counter" not in st.session_state:
    st.session_state.widget_counter = 0

if "cleared" not in st.session_state:
    st.session_state.cleared = False


# ─────────────────────────────────────────────────────────
# LOCAL DEEP LEARNING INFERENCE (BiLSTM)
# ─────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.45


def predict_bilstm(text: str, is_rollback: bool = False):
    """
    Classify user intent using the custom trained BiLSTM neural network.
    Acts as the primary offline model and the automated rollback engine.
    """
    cleaned = text.lower().strip()
    words = [w for w in cleaned.split() if len(w) > 1]
    stopwords = {
        "what", "is", "a", "an", "the", "tell", "me", "about", "how", "do",
        "does", "explain", "who", "which", "can", "you", "of", "in", "to", "for", "are"
    }
    content_words = [w for w in words if w not in stopwords]
    oov_content = [w for w in content_words if w not in tokenizer.word_index]

    # Out-of-Domain protection: if content words are mostly unlearned/unknown
    if content_words and (len(oov_content) / len(content_words) >= 0.6):
        if is_rollback:
            reply = (
                "I am currently operating in offline rollback mode (BiLSTM) because the cloud AI quota is temporarily refreshing. "
                "My local model is trained specifically on Computer Science and Artificial Intelligence questions. "
                "Please ask an AI/programming question, or try again in a few moments once the cloud service restores!"
            )
        else:
            reply = (
                "That topic appears to be outside my local training dataset. "
                "Please ask about Artificial Intelligence, Machine Learning, Deep Learning, Python, or related topics."
            )
        return {
            "reply": reply,
            "engine": "BiLSTM (Auto Rollback)" if is_rollback else "BiLSTM Neural Network",
            "intent": "out_of_domain",
            "confidence": 0.0,
        }

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    probabilities = bilstm_model.predict(padded, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    intent_tag = str(label_encoder.inverse_transform([predicted_index])[0])

    if confidence < CONFIDENCE_THRESHOLD:
        reply = (
            "I'm not completely sure what you mean. "
            "Could you please rephrase your question?"
        )
    else:
        candidates = intent_responses.get(intent_tag, ["I don't have a response for that."])
        reply = random.choice(candidates)

    engine_label = "BiLSTM (Auto Rollback)" if is_rollback else "BiLSTM Neural Network"

    return {
        "reply": reply,
        "engine": engine_label,
        "intent": intent_tag,
        "confidence": confidence,
    }


# ─────────────────────────────────────────────────────────
# CLOUD LLM APIS (GEMINI / GROQ / OPENAI)
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are VoiceBot AI, an intelligent, highly accurate voice assistant. "
    "Keep your spoken answers concise (2 to 4 sentences max), strictly accurate and factual, and natural so they sound great when read aloud via text-to-speech. "
    "Never invent or hallucinate movie titles, names, dates, or false facts. "
    "Avoid markdown tables, asterisks, bullet lists, URLs, or complex ASCII formatting."
)


def try_gemini(user_text: str, api_key: str):
    """
    Attempt generation via Gemini API (cycling through supported flash/preview models with multi-turn memory).
    Returns None on failure so automatic rollback to BiLSTM seamlessly kicks in.
    """
    if not api_key:
        return None

    # Build multi-turn context
    contents = []
    if "messages" in st.session_state:
        for msg in st.session_state.messages[-6:]:
            role = "user" if msg.get("role") == "user" else "model"
            content = msg.get("content", "").strip()
            if content:
                contents.append({"role": role, "parts": [{"text": content}]})
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    candidate_models = [
        "gemini-2.5-flash",
        "gemini-flash-latest",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
    ]

    for model_name in candidate_models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key.strip()}"
            payload = {
                "systemInstruction": {
                    "parts": [{"text": SYSTEM_PROMPT}]
                },
                "contents": contents,
                "generationConfig": {
                    "maxOutputTokens": 350,
                    "temperature": 0.3,
                    "thinkingConfig": {"thinkingBudget": 0}
                }
            }
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=8)
            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        content = parts[0]["text"].strip()
                        if content:
                            return {
                                "reply": content,
                                "engine": f"Cloud AI ({model_name.replace('-', ' ').title()})",
                                "intent": "Generative AI",
                                "confidence": 1.0
                            }
        except Exception:
            continue
    return None


def try_groq(user_text: str, api_key: str):
    """Attempt generation via Groq API (prioritizing GPT-OSS-120B / GPT-OSS-20B for maximum factual accuracy)."""
    if not api_key:
        return None

    groq_models = [
        ("openai/gpt-oss-120b", 550, 0.2),
        ("openai/gpt-oss-20b", 500, 0.2),
        ("qwen/qwen3.8-27b", 350, 0.2),
        ("allam-2-7b", 350, 0.3),
    ]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if "messages" in st.session_state:
        for m in st.session_state.messages[-4:]:
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_text})

    for model, max_tok, temp in groq_models:
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": max_tok,
            }
            res = requests.post(url, json=payload, headers=headers, timeout=8)
            if res.status_code == 200:
                raw = res.json()["choices"][0]["message"]["content"]
                if raw:
                    content = (
                        raw.replace("\u202f", " ")
                        .replace("**", "")
                        .replace("###", "")
                        .replace("##", "")
                        .strip()
                    )
                    if content:
                        model_display = model.split("/")[-1].replace("-", " ").title()
                        return {
                            "reply": content,
                            "engine": f"Cloud AI (Groq {model_display})",
                            "intent": "Generative AI",
                            "confidence": 1.0
                        }
        except Exception:
            continue
    return None


def try_openai(user_text: str, api_key: str):
    """Attempt generation via OpenAI API. Returns None on failure."""
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ]
        payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 250,
        }
        res = requests.post(url, json=payload, headers=headers, timeout=8)
        if res.status_code == 200:
            content = res.json()["choices"][0]["message"]["content"].strip()
            if content:
                return {
                    "reply": content,
                    "engine": "Cloud AI (OpenAI GPT-4o)",
                    "intent": "Generative AI",
                    "confidence": 1.0
                }
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────
# RESILIENT RESPONSE ROUTER (ZERO-FRICTION ROLLBACK)
# ─────────────────────────────────────────────────────────

def get_agent_response(user_text: str, force_local: bool = False):
    """
    Intelligent router with automatic rollback:
      1. If user forces local mode -> Use BiLSTM immediately.
      2. If cloud API key exists -> Try Groq (blazing fast) then Gemini or OpenAI.
      3. If Cloud API fails or is unavailable -> Automatically and silently
         shift to local BiLSTM deep learning model as fallback.
    """
    if force_local:
        return predict_bilstm(user_text, is_rollback=False)

    gemini_key = get_secret("GEMINI_API_KEY")
    groq_key = get_secret("GROQ_API_KEY")
    openai_key = get_secret("OPENAI_API_KEY")

    has_cloud_key = bool(gemini_key or groq_key or openai_key)

    if has_cloud_key:
        # 1. Prioritize Groq: ultra-fast (~0.8s) and high rate limits
        if groq_key:
            res = try_groq(user_text, groq_key)
            if res:
                return res

        # 2. Try Gemini
        if gemini_key:
            res = try_gemini(user_text, gemini_key)
            if res:
                return res

        # 3. Try OpenAI
        if openai_key:
            res = try_openai(user_text, openai_key)
            if res:
                return res

        # Automatic Rollback: Cloud API failed or rate-limited, shift to local BiLSTM
        return predict_bilstm(user_text, is_rollback=True)

    # Default: No cloud key present, use local BiLSTM directly
    return predict_bilstm(user_text, is_rollback=False)


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────

with st.sidebar:
    # Futuristic Glowing Header
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; padding: 14px 16px; background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; margin-bottom: 16px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);">
        <div style="width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, #2563eb, #8b5cf6); display: flex; align-items: center; justify-content: center; font-size: 1.2rem; box-shadow: 0 0 14px rgba(59, 130, 246, 0.45); flex-shrink: 0;">
            ⚡
        </div>
        <div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; letter-spacing: -0.01em;">Neural Engine</div>
            <div style="font-size: 0.72rem; color: #94a3b8; letter-spacing: 0.04em; text-transform: uppercase;">System Architecture</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    gemini_k = get_secret("GEMINI_API_KEY")
    groq_k = get_secret("GROQ_API_KEY")
    openai_k = get_secret("OPENAI_API_KEY")
    has_api = bool(gemini_k or groq_k or openai_k)

    if has_api:
        if groq_k:
            api_name = "Groq (GPT-OSS-120B / Fast)"
        elif gemini_k:
            api_name = "Gemini 2.5 Flash"
        else:
            api_name = "OpenAI GPT-4o-mini"

        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.72); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 14px; padding: 14px 16px; margin-bottom: 16px; box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35); backdrop-filter: blur(14px);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-size: 0.72rem; font-weight: 700; color: #38bdf8; letter-spacing: 0.06em; text-transform: uppercase;">ACTIVE ENGINE</span>
                <span style="display: inline-flex; align-items: center; gap: 5px; font-size: 0.72rem; font-weight: 600; color: #4ade80; background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.3); padding: 2px 8px; border-radius: 20px;">
                    <span style="width: 6px; height: 6px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 6px #22c55e;"></span>
                    Online
                </span>
            </div>
            <div style="font-size: 0.95rem; font-weight: 700; color: #f1f5f9; margin-bottom: 6px;">{api_name}</div>
            <div style="font-size: 0.78rem; color: #94a3b8; display: flex; align-items: center; gap: 6px;">
                <span>🛡️</span>
                <span>Auto-Rollback: <b style="color: #60a5fa;">BiLSTM Ready</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.72); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 14px; padding: 14px 16px; margin-bottom: 16px; box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35); backdrop-filter: blur(14px);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-size: 0.72rem; font-weight: 700; color: #60a5fa; letter-spacing: 0.06em; text-transform: uppercase;">ACTIVE ENGINE</span>
                <span style="display: inline-flex; align-items: center; gap: 5px; font-size: 0.72rem; font-weight: 600; color: #60a5fa; background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); padding: 2px 8px; border-radius: 20px;">
                    Local Mode
                </span>
            </div>
            <div style="font-size: 0.95rem; font-weight: 700; color: #f1f5f9; margin-bottom: 6px;">BiLSTM Deep Neural Net</div>
            <div style="font-size: 0.78rem; color: #94a3b8;">Self-contained offline model</div>
        </div>
        """, unsafe_allow_html=True)

    force_bilstm = st.toggle(
        "🧠 Force Local BiLSTM (Lab Mode)",
        value=False,
        help="Enable to test exclusively with the custom trained 28-intent BiLSTM deep learning model."
    )

    st.session_state.auto_tts = st.toggle(
        "🔊 Auto Speak Responses (TTS)",
        value=st.session_state.auto_tts,
        help="When enabled, the browser will automatically speak the chatbot's answers aloud."
    )

    st.markdown("<div style='margin: 12px 0;'></div>", unsafe_allow_html=True)

    with st.expander("📊 Lab Model Specifications"):
        st.markdown("""
        <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.8;">
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding: 5px 0;">
                <span style="color: #94a3b8;">Architecture</span>
                <span style="font-weight: 600; color: #60a5fa;">BiLSTM Neural Net</span>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding: 5px 0;">
                <span style="color: #94a3b8;">Intent Classes</span>
                <span style="font-weight: 600; color: #38bdf8;">28 Categories</span>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding: 5px 0;">
                <span style="color: #94a3b8;">Training Corpus</span>
                <span style="font-weight: 600; color: #f1f5f9;">616 Utterances V2</span>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding: 5px 0;">
                <span style="color: #94a3b8;">Test Accuracy</span>
                <span style="font-weight: 700; color: #34d399;">60.22%</span>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding: 5px 0;">
                <span style="color: #94a3b8;">Inference Speed</span>
                <span style="font-weight: 600; color: #facc15;">&lt; 15ms</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 5px 0;">
                <span style="color: #94a3b8;">Random Baseline</span>
                <span style="color: #94a3b8;">3.57% (1/28)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# FLOATING CLEAR CHAT BUTTON (OUTSIDE PANEL, BOTTOM-LEFT CORNER)
# ─────────────────────────────────────────────────────────

if st.button("🗑️ Clear Chat", key="main_corner_clear_btn", help="Clear conversation history"):
    st.session_state.messages = []
    st.session_state.speech_to_speak = ""
    st.session_state.widget_counter += 1
    st.session_state.cleared = True
    st.components.v1.html("<script>try { window.speechSynthesis.cancel(); if(window.parent && window.parent.speechSynthesis) window.parent.speechSynthesis.cancel(); } catch(e){}</script>", height=0)
    st.rerun()


# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────

st.markdown("""
<div class="instagram-top-bar">
    <div class="instagram-header-center">
        <div class="insta-avatar-ring">
            <span style="font-size: 1.35rem;">🎙️</span>
            <span class="insta-active-dot"></span>
        </div>
        <div class="insta-title-wrap">
            <div class="insta-main-title">VoiceBot AI</div>
            <div class="insta-sub-title">Speech Recognition & Deep Learning Conversational Agent</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# BROWSER TEXT-TO-SPEECH HELPER
# ─────────────────────────────────────────────────────────

def trigger_browser_tts(text_to_speak: str):
    """Speaks the response aloud via Web Speech API in parallel with live caption streaming."""
    clean_text = (
        text_to_speak.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace('"', '\\"')
        .replace("\n", " ")
    )
    tts_js = f"""
    <script>
        (function() {{
            var synth = window.speechSynthesis;
            try {{
                if (window.parent && window.parent.speechSynthesis) {{
                    synth = window.parent.speechSynthesis;
                }}
            }} catch(e) {{}}

            if (synth) {{
                synth.cancel();
                var utterance = new SpeechSynthesisUtterance("{clean_text}");
                utterance.rate = 1.0;
                utterance.pitch = 1.0;

                function setVoiceAndSpeak() {{
                    var voices = synth.getVoices();
                    var preferred = voices.find(function(v) {{
                        return v.name.includes('Google UK English Female') || 
                               v.name.includes('Natural') || 
                               v.name.includes('Samantha') || 
                               v.name.includes('Zira') ||
                               (v.lang && v.lang.startsWith('en'));
                    }});
                    if (preferred) utterance.voice = preferred;
                    synth.speak(utterance);
                }}

                if (synth.getVoices().length > 0) {{
                    setVoiceAndSpeak();
                }} else {{
                    synth.onvoiceschanged = setVoiceAndSpeak;
                }}
            }}
        }})();
    </script>
    """
    st.components.v1.html(tts_js, height=0)


# ─────────────────────────────────────────────────────────
# CONVERSATION CHAT CONTAINER & HISTORY
# ─────────────────────────────────────────────────────────

chat_container = st.container()

# Render unified bottom dock (Mic + Text Input + Stop button)
dock_container = st.container()
with dock_container:
    spoken_data = voice_input_widget(key=f"unified_input_bar_{st.session_state.widget_counter}")

# Guard against clearing
if st.session_state.get("cleared", False):
    st.session_state.cleared = False
    spoken_data = None

# Process submitted query (from either voice recognition or typed text)
if spoken_data:
    if isinstance(spoken_data, dict):
        user_query = str(spoken_data.get("text", "")).strip()
        query_id = str(spoken_data.get("ts", user_query))
    else:
        user_query = str(spoken_data).strip()
        query_id = user_query

    is_duplicate = (query_id == st.session_state.get("last_processed_id"))

    if user_query and not is_duplicate:
        st.session_state.last_processed_id = query_id

        with chat_container:
            # First render prior history
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant":
                        engine_name = msg.get("engine", "")
                        if "Rollback" in engine_name:
                            st.markdown("<span class='badge-fallback'>🛡️ Local Rollback (BiLSTM)</span>", unsafe_allow_html=True)
                        elif "BiLSTM" in engine_name and force_bilstm:
                            st.markdown("<span class='badge-bilstm'>🧠 BiLSTM</span>", unsafe_allow_html=True)

            # Render current user question
            with st.chat_message("user"):
                st.markdown(user_query)

            # Auto-scroll up to display the newly posted question
            st.components.v1.html("<script>try{window.parent.scrollTo({top: window.parent.document.body.scrollHeight, behavior: 'smooth'});}catch(e){}</script>", height=0)

            # Render assistant message with dynamic live caption streaming!
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    agent_data = get_agent_response(user_query, force_local=force_bilstm)

                reply_text = agent_data["reply"]

                # Trigger browser TTS immediately so words stream in sync with spoken voice
                if st.session_state.auto_tts:
                    trigger_browser_tts(reply_text)

                # Live caption streaming generator: parallelized to voice speech timing (~142 WPM)
                def stream_live_captions():
                    if st.session_state.auto_tts:
                        # Synchronize with Web Speech API audio initialization buffer
                        time.sleep(0.65)

                    tokens = re.split(r'(\s+)', reply_text)
                    for token in tokens:
                        if token:
                            yield token
                            if not token.isspace():
                                if st.session_state.auto_tts:
                                    # Paced in parallel to spoken vocal delivery
                                    word_clean = token.strip()
                                    delay = 0.28 + (len(word_clean) * 0.022)
                                    if word_clean.endswith((',', ';', ':', '—', '-')):
                                        delay += 0.28
                                    elif word_clean.endswith(('.', '!', '?')):
                                        delay += 0.46
                                    time.sleep(delay)
                                else:
                                    time.sleep(0.02)

                st.write_stream(stream_live_captions)

                engine_name = agent_data.get("engine", "")
                if "Rollback" in engine_name:
                    st.markdown("<span class='badge-fallback'>🛡️ Local Rollback (BiLSTM)</span>", unsafe_allow_html=True)
                elif "BiLSTM" in engine_name and force_bilstm:
                    st.markdown("<span class='badge-bilstm'>🧠 BiLSTM</span>", unsafe_allow_html=True)

            # Auto-scroll down to ensure complete reply is in full view above the dock
            st.components.v1.html("<script>try{window.parent.scrollTo({top: window.parent.document.body.scrollHeight, behavior: 'smooth'});}catch(e){}</script>", height=0)

        # Save to session state
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({
            "role": "assistant",
            "content": reply_text,
            "engine": agent_data.get("engine", "BiLSTM"),
            "intent": agent_data.get("intent", ""),
            "confidence": agent_data.get("confidence", 1.0),
        })

else:
    with chat_container:
        if len(st.session_state.messages) == 0:
            st.markdown("""
            <div style="text-align: center; padding: 45px 20px; background: rgba(15, 23, 42, 0.45); border-radius: 16px; border: 1px dashed rgba(148, 163, 184, 0.2); margin: 25px 0;">
                <div style="font-size: 2.8rem; margin-bottom: 10px;">🎙️</div>
                <h3 style="margin: 0 0 6px 0; color: #f1f5f9; font-weight: 700;">Ready to Chat</h3>
                <p style="margin: 0; color: #94a3b8; font-size: 0.9rem;">Tap the microphone below to speak naturally, or type your question in the text box.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant":
                        engine_name = msg.get("engine", "")
                        if "Rollback" in engine_name:
                            st.markdown("<span class='badge-fallback'>🛡️ Local Rollback (BiLSTM)</span>", unsafe_allow_html=True)
                        elif "BiLSTM" in engine_name and force_bilstm:
                            st.markdown("<span class='badge-bilstm'>🧠 BiLSTM</span>", unsafe_allow_html=True)

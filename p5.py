# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        DocMind AI  –  Premium RAG Chatbot                   ║
# ║              Built with Streamlit · LangChain · Groq · HuggingFace          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os, shutil, time, json, re, base64
from datetime import datetime

import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─── Groq API key ─────────────────────────────────────────────────────────────
os.environ["GROQ_API_KEY"] = "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"

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DocMind AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  ─  luxury dark theme, Bebas Neue headline + DM Sans body
# ══════════════════════════════════════════════════════════════════════════════
GLOBAL_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── CSS Variables ────────────────────────────────────────────────────────── */
:root {
  --bg:           #0A0E1A;
  --surface:      #111827;
  --surface2:     #1A2235;
  --border:       #1E2D45;
  --border2:      #253352;
  --primary:      #4F8CFF;
  --primary-glow: rgba(79,140,255,.18);
  --secondary:    #7C5CFF;
  --accent:       #00D4AA;
  --accent-glow:  rgba(0,212,170,.14);
  --danger:       #FF4F6A;
  --text:         #F0F4FF;
  --muted:        #6B7FA3;
  --muted2:       #8E9FC0;
  --card-shadow:  0 8px 32px rgba(0,0,0,.45);
  --radius:       14px;
  --radius-sm:    8px;
  --font-head:    'Bebas Neue', sans-serif;
  --font-body:    'DM Sans', sans-serif;
  --font-mono:    'DM Mono', monospace;
}

/* ── Reset & Base ─────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text);
  font-family: var(--font-body);
  font-size: 15px;
  line-height: 1.65;
}

/* hide streamlit chrome */
#MainMenu, footer, header { display: none !important; }
.block-container { padding: 0 !important; }

/* ── Scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 99px; }

/* ── Top Navigation Bar ───────────────────────────────────────────────────── */
.nav-bar {
  position: sticky; top: 0; z-index: 999;
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 36px;
  height: 64px;
  background: rgba(10,14,26,.85);
  backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--border);
}
.nav-logo {
  display: flex; align-items: center; gap: 12px;
  font-family: var(--font-head);
  font-size: 1.6rem;
  letter-spacing: .04em;
  color: var(--text);
}
.nav-logo-hex {
  width: 34px; height: 34px;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
  display: flex; align-items: center; justify-content: center;
  font-size: 15px; color: #fff;
  animation: hexPulse 3s ease-in-out infinite;
}
@keyframes hexPulse {
  0%,100% { box-shadow: 0 0 0 0 rgba(79,140,255,.4); }
  50%      { box-shadow: 0 0 0 10px rgba(79,140,255,0); }
}
.nav-status {
  display: flex; align-items: center; gap: 8px;
  font-size: .8rem; color: var(--muted2);
}
.status-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 8px var(--accent);
  animation: blink 2s ease-in-out infinite;
}
@keyframes blink {
  0%,100% { opacity: 1; }
  50%      { opacity: .35; }
}
.nav-badge {
  padding: 4px 14px; border-radius: 99px;
  font-size: .72rem; font-weight: 600; letter-spacing: .05em;
  background: linear-gradient(135deg, rgba(79,140,255,.2), rgba(124,92,255,.2));
  border: 1px solid rgba(79,140,255,.3);
  color: var(--primary);
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
  min-width: 280px !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stMarkdown p { color: var(--muted2) !important; }

.sidebar-section {
  padding: 18px 20px;
  border-bottom: 1px solid var(--border);
}
.sidebar-title {
  font-family: var(--font-head);
  font-size: 1.05rem; letter-spacing: .06em;
  color: var(--muted2); margin-bottom: 14px;
  text-transform: uppercase;
}
.stat-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 9px 0; border-bottom: 1px solid rgba(30,45,69,.6);
}
.stat-label { font-size: .82rem; color: var(--muted); }
.stat-value {
  font-family: var(--font-mono); font-size: .85rem;
  color: var(--primary); font-weight: 500;
}
.file-chip {
  display: flex; align-items: center; gap: 8px;
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 8px 12px; margin: 6px 0;
  font-size: .8rem; color: var(--muted2);
}
.file-chip-icon { color: var(--accent); font-size: 1rem; }

/* ── Sidebar Buttons ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stButton button {
  background: var(--surface2) !important;
  color: var(--muted2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--radius-sm) !important;
  width: 100% !important;
  font-family: var(--font-body) !important;
  font-size: .85rem !important;
  padding: 8px 0 !important;
  transition: all .2s ease !important;
}
[data-testid="stSidebar"] .stButton button:hover {
  background: rgba(79,140,255,.12) !important;
  border-color: rgba(79,140,255,.4) !important;
  color: var(--primary) !important;
}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
  background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  width: 100% !important;
  font-family: var(--font-body) !important;
  font-size: .85rem !important;
  padding: 9px 0 !important;
  margin-top: 4px !important;
}

/* ── Main content area ────────────────────────────────────────────────────── */
.main-content {
  max-width: 920px;
  margin: 0 auto;
  padding: 32px 24px 160px;
  position: relative;
}

/* ── Hero Section ─────────────────────────────────────────────────────────── */
.hero {
  text-align: center;
  padding: 72px 24px 56px;
  position: relative;
}
.hero-glow {
  position: absolute;
  top: 0; left: 50%; transform: translateX(-50%);
  width: 600px; height: 300px;
  background: radial-gradient(ellipse at center,
    rgba(79,140,255,.12) 0%, transparent 70%);
  pointer-events: none;
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 6px 18px; border-radius: 99px;
  background: rgba(79,140,255,.1);
  border: 1px solid rgba(79,140,255,.3);
  font-size: .78rem; font-weight: 600; letter-spacing: .08em;
  color: var(--primary); margin-bottom: 28px;
  animation: fadeInUp .6s ease both;
}
.hero-title {
  font-family: var(--font-head);
  font-size: clamp(3rem, 7vw, 5.2rem);
  letter-spacing: .02em;
  line-height: 1;
  background: linear-gradient(135deg, #fff 30%, var(--primary) 70%, var(--secondary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 20px;
  animation: fadeInUp .7s .1s ease both;
}
.hero-subtitle {
  font-size: 1.05rem; color: var(--muted2);
  max-width: 500px; margin: 0 auto 48px;
  animation: fadeInUp .7s .2s ease both;
}
.feature-grid {
  display: grid; grid-template-columns: repeat(3,1fr);
  gap: 14px; max-width: 680px; margin: 0 auto 48px;
  animation: fadeInUp .7s .3s ease both;
}
.feature-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 16px;
  transition: all .25s ease;
  cursor: default;
}
.feature-card:hover {
  border-color: rgba(79,140,255,.45);
  background: var(--surface2);
  transform: translateY(-3px);
  box-shadow: 0 12px 32px rgba(79,140,255,.12);
}
.feature-icon {
  font-size: 1.6rem; margin-bottom: 10px;
  display: block;
  animation: float 3.5s ease-in-out infinite;
}
.feature-card:nth-child(2) .feature-icon { animation-delay: .5s; }
.feature-card:nth-child(3) .feature-icon { animation-delay: 1s; }
@keyframes float {
  0%,100% { transform: translateY(0); }
  50%      { transform: translateY(-6px); }
}
.feature-card h4 { font-size: .9rem; font-weight: 600; color: var(--text); margin-bottom: 4px; }
.feature-card p  { font-size: .78rem; color: var(--muted); line-height: 1.5; }

@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(24px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ── Upload Area — animated gradient border + glow ──────────────────────── */
.upload-wrapper {
  position: relative;
  background: var(--surface);
  border-radius: 22px;
  padding: 56px 32px;
  text-align: center;
  overflow: hidden;
  z-index: 0;
  animation: fadeInUp .6s .4s ease both;
}
/* rotating gradient border */
.upload-wrapper::before {
  content: '';
  position: absolute;
  inset: -2px;
  border-radius: 22px;
  background: conic-gradient(from 0deg,
    var(--primary), var(--secondary), var(--accent), var(--primary));
  animation: rotateBorder 5s linear infinite;
  z-index: -2;
}
.upload-wrapper::after {
  content: '';
  position: absolute;
  inset: 2px;
  border-radius: 20px;
  background: var(--surface);
  z-index: -1;
}
@keyframes rotateBorder { to { transform: rotate(360deg); } }

/* floating glow particles */
.upload-glow {
  position: absolute; inset: 0; pointer-events: none; z-index: 0;
  overflow: hidden; border-radius: 20px;
}
.upload-glow span {
  position: absolute;
  width: 120px; height: 120px;
  border-radius: 50%;
  filter: blur(30px);
  opacity: .35;
  animation: drift 8s ease-in-out infinite;
}
.upload-glow span:nth-child(1) { background: var(--primary);   top: -30px; left: 10%;  animation-delay: 0s; }
.upload-glow span:nth-child(2) { background: var(--secondary); top: 40%;  right: 5%;   animation-delay: 2.2s; }
.upload-glow span:nth-child(3) { background: var(--accent);    bottom: -40px; left: 40%; animation-delay: 4.4s; }
@keyframes drift {
  0%,100% { transform: translate(0,0) scale(1); }
  50%     { transform: translate(20px,-18px) scale(1.25); }
}

.upload-content { position: relative; z-index: 1; }
.upload-icon {
  font-size: 3.4rem; display: block; margin-bottom: 18px;
  filter: drop-shadow(0 0 18px rgba(79,140,255,.6));
  animation: float 2.8s ease-in-out infinite;
}
.upload-title {
  font-family: var(--font-head);
  font-size: 2rem; letter-spacing: .04em;
  background: linear-gradient(135deg, #fff, var(--primary), var(--secondary));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 8px;
}
.upload-sub { font-size: .9rem; color: var(--muted2); margin-bottom: 10px; }
.upload-hint {
  display: inline-block;
  padding: 5px 16px; border-radius: 99px;
  background: rgba(0,212,170,.12);
  border: 1px solid rgba(0,212,170,.3);
  font-size: .75rem; color: var(--accent);
  font-weight: 700; letter-spacing: .06em;
  margin-top: 6px;
  box-shadow: 0 0 16px rgba(0,212,170,.25);
}

/* ── File preview grid ────────────────────────────────────────────────────── */
.file-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 12px;
  margin: 22px 0;
}
.file-card {
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-radius: var(--radius);
  padding: 16px 12px;
  text-align: center;
  transition: all .2s ease;
  animation: fadeInUp .4s ease both;
}
.file-card:hover {
  border-color: var(--primary);
  transform: translateY(-3px);
  box-shadow: 0 8px 20px rgba(79,140,255,.18);
}
.file-card-icon {
  font-size: 1.8rem; display: block; margin-bottom: 8px;
  filter: drop-shadow(0 0 8px rgba(0,212,170,.45));
}
.file-card-name {
  font-size: .78rem; color: var(--text); font-weight: 500;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.file-card-size {
  font-size: .7rem; color: var(--muted); margin-top: 4px;
  font-family: var(--font-mono);
}

/* ── Streamlit uploader overrides ─────────────────────────────────────────── */
[data-testid="stFileUploader"] { margin-top: 16px; }
[data-testid="stFileUploader"] section {
  background: transparent !important;
  border: none !important;
}
[data-testid="stFileUploader"] label { color: var(--muted2) !important; }

/* ── Process Button ───────────────────────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius) !important;
  font-family: var(--font-body) !important;
  font-size: .92rem !important;
  font-weight: 600 !important;
  padding: 14px 32px !important;
  width: 100% !important;
  letter-spacing: .04em !important;
  transition: all .25s ease !important;
  box-shadow: 0 6px 24px rgba(79,140,255,.35) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 32px rgba(79,140,255,.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Processing Steps Timeline ────────────────────────────────────────────── */
.timeline {
  display: flex; flex-direction: column; gap: 0;
  max-width: 520px; margin: 0 auto;
}
.timeline-step {
  display: flex; align-items: center; gap: 16px;
  padding: 14px 20px;
  border-radius: var(--radius);
  transition: background .3s;
  position: relative;
}
.timeline-step.active  { background: rgba(79,140,255,.08); }
.timeline-step.done    { background: rgba(0,212,170,.06); }
.timeline-step.pending { opacity: .4; }
.step-icon {
  width: 38px; height: 38px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem; flex-shrink: 0;
}
.step-icon.active  { background: rgba(79,140,255,.2); border: 2px solid var(--primary); animation: stepPulse 1.2s ease-in-out infinite; }
.step-icon.done    { background: rgba(0,212,170,.2);  border: 2px solid var(--accent); }
.step-icon.pending { background: var(--surface2);     border: 2px solid var(--border); }
@keyframes stepPulse {
  0%,100% { box-shadow: 0 0 0 0 rgba(79,140,255,.5); }
  50%      { box-shadow: 0 0 0 8px rgba(79,140,255,0); }
}
.step-text { flex: 1; }
.step-label { font-size: .9rem; font-weight: 500; color: var(--text); }
.step-desc  { font-size: .75rem; color: var(--muted); margin-top: 2px; }
.step-badge {
  font-size: .7rem; font-weight: 700; letter-spacing: .06em;
  padding: 3px 10px; border-radius: 99px;
}
.step-badge.done    { background: rgba(0,212,170,.15); color: var(--accent); border: 1px solid rgba(0,212,170,.3); }
.step-badge.active  { background: rgba(79,140,255,.15); color: var(--primary); border: 1px solid rgba(79,140,255,.3); }
.step-badge.pending { background: var(--surface2); color: var(--muted); border: 1px solid var(--border); }

/* connector line between steps */
.timeline-step:not(:last-child)::after {
  content: '';
  position: absolute; bottom: -1px; left: 38px;
  width: 2px; height: 2px;
  background: var(--border2);
}

/* ── Chat Messages ────────────────────────────────────────────────────────── */
.chat-spacer { height: 24px; }

.msg-row {
  display: flex; gap: 12px; align-items: flex-start;
  padding: 8px 0;
  animation: msgFadeIn .35s ease both;
}
@keyframes msgFadeIn {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
.msg-row.user  { flex-direction: row-reverse; }

.avatar {
  width: 36px; height: 36px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem; flex-shrink: 0;
}
.avatar.user {
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  box-shadow: 0 4px 14px rgba(79,140,255,.4);
}
.avatar.ai {
  background: linear-gradient(135deg, #1a2235, #253352);
  border: 1.5px solid var(--border2);
}

.bubble-wrap { max-width: 75%; display: flex; flex-direction: column; gap: 4px; }
.msg-row.user .bubble-wrap { align-items: flex-end; }

.bubble {
  padding: 13px 18px;
  border-radius: 18px;
  font-size: .93rem; line-height: 1.65;
  word-break: break-word;
}
.bubble.user {
  background: linear-gradient(135deg, rgba(79,140,255,.3), rgba(124,92,255,.2));
  border: 1px solid rgba(79,140,255,.35);
  color: var(--text);
  border-bottom-right-radius: 5px;
}
.bubble.ai {
  background: var(--surface2);
  border: 1px solid var(--border);
  color: var(--text);
  border-bottom-left-radius: 5px;
}
.bubble-meta {
  font-size: .7rem; color: var(--muted);
  display: flex; align-items: center; gap: 6px;
}
.bubble-meta span { cursor: pointer; }
.bubble-meta span:hover { color: var(--primary); }

/* typing indicator */
.typing-indicator {
  display: flex; align-items: center; gap: 9px;
  padding: 8px 0;
  animation: msgFadeIn .3s ease;
}
.typing-dots { display: flex; gap: 5px; }
.dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--primary);
  animation: dotBounce 1.2s ease-in-out infinite;
}
.dot:nth-child(2) { animation-delay: .2s; }
.dot:nth-child(3) { animation-delay: .4s; }
@keyframes dotBounce {
  0%,80%,100% { transform: translateY(0); opacity: .4; }
  40%          { transform: translateY(-7px); opacity: 1; }
}
.typing-label { font-size: .8rem; color: var(--muted); }

/* ── Chat Input ───────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] > div {
  background: var(--surface) !important;
  border: 1.5px solid var(--border2) !important;
  border-radius: 16px !important;
  box-shadow: 0 4px 24px rgba(0,0,0,.35) !important;
  transition: border-color .2s !important;
}
[data-testid="stChatInput"] > div:focus-within {
  border-color: var(--primary) !important;
  box-shadow: 0 4px 24px rgba(79,140,255,.2) !important;
}
[data-testid="stChatInput"] textarea {
  background: transparent !important;
  color: var(--text) !important;
  font-family: var(--font-body) !important;
  font-size: .93rem !important;
}
[data-testid="stChatInput"] button {
  background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
  border-radius: 10px !important;
  border: none !important;
}

/* ── Info / Error Boxes ───────────────────────────────────────────────────── */
.info-box {
  background: rgba(79,140,255,.08);
  border: 1px solid rgba(79,140,255,.25);
  border-left: 3px solid var(--primary);
  border-radius: var(--radius-sm);
  padding: 14px 18px;
  font-size: .88rem; color: var(--muted2);
  margin: 12px 0;
}
.success-box {
  background: rgba(0,212,170,.08);
  border: 1px solid rgba(0,212,170,.25);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius-sm);
  padding: 14px 18px;
  font-size: .88rem; color: #90f0dc;
}
.error-box {
  background: rgba(255,79,106,.08);
  border: 1px solid rgba(255,79,106,.25);
  border-left: 3px solid var(--danger);
  border-radius: var(--radius-sm);
  padding: 14px 18px;
  font-size: .88rem; color: #ffaab8;
}

/* ── Knowledge Base Banner ────────────────────────────────────────────────── */
.kb-banner {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 18px 24px;
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 14px;
  margin-bottom: 28px;
}
.kb-stat {
  text-align: center;
}
.kb-stat-value {
  font-family: var(--font-head);
  font-size: 1.6rem; letter-spacing: .04em;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.kb-stat-label {
  font-size: .73rem; color: var(--muted);
  font-weight: 500; text-transform: uppercase; letter-spacing: .06em;
  margin-top: 2px;
}
.kb-divider {
  width: 1px; height: 40px;
  background: var(--border);
}

/* ── Streamlit overrides ──────────────────────────────────────────────────── */
div[data-testid="stChatMessage"]   { background: transparent !important; padding: 0 !important; }
div[data-testid="stVerticalBlock"] { gap: 0 !important; }
.stSpinner > div > div > div      { border-top-color: var(--primary) !important; }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "documents_uploaded": False,
    "vector_store": None,
    "messages": [],
    "uploaded_file_names": [],
    "total_chunks": 0,
    "questions_asked": 0,
    "session_start": datetime.now().strftime("%H:%M"),
    "processing_error": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

DOC_DIR = "./docmind_uploads/"

# ══════════════════════════════════════════════════════════════════════════════
#  CORE RAG FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Load HuggingFace embeddings once and cache."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def process_files(path: str) -> int:
    """Load PDFs → chunk → embed → build vector store. Returns chunk count."""
    loader = PyPDFDirectoryLoader(path)
    docs   = loader.load()
    if not docs:
        raise ValueError("No text could be extracted from the uploaded PDFs.")

    splitter   = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    split_docs = splitter.split_documents(docs)

    embeddings   = get_embeddings()
    vector_store = InMemoryVectorStore.from_documents(split_docs, embeddings)

    st.session_state.vector_store       = vector_store
    st.session_state.documents_uploaded = True
    st.session_state.processing_error   = None
    return len(split_docs)


def get_answer(question: str, chat_history: list) -> str:
    """Retrieve context and generate answer via Groq LLM."""
    from langchain_groq import ChatGroq

    docs    = st.session_state.vector_store.similarity_search(question, k=3)
    context = "\n\n".join(d.page_content for d in docs)

    history_str = ""
    for msg in chat_history[-8:]:
        role         = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    prompt = ChatPromptTemplate.from_template("""
You are DocMind AI, an expert document analyst.

CHAT HISTORY:
{history}

DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

RULES:
- Answer based ONLY on the document context.
- If not found, reply: "I couldn't find that information in your uploaded documents."
- Format responses with clear structure (use markdown when helpful).
- Be precise, insightful, and concise.

ANSWER:
""")

    llm   = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "history":  history_str,
        "context":  context,
        "question": question,
    }).strip()


def export_chat() -> str:
    """Export chat history as formatted text."""
    lines = ["# DocMind AI – Chat Export", f"Session: {st.session_state.session_start}", "---"]
    for m in st.session_state.messages:
        role = "You" if m["role"] == "user" else "DocMind AI"
        lines.append(f"\n**{role}**\n{m['content']}\n")
    return "\n".join(lines)


def reset_session():
    """Clear everything and start fresh."""
    for k in ["documents_uploaded", "vector_store", "messages",
              "uploaded_file_names", "total_chunks", "questions_asked",
              "processing_error"]:
        st.session_state[k] = (
            False if k == "documents_uploaded"
            else None if k in ("vector_store", "processing_error")
            else 0    if k in ("total_chunks", "questions_asked")
            else []
        )
    st.session_state.session_start = datetime.now().strftime("%H:%M")
    if os.path.exists(DOC_DIR):
        shutil.rmtree(DOC_DIR)

# ══════════════════════════════════════════════════════════════════════════════
#  TOP NAVIGATION BAR
# ══════════════════════════════════════════════════════════════════════════════
status_text  = "Knowledge Base Active" if st.session_state.documents_uploaded else "Waiting for Documents"
status_badge = "READY" if st.session_state.documents_uploaded else "IDLE"

st.markdown(f"""
<div class="nav-bar">
  <div class="nav-logo">
    <div class="nav-logo-hex">⬡</div>
    DocMind AI
  </div>
  <div class="nav-status">
    <div class="status-dot"></div>
    {status_text}
  </div>
  <div class="nav-badge">{status_badge}</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Brand ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:22px 20px 14px; border-bottom:1px solid #1E2D45;">
      <div style="font-family:'Bebas Neue',sans-serif;font-size:1.4rem;letter-spacing:.06em;
                  background:linear-gradient(135deg,#4F8CFF,#7C5CFF);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;">
        DocMind AI
      </div>
      <div style="font-size:.75rem;color:#6B7FA3;margin-top:3px;">
        Document Intelligence Platform
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Session Stats ────────────────────────────────────────────────────────
    st.markdown("""<div class="sidebar-section">
      <div class="sidebar-title">Session Stats</div>""", unsafe_allow_html=True)

    stats = [
        ("📄 Documents",   len(st.session_state.uploaded_file_names)),
        ("🧩 Chunks",      st.session_state.total_chunks),
        ("💬 Questions",   st.session_state.questions_asked),
        ("🕐 Started",     st.session_state.session_start),
    ]
    for label, value in stats:
        st.markdown(f"""
        <div class="stat-row">
          <span class="stat-label">{label}</span>
          <span class="stat-value">{value}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Uploaded Files ────────────────────────────────────────────────────────
    if st.session_state.uploaded_file_names:
        st.markdown("""<div class="sidebar-section">
          <div class="sidebar-title">Knowledge Base</div>""", unsafe_allow_html=True)

        for name in st.session_state.uploaded_file_names:
            short = name if len(name) <= 22 else name[:20] + "…"
            st.markdown(f"""
            <div class="file-chip">
              <span class="file-chip-icon">📑</span>
              <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                {short}
              </span>
              <span style="color:#00D4AA;font-size:.65rem;">✓</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Action Buttons ────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section" style="border-bottom:none;">', unsafe_allow_html=True)

    if st.session_state.documents_uploaded:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🧹  Clear", use_container_width=True):
                st.session_state.messages        = []
                st.session_state.questions_asked = 0
                st.rerun()
        with c2:
            if st.button("🔄  Reset", use_container_width=True):
                reset_session()
                st.rerun()

        export_data = export_chat()
        st.download_button(
            label="📥  Export Chat",
            data=export_data,
            file_name=f"docmind_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        st.markdown(
            '<p style="font-size:.8rem;color:#6B7FA3;">Upload PDFs in the main panel to get started.</p>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Footer (normal flow, not absolute) ─────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;font-size:.7rem;color:#3a4a6b;
                padding:16px 20px;border-top:1px solid #1E2D45;margin-top:8px;">
      Powered by Groq · LangChain · HuggingFace<br>
      <span style="color:#4F8CFF;">DocMind AI</span> · 2025
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT — wrapping div
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  UPLOAD / HERO SCREEN
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.documents_uploaded:

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <div class="hero-glow"></div>
      <div class="hero-badge">⚡ AI-POWERED DOCUMENT INTELLIGENCE</div>
      <div class="hero-title">Transform Documents<br>Into Conversations</div>
      <div class="hero-subtitle">
        Upload your PDFs and instantly unlock insights, answers,<br>
        and deep understanding through natural conversation.
      </div>

      <div class="feature-grid">
        <div class="feature-card">
          <span class="feature-icon">⚡</span>
          <h4>Instant Answers</h4>
          <p>Ask anything — get precise answers from your documents in seconds.</p>
        </div>
        <div class="feature-card">
          <span class="feature-icon">🧠</span>
          <h4>Smart Retrieval</h4>
          <p>Semantic vector search finds the most relevant context, always.</p>
        </div>
        <div class="feature-card">
          <span class="feature-icon">🔒</span>
          <h4>Fully Private</h4>
          <p>Documents stay in your session — nothing is stored externally.</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload Card (animated gradient border + glow) ─────────────────────────
    st.markdown("""
    <div class="upload-wrapper">
      <div class="upload-glow"><span></span><span></span><span></span></div>
      <div class="upload-content">
        <span class="upload-icon">📂</span>
        <div class="upload-title">Drop Your PDFs Here</div>
        <div class="upload-sub">Supports multiple files · Drag & drop or click to browse</div>
        <div class="upload-hint">PDF FILES ONLY</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        # Preview grid
        grid_html = '<div class="file-grid">'
        for f in uploaded_files:
            size_kb = round(f.size / 1024, 1)
            grid_html += f"""
            <div class="file-card">
              <span class="file-card-icon">📑</span>
              <div class="file-card-name" title="{f.name}">{f.name}</div>
              <div class="file-card-size">{size_kb} KB</div>
            </div>"""
        grid_html += "</div>"
        st.markdown(grid_html, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1.5, 3, 1.5])
        with col2:
            if st.button("🚀  Process Documents", use_container_width=True):

                # ── Animated Processing Timeline ─────────────────────────────
                STEPS = [
                    ("📤", "Uploading Files",          "Saving PDFs to disk",            .3),
                    ("📖", "Analyzing Documents",      "Extracting text and structure",   .7),
                    ("✂️",  "Creating Chunks",          "Splitting into semantic pieces",  .7),
                    ("🔬", "Generating Embeddings",    "Encoding with MiniLM-L6-v2",     2.5),
                    ("🗄️",  "Building Knowledge Base",  "Indexing vector store",           .5),
                    ("✅", "Ready for Questions",       "All done!",                       .2),
                ]

                timeline_ph = st.empty()
                error_ph    = st.empty()

                def render_timeline(current_idx: int):
                    rows = ""
                    for i, (icon, label, desc, _) in enumerate(STEPS):
                        state   = "done" if i < current_idx else ("active" if i == current_idx else "pending")
                        badge_t = "DONE" if state == "done" else ("RUNNING" if state == "active" else "PENDING")
                        rows   += f"""
                        <div class="timeline-step {state}">
                          <div class="step-icon {state}">{icon}</div>
                          <div class="step-text">
                            <div class="step-label">{label}</div>
                            <div class="step-desc">{desc}</div>
                          </div>
                          <div class="step-badge {state}">{badge_t}</div>
                        </div>"""
                    timeline_ph.markdown(
                        f'<div style="margin:24px 0;"><div class="timeline">{rows}</div></div>',
                        unsafe_allow_html=True
                    )

                try:
                    # Step 0 – save files
                    render_timeline(0)
                    os.makedirs(DOC_DIR, exist_ok=True)
                    for f in uploaded_files:
                        with open(os.path.join(DOC_DIR, f.name), "wb") as fp:
                            fp.write(f.getvalue())
                    st.session_state.uploaded_file_names = [f.name for f in uploaded_files]
                    time.sleep(STEPS[0][3])

                    # Steps 1-4 handled inside process_files but we animate them here
                    for step_idx in range(1, len(STEPS) - 1):
                        render_timeline(step_idx)
                        time.sleep(STEPS[step_idx][3])

                    # Actual processing
                    chunks = process_files(DOC_DIR)
                    st.session_state.total_chunks = chunks

                    # Step 5 – done
                    render_timeline(len(STEPS))
                    time.sleep(STEPS[-1][3])

                    if st.session_state.processing_error:
                        error_ph.markdown(
                            f'<div class="error-box">❌ {st.session_state.processing_error}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        error_ph.markdown(
                            f'<div class="success-box">✅ Knowledge base built — {chunks} chunks indexed across {len(uploaded_files)} file(s). Ready!</div>',
                            unsafe_allow_html=True
                        )
                        time.sleep(1.2)
                        st.rerun()

                except Exception as e:
                    error_ph.markdown(
                        f'<div class="error-box">❌ Processing failed: {e}</div>',
                        unsafe_allow_html=True
                    )

# ══════════════════════════════════════════════════════════════════════════════
#  CHAT SCREEN
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.documents_uploaded and st.session_state.vector_store:

    # ── Knowledge Base Banner ────────────────────────────────────────────────
    st.markdown(f"""
    <div class="kb-banner">
      <div class="kb-stat">
        <div class="kb-stat-value">{len(st.session_state.uploaded_file_names)}</div>
        <div class="kb-stat-label">Documents</div>
      </div>
      <div class="kb-divider"></div>
      <div class="kb-stat">
        <div class="kb-stat-value">{st.session_state.total_chunks:,}</div>
        <div class="kb-stat-label">Chunks</div>
      </div>
      <div class="kb-divider"></div>
      <div class="kb-stat">
        <div class="kb-stat-value">{st.session_state.questions_asked}</div>
        <div class="kb-stat-label">Questions</div>
      </div>
      <div class="kb-divider"></div>
      <div style="display:flex;align-items:center;gap:8px;font-size:.85rem;color:#00D4AA;">
        <div class="status-dot"></div> Knowledge Base Active
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Welcome tip (when no messages yet) ──────────────────────────────────
    if not st.session_state.messages:
        files_joined = ", ".join(
            f"`{n}`" for n in st.session_state.uploaded_file_names
        )
        st.markdown(f"""
        <div class="info-box">
          💡 <strong>Knowledge base ready.</strong> I've indexed your documents — 
          {files_joined}. Ask me anything about their contents!
        </div>
        """, unsafe_allow_html=True)

    # ── Message History ──────────────────────────────────────────────────────
    st.markdown('<div class="chat-spacer"></div>', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        ts = msg.get("time", "")
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-row user">
              <div class="avatar user">👤</div>
              <div class="bubble-wrap">
                <div class="bubble user">{msg["content"]}</div>
                <div class="bubble-meta" style="justify-content:flex-end;">
                  <span>{ts}</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-row">
              <div class="avatar ai">⬡</div>
              <div class="bubble-wrap">
                <div class="bubble ai">{msg["content"]}</div>
                <div class="bubble-meta">
                  <span>DocMind AI</span> · <span>{ts}</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="chat-spacer"></div>', unsafe_allow_html=True)

    # ── Chat Input ───────────────────────────────────────────────────────────
    query = st.chat_input("Ask anything about your documents…")

    if query:
        now = datetime.now().strftime("%H:%M")

        # Add user message
        st.session_state.messages.append({
            "role": "user", "content": query, "time": now
        })
        st.session_state.questions_asked += 1

        # Typing indicator
        typing_ph = st.empty()
        typing_ph.markdown("""
        <div class="typing-indicator">
          <div class="avatar ai" style="width:30px;height:30px;font-size:.85rem;">⬡</div>
          <div class="typing-dots">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
          </div>
          <div class="typing-label">DocMind AI is thinking…</div>
        </div>
        """, unsafe_allow_html=True)

        # Get answer
        answer = get_answer(query, st.session_state.messages)

        typing_ph.empty()

        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant", "content": answer,
            "time": datetime.now().strftime("%H:%M")
        })

        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)   # close .main-content

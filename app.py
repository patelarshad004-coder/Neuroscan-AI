import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage import measure, filters
from nilearn import datasets, surface
import h5py
import os
import datetime

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI | Clinical Brain Imaging",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #020b18 0%, #041225 50%, #020b18 100%);
    color: #e0eaf5;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020d1e 0%, #031628 100%);
    border-right: 1px solid #0d3a5c;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #041830 0%, #062040 100%);
    border: 1px solid #0d4a7a;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0, 100, 200, 0.1);
    transition: all 0.3s ease;
}
[data-testid="stMetric"]:hover {
    border-color: #1a7abf;
    box-shadow: 0 4px 30px rgba(0, 150, 255, 0.2);
    transform: translateY(-2px);
}
[data-testid="stMetricLabel"] {
    color: #5a9fd4 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #e0f0ff !important;
    font-size: 22px !important;
    font-weight: 700 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #020d1e;
    border-bottom: 1px solid #0d3a5c;
    gap: 4px;
    padding: 0 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #4a7a9b;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 12px 24px;
    font-weight: 500;
    font-size: 13px;
    letter-spacing: 0.3px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #4a9eff !important;
    border-bottom: 2px solid #4a9eff !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #7ab8ff !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0d4a7a, #1a7abf);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(26, 122, 191, 0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1a6aaa, #2a9adf);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(26, 122, 191, 0.5);
}

/* Cards */
.clinical-card {
    background: linear-gradient(135deg, #041830, #062040);
    border: 1px solid #0d3a5c;
    border-radius: 16px;
    padding: 24px;
    margin: 8px 0;
    box-shadow: 0 4px 20px rgba(0, 50, 100, 0.2);
}

.status-detected {
    background: linear-gradient(135deg, #4a0a0a, #7f1d1d);
    color: #fca5a5;
    padding: 8px 20px;
    border-radius: 25px;
    font-weight: 700;
    font-size: 13px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid #dc2626;
    box-shadow: 0 0 20px rgba(220, 38, 38, 0.3);
    animation: pulse-red 2s infinite;
}

.status-clear {
    background: linear-gradient(135deg, #0a2a0a, #14532d);
    color: #86efac;
    padding: 8px 20px;
    border-radius: 25px;
    font-weight: 700;
    font-size: 13px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid #16a34a;
    box-shadow: 0 0 20px rgba(22, 163, 74, 0.3);
}

@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 20px rgba(220, 38, 38, 0.3); }
    50% { box-shadow: 0 0 30px rgba(220, 38, 38, 0.6); }
}

.section-label {
    color: #3a7abf;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #0d3a5c;
}

.progress-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    color: #4a9eff;
    font-size: 13px;
    font-weight: 500;
}

.progress-step-done {
    color: #4ade80;
}

.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #0d2a40;
    font-size: 13px;
}

.info-label { color: #4a7a9b; }
.info-value { color: #e0f0ff; font-weight: 600; }

#MainMenu, footer, header { visibility: hidden; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #041830;
    border: 2px dashed #0d4a7a;
    border-radius: 12px;
    padding: 16px;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #1a7abf, #4a9eff);
    border-radius: 10px;
}

/* Selectbox */
[data-testid="stSelectbox"] > div {
    background: #041830;
    border: 1px solid #0d3a5c;
    border-radius: 8px;
    color: #e0f0ff;
}
</style>
""", unsafe_allow_html=True)

# ── Model ──────────────────────────────────────────────
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))

class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.block(4, 32)
        self.enc2 = self.block(32, 64)
        self.enc3 = self.block(64, 128)
        self.enc4 = self.block(128, 256)
        self.bottleneck = self.block(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att4 = AttentionBlock(256, 256, 128)
        self.dec4 = self.block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = AttentionBlock(128, 128, 64)
        self.dec3 = self.block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att2 = AttentionBlock(64, 64, 32)
        self.dec2 = self.block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.att1 = AttentionBlock(32, 32, 16)
        self.dec1 = self.block(64, 32)
        self.output = nn.Conv2d(32, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), self.att4(self.up4(b), e4)], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), self.att3(self.up3(d4), e3)], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), self.att2(self.up2(d3), e2)], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.att1(self.up1(d2), e1)], dim=1))
        return torch.sigmoid(self.output(d1))

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = AttentionUNet().to(device)
    model.load_state_dict(torch.load("best_attention_model.pth", map_location=device))
    model.eval()
    return model, device

@st.cache_resource
def load_brain_surface():
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    coords_l, faces_l = surface.load_surf_mesh(fsaverage.pial_left)
    coords_r, faces_r = surface.load_surf_mesh(fsaverage.pial_right)
    return coords_l, faces_l, coords_r, faces_r

def get_location(pred_binary):
    cy, cx = np.where(pred_binary > 0)
    if len(cx) == 0:
        return "Not detected"
    mean_x, mean_y = cx.mean(), cy.mean()
    loc    = "Left"     if mean_x < 120 else "Right"
    region = "Frontal"  if mean_y < 80  else "Parietal" if mean_y < 140 else "Occipital"
    return f"{loc} {region} Lobe"

def build_3d_figure(pred_binary):
    tumor_vol = np.stack([pred_binary * (0.6 + 0.4*(i/20)) for i in range(20)], axis=0)
    tumor_vol[0] *= 0.2; tumor_vol[-1] *= 0.2
    tumor_smooth = filters.gaussian(tumor_vol, sigma=1)
    try:
        t_verts, t_faces, _, _ = measure.marching_cubes(tumor_smooth, level=0.25, step_size=1)
    except:
        return None

    coords_l, faces_l, coords_r, faces_r = load_brain_surface()
    def norm(c, ref):
        out = c.copy().astype(float)
        for i in range(3):
            mn, mx = ref[:,i].min(), ref[:,i].max()
            out[:,i] = (out[:,i]-mn)/(mx-mn)*200+20
        return out

    bn_l = norm(coords_l, coords_l)
    bn_r = norm(coords_r, coords_l)
    bc   = [(bn_l[:,i].min()+bn_l[:,i].max())/2 for i in range(3)]
    tc   = [(t_verts[:,2].min()+t_verts[:,2].max())/2,
            (t_verts[:,1].min()+t_verts[:,1].max())/2,
            (t_verts[:,0].min()+t_verts[:,0].max())/2]
    tv   = t_verts.copy()
    tv[:,2] += bc[0]-tc[0]; tv[:,1] += bc[1]-tc[1]; tv[:,0] += bc[2]-tc[2]

    fig = go.Figure()
    for bn, fn, name in [(bn_l, faces_l, 'Left Hemisphere'), (bn_r, faces_r, 'Right Hemisphere')]:
        fig.add_trace(go.Mesh3d(
            x=bn[:,0], y=bn[:,1], z=bn[:,2],
            i=fn[:,0], j=fn[:,1], k=fn[:,2],
            colorscale=[[0,'#020d1e'],[1,'#0d4a7a']],
            intensity=bn[:,2], opacity=0.12, name=name,
            showlegend=True, flatshading=False,
            lighting=dict(ambient=0.7, diffuse=0.6, specular=0.5, fresnel=0.5),
            lightposition=dict(x=300, y=300, z=500)
        ))
    fig.add_trace(go.Mesh3d(
        x=tv[:,2], y=tv[:,1], z=tv[:,0],
        i=t_faces[:,0], j=t_faces[:,1], k=t_faces[:,2],
        colorscale=[[0,'#7f1d1d'],[0.4,'#dc2626'],[0.7,'#ef4444'],[1,'#fca5a5']],
        intensity=tv[:,0], opacity=0.95,
        name='Tumor Region', showlegend=True, flatshading=False,
        lighting=dict(ambient=0.3, diffuse=0.8, specular=1.0, fresnel=0.8),
        lightposition=dict(x=300, y=300, z=500)
    ))
    fig.update_layout(
        paper_bgcolor='#020b18',
        scene=dict(
            bgcolor='#020b18',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showspikes=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showspikes=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showspikes=False),
            camera=dict(up=dict(x=0,y=0,z=1), center=dict(x=0,y=0,z=0),
                       eye=dict(x=1.5, y=1.5, z=0.8)),
            aspectmode='data'
        ),
        legend=dict(font=dict(color='#7ab8ff', size=12),
                   bgcolor='rgba(2,11,24,0.9)', bordercolor='#0d3a5c',
                   borderwidth=1, x=0.02, y=0.98),
        height=620,
        margin=dict(l=0, r=0, t=0, b=0),
        annotations=[dict(
            text="Rotate: Click+Drag  •  Zoom: Scroll  •  Pan: Right-click+Drag",
            xref="paper", yref="paper", x=0.5, y=0.01,
            showarrow=False, font=dict(color='#1a4a6a', size=11), xanchor='center'
        )]
    )
    return fig

# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:24px 0 16px;'>
        <div style='font-size:48px; margin-bottom:8px;'>🧠</div>
        <div style='font-size:22px; font-weight:700; color:#4a9eff; letter-spacing:1px;'>
            NeuroScan AI
        </div>
        <div style='font-size:10px; color:#2a5a7a; letter-spacing:3px;
                    text-transform:uppercase; margin-top:6px;'>
            Clinical Decision Support
        </div>
        <div style='margin-top:12px; display:inline-block; background:#041830;
                    border:1px solid #0d4a7a; border-radius:20px;
                    padding:4px 14px; font-size:11px; color:#4a9eff;'>
            v2.0 — Research Edition
        </div>
    </div>
    <hr style='border-color:#0d3a5c; margin:8px 0 16px;'>
    """, unsafe_allow_html=True)

    # Patient Info
    st.markdown('<div class="section-label">Patient Information</div>', unsafe_allow_html=True)
    patient_id   = st.text_input("Patient ID", value="BRT-1024", label_visibility="visible")
    patient_name = st.text_input("Patient Name", value="Anonymous")
    scan_date    = st.date_input("Scan Date", value=datetime.date.today())
    referring_dr = st.text_input("Referring Physician", value="Dr. —")

    st.markdown('<hr style="border-color:#0d3a5c; margin:16px 0;">', unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="section-label">Scan Upload</div>', unsafe_allow_html=True)

    use_sample = st.checkbox("Use sample MRI scan", value=False)
    uploaded_file = None
    if not use_sample:
        uploaded_file = st.file_uploader("Upload BraTS .h5 file", type=["h5"],
                                          label_visibility="collapsed")
    analyze = st.button("▶  Run AI Analysis", use_container_width=True)

    st.markdown('<hr style="border-color:#0d3a5c; margin:16px 0;">', unsafe_allow_html=True)

    # Platform info
    st.markdown('<div class="section-label">Platform</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:12px; color:#2a5a7a; line-height:2;'>
        🤖 Model: Attention U-Net<br>
        📊 Dataset: BraTS 2020<br>
        🎯 Dice Score: 93.1%<br>
        ⚙️ Parameters: 7.8M<br>
        🏥 Sequences: T1, T1ce, T2, FLAIR
    </div>
    <div style='margin-top:16px; background:#041018; border:1px solid #4a1010;
                border-radius:8px; padding:10px; font-size:11px; color:#7a3a3a;'>
        ⚠️ For research purposes only.<br>Not approved for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────
st.markdown("""
<div style='display:flex; align-items:center; justify-content:space-between;
            padding:20px 0 16px; border-bottom:1px solid #0d3a5c; margin-bottom:28px;'>
    <div>
        <div style='font-size:11px; color:#2a5a7a; letter-spacing:3px;
                    text-transform:uppercase; margin-bottom:6px;'>
            Clinical Brain Imaging Platform
        </div>
        <div style='font-size:28px; font-weight:700; color:#e0f0ff; letter-spacing:0.3px;'>
            Brain MRI Analysis
        </div>
    </div>
    <div style='text-align:right;'>
        <div style='font-size:10px; color:#2a5a7a; letter-spacing:2px;
                    text-transform:uppercase;'>Powered by</div>
        <div style='font-size:18px; font-weight:700;
                    background:linear-gradient(135deg, #4a9eff, #7ab8ff);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            NeuroScan AI
        </div>
        <div style='font-size:11px; color:#2a5a7a; margin-top:2px;'>
            {date}
        </div>
    </div>
</div>
""".format(date=datetime.date.today().strftime("%d %B %Y")), unsafe_allow_html=True)

# ── Main Logic ─────────────────────────────────────────
run_analysis = analyze and (uploaded_file is not None or use_sample)

if run_analysis:
    # Progress indicator
    st.markdown("""
    <div class="clinical-card" style='margin-bottom:24px;'>
        <div class="section-label">Analysis Pipeline</div>
        <div style='display:flex; gap:0; align-items:center; flex-wrap:wrap;'>
            <div style='display:flex; align-items:center; gap:8px; padding:6px 16px 6px 0;'>
                <div style='width:24px; height:24px; background:#4ade80; border-radius:50%;
                            display:flex; align-items:center; justify-content:center;
                            font-size:12px;'>✓</div>
                <span style='font-size:12px; color:#4ade80; font-weight:500;'>MRI Upload</span>
            </div>
            <div style='color:#0d3a5c; margin:0 4px;'>→</div>
            <div style='display:flex; align-items:center; gap:8px; padding:6px 16px;'>
                <div style='width:24px; height:24px; background:#4ade80; border-radius:50%;
                            display:flex; align-items:center; justify-content:center;
                            font-size:12px;'>✓</div>
                <span style='font-size:12px; color:#4ade80; font-weight:500;'>Preprocessing</span>
            </div>
            <div style='color:#0d3a5c; margin:0 4px;'>→</div>
            <div style='display:flex; align-items:center; gap:8px; padding:6px 16px;'>
                <div style='width:24px; height:24px; background:#4ade80; border-radius:50%;
                            display:flex; align-items:center; justify-content:center;
                            font-size:12px;'>✓</div>
                <span style='font-size:12px; color:#4ade80; font-weight:500;'>Segmentation</span>
            </div>
            <div style='color:#0d3a5c; margin:0 4px;'>→</div>
            <div style='display:flex; align-items:center; gap:8px; padding:6px 16px;'>
                <div style='width:24px; height:24px; background:#4ade80; border-radius:50%;
                            display:flex; align-items:center; justify-content:center;
                            font-size:12px;'>✓</div>
                <span style='font-size:12px; color:#4ade80; font-weight:500;'>Analysis</span>
            </div>
            <div style='color:#0d3a5c; margin:0 4px;'>→</div>
            <div style='display:flex; align-items:center; gap:8px; padding:6px 16px;'>
                <div style='width:24px; height:24px; background:#4ade80; border-radius:50%;
                            display:flex; align-items:center; justify-content:center;
                            font-size:12px;'>✓</div>
                <span style='font-size:12px; color:#4ade80; font-weight:500;'>3D Reconstruction</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    if use_sample:
        # Load sample from BraTS data
        sample_files = [f for f in os.listdir("data/brats2020/BraTS2020_training_data/content/data/")
                       if "volume_100" in f and f.endswith(".h5")]
        if sample_files:
            sample_path = f"data/brats2020/BraTS2020_training_data/content/data/{sample_files[50]}"
            with h5py.File(sample_path, "r") as f:
                image = f['image'][:]
        else:
            st.error("Sample data not available in this environment.")
            st.stop()
    else:
        with h5py.File(uploaded_file, "r") as f:
            image = f['image'][:]

    # Run inference
    with st.spinner("Running Attention U-Net inference..."):
        model, device = load_model()
        img_tensor = torch.tensor(image, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_tensor)
        pred_np      = pred[0,0].cpu().numpy()
        pred_binary  = (pred_np > 0.3).astype(float)
        tumor_pixels = int(pred_binary.sum())
        tumor_pct    = (tumor_pixels / (240*240)) * 100
        confidence   = float(pred_np.max()) * 100
        volume_cm3   = round(tumor_pixels * 0.001 * 8.4, 1)
        location     = get_location(pred_binary)
        detected     = tumor_pixels > 50

    # ── Patient Bar ──
    status_html = (
        '<span class="status-detected">⚠ TUMOR DETECTED</span>'
        if detected else
        '<span class="status-clear">✓ NO ANOMALY</span>'
    )
    st.markdown(f"""
    <div class="clinical-card" style='margin-bottom:24px;'>
        <div style='display:flex; gap:48px; align-items:center; flex-wrap:wrap;'>
            <div>
                <div style='font-size:10px; color:#2a5a7a; letter-spacing:2px;
                            text-transform:uppercase;'>Patient ID</div>
                <div style='font-size:18px; font-weight:700; color:#e0f0ff;
                            margin-top:4px;'>{patient_id}</div>
            </div>
            <div>
                <div style='font-size:10px; color:#2a5a7a; letter-spacing:2px;
                            text-transform:uppercase;'>Name</div>
                <div style='font-size:16px; font-weight:600; color:#e0f0ff;
                            margin-top:4px;'>{patient_name}</div>
            </div>
            <div>
                <div style='font-size:10px; color:#2a5a7a; letter-spacing:2px;
                            text-transform:uppercase;'>Scan Date</div>
                <div style='font-size:16px; font-weight:600; color:#e0f0ff;
                            margin-top:4px;'>{scan_date.strftime("%d %b %Y")}</div>
            </div>
            <div>
                <div style='font-size:10px; color:#2a5a7a; letter-spacing:2px;
                            text-transform:uppercase;'>Physician</div>
                <div style='font-size:16px; font-weight:600; color:#e0f0ff;
                            margin-top:4px;'>{referring_dr}</div>
            </div>
            <div style='margin-left:auto;'>{status_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics ──
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Tumor Status",   "Detected" if detected else "Clear",
              "⚠ Abnormal" if detected else "✓ Normal")
    m2.metric("Location",       location)
    m3.metric("Est. Volume",    f"{volume_cm3} cm³")
    m4.metric("AI Confidence",  f"{confidence:.1f}%")
    m5.metric("Model Accuracy", "93.1% Dice")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬  Analysis",
        "📐  Multi-Planar View",
        "🔮  3D Reconstruction",
        "📄  Clinical Report"
    ])

    # ── Tab 1: Analysis ──
    with tab1:
        col_l, col_r = st.columns([3, 2])

        with col_l:
            st.markdown('<div class="section-label">MRI Segmentation Results</div>',
                        unsafe_allow_html=True)
            fig2d, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig2d.patch.set_facecolor('#020b18')
            img_display = image[:,:,1]  # T1ce for best contrast

            for ax in axes:
                ax.set_facecolor('#020b18')
                ax.axis("off")

            axes[0].imshow(img_display, cmap="gray")
            axes[0].set_title("Original MRI (T1ce)", color='#5a9fd4',
                               fontsize=12, pad=10)

            axes[1].imshow(img_display, cmap="gray")
            axes[1].imshow(pred_binary, cmap="Reds", alpha=0.65)
            axes[1].set_title("AI Segmentation Overlay", color='#5a9fd4',
                               fontsize=12, pad=10)

            axes[2].imshow(pred_binary, cmap="hot")
            axes[2].set_title("Tumor Probability Map", color='#5a9fd4',
                               fontsize=12, pad=10)

            # Add tumor boundary
            if detected:
                from skimage import measure as skm
                contours = skm.find_contours(pred_binary, 0.5)
                for contour in contours:
                    axes[1].plot(contour[:, 1], contour[:, 0],
                                linewidth=1.5, color='#ff4444', alpha=0.8)

            plt.tight_layout(pad=0.5)
            st.pyplot(fig2d)
            plt.close()

        with col_r:
            st.markdown('<div class="section-label">Segmentation Analysis</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="clinical-card">
                <div class="info-row">
                    <span class="info-label">Tumor Status</span>
                    <span class="info-value" style='color:{"#ef4444" if detected else "#4ade80"};'>
                        {"Detected" if detected else "Not Detected"}
                    </span>
                </div>
                <div class="info-row">
                    <span class="info-label">Location</span>
                    <span class="info-value">{location}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Est. Volume</span>
                    <span class="info-value">{volume_cm3} cm³</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Scan Coverage</span>
                    <span class="info-value">{tumor_pct:.2f}%</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Pixel Count</span>
                    <span class="info-value">{tumor_pixels:,} px</span>
                </div>
                <div class="info-row">
                    <span class="info-label">AI Confidence</span>
                    <span class="info-value" style='color:#4ade80;'>{confidence:.1f}%</span>
                </div>
                <div class="info-row" style='border:none;'>
                    <span class="info-label">Model</span>
                    <span class="info-value">Attention U-Net</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bar
            st.markdown(f"""
            <div class="clinical-card" style='margin-top:12px;'>
                <div class="section-label">Confidence Score</div>
                <div style='font-size:32px; font-weight:700;
                            color:{"#ef4444" if confidence > 80 else "#f59e0b"};
                            margin-bottom:8px;'>{confidence:.1f}%</div>
                <div style='background:#041018; border-radius:8px; height:8px; overflow:hidden;'>
                    <div style='width:{min(confidence,100)}%; height:100%;
                                background:linear-gradient(90deg, #1a7abf, #4a9eff);
                                border-radius:8px; transition:width 0.5s ease;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Clinical note
            st.markdown(f"""
            <div class="clinical-card" style='margin-top:12px;
                        border-left:3px solid {"#dc2626" if detected else "#16a34a"};'>
                <div class="section-label">AI Clinical Note</div>
                <div style='font-size:13px; color:#a0c0d8; line-height:1.7;'>
                    {"Attention U-Net identified a hyperintense region in the <b style='color:#e0f0ff;'>" + location + "</b>. Estimated volume <b style='color:#e0f0ff;'>" + str(volume_cm3) + " cm³</b> with <b style='color:#4ade80;'>" + str(round(confidence,1)) + "%</b> confidence. Recommend radiologist review." if detected else "No significant anomaly detected in this MRI slice. Confidence score indicates normal brain tissue patterns across all analyzed regions."}
                </div>
                <div style='margin-top:12px; font-size:11px; color:#2a4a5a;'>
                    ⚠ This analysis is AI-generated and must be validated by
                    a qualified radiologist before clinical use.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 2: Multi-Planar ──
    with tab2:
        st.markdown('<div class="section-label">Multi-Sequence MRI Analysis</div>',
                    unsafe_allow_html=True)
        seqs = [("T1", 0, "Anatomical structure"),
                ("T1ce", 1, "Contrast enhanced — tumor enhancement"),
                ("T2", 2, "Fluid detection — edema"),
                ("FLAIR", 3, "Fluid suppression — lesion detection")]

        for seq_name, idx, desc in seqs:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"""
                <div class="clinical-card" style='height:100%;'>
                    <div style='font-size:18px; font-weight:700; color:#4a9eff;'>{seq_name}</div>
                    <div style='font-size:12px; color:#4a7a9b; margin-top:4px;'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                fig_s, ax = plt.subplots(1, 1, figsize=(8, 3))
                fig_s.patch.set_facecolor('#020b18')
                ax.imshow(image[:,:,idx], cmap="gray")
                ax.imshow(pred_binary, cmap="Reds", alpha=0.4)
                ax.axis("off")
                ax.set_facecolor('#020b18')
                plt.tight_layout(pad=0)
                st.pyplot(fig_s)
                plt.close()

    # ── Tab 3: 3D ──
    with tab3:
        st.markdown('<div class="section-label">Interactive 3D Brain + Tumor Reconstruction</div>',
                    unsafe_allow_html=True)

        col_ctrl, col_info = st.columns([3, 1])
        with col_info:
            st.markdown(f"""
            <div class="clinical-card">
                <div class="section-label">3D Info</div>
                <div class="info-row">
                    <span class="info-label">Brain Atlas</span>
                    <span class="info-value" style='font-size:12px;'>MNI fsaverage</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Hemispheres</span>
                    <span class="info-value" style='font-size:12px;'>Both</span>
                </div>
                <div class="info-row">
                    <span class="info-row">Tumor Vol.</span>
                    <span class="info-value" style='font-size:12px;'>{volume_cm3} cm³</span>
                </div>
                <div class="info-row" style='border:none;'>
                    <span class="info-label">Render</span>
                    <span class="info-value" style='font-size:12px;'>WebGL</span>
                </div>
            </div>
            <div class="clinical-card" style='margin-top:8px;'>
                <div class="section-label">Controls</div>
                <div style='font-size:11px; color:#4a7a9b; line-height:2.2;'>
                    🖱 Rotate: Click+Drag<br>
                    🔍 Zoom: Scroll<br>
                    ✋ Pan: Right-click<br>
                    👁 Reset: Double-click
                </div>
            </div>
            """, unsafe_allow_html=True)

        if detected:
            with col_ctrl:
                with st.spinner("Rendering 3D brain model..."):
                    fig3d = build_3d_figure(pred_binary)
                if fig3d:
                    st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.info("No significant tumor detected — 3D reconstruction not available.")

    # ── Tab 4: Report ──
    with tab4:
        st.markdown('<div class="section-label">Clinical Report</div>',
                    unsafe_allow_html=True)

        report_date = datetime.date.today().strftime("%d %B %Y")
        report = f"""
NEUROSCAN AI — CLINICAL IMAGING REPORT
{'='*55}
PATIENT INFORMATION
{'─'*55}
Patient ID:          {patient_id}
Patient Name:        {patient_name}
Scan Date:           {scan_date.strftime("%d %B %Y")}
Report Date:         {report_date}
Referring Physician: {referring_dr}
Modality:            MRI Brain (Multi-sequence)

FINDINGS
{'─'*55}
AI Analysis Status:  {'TUMOR DETECTED — REVIEW REQUIRED' if detected else 'NO ANOMALY DETECTED'}
Tumor Location:      {location}
Estimated Volume:    {volume_cm3} cm³
Scan Area Coverage:  {tumor_pct:.2f}%
AI Confidence Score: {confidence:.1f}%

SEQUENCES ANALYZED
{'─'*55}
- T1 — Anatomical structure
- T1ce — Contrast enhanced (tumor enhancement)
- T2 — Fluid detection / edema
- FLAIR — Fluid suppression / lesion detection

AI MODEL SPECIFICATIONS
{'─'*55}
Model Architecture:  Attention U-Net
Training Dataset:    BraTS 2020 (Brain Tumor Segmentation)
Model Parameters:    7,855,005
Validation Dice:     93.1%
Training Epochs:     90

CLINICAL RECOMMENDATION
{'─'*55}
{'This AI analysis has identified a region of interest in the ' + location + '. The detected hyperintense region with estimated volume of ' + str(volume_cm3) + ' cm³ warrants further evaluation. Recommend correlation with clinical symptoms and follow-up imaging.' if detected else 'AI analysis did not identify significant anomalies in this MRI scan. Continue routine monitoring as clinically indicated.'}

DISCLAIMER
{'─'*55}
This report is generated by NeuroScan AI, an experimental
AI system for research purposes only. It has NOT been
approved for clinical diagnosis by any regulatory authority.
All findings must be reviewed and validated by a qualified
radiologist or physician before any clinical decision.

NeuroScan AI v2.0 | Built at VIT Pune
Research Prototype | Not for Clinical Use
        """

        st.markdown(f"""
        <div class="clinical-card">
            <pre style='font-family:monospace; font-size:12px; color:#a0c0d8;
                        line-height:1.6; white-space:pre-wrap;'>{report}</pre>
        </div>
        """, unsafe_allow_html=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "⬇  Download Report (.txt)",
                report,
                file_name=f"NeuroScan_{patient_id}_{scan_date}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col_d2:
            if st.button("🖨  Print Report", use_container_width=True):
                st.info("Use Ctrl+P or Cmd+P to print this page.")

else:
    # ── Landing Page ──
    st.markdown("""
    <div style='text-align:center; padding:60px 20px 40px;'>
        <div style='font-size:80px; margin-bottom:20px;'>🧠</div>
        <div style='font-size:32px; font-weight:700; color:#e0f0ff;
                    margin-bottom:12px; letter-spacing:0.3px;'>
            AI-Assisted Brain Tumor Detection
        </div>
        <div style='font-size:16px; color:#4a7a9b; max-width:600px;
                    margin:0 auto; line-height:1.8;'>
            Upload a multi-sequence brain MRI scan and receive instant AI-powered
            tumor detection, segmentation, and interactive 3D visualization.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    c1, c2, c3, c4 = st.columns(4)
    features = [
        ("🤖", "AI Detection", "Attention U-Net with 93.1% Dice Score accuracy on BraTS 2020"),
        ("📊", "Multi-Sequence", "Analyzes T1, T1ce, T2, and FLAIR MRI sequences simultaneously"),
        ("🔮", "3D Visualization", "Interactive 3D brain + tumor reconstruction using MNI atlas"),
        ("📄", "Clinical Report", "Auto-generated downloadable clinical imaging report"),
    ]
    for col, (icon, title, desc) in zip([c1,c2,c3,c4], features):
        col.markdown(f"""
        <div class="clinical-card" style='text-align:center; height:180px;'>
            <div style='font-size:36px; margin-bottom:12px;'>{icon}</div>
            <div style='font-size:14px; font-weight:700; color:#e0f0ff;
                        margin-bottom:8px;'>{title}</div>
            <div style='font-size:12px; color:#4a7a9b; line-height:1.6;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Stats row
    st.markdown("""
    <div style='display:flex; gap:20px; justify-content:center; flex-wrap:wrap; margin:20px 0;'>
        <div class="clinical-card" style='text-align:center; min-width:140px; flex:1; max-width:180px;'>
            <div style='font-size:32px; font-weight:700;
                        background:linear-gradient(135deg, #4a9eff, #7ab8ff);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>93.1%</div>
            <div style='font-size:11px; color:#2a5a7a; margin-top:4px;
                        letter-spacing:1px; text-transform:uppercase;'>Dice Score</div>
        </div>
        <div class="clinical-card" style='text-align:center; min-width:140px; flex:1; max-width:180px;'>
            <div style='font-size:32px; font-weight:700;
                        background:linear-gradient(135deg, #4a9eff, #7ab8ff);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>7.8M</div>
            <div style='font-size:11px; color:#2a5a7a; margin-top:4px;
                        letter-spacing:1px; text-transform:uppercase;'>Parameters</div>
        </div>
        <div class="clinical-card" style='text-align:center; min-width:140px; flex:1; max-width:180px;'>
            <div style='font-size:32px; font-weight:700;
                        background:linear-gradient(135deg, #4a9eff, #7ab8ff);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>4</div>
            <div style='font-size:11px; color:#2a5a7a; margin-top:4px;
                        letter-spacing:1px; text-transform:uppercase;'>MRI Sequences</div>
        </div>
        <div class="clinical-card" style='text-align:center; min-width:140px; flex:1; max-width:180px;'>
            <div style='font-size:32px; font-weight:700;
                        background:linear-gradient(135deg, #4a9eff, #7ab8ff);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>3D</div>
            <div style='font-size:11px; color:#2a5a7a; margin-top:4px;
                        letter-spacing:1px; text-transform:uppercase;'>Brain Viewer</div>
        </div>
        <div class="clinical-card" style='text-align:center; min-width:140px; flex:1; max-width:180px;'>
            <div style='font-size:32px; font-weight:700;
                        background:linear-gradient(135deg, #4a9eff, #7ab8ff);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>90</div>
            <div style='font-size:11px; color:#2a5a7a; margin-top:4px;
                        letter-spacing:1px; text-transform:uppercase;'>Training Epochs</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Getting started
    st.markdown("""
    <div class="clinical-card" style='max-width:600px; margin:20px auto; text-align:center;'>
        <div style='font-size:16px; font-weight:600; color:#e0f0ff; margin-bottom:12px;'>
            Getting Started
        </div>
        <div style='font-size:13px; color:#4a7a9b; line-height:2;'>
            1. Enter patient information in the sidebar<br>
            2. Upload a BraTS 2020 .h5 MRI file <b style='color:#4a9eff;'>or</b>
               check "Use sample scan"<br>
            3. Click <b style='color:#4a9eff;'>▶ Run AI Analysis</b><br>
            4. Explore results across Analysis, Multi-Planar, 3D, and Report tabs
        </div>
    </div>
    """, unsafe_allow_html=True)

print("✅ Premium UI app.py created!")

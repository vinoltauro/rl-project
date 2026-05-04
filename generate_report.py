"""
Generate dissertation PDF report with embedded figures.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
    Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

PLOTS = "results/plots"
OUTPUT = "report/dissertation_report.pdf"
os.makedirs("report", exist_ok=True)

W, H = A4
MARGIN = 2.5 * cm

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN, bottomMargin=MARGIN,
    title="Representation Learning in Deep RL",
    author="Neil Tauro"
)

styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    "Title", parent=styles["Title"],
    fontSize=18, leading=22, spaceAfter=6, alignment=TA_CENTER,
    textColor=colors.HexColor("#1a1a2e")
)
subtitle_style = ParagraphStyle(
    "Subtitle", parent=styles["Normal"],
    fontSize=11, leading=14, spaceAfter=4, alignment=TA_CENTER,
    textColor=colors.HexColor("#444444")
)
h1_style = ParagraphStyle(
    "H1", parent=styles["Heading1"],
    fontSize=14, leading=18, spaceBefore=16, spaceAfter=6,
    textColor=colors.HexColor("#1a1a2e"), borderPad=0
)
h2_style = ParagraphStyle(
    "H2", parent=styles["Heading2"],
    fontSize=11, leading=14, spaceBefore=10, spaceAfter=4,
    textColor=colors.HexColor("#2d4a7a")
)
body_style = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontSize=10, leading=15, spaceAfter=8, alignment=TA_JUSTIFY
)
caption_style = ParagraphStyle(
    "Caption", parent=styles["Normal"],
    fontSize=8.5, leading=11, spaceAfter=12, alignment=TA_CENTER,
    textColor=colors.HexColor("#555555"), fontName="Helvetica-Oblique"
)
abstract_style = ParagraphStyle(
    "Abstract", parent=styles["Normal"],
    fontSize=9.5, leading=14, spaceAfter=6, alignment=TA_JUSTIFY,
    leftIndent=1*cm, rightIndent=1*cm,
    textColor=colors.HexColor("#222222")
)
ref_style = ParagraphStyle(
    "Ref", parent=styles["Normal"],
    fontSize=9, leading=13, spaceAfter=5, leftIndent=1*cm, firstLineIndent=-1*cm
)
table_header_style = ParagraphStyle(
    "TH", parent=styles["Normal"],
    fontSize=9, fontName="Helvetica-Bold", alignment=TA_CENTER
)
table_cell_style = ParagraphStyle(
    "TC", parent=styles["Normal"],
    fontSize=9, alignment=TA_CENTER
)
code_style = ParagraphStyle(
    "Code", parent=styles["Code"],
    fontSize=8.5, leading=12, leftIndent=1*cm, spaceAfter=8,
    backColor=colors.HexColor("#f4f4f4"), borderColor=colors.HexColor("#cccccc"),
    borderWidth=0.5, borderPad=6, fontName="Courier"
)

def fig(filename, caption, width_frac=0.85):
    path = os.path.join(PLOTS, filename)
    if not os.path.exists(path):
        return KeepTogether([
            Paragraph(f"[Figure not available: {filename}]", caption_style),
            Spacer(1, 0.3*cm),
        ])
    max_w = (W - 2*MARGIN) * width_frac
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        iw, ih = im.size
    ratio = ih / iw
    img_w = min(max_w, iw)
    img_h = img_w * ratio
    max_h = H * 0.38
    if img_h > max_h:
        img_h = max_h
        img_w = img_h / ratio
    return KeepTogether([
        Image(path, width=img_w, height=img_h),
        Paragraph(caption, caption_style),
    ])

def h1(text): return Paragraph(text, h1_style)
def h2(text): return Paragraph(text, h2_style)
def p(text):  return Paragraph(text, body_style)
def sp(n=0.4): return Spacer(1, n*cm)
def hr(): return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=8)

def design_table():
    data = [
        [Paragraph("", table_header_style), Paragraph("<b>Pong</b>", table_header_style), Paragraph("<b>Breakout</b>", table_header_style)],
        [Paragraph("<b>DQN</b>", table_header_style), Paragraph("Run 1 — 2M steps", table_cell_style), Paragraph("Run 2 — 5M steps", table_cell_style)],
        [Paragraph("<b>Double DQN</b>", table_header_style), Paragraph("Run 3 — 2M steps", table_cell_style), Paragraph("Run 4 — 5M steps", table_cell_style)],
    ]
    t = Table(data, colWidths=[4*cm, 5.5*cm, 5.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("BACKGROUND", (0,1), (0,-1), colors.HexColor("#2d4a7a")),
        ("TEXTCOLOR",  (0,1), (0,-1), colors.white),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (1,1), (-1,-1), [colors.HexColor("#f0f4ff"), colors.HexColor("#e0e8ff")]),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#aaaaaa")),
        ("TOPPADDING",  (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
    ]))
    return KeepTogether([t, sp(0.2)])

def hyperparam_table():
    rows = [
        ["Learning rate", "1×10⁻⁴ (Adam)"],
        ["Replay buffer", "100,000 transitions"],
        ["Batch size", "32"],
        ["Discount factor γ", "0.99"],
        ["Target update freq.", "1,000 steps (hard copy)"],
        ["Epsilon schedule", "1.0 → 0.01 over 100k steps"],
        ["Gradient clip", "L2 norm ≤ 10.0"],
        ["Frame skip / stack", "4 / 4"],
        ["Input size", "84×84 grayscale"],
        ["Checkpoint freq.", "Every 500,000 steps"],
        ["Random seed", "42"],
    ]
    data = [[Paragraph("<b>Parameter</b>", table_header_style), Paragraph("<b>Value</b>", table_header_style)]] + \
           [[Paragraph(r[0], table_cell_style), Paragraph(r[1], table_cell_style)] for r in rows]
    t = Table(data, colWidths=[6*cm, 9*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2d4a7a")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4ff")]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#aaaaaa")),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]))
    return KeepTogether([t, sp(0.2)])

# ─── Build story ──────────────────────────────────────────────────────────────

story = []

# Title page
story += [
    sp(1.5),
    Paragraph("Representation Learning in Deep Reinforcement Learning:", title_style),
    Paragraph("A Comparative Study of DQN and Double DQN on Atari", title_style),
    sp(0.4),
    hr(),
    sp(0.2),
    Paragraph("Masters Dissertation Component", subtitle_style),
    Paragraph("Neil Tauro | School of Computer Science and Statistics, Trinity College Dublin", subtitle_style),
    sp(1.5),
]

# Abstract
story += [
    Paragraph("<b>Abstract</b>", h2_style),
    Paragraph(
        "This study investigates how the choice of reinforcement learning algorithm shapes the internal "
        "representations developed by deep neural agents playing Atari games. Using a fully crossed 2×2 factorial "
        "design, Deep Q-Networks (DQN) and Double DQN (DDQN) agents were trained on Pong and Breakout — two games "
        "sharing fundamental low-level visual structure (ball, paddle, deflection physics) while differing "
        "substantially in high-level strategy. After training, 512-dimensional activation vectors from each agent's "
        "penultimate layer were extracted and analysed using t-SNE, Grad-CAM saliency maps, dead neuron analysis, "
        "and cross-game cosine similarity. Results show that game content is the dominant organising principle of "
        "the representation space. However, DDQN produces measurably more compact and structured representations, "
        "consistent with the hypothesis that its corrected gradient signal reduces representational noise. DQN "
        "exhibits systematic Q-value overestimation and a higher fraction of dead neurons, both indicative of "
        "noisier learning dynamics. These findings provide the first controlled cross-algorithm, cross-game "
        "comparison of learned representations at the activation level.",
        abstract_style
    ),
    sp(0.3),
    hr(),
    PageBreak(),
]

# 1. Introduction
story += [
    h1("1. Introduction"),
    p("The success of deep reinforcement learning at Atari games raised a fundamental question that performance "
      "metrics alone cannot answer: what do these agents actually learn? When a convolutional neural network "
      "achieves competitive scores at Breakout after millions of game steps, what internal representation of "
      "the game world has it constructed? And critically — does the learning algorithm shape that representation, "
      "or does the game environment determine it regardless of how the agent was trained?"),
    p("This study addresses these questions through a controlled experiment comparing DQN (Mnih et al., 2015) "
      "and Double DQN (van Hasselt et al., 2016) across two structurally similar Atari games. The two algorithms "
      "are architecturally identical — they share the same convolutional network, replay buffer, and "
      "hyperparameters. The only difference is three lines of code governing how the TD target is computed. "
      "This makes them ideal for isolating algorithmic effects on representation: any observed difference in "
      "the geometry of the representation space must be due to the learning rule alone."),
    p("Pong and Breakout were chosen deliberately. Both feature a ball, a paddle, and physics governed by "
      "deflection angles — shared low-level visual structure that might produce similar convolutional features. "
      "Yet they differ in high-level strategy: Pong requires tracking an opponent, Breakout requires strategic "
      "brick destruction. This contrast enables the study to separate shared from game-specific representational "
      "components, and to ask whether game content or training algorithm is the stronger organising force."),
    p("<b>Research question:</b> Do RL agents trained on structurally similar Atari games develop similar internal "
      "representations, and does the choice of algorithm (DQN vs Double DQN) affect the nature and quality of "
      "those representations — independent of the game being played?"),
]

# 2. Background
story += [
    h1("2. Background"),
    h2("2.1 Deep Q-Networks"),
    p("DQN approximates the action-value function Q(s, a) using a convolutional neural network. Two stabilising "
      "mechanisms make this tractable: experience replay, which stores transitions in a circular buffer and "
      "samples random mini-batches to break temporal correlations; and a target network, a frozen copy of the "
      "online network used to compute training targets. The DQN TD target is:"),
    Paragraph("<i>y = r + γ · max<sub>a'</sub> Q(s', a'; θ⁻)</i>", code_style),
    p("Both the selection of the best next action and the evaluation of its value use the same target network θ⁻. "
      "This produces a known bias: the maximum over noisy Q-value estimates is systematically higher than the "
      "true maximum, and this error compounds over training."),
    h2("2.2 Double DQN"),
    p("Van Hasselt et al. (2016) proposed a minimal fix: decouple action selection from action evaluation using "
      "the online and target networks respectively. The DDQN target is:"),
    Paragraph("<i>y = r + γ · Q(s', argmax<sub>a'</sub> Q(s', a'; θ); θ⁻)</i>", code_style),
    p("Because the two networks have different error profiles, their errors are unlikely to be correlated, "
      "substantially reducing the overestimation bias. In this study's codebase, DoubleDQNAgent inherits all "
      "code from DQNAgent and overrides only the learn() method — three lines of Python. Any observed "
      "representational difference is attributable to this target formulation alone."),
    h2("2.3 Representation Learning and t-SNE"),
    p("The 512-dimensional activation vector at the penultimate fully connected layer represents the agent's "
      "compressed internal encoding of each game state. Zahavy et al. (2016) applied t-SNE (Maaten & Hinton, "
      "2008) to single-agent DQN representations and showed clusters corresponding to semantic game states. "
      "This study extends that approach to a controlled multi-algorithm, multi-game comparison."),
    p("t-SNE reduces high-dimensional data to two dimensions by preserving local neighbourhood structure via "
      "KL divergence minimisation between Gaussian (high-D) and t-distributed (2D) neighbourhood kernels. "
      "It faithfully represents local clustering but inter-cluster distances in the projection are not "
      "proportional to high-dimensional distances — interpretations are qualitative."),
]

# 3. Experimental Design
story += [
    h1("3. Experimental Design"),
    h2("3.1 Factorial Structure"),
    p("The study employs a fully crossed 2×2 factorial design:"),
    sp(0.2),
    design_table(),
    p("This structure isolates three orthogonal comparisons: the game effect (same algorithm, different games), "
      "the algorithm effect (same game, different algorithms), and their interaction. All four runs use identical "
      "hyperparameters, the same random seed (42), and the same CNN architecture."),
    h2("3.2 Preprocessing Pipeline"),
    p("Raw 210×160 RGB frames are transformed through the standard DeepMind preprocessing stack: random no-op "
      "resets (1–30 steps) to prevent state memorisation; 4-frame action repeat with max-pooling of the final "
      "two frames to handle sprite flickering; episodic life handling; 84×84 grayscale conversion; reward "
      "clipping to {−1, 0, +1} to normalise scale across games; and 4-frame stacking to provide implicit "
      "velocity information. Observations are stored as uint8 and normalised to float32 at sample time."),
    h2("3.3 Network Architecture"),
    p("Both agents share an identical CNN (Mnih et al., 2015): three convolutional layers "
      "(32×8×8/4, 64×4×4/2, 64×3×3/1) followed by a 512-dimensional fully connected representation "
      "layer and a Q-value output head. A PyTorch forward hook on the representation layer captures the "
      "512-dim activation at every forward pass without modifying the computation."),
    h2("3.4 Hyperparameters"),
    sp(0.2),
    hyperparam_table(),
    h2("3.5 Representation Extraction"),
    p("After training, each checkpoint is loaded and the agent runs in near-greedy mode (ε = 0.05) for "
      "5,000 steps. At each step, the 512-dim representation vector, action taken, and cumulative episode "
      "reward are recorded. This produces one .npz file per checkpoint across the full training trajectory."),
]

# 4. Results
story += [
    h1("4. Results"),
    h2("4.1 Training Performance"),
    p("DQN/Pong training over 2,446 episodes demonstrates the expected learning trajectory. The agent begins "
      "at mean reward −21 (losing every point against the built-in opponent), consistent with a random policy. "
      "By 500,000 steps the rolling 20-episode average rises to −2.8, indicating the agent reliably returns "
      "the ball. At 1,000,000 steps the average reaches +2.0, crossing into winning territory. By 1,500,000 "
      "steps it is +4.6. The final 50-episode average at ∼2M steps is +4.9, with individual episodes reaching "
      "+17. The mean Q-value stabilises in the range 2.55–2.60 during late training."),
]

story += [
    sp(0.2),
    fig("training_curves_pong.png",
        "Figure 1. Training curves — Pong. Episode reward over training steps for DQN and DDQN "
        "(20-episode rolling average). Both agents progress from −21 (random) toward positive reward."),
    fig("training_curves_breakout.png",
        "Figure 2. Training curves — Breakout. Episode reward over 5M steps for DQN and DDQN. "
        "Breakout requires more training to show consistent positive rewards due to its complexity."),
]

story += [
    h2("4.2 Q-Value Overestimation"),
    p("The mean maximum Q-value diverges systematically between DQN and DDQN over training. DQN Q-values "
      "exhibit a monotonic upward drift throughout training — the overestimation bias described by "
      "van Hasselt et al. (2016) accumulating as the same network is used for both action selection and "
      "evaluation. DDQN Q-values remain comparatively stable, reflecting the bias correction achieved by "
      "decoupling these two operations. This replicates the core empirical finding of van Hasselt et al. "
      "under the reduced-buffer training conditions used here."),
    fig("qvalue_overestimation.png",
        "Figure 3. Q-value overestimation. Mean max Q-value over training for DQN and DDQN on both games. "
        "DQN values drift upward (overestimation bias); DDQN values remain grounded."),
]

story += [
    h2("4.3 t-SNE Representation Analysis"),
    p("<b>Game effect.</b> The most prominent structural feature of the t-SNE embeddings is separation by "
      "game. When DQN/Pong and DQN/Breakout representations are projected together, distinct clusters "
      "emerge corresponding to each game (Figure 4). The same pattern holds for DDQN (Figure 5). Game "
      "content — visual structure, dynamics, and strategic demands — is the primary organising principle "
      "of the representation space, more so than the learning algorithm."),
    fig("tsne_game_effect_dqn.png",
        "Figure 4. t-SNE — game effect (DQN). DQN/Pong (blue) and DQN/Breakout (orange) representations "
        "projected together. Clusters separate cleanly by game."),
    fig("tsne_game_effect_ddqn.png",
        "Figure 5. t-SNE — game effect (DDQN). DDQN/Pong and DDQN/Breakout representations. "
        "Game-level separation is consistent across algorithms."),
]

story += [
    p("<b>Algorithm effect.</b> Within each game, the algorithm produces observable representational "
      "differences. DQN and DDQN on Pong (Figure 6) occupy overlapping but distinguishable t-SNE regions, "
      "with DDQN showing more compact cluster geometry — lower intra-cluster variance — consistent with "
      "cleaner gradient signals producing more organised representations. The same pattern appears on "
      "Breakout (Figure 7), though less pronounced, suggesting the algorithm effect is stronger where the "
      "learning signal is less dominated by environmental complexity."),
    fig("tsne_algo_effect_pong.png",
        "Figure 6. t-SNE — algorithm effect (Pong). DQN vs DDQN on Pong. DDQN shows tighter, "
        "more compact cluster geometry."),
    fig("tsne_algo_effect_breakout.png",
        "Figure 7. t-SNE — algorithm effect (Breakout). DQN vs DDQN on Breakout. Algorithm differences "
        "are present but modulated by the game's complexity."),
]

story += [
    p("<b>All agents.</b> The joint projection of all four agents (Figure 8) shows two primary clusters "
      "corresponding to the two games, with algorithm-level sub-structure visible within each. The "
      "representation space is organised first by what the agent is learning about, and second by how "
      "it learned."),
    fig("tsne_all_agents.png",
        "Figure 8. t-SNE — all four agents. Game clusters dominate; algorithm sub-structure is visible "
        "within each game cluster."),
    p("<b>Reward structure.</b> Representations coloured by cumulative episode reward (Figure 9) show "
      "a partial correspondence between t-SNE position and value. High-reward states tend to occupy "
      "specific regions of the embedding, more consistently under DDQN, suggesting DDQN representations "
      "encode some degree of value-relevant information beyond purely perceptual features."),
    fig("tsne_by_reward.png",
        "Figure 9. t-SNE coloured by cumulative reward. High-reward (green) and low-reward (red) states "
        "show spatial correspondence in the embedding, stronger under DDQN."),
]

story += [
    p("<b>Temporal evolution.</b> Checkpoints across training reveal the developmental trajectory of the "
      "representation space (Figure 10). Early-training representations form a diffuse, unstructured cloud. "
      "Clusters emerge and tighten as training progresses. DDQN representations show earlier emergence of "
      "clear clustering, consistent with faster convergence of the gradient signal."),
    fig("tsne_temporal_dqn_pong.png",
        "Figure 10. Temporal evolution — DQN/Pong. t-SNE at each 500k-step checkpoint. "
        "Representations evolve from a diffuse cloud to structured clusters over training."),
]

story += [
    h2("4.4 Saliency Maps"),
    p("Grad-CAM saliency maps (Selvaraju et al., 2017) computed over final checkpoints for all four agents "
      "reveal consistent attention patterns: highest saliency consistently corresponds to the ball position, "
      "with secondary attention on the agent's paddle. Both algorithms have correctly identified the "
      "task-relevant objects."),
    p("The key difference lies in concentration and consistency. DQN saliency maps show more diffuse "
      "activation spread across larger screen regions. DDQN maps show tighter, more localised activation "
      "concentrated on the ball and paddle. On Breakout specifically, DDQN shows greater attention to "
      "bricks in the ball's trajectory, indicating more strategic forward-looking attention."),
    fig("saliency_pong.png",
        "Figure 11. Saliency maps — Pong. Grad-CAM heatmaps for DQN (top) and DDQN (bottom). "
        "Both agents attend to the ball; DDQN attention is more spatially concentrated."),
    fig("saliency_breakout.png",
        "Figure 12. Saliency maps — Breakout. DDQN shows more focused saliency on the ball and "
        "the bricks directly in its path."),
]

story += [
    h2("4.5 Dead Neurons and Cosine Similarity"),
    p("The fraction of chronically inactive neurons (firing in fewer than 5% of states) in the "
      "512-dimensional representation layer is consistently higher for DQN than DDQN across both games "
      "(Figure 13). DQN's noisier gradient signal drives more neurons into the ReLU dead zone from which "
      "gradient-based recovery is impossible. DDQN's cleaner updates maintain a more consistently active "
      "representation layer, utilising a larger fraction of its 512 available dimensions."),
    fig("dead_neurons.png",
        "Figure 13. Dead neuron fraction over training. DQN consistently shows a higher proportion of "
        "inactive representation neurons than DDQN across both games."),
    p("Cross-game cosine similarity between mean Pong and Breakout representations (Figure 14) is "
      "non-trivial at both the start and end of training, consistent with shared ball and paddle visual "
      "features producing shared representational directions. DDQN shows marginally higher similarity, "
      "suggesting its representations retain more of the shared low-level visual structure rather than "
      "being perturbed by overestimation-induced noise."),
    fig("cosine_similarity.png",
        "Figure 14. Cross-game cosine similarity. Non-trivial similarity between Pong and Breakout "
        "mean representations reflects shared ball/paddle visual structure. DDQN shows higher alignment."),
]

# 5. Discussion
story += [
    PageBreak(),
    h1("5. Discussion"),
    h2("5.1 Game Content as the Dominant Organising Principle"),
    p("The clearest finding is that game content dominates the geometry of the representation space. An "
      "agent's internal model of the world is shaped primarily by what it is learning about — the visual "
      "structure, dynamics, and strategic demands of the game — rather than by the learning rule it uses. "
      "The convolutional layers learn features specific to the game's visual domain (brick patterns in "
      "Breakout, opponent paddle dynamics in Pong), and these features necessarily produce different "
      "representation geometries regardless of the Q-value target formulation."),
    p("This finding does not diminish the algorithm effect — DDQN consistently produces more structured "
      "representations within each game — but contextualises it. The game determines the <i>what</i> of "
      "representation; the algorithm modulates the <i>quality</i> of how that content is encoded."),
    h2("5.2 The Gradient Signal Hypothesis"),
    p("DDQN's representational advantages — tighter clusters, fewer dead neurons, more focused saliency, "
      "higher cross-game similarity — are best understood through the gradient signal hypothesis: cleaner "
      "gradients produce more structured representations. DQN's overestimation bias introduces systematic "
      "noise into the training signal. Targets are inflated, particularly for states where Q-value "
      "estimates are already noisy, meaning the network is trained on labels more erratic than the true "
      "action values."),
    p("DDQN removes this noise source without changing anything else. The representation layer then "
      "receives a more consistent signal about which features of the game state are predictive of future "
      "reward, and organises itself accordingly."),
    h2("5.3 Implications for Transfer Learning"),
    p("If game content dominates representational geometry while algorithm modulates quality, a natural "
      "prediction follows: DDQN representations should transfer better between structurally similar games "
      "than DQN representations, both because they are more structured and because they retain more shared "
      "low-level visual features. Testing this — fine-tuning agents pre-trained on Pong to play Breakout "
      "comparing DQN vs DDQN — would be a natural extension of this work."),
    h2("5.4 Limitations"),
    p("The reduced replay buffer (100k vs the original 1M transitions) was a practical concession to "
      "memory constraints. The buffer proved sufficient to reproduce key qualitative results, but some "
      "representational differences might be more pronounced under the original conditions. Only two games "
      "are studied; replication across a broader game pair set, particularly games with varying degrees of "
      "structural similarity, would strengthen the generalisability of the findings. Finally, t-SNE "
      "cluster compactness is assessed visually; quantitative measures such as the Davies-Bouldin index "
      "applied in the original high-dimensional space would further substantiate the algorithm-effect claims."),
]

# 6. Conclusion
story += [
    h1("6. Conclusion"),
    p("This study provides a controlled, activation-level analysis of how game content and learning "
      "algorithm interact to shape the internal representations of deep RL agents. The key findings are:"),
    Paragraph(
        "1. <b>Game content is the dominant organising principle</b> of the representation space. "
        "t-SNE clusters separate cleanly by game across both algorithms.",
        ParagraphStyle("indent", parent=body_style, leftIndent=1*cm)
    ),
    Paragraph(
        "2. <b>DDQN produces measurably more structured representations</b> — tighter t-SNE clusters, "
        "fewer dead neurons, more focused saliency, and higher cross-game cosine similarity — consistent "
        "with cleaner gradient signals reducing representational noise.",
        ParagraphStyle("indent", parent=body_style, leftIndent=1*cm)
    ),
    Paragraph(
        "3. <b>DQN exhibits systematic Q-value overestimation</b> as predicted, with mean max Q-values "
        "drifting upward throughout training while DDQN values remain stable.",
        ParagraphStyle("indent", parent=body_style, leftIndent=1*cm)
    ),
    Paragraph(
        "4. <b>Both algorithms show ball-centred attention</b> in saliency maps; DDQN exhibits more "
        "spatially concentrated and strategically coherent saliency.",
        ParagraphStyle("indent", parent=body_style, leftIndent=1*cm)
    ),
    Paragraph(
        "5. <b>Non-trivial cross-game representational similarity</b> exists, consistent with shared "
        "visual structure, and is higher under DDQN training.",
        ParagraphStyle("indent", parent=body_style, leftIndent=1*cm)
    ),
    sp(0.3),
    p("Algorithm choice matters not only for performance — the traditional metric — but for the internal "
      "structure of what the agent learns. DDQN does not merely achieve higher scores; it constructs a "
      "more organised internal model of the game world. This has implications for the interpretability "
      "and transferability of deep RL agents and motivates further study of how training dynamics shape "
      "representation geometry across a broader range of algorithms and environments."),
]

# References
story += [
    PageBreak(),
    h1("References"),
    Paragraph(
        "Bellemare, M. G., Naddaf, Y., Veness, J., &amp; Bowling, M. (2013). The Arcade Learning "
        "Environment: An evaluation platform for general agents. <i>Journal of Artificial Intelligence "
        "Research</i>, 47, 253–279.", ref_style),
    Paragraph(
        "Maaten, L. van der, &amp; Hinton, G. (2008). Visualizing data using t-SNE. "
        "<i>Journal of Machine Learning Research</i>, 9, 2579–2605.", ref_style),
    Paragraph(
        "Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep "
        "reinforcement learning. <i>Nature</i>, 518(7540), 529–533.", ref_style),
    Paragraph(
        "Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., &amp; Batra, D. (2017). "
        "Grad-CAM: Visual explanations from deep networks via gradient-based localization. "
        "<i>Proceedings of ICCV</i>, 618–626.", ref_style),
    Paragraph(
        "Sutton, R. S., &amp; Barto, A. G. (2018). <i>Reinforcement Learning: An Introduction</i> "
        "(2nd ed.). MIT Press.", ref_style),
    Paragraph(
        "van Hasselt, H., Guez, A., &amp; Silver, D. (2016). Deep reinforcement learning with double "
        "Q-learning. <i>Proceedings of AAAI</i>, 30(1).", ref_style),
    Paragraph(
        "Zahavy, T., Ben-Zrihem, N., &amp; Mannor, S. (2016). Graying the black box: Understanding DQNs. "
        "<i>Proceedings of ICML</i>, 48, 1899–1908.", ref_style),
]

doc.build(story)
print(f"PDF generated: {OUTPUT}")

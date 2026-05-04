"""
Generate a plain-English explainer PDF for personal study.
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
from PIL import Image as PILImage
import os

PLOTS = "results/plots"
OUTPUT = "report/explainer.pdf"
os.makedirs("report", exist_ok=True)

W, H = A4
MARGIN = 2.5 * cm

doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN, bottomMargin=MARGIN,
    title="RL Project — Plain English Explainer",
    author="Neil Tauro"
)

styles = getSampleStyleSheet()

title_s = ParagraphStyle("T", parent=styles["Title"],
    fontSize=22, leading=26, spaceAfter=6, alignment=TA_CENTER,
    textColor=colors.HexColor("#1a1a2e"))

subtitle_s = ParagraphStyle("Sub", parent=styles["Normal"],
    fontSize=12, leading=16, spaceAfter=4, alignment=TA_CENTER,
    textColor=colors.HexColor("#555555"))

h1_s = ParagraphStyle("H1", parent=styles["Heading1"],
    fontSize=15, leading=19, spaceBefore=18, spaceAfter=8,
    textColor=colors.HexColor("#1a1a2e"),
    borderPad=4, backColor=colors.HexColor("#eef2ff"),
    borderColor=colors.HexColor("#2d4a7a"), borderWidth=0,
    leftIndent=-0.3*cm, rightIndent=-0.3*cm)

h2_s = ParagraphStyle("H2", parent=styles["Heading2"],
    fontSize=12, leading=15, spaceBefore=12, spaceAfter=5,
    textColor=colors.HexColor("#2d4a7a"))

body_s = ParagraphStyle("B", parent=styles["Normal"],
    fontSize=10.5, leading=16, spaceAfter=9, alignment=TA_JUSTIFY)

callout_s = ParagraphStyle("C", parent=styles["Normal"],
    fontSize=10, leading=15, spaceAfter=8,
    leftIndent=1*cm, rightIndent=1*cm,
    backColor=colors.HexColor("#fff8e7"),
    borderColor=colors.HexColor("#f0a500"), borderWidth=1.5,
    borderPad=8, textColor=colors.HexColor("#333333"))

tip_s = ParagraphStyle("Tip", parent=styles["Normal"],
    fontSize=10, leading=15, spaceAfter=8,
    leftIndent=1*cm, rightIndent=1*cm,
    backColor=colors.HexColor("#e8f5e9"),
    borderColor=colors.HexColor("#2e7d32"), borderWidth=1.5,
    borderPad=8, textColor=colors.HexColor("#1b5e20"))

caption_s = ParagraphStyle("Cap", parent=styles["Normal"],
    fontSize=8.5, leading=12, spaceAfter=14, alignment=TA_CENTER,
    textColor=colors.HexColor("#555555"), fontName="Helvetica-Oblique")

bullet_s = ParagraphStyle("Bul", parent=styles["Normal"],
    fontSize=10.5, leading=16, spaceAfter=5,
    leftIndent=1.2*cm, firstLineIndent=-0.7*cm)

def h1(t): return Paragraph(t, h1_s)
def h2(t): return Paragraph(t, h2_s)
def p(t):  return Paragraph(t, body_s)
def tip(t): return Paragraph(t, tip_s)
def note(t): return Paragraph(t, callout_s)
def sp(n=0.4): return Spacer(1, n*cm)
def hr(): return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=10)
def bullet(t): return Paragraph(f"• {t}", bullet_s)

def fig(filename, caption, width_frac=0.88):
    path = os.path.join(PLOTS, filename)
    if not os.path.exists(path):
        return KeepTogether([
            Paragraph(f"[Figure not found: {filename}]", caption_s),
            sp(0.2),
        ])
    max_w = (W - 2*MARGIN) * width_frac
    with PILImage.open(path) as im:
        iw, ih = im.size
    ratio = ih / iw
    img_w = min(max_w, iw)
    img_h = img_w * ratio
    max_h = H * 0.40
    if img_h > max_h:
        img_h = max_h
        img_w = img_h / ratio
    return KeepTogether([
        Image(path, width=img_w, height=img_h),
        Paragraph(caption, caption_s),
    ])

story = []

# ── Cover ─────────────────────────────────────────────────────────────────────
story += [
    sp(1.5),
    Paragraph("My RL Project — Plain English", title_s),
    Paragraph("Everything you need to understand and explain it", subtitle_s),
    sp(0.3),
    hr(),
    Paragraph("Neil Tauro | RL Dissertation Component | TCD", subtitle_s),
    sp(2),
    note("📖  This is NOT the formal report. This is the version you read the night before "
         "to actually understand what you built, why it works, and how to talk about it confidently."),
    PageBreak(),
]

# ── What Did I Actually Build? ────────────────────────────────────────────────
story += [
    h1("1. What Did I Actually Build?"),
    p("You trained two AI agents to play two Atari games. Then — instead of just measuring the scores — "
      "you looked <i>inside</i> the AI's brain to see what it had learned internally."),
    p("The two agents are called <b>DQN</b> and <b>Double DQN (DDQN)</b>. They are almost identical — "
      "same neural network, same settings, same everything. The only difference is three lines of code "
      "in how they compute their training targets. You'll explain why that matters in a minute."),
    p("The two games are <b>Pong</b> and <b>Breakout</b>. You picked these deliberately because they share "
      "key visual features (ball, paddle, physics) but play very differently at a strategic level."),
    sp(0.2),
    tip("💡  One sentence to remember: \"I trained four agents in a 2×2 experiment — two algorithms × two games "
        "— and compared the internal representations they developed, not just their scores.\""),
]

# ── The 2x2 Design ────────────────────────────────────────────────────────────
story += [
    h1("2. The Experiment — The 2×2 Design"),
    p("Think of it like a table:"),
    sp(0.2),
]

data = [
    [Paragraph("", ParagraphStyle("x", fontSize=10, fontName="Helvetica-Bold", alignment=TA_CENTER)),
     Paragraph("Pong", ParagraphStyle("x", fontSize=10, fontName="Helvetica-Bold", alignment=TA_CENTER)),
     Paragraph("Breakout", ParagraphStyle("x", fontSize=10, fontName="Helvetica-Bold", alignment=TA_CENTER))],
    [Paragraph("DQN", ParagraphStyle("x", fontSize=10, fontName="Helvetica-Bold", alignment=TA_CENTER)),
     Paragraph("Run 1\n2M steps ✅", ParagraphStyle("x", fontSize=10, alignment=TA_CENTER, leading=14)),
     Paragraph("Run 2\n5M steps ✅", ParagraphStyle("x", fontSize=10, alignment=TA_CENTER, leading=14))],
    [Paragraph("Double DQN", ParagraphStyle("x", fontSize=10, fontName="Helvetica-Bold", alignment=TA_CENTER)),
     Paragraph("Run 3\n2M steps ✅", ParagraphStyle("x", fontSize=10, alignment=TA_CENTER, leading=14)),
     Paragraph("Run 4\n5M steps ✅", ParagraphStyle("x", fontSize=10, alignment=TA_CENTER, leading=14))],
]
t = Table(data, colWidths=[4*cm, 5*cm, 5*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a2e")),
    ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
    ("BACKGROUND", (0,1), (0,-1), colors.HexColor("#2d4a7a")),
    ("TEXTCOLOR",  (0,1), (0,-1), colors.white),
    ("ROWBACKGROUNDS", (1,1), (-1,-1), [colors.HexColor("#f0f4ff"), colors.HexColor("#dce8ff")]),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#aaaaaa")),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 8),
    ("BOTTOMPADDING", (0,0), (-1,-1), 8),
]))
story += [t, sp(0.3)]

story += [
    p("This design lets you isolate two things:"),
    bullet("<b>Game effect:</b> Compare Run 1 vs Run 2 — same algorithm, different game. Do the games produce "
           "different internal representations?"),
    bullet("<b>Algorithm effect:</b> Compare Run 1 vs Run 3 — same game, different algorithm. Does the learning "
           "rule change what gets encoded internally?"),
    sp(0.2),
    tip("💡  Why 2M steps for Pong but 5M for Breakout? Breakout is harder. The AI needs more practice. "
        "2M steps ≈ 33 hours of gameplay. Pong has a simpler strategy; it converges faster."),
]

# ── DQN vs DDQN ───────────────────────────────────────────────────────────────
story += [
    h1("3. DQN vs Double DQN — What's the Actual Difference?"),
    h2("The core idea of DQN"),
    p("DQN is a neural network that learns to predict how good each action is in any given situation — "
      "this prediction is called a Q-value. 'Q' stands for quality. Higher Q = better action."),
    p("It learns by playing the game over and over. Every time something happens, it updates its Q-value "
      "predictions based on what reward it got. The target it tries to hit is called the <b>TD target</b>."),
    h2("The problem with DQN — overestimation"),
    p("Here's the flaw. When DQN computes its training target, it does two things with the same network:"),
    bullet("Picks the best next action (the argmax)"),
    bullet("Estimates how good that action is (the Q-value)"),
    p("Using the same noisy network for both steps causes it to consistently <b>overestimate</b> how good "
      "actions are. The maximum of a bunch of noisy estimates is always biased upward. Over millions of "
      "training steps this bias compounds — the AI thinks it's doing better than it really is."),
    h2("The Double DQN fix — three lines of code"),
    p("DDQN fixes this by using <b>two different networks</b> for the two steps:"),
    bullet("The <b>online network</b> picks which action to take"),
    bullet("The <b>target network</b> estimates how good that action is"),
    p("Because the two networks were trained at different times, their errors don't overlap. The overestimation "
      "bias cancels out. That's it. That's the entire difference. Three lines of code in the training loop."),
    sp(0.2),
    note("🧠  The key insight for your professor: \"The two algorithms are architecturally identical — same CNN, "
         "same hyperparameters, same everything. The only difference is which network evaluates the action. "
         "So any difference I find in the representations is purely down to this one algorithmic change.\""),
]

# ── What is a Representation? ─────────────────────────────────────────────────
story += [
    h1("4. What Is a 'Representation' and Why Does It Matter?"),
    p("The neural network has several layers. The last few layers turn game pixels into Q-values. "
      "But right before the final Q-value layer, there's a layer with 512 neurons. These 512 numbers "
      "are the AI's internal <b>summary</b> of what it's seeing."),
    p("Think of it like this: if you show the AI a game frame of Pong, it processes it through convolutional "
      "layers (which detect edges, shapes, motion) and compresses everything into a 512-number vector. "
      "That vector is the representation — the AI's internal description of the game state."),
    p("The hypothesis is: if two game states are similar (ball in same position, same velocity), their "
      "representations should be similar. If two states are very different, their representations should be "
      "very different. A well-trained AI should have a representation space that mirrors the structure of "
      "the game world."),
    sp(0.2),
    tip("💡  Analogy: it's like asking 'when you picture a game of Pong in your head, what do you actually "
        "think about?' The representation is what the AI 'thinks about' when it sees a game frame."),
    h2("How you extracted the representations"),
    p("You added a forward hook to the neural network — a piece of code that fires every time the network "
      "processes a frame and saves that 512-number vector. Then you let each trained agent play 5,000 steps "
      "and collected 5,000 of these vectors. That's your dataset for analysis."),
]

# ── t-SNE ─────────────────────────────────────────────────────────────────────
story += [
    h1("5. t-SNE — How You Visualised 512 Dimensions"),
    p("You can't visualise a 512-dimensional space. So you used a tool called <b>t-SNE</b> to compress "
      "it down to 2D while preserving the clustering structure. Points that were close in 512D end up "
      "close in 2D. Points that were far apart stay far apart."),
    p("The result is a scatter plot where each dot is one game frame (one moment in the game). The colour "
      "tells you which agent produced it. If dots of the same colour cluster together, those game states "
      "have similar internal representations."),
    sp(0.2),
    note("⚠️  One thing to know: the distance <i>between</i> clusters in a t-SNE plot is not meaningful — "
         "only whether points cluster together or not. Don't say 'cluster A is twice as far from cluster B "
         "as from C' — that's not a valid interpretation."),
]

# ── Results with figures ──────────────────────────────────────────────────────
story += [
    h1("6. Results — What Did You Find?"),
    h2("Training performance"),
    p("All four agents learned successfully. DQN on Pong started at −21 (losing every point) and "
      "reached a rolling average of around +5 by 2M steps, with individual games reaching +17. "
      "The learning curve shows the classic progression: random play → learning to hit the ball → "
      "winning some points → winning most games."),
]
story += [
    fig("training_curves_pong.png",
        "Figure 1 — Training curves for Pong. Both DQN and DDQN improve from −21 toward positive reward. "
        "The rolling average smooths out the episode-to-episode noise."),
    fig("training_curves_breakout.png",
        "Figure 2 — Training curves for Breakout (5M steps). Breakout takes longer to show "
        "consistent improvement because it's a harder, more strategic game."),
    h2("Q-value overestimation — the smoking gun"),
    p("This is one of your strongest results. DQN's predicted Q-values keep drifting upward over "
      "training — the AI is increasingly overconfident about how good its actions are. DDQN's Q-values "
      "stay flat and realistic. This is direct empirical evidence of the overestimation bias you're studying."),
]
story += [
    fig("qvalue_overestimation.png",
        "Figure 3 — Q-value overestimation. DQN's mean max Q-value rises continuously (overestimation); "
        "DDQN's stays stable. This reproduces the key finding of van Hasselt et al. (2016)."),
    h2("Game effect — the game shapes representations more than the algorithm"),
    p("The biggest finding: when you plot Pong representations and Breakout representations together in "
      "t-SNE, they form completely separate clusters. It doesn't matter whether it's DQN or DDQN — "
      "the game you play is the dominant organising force of your internal representation space."),
    p("This makes intuitive sense: Pong requires tracking an opponent and a horizontally-moving ball. "
      "Breakout requires tracking a ball bouncing through brick layers. The networks have to learn "
      "fundamentally different spatial features for each game, so their internal summaries (representations) "
      "end up very different."),
]
story += [
    fig("tsne_game_effect_dqn.png",
        "Figure 4 — t-SNE game effect (DQN). Each dot is one game frame. Blue = Pong, Orange = Breakout. "
        "Separate clusters mean the two games produce different internal representations."),
    fig("tsne_game_effect_ddqn.png",
        "Figure 5 — t-SNE game effect (DDQN). Same separation pattern holds for DDQN. "
        "The game effect is consistent across both algorithms."),
    h2("Algorithm effect — DDQN produces cleaner representations"),
    p("Within each game, the algorithm does make a difference. DDQN representations form tighter, "
      "more compact clusters than DQN. The points don't spread out as much — they're more consistent. "
      "This is the visual signature of a cleaner learning signal: DDQN's corrected gradient means the "
      "network learns more consistently what each game state 'means'."),
]
story += [
    fig("tsne_algo_effect_pong.png",
        "Figure 6 — Algorithm effect on Pong. DQN vs DDQN on the same game. DDQN clusters are tighter "
        "— more compact geometry — suggesting more structured internal representations."),
    fig("tsne_algo_effect_breakout.png",
        "Figure 7 — Algorithm effect on Breakout. Similar pattern, slightly less pronounced "
        "because Breakout's complexity partially dominates the signal."),
    fig("tsne_all_agents.png",
        "Figure 8 — All four agents together. Two main clusters (one per game) with algorithm-level "
        "sub-structure inside each. The representation space is organised first by game, then by algorithm."),
]

story += [
    h2("Reward structure in representations"),
    p("When you colour the t-SNE dots by how much reward the agent was getting at that moment, you see "
      "that high-reward and low-reward states tend to cluster in different parts of the map. The AI's "
      "internal representation isn't just about what it sees — it also encodes something about how good "
      "the situation is. This is cleaner in DDQN."),
    fig("tsne_by_reward.png",
        "Figure 9 — t-SNE coloured by reward. Green = high reward, Red = low reward. "
        "Reward-relevant structure is visible in the representation space, especially for DDQN."),
    h2("How representations evolve over training"),
    p("By saving checkpoints every 500,000 steps and extracting representations from each, you can watch "
      "the representation space develop. Early in training it's a random cloud — the AI hasn't learned "
      "anything yet. As training progresses, structure emerges and clusters form. By the end, it's "
      "well-organised. DDQN reaches a structured state earlier."),
    fig("tsne_temporal_dqn_pong.png",
        "Figure 10 — Temporal evolution of DQN/Pong representations at each checkpoint (500k, 1M, 1.5M, 2M steps). "
        "The representation space goes from a diffuse cloud to organised clusters as training progresses."),
]

story += [
    h2("Saliency maps — what is the AI actually looking at?"),
    p("Saliency maps (using a technique called Grad-CAM) show you which pixels in a game frame most "
      "influenced the AI's decision. The redder the pixel, the more the AI was 'looking at' it."),
    p("Both DQN and DDQN correctly focus on the ball — which is the most important thing to track in "
      "both games. But DDQN's attention is more focused and concentrated. DQN tends to spread its "
      "attention around more, including some irrelevant background pixels."),
    fig("saliency_pong.png",
        "Figure 11 — Saliency maps on Pong. Bright regions show where the AI is 'looking'. "
        "Both agents focus on the ball; DDQN's attention is more tightly concentrated."),
    fig("saliency_breakout.png",
        "Figure 12 — Saliency maps on Breakout. DDQN shows more strategic attention — "
        "tracking the ball and the bricks directly in its path."),
    h2("Dead neurons — wasted capacity"),
    p("A dead neuron is one that has stopped firing — it always outputs 0 no matter what the input is. "
      "This happens when DQN's noisy training pushes neurons into a permanently inactive state they "
      "can't recover from (because the gradient through a ReLU at 0 is also 0 — no learning signal)."),
    p("DQN has more dead neurons than DDQN throughout training. This means DQN is wasting some of its "
      "512 representation dimensions. DDQN uses them all more consistently."),
    fig("dead_neurons.png",
        "Figure 13 — Fraction of dead neurons in the 512-dim representation layer over training. "
        "DQN consistently has more inactive neurons than DDQN."),
    h2("Cross-game similarity — shared features"),
    p("Even though Pong and Breakout produce different representation clusters, the mean representations "
      "are not completely orthogonal — there's some similarity. This makes sense: both games have a ball "
      "and a paddle. The early convolutional layers develop ball-detection and paddle-detection filters "
      "that are useful for both games, and this shared structure shows up as non-zero cosine similarity "
      "between the mean representations. DDQN shows slightly higher similarity, meaning it retains more "
      "of this shared visual structure."),
    fig("cosine_similarity.png",
        "Figure 14 — Cross-game cosine similarity over training. Non-trivial similarity reflects the "
        "shared ball/paddle visual features. DDQN is slightly better aligned across games."),
]

# ── How to Talk to Professor ──────────────────────────────────────────────────
story += [
    PageBreak(),
    h1("7. How to Talk About This to Your Professor"),
    h2("Opening — one sentence summary"),
    note("\"I trained DQN and Double DQN on Pong and Breakout in a 2×2 factorial design, then analysed "
         "the 512-dimensional representations the agents develop internally — not just their scores.\""),
    h2("The three things you found"),
    bullet("<b>Game effect dominates:</b> The game you train on is the biggest factor in shaping internal "
           "representations. Pong and Breakout produce clearly separate clusters in t-SNE, regardless of algorithm."),
    bullet("<b>DDQN is more structured:</b> Within each game, DDQN produces tighter, more compact representation "
           "clusters, fewer dead neurons, and more focused saliency — consistent with its cleaner gradient signal."),
    bullet("<b>Q-value overestimation is real:</b> DQN's Q-values drift upward over training (overestimation bias). "
           "DDQN's stay flat. This reproduces the core finding of van Hasselt et al. (2016)."),
    sp(0.3),
    h2("Questions your professor might ask — and how to answer"),
    h2("Why did you pick these two games?"),
    p("They share low-level visual features — ball, paddle, deflection physics — but differ in high-level "
      "strategy. This lets you test whether shared visual structure produces shared representations, and "
      "whether strategic differences override that. If you'd picked completely different games (Pac-Man and "
      "Space Invaders) you couldn't isolate the visual structure variable."),
    h2("Why is the 2×2 design important?"),
    p("It lets you isolate the effect of each variable independently. If you only trained DQN on Pong and "
      "DDQN on Breakout, you couldn't tell if any difference was due to the algorithm or the game. The "
      "factorial design controls for that."),
    h2("What does it mean that game effect dominates?"),
    p("It means the visual and strategic content of what you're learning about shapes your internal model "
      "more than the rule you use to learn it. The game determines the 'what' of representation; the "
      "algorithm modulates the 'quality'. This is interesting for transfer learning — if games shape "
      "representations, then training on similar games might give you a head start."),
    h2("Why does DDQN produce better representations?"),
    p("DQN's overestimation bias injects noise into the gradient signal. The network is being trained "
      "toward inflated, incorrect targets. This noise makes it harder for the representation layer to "
      "develop consistent, organised structure. DDQN removes that noise source, so the gradient signal "
      "is cleaner, and the representation layer organises itself more effectively."),
    h2("Why do you only have 14 figures not 17?"),
    p("The ablation figures (network size, learning rate, buffer size) are still running on the VM — "
      "those require separate training runs. The 14 core figures cover all four main runs and the "
      "complete representation analysis."),
    sp(0.5),
    tip("💡  If you get stuck on any question, the best fallback is: \"That's a good question — what "
        "I can say is that the representation analysis shows X, which suggests Y. I'd need to run "
        "additional experiments to say more definitively.\" Honest and shows you understand the limits."),
]

# ── Glossary ──────────────────────────────────────────────────────────────────
story += [
    h1("8. Quick Glossary — Key Terms"),
]

terms = [
    ("Q-value", "A score representing how good an action is in a given state. Higher = better. "
     "The AI learns these by playing the game."),
    ("TD target", "The 'correct answer' the neural network is trying to predict at each training step. "
     "Computed using the Bellman equation."),
    ("Overestimation bias", "DQN's tendency to predict Q-values that are too high. Caused by using "
     "the same network to both select and evaluate actions."),
    ("Replay buffer", "A memory bank of past experiences (game frames, actions, rewards) that the AI "
     "samples randomly from during training. Prevents it from only learning from recent events."),
    ("Target network", "A frozen copy of the online network updated every 1,000 steps. Provides stable "
     "training targets. Without it, training would be like chasing a moving target."),
    ("Representation layer", "The 512-neuron layer just before the Q-value output. Its activations "
     "are the AI's internal summary of the game state — what we analyse."),
    ("t-SNE", "A tool that compresses high-dimensional data (512D) down to 2D for visualisation, "
     "preserving cluster structure. Points that are similar in 512D appear nearby in 2D."),
    ("Grad-CAM / Saliency map", "A technique that shows which pixels most influenced the AI's decision. "
     "Essentially: what is the AI 'looking at' when it makes a move?"),
    ("Dead neuron", "A neuron that always outputs 0 regardless of input. Stuck permanently in the "
     "ReLU dead zone. Wasted capacity — that neuron contributes nothing to the representation."),
    ("Cosine similarity", "A measure of how similar two vectors are, between -1 and 1. "
     "1 = identical direction, 0 = completely different, -1 = opposite."),
    ("Forward hook", "A piece of code that fires every time the network processes an input and "
     "captures intermediate layer activations — how you extracted the 512-dim representations."),
    ("Frame stack", "Feeding the last 4 game frames as input instead of just 1. This gives the AI "
     "velocity information — it can tell which way the ball is moving."),
    ("Epsilon-greedy", "The exploration strategy: with probability ε, take a random action (explore); "
     "otherwise take the best known action (exploit). ε starts at 1.0 and decays to 0.01."),
]

for term, defn in terms:
    story.append(KeepTogether([
        Paragraph(f"<b>{term}</b>", ParagraphStyle("TH", parent=styles["Normal"],
            fontSize=10.5, leading=14, spaceBefore=8, textColor=colors.HexColor("#1a1a2e"))),
        Paragraph(defn, ParagraphStyle("TD", parent=styles["Normal"],
            fontSize=10, leading=14, spaceAfter=6, leftIndent=1*cm,
            textColor=colors.HexColor("#333333"))),
    ]))

story += [
    sp(0.5),
    hr(),
    Paragraph("Good luck with the meeting. You built this — you understand it.", ParagraphStyle(
        "end", parent=styles["Normal"], fontSize=11, leading=16, alignment=TA_CENTER,
        textColor=colors.HexColor("#2d4a7a"), spaceBefore=12)),
]

doc.build(story)
print(f"PDF generated: {OUTPUT}")

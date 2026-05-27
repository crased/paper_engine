# Paper Engine — Roadmap

**Status:** direction reset (2026-05-26). The project is pivoting away from
Cuphead-specific memory reading + bot automation toward a **general-purpose
game object detector**.

---

## North Star

> Point Paper Engine at *any* game and have it detect the things that matter —
> ideally with zero per-game setup.

"The things that matter" can mean two different things, and the choice decides
how far we have to go (see D1 and Phase 4):

- **A fixed, universal set** (character, projectile, UI, interactive, terrain) →
  a closed-vocab YOLO trained across many games is sufficient. Simpler, faster,
  more accurate. We stop at the generalist.
- **Whatever a specific game cares about** (rival car, resource node, quest
  marker — unbounded, per-game) → no fixed taxonomy can cover it, and we need an
  **open-vocabulary detector** (text-prompted at runtime) as the end state.

Either way, we do **not** get there by training one fixed-class YOLO on one
game. We get there by building a data engine first, accumulating a multi-game
dataset, and training/fine-tuning on what it produces.

### Two axes of generalization (don't conflate them)

| Axis | Generalize to… | What handles it |
|---|---|---|
| **Visual** | a new *game*, detecting the *same* object types | closed-vocab multi-game YOLO |
| **Semantic** | *new object types* named at runtime | open-vocab only |

A multi-game closed model already covers the visual axis. Open-vocab is only
required for the semantic axis — arbitrary, un-pre-enumerated concepts.

---

## Strategy: a flywheel, not a ladder

```
  Per-game auto-annotation pipeline        ← THE DATA ENGINE (foundation)
   record gameplay → LLM annotates → labels
            │
            ▼  accumulate labels across many games (consistent taxonomy)
     Multi-game dataset
            │
            ├───────────────►  Multi-game generalist YOLO        (milestone)
            │
            └───────────────►  fine-tune open-vocab backbone     (north star)
                               (YOLO-World / Grounding DINO)
                               → approaches one-shot-anything
```

The per-game pipeline is the bottom of the stack. Every game it onboards adds
to the multi-game pile. The generalist model and the open-vocab fine-tune both
*consume* that pile — they are not separate efforts, they are downstream of the
same data engine. So the build order is:

1. Make the data engine reliable and game-agnostic.
2. Accumulate a diverse multi-game dataset through it.
3. Train a multi-game generalist from that data (proves the data is good).
4. Fine-tune an open-vocab model on the same data (the north star).

What we already have (Cuphead-era) maps directly onto step 1:
`gameplay_recorder` → `batch_annotator` (Gemini) → `audit_labels` /
`example_bank` / `golden_rules` → `training_model` / `test_model`.

### The data engine: a cold → hot crossover

Who labels a new game changes as the model gets more general. There is a
crossover point:

```
model competence on an UNSEEN game
   │
hot├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ╭──────────  north star: model labels new games itself;
   │                     ╭─╯              LLM nearly retired
   │                   ╭─╯
   │              ╭────╯   CROSSOVER: model pre-labels, human/LLM only corrects
   │         ╭────╯
cold├────────╯            bootstrap: model blind to new games → LLM labels
   └──────────────────────────────────────► games onboarded over time
```

- **Before the crossover** (now → first few games): the model is cold on
  anything new, so the **LLM is the cold-start labeler.** You cannot hot-start
  from generality you do not have yet.
- **After the crossover**: the model is warm/hot on new games → it pre-labels
  itself → the LLM only corrects, then becomes redundant.

**The LLM is temporary scaffolding** — its job is to carry the project across
the bootstrap until the model can self-label. Past the crossover the engine
feeds itself. Two honest caveats: "hot" is really "warm that keeps heating"
(novel art/objects always need *some* correction — the audit gate never hits
zero), and the bootstrap cannot be skipped.

### Alternative bootstrap: open-vocab model as the labeler (today)

The "general form" that warm-starts a new game does not have to be *our* trained
model. An **off-the-shelf open-vocab detector (YOLO-World) used as the labeler**
gives a warm start on any game *right now* — pulling the open-vocab work forward
to act as the Phase 1 labeler instead of (or alongside) the LLM. This can shorten
or skip the LLM cold-start entirely. Worth piloting in Phase 0/1 even if the
*shipped* detector stays a fast closed-vocab YOLO.

---

## Cross-cutting decisions (settle these before scaling)

These are not model choices — they determine whether the whole flywheel works.

### D1. Taxonomy must be cross-game, not Cuphead-shaped
The current 4 classes (player, enemy, projectile, platform) are role labels.
Roles look completely different across games, which a pixel detector cannot
learn. Define a small, **visually/functionally grounded** taxonomy that means
the same thing everywhere. Candidate direction (to be decided in Phase 0):

| Proposed class | Definition (visual/functional, not role) |
|---|---|
| character | any agent sprite/model (player or NPC) |
| projectile | any small moving hazard / fired object |
| interactive | doors, pickups, switches, nodes |
| ui | menus, HUD, text, buttons |
| platform/terrain | traversable surfaces (2D games) |

The "player vs enemy" distinction is a *downstream* problem (memory, tracking,
or heuristics) — not something the detector should be asked to learn visually.

**This decision also gates Phase 4.** If a fixed taxonomy like the above
genuinely covers "what matters," a closed-vocab generalist is the destination
and open-vocab is unnecessary. If "what matters" is open-ended and game-specific
(needs concepts the taxonomy can't pre-enumerate), open-vocab becomes required.
Decide the taxonomy *first*; it tells you whether Phase 4 exists.

### D2. Label quality is mission-critical
Multi-game means LLM annotation is the **only** scalable labeler — memory
reading cannot help on games we haven't reverse-engineered. Noisy or
inconsistent labels across games poison the generalist. Treat the existing
quality machinery as core infrastructure:
- `audit_labels` → gate every new game's labels before they enter the dataset.
- `golden_rules` / `example_bank` → keep the LLM annotator consistent.
- A human spot-check step stays in the loop until label quality is proven.

### D3. Memory reading: role shrinks (final decision deferred)
The Mono/`process_vm_readv` stack is Cuphead-only and cannot generalize. It is
no longer the point. Likely future role: an *optional* per-game "ground truth"
oracle to validate/bootstrap labels where available — never a dependency.
Decision on keep / demote / remove is deferred until after Phase 0.

---

## Phases

### Phase 0 — Validate the data engine on a second game *(do this first)*
The cheapest experiment with the most information. Premise check before any
refactor.

- Pick one **non-Cuphead** game, ideally a different art style.
- Record ~100 frames, run `batch_annotator`, eyeball the labels.
- Measure how a Cuphead-trained model degrades on game #2 (generalization baseline).

**Exit criteria:**
- LLM annotator produces usable labels on an unseen game.
- Current/Phase-0 taxonomy (D1) holds up on game #2.
- Decision recorded: is the per-game engine solid enough to scale?

### Phase 1 — Generalize the pipeline (remove Cuphead assumptions)
- Make `gameplay_recorder` capture work without memory reading (screenshot-only
  mode is already there via `--no-memory`; make it the default path).
- Lock in the cross-game taxonomy (D1) and migrate dataset configs.
- Harden the label-quality gate (D2) as a required pipeline step.

**Exit criteria:** onboarding a new game is a documented, repeatable process
that needs no code changes — just screenshots/recordings.

### Phase 2 — Accumulate a multi-game dataset
- Onboard N games (target: 5–10, spread across art styles and perspectives).
- Track per-game and per-class counts; enforce taxonomy consistency.
- Continuous audit; reject games whose labels can't pass the quality gate.

**Exit criteria:** a diverse, audited multi-game dataset large enough to train
a generalist (rough target: several thousand images across ≥5 games).

### Phase 3 — Train the multi-game generalist
- Train a single YOLO on the full multi-game set.
- Evaluate **held-out games** (train on some, test on entirely unseen ones) —
  this is the real generalization metric, not in-distribution mAP.

**Exit criteria:** measurable detection on a held-out game. This validates that
the dataset captures cross-game signal, not per-game memorization.

### Phase 4 — Open-vocabulary fine-tune (north star) — *contingent on D1*
**Only pursue this if the taxonomy decision (D1) came out "open-ended."** If a
fixed taxonomy covers what matters, the Phase 3 generalist is the destination
and this phase does not exist.

- Adopt an open-vocab backbone (YOLO-World / Grounding DINO / OWLv2).
- Baseline its zero-shot performance on games as-is.
- Fine-tune on the multi-game dataset to specialize it for the game domain.
- Goal: text-prompted detection on an unseen game with no per-game training.

**Exit criteria:** prompt-driven detection on a brand-new game that beats both
the zero-shot baseline and the fixed-class generalist.

> Note: an open-vocab model may still appear *earlier* as a **labeler** (see
> "Alternative bootstrap" above) even if this phase is skipped as a final model.

---

## Immediate next step

Run **Phase 0**: choose a second game (different visual style than Cuphead),
push ~100 frames through the existing pipeline, and look at the output. That
single experiment tells us whether the data engine — the foundation of
everything above — actually generalizes.

*Open question:* which second game? Needs to be installed/accessible and
ideally visually distinct from Cuphead.

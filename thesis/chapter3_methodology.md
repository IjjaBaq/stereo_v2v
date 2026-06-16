# Chapter 3: Methodology and Proposed Framework

## 3.1 Introduction

Chapter 1 framed the central question of this thesis: whether perception built
solely on stereo cameras can be made accurate enough that sharing detections
between two cooperating vehicles produces a measurable improvement in what each
one perceives. This chapter answers the *how*. It presents the end-to-end
engineering architecture of the proposed system — a four-stage perception and
fusion pipeline — and the development process used to bring it from requirements
to a validated implementation.

The chapter is organized around two complementary views of the same system. The
first is *structural*: the decomposition of the perception problem into a chain
of well-defined stages, each with a single responsibility, a typed interface to
its neighbours, and a standalone validation step. The second is *procedural*: the
**V-model** development process adopted to keep design, implementation, and
verification in deliberate correspondence, so that confidence in the final
cooperative result rests on stage-by-stage evidence rather than on the end-to-end
output alone.

Throughout, the design is governed by the central concern of Chapter 1: that
stereo localisation error is *range-dependent* — it grows quadratically with
distance — so the system must be designed around what stereo can recover reliably,
and honest about what it cannot. Several of the most consequential decisions in
this chapter — most notably the choice to output 3D *position* rather than full 3D
bounding boxes — follow directly from these physical limits of stereo sensing
rather than from convenience. The chapter sets out those decisions and the
reasoning behind them; the empirical evidence that justifies them is deferred to
the results in Chapter 5.

## 3.2 Design Requirements and Principles

Before any architecture can be proposed, the objectives of Chapter 1 must be
translated into concrete engineering requirements. These fall into functional
requirements (what the system must do) and non-functional requirements (the
qualities it must exhibit).

**Functional requirements.**

- **FR1 — Depth from stereo.** Given a rectified stereo image pair from one
  vehicle, the system shall produce a dense per-pixel estimate of scene depth.
- **FR2 — Object detection.** The system shall detect the surrounding vehicles in
  a single camera image as two-dimensional regions.
- **FR3 — Lifting to 3D.** The system shall combine detections with depth to place
  each detected vehicle at a three-dimensional position in the observing
  vehicle's coordinate frame.
- **FR4 — Cooperative fusion.** Given the independent 3D observations of two
  vehicles and their relative pose, the system shall register both sets of
  observations into a common frame and combine them into a single, consistent
  scene.
- **FR5 — Symmetric benefit measurement.** The system shall make it possible to
  quantify, for *each* of the two vehicles, the benefit obtained from
  cooperation — both the objects recovered that the vehicle could not see alone
  and any change in localization accuracy.

**Non-functional requirements.**

- **NFR1 — Modularity.** Each perception stage shall be independently runnable,
  testable, and replaceable, communicating with its neighbours only through a
  documented data interface.
- **NFR2 — Reproducibility.** Every run shall be deterministic given its
  configuration, and all parameters and outcomes shall be recorded so that any
  result can be regenerated and audited.
- **NFR3 — Stagewise verifiability.** No stage shall depend on the correctness of
  a later stage for its own validation; each shall be verifiable against ground
  truth in isolation.
- **NFR4 — Honesty of output.** The system shall report only those quantities that
  the sensing modality can reliably support, and shall explicitly decline to
  report those it cannot.

These requirements are not merely a checklist; they shape the architecture
directly. NFR1 and NFR3 motivate the staged decomposition and the per-stage
validation harness; NFR2 motivates the configuration, seeding, and
experiment-tracking infrastructure described in Section 3.9; and NFR4 is the
origin of the position-only output decision in Section 3.7.

## 3.3 The V-Model Development Process

The system was developed under a **V-model** lifecycle. The V-model is well suited
to this work for two reasons. First, the perception pipeline is naturally
decomposable into stages whose interfaces can be specified up front, which favours
a process that fixes requirements and design before implementation. Second, and
more importantly, the thesis question is ultimately a question of *trust*: a
cooperative gain is only meaningful if each link in the chain that produces it has
been independently verified. The V-model makes this verification structural rather
than incidental — every level of design is paired, from the outset, with the level
of testing that will confirm it.

The left arm of the V descends from abstract intent to concrete code: research
objectives are refined into the system requirements of Section 3.2, the
requirements into the architecture of Section 3.4, the architecture into the
detailed design of each individual stage, and the stage designs into
implementation. The right arm ascends back up through progressively broader levels
of testing, and — this is the defining feature of the model — each ascending
verification level is tied horizontally to the descending design level it
confirms.

```
   REQUIREMENTS / OBJECTIVES                          ACCEPTANCE VALIDATION
   (Ch. 1: can camera-only V2V      <------------>    Characterise the cooperative
    improve 3D localisation over                       localisation dynamics
    single-agent stereo, and what                      (recall / precision / loc-
    are its operating limits?)                         error vs range) & its limits/
         \                                                                        /
          SYSTEM ARCHITECTURE          <-------->   SYSTEM TESTING               /
          (Sec. 3.4: the 4-stage                    End-to-end run of the full
           pipeline + dual data                     pipeline on a vehicle pair
           strategy)                                (detector path, two agents)
            \                                                                   /
             STAGE / MODULE DESIGN     <------->   STAGE VALIDATION
             (Sec. 3.5-3.8: algorithm             Standalone, ground-truth
              choice & interface of                validation of each stage
              each stage)                          in isolation (NFR3)
              \                                                              /
               DETAILED DESIGN &       <----->    UNIT TESTING
               IMPLEMENTATION                     Per-stage automated tests
               (the stage modules &               of each component's logic
                shared utilities)
                          \                                       /
                           \-------  IMPLEMENTATION  -----------/
                                     (the codebase)
```
*Figure 3.1 — The V-model as instantiated in this project. Each descending design
level (left) is paired with the ascending verification level (right) that
confirms it: implementation is checked by unit tests, individual stage designs by
standalone stage validation, the overall architecture by an end-to-end system
run, and the original research objectives by the final acceptance measurement of
the cooperative gain. The horizontal arrows are the verification correspondences;
the descent and ascent are read down the left arm and up the right arm.*

The horizontal correspondences are realized concretely in this project:

- **Implementation ↔ Unit testing.** Each stage is accompanied by an automated
  test module exercising its internal logic — depth conversion, class mapping,
  geometric unprojection, box registration, matching, and confidence merging —
  in isolation from the rest of the pipeline and from real data. These tests
  encode the detailed design contracts and are the first gate any change must
  pass.
- **Stage design ↔ Stage validation.** Each stage has a dedicated, standalone
  validation procedure that scores its output against ground truth without
  invoking later stages (NFR3). This is the level at which a stage's *design* —
  not just its code — is confirmed to behave as intended on real inputs.
- **System architecture ↔ System testing.** The architecture is exercised as a
  whole by running the complete chain — depth, detection, lifting, and fusion —
  for a genuine pair of cooperating agents, confirming that the stage interfaces
  compose correctly end to end.
- **Objectives ↔ Acceptance validation.** The top-level question is answered by
  the final measurement of the cooperative gain in a controlled multi-agent
  setting, which is the engineering acceptance criterion for the thesis as a
  whole.

A practical consequence of this discipline is that defects are localized to the
narrowest possible level. A regression in a geometric routine surfaces at the unit
level; a degradation in depth quality surfaces at stage validation; an interface
mismatch surfaces at system testing. Because each level has an explicit owner in
the design, failures are diagnosed against the corresponding specification rather
than against the opaque end-to-end behaviour.

## 3.4 System Architecture Overview

The proposed system is a **four-stage pipeline**. Each stage consumes the output
of its predecessor through a documented interface and emits artifacts that are
both the input to the next stage and an independently inspectable result. The
stages and their data interfaces are summarized below.

| Stage | Name | Input → Output |
|-------|------|----------------|
| 1 | Depth | Rectified stereo pair → dense disparity / metric depth |
| 2 | Detect | Single (left) image → 2D vehicle regions |
| 3 | Lift | 2D regions + depth → 3D position per vehicle (+ source region) |
| 4 | Fusion | Two agents' 3D observations + relative pose → one fused, registered scene |

*Table 3.1 — The four pipeline stages and their interfaces.*

Stages 1 through 3 constitute the **single-vehicle perception chain**: they take
one vehicle's raw stereo imagery and turn it into a set of 3D vehicle positions
expressed in that vehicle's own camera frame. Stage 4 is the **cooperative
layer**: it takes the perception output of two such vehicles and the geometric
relationship between them, and produces a single shared scene. The first three
stages realize Objective 1 (the single-agent 3D localisation baseline); Stage 4
realizes Objective 2 (the decentralised object-level fusion layer); and the
stagewise validation harness, culminating in the cooperative evaluation, serves
Objective 3 (the empirical characterisation of cooperative localisation dynamics).

```
        VEHICLE A                                   VEHICLE B
   ┌───────────────────┐                       ┌───────────────────┐
   │ stereo pair (L,R)  │                       │ stereo pair (L,R)  │
   └─────────┬─────────┘                        └─────────┬─────────┘
             │                                            │
    ┌────────▼────────┐                          ┌────────▼────────┐
    │ S1  Depth       │                          │ S1  Depth       │
    │ disparity→depth │                          │ disparity→depth │
    └────────┬────────┘                          └────────┬────────┘
             │ depth map        left image                │
    ┌────────▼────────┐  ┌─────────────┐          ┌───────▼─────────┐
    │ S2  Detection   │◄─┤ left image  │          │ S2  Detection   │
    │ 2D vehicle boxes│  └─────────────┘          │ 2D vehicle boxes│
    └────────┬────────┘                           └────────┬────────┘
             │ boxes + depth                               │
    ┌────────▼────────┐                           ┌────────▼────────┐
    │ S3  Lift to 3D  │                           │ S3  Lift to 3D  │
    │ 3D positions    │                           │ 3D positions    │
    └────────┬────────┘                           └────────┬────────┘
             │  boxes_A (A's frame)        boxes_B (B's frame) │
             └───────────────┐         ┌───────────────────────┘
                             ▼         ▼
                     ┌───────────────────────┐     relative pose
                     │  S4  V2V Fusion        │◄──  (T: B's frame → A's)
                     │  register · match ·    │
                     │  merge → shared scene  │
                     └───────────────────────┘
                                 │
                                 ▼
                       fused scene in A's frame
                  (each vehicle's symmetric gain measurable)
```
*Figure 3.2 — End-to-end data flow. The single-vehicle perception chain (Stages
1–3) runs independently and identically on each agent; the cooperative layer
(Stage 4) registers the two agents' 3D observations into a common frame using
their relative pose and fuses them. The same architecture serves both agents,
underscoring that the cooperative benefit is symmetric.*

Two architectural properties follow from the requirements of Section 3.2 and merit
emphasis:

- **Symmetry of the perception chain.** The identical Stage 1–3 chain is applied
  to each vehicle. Neither agent is privileged; cooperation is a relationship
  between two equal perception systems, which is what makes a *symmetric* benefit
  measurement (FR5) meaningful.
- **A single point of coupling.** The two perception chains are entirely
  independent until Stage 4. All inter-agent coupling — and therefore all
  assumptions about communication and coordinate alignment — is confined to the
  fusion layer, keeping the single-vehicle design free of cooperation-specific
  concerns.

## 3.5 Data Sources and the Dual-Dataset Strategy

A recurring difficulty identified in Chapter 1 is that no single dataset offers
both *real* stereo imagery and *simultaneous multi-vehicle* recordings with
trustworthy ground truth. The architecture resolves this with a deliberate
**dual-dataset strategy**, assigning each data source the role it is uniquely
suited to.

- **Real-image stereo source (single-vehicle chain).** Stages 1–3 are designed and
  validated against real-world driving stereo data. This is the only way to
  confirm that the camera-only perception chain behaves on genuine imagery, with
  its real noise, lighting, and scene complexity. Because this source provides one
  vehicle's view at a time, it cannot exercise cooperation, but it is the
  authoritative ground for the *single-vehicle* claim.
- **Simultaneous multi-agent source (cooperative layer).** Stage 4 requires two
  vehicles observing the *same* scene at the *same* instant, together with exact
  relative pose and a complete inventory of which objects each vehicle can truly
  see. These conditions are obtainable from a controlled simulation environment
  [CITATION: CARLA simulator] that records two moving ego vehicles in a shared
  scene, with per-object,
  per-agent visibility derived from the rendered view rather than guessed
  geometrically. This source is what makes the cooperative gain *measurable* (FR5)
  with reliable ground truth.

This division is an engineering decision rather than a compromise: each stage is
exercised on the data that can actually test it. The single-vehicle chain earns
its credibility on real images; the cooperative claim earns its credibility on
data where simultaneous, occlusion-truthful, multi-agent ground truth exists. The
shared interface contract between the stages (Table 3.1) is what allows the same
Stage 1–3 design to operate on either source.

A point of consistency worth noting is that the cooperative source's per-agent
ground truth is built from *true visibility*: an object counts as seen by an agent
only when a sufficient number of its pixels are actually rendered to that agent's
camera. This is occlusion-truthful — objects hidden behind buildings or other
vehicles correctly drop out of an agent's view — which is precisely the
blind-spot condition cooperation is meant to address. A geometric
field-of-view test would have falsely credited an agent with seeing occluded
objects and thereby understated the cooperative benefit.

## 3.6 Stage 1 — Stereo Depth Estimation

The remaining sections of this chapter are stated mathematically. To keep them
self-contained, the notation used throughout is collected once in Table 3.2, and
the two coordinate frames the pipeline operates in are shown in Figure 3.3: the
**camera optical frame**, in which every stage produces its 3D output, and the
**vehicle (agent) frame**, in which the agent poses that drive cooperative
registration are expressed.

| Symbol | Meaning |
|--------|---------|
| $f,\; B,\; d$ | focal length (px), stereo baseline (m), disparity (px) |
| $Z$ | depth = camera-frame forward coordinate (m) |
| $(u, v)$ | pixel coordinates in the image |
| $f_x, f_y$ | camera focal lengths along the image axes (px); $f_x \equiv f$ |
| $(c_x, c_y)$ | principal point of the left camera (px) |
| $(X, Y, Z)$ | 3D point in the camera frame (m): $X$ right, $Y$ down, $Z$ forward |
| $\rho$ | depth coverage ratio of a detection region, $\rho \in [0, 1]$ |
| $c_{2D},\; c_{3D}$ | 2D-detection confidence and propagated 3D confidence |
| $c_A, c_B,\; \hat{c}$ | the two agents' confidences and the fused confidence |
| $\tau$ | per-class BEV matching distance threshold (m) |
| $d_{\text{BEV}}$ | bird's-eye-view (ground-plane) centre distance (m) |
| $T_{W,A},\; T_{W,B}$ | world-from-camera transforms of Vehicles A and B (4×4) |
| $T_{B \rightarrow A}$ | rigid transform mapping B's camera frame into A's (4×4) |
| $M$ | fixed axis map from the vehicle frame to the camera frame |
| $R$ | rotation (3×3) block of a homogeneous transform |
| $\theta,\; \hat{\theta}$ | object heading and fused heading (rad) |
| $\tilde{p}$ | a 3D point in homogeneous coordinates |

*Table 3.2 — Notation used throughout the methodology (Sections 3.6–3.9).*

```
        KITTI CAMERA FRAME                      VEHICLE (AGENT) FRAME
        (optical, right-handed)                 (left-handed, CARLA)

              Z  (forward,                              z (up)
             ↗    into scene)                           │
            ╱                                           │
           ╱                                            o────► y (right)
          o ─────► X (right)                           ╱
          │                                           ╱
          │                                          ↙
          ▼ Y (down)                                x (forward)

     Stage 1–3 outputs and all fusion             Agent poses are recorded here;
     geometry live in this frame.                 the camera mount sits on it.

     Axis map (Section 3.9):  (X, Y, Z)_camera  =  ( y,  −z,  x )_vehicle
```
*Figure 3.3 — The two coordinate frames. Every stage emits 3D positions in the
KITTI camera optical frame ($X$ right, $Y$ down, $Z$ forward). Vehicle poses are
expressed in the left-handed vehicle frame ($x$ forward, $y$ right, $z$ up); the
constant axis map $M$ converts between them and is the bridge used when one
agent's detections are registered into the other's frame (Section 3.9).*

**Purpose.** Stage 1 converts a rectified stereo image pair into a dense estimate
of per-pixel disparity, from which metric depth is recovered by the standard
stereo triangulation relationship [CITATION: stereo depth / pseudo-LiDAR]

$$ Z = \frac{f \cdot B}{d} $$

where $Z$ is the depth of a pixel, $f$ the focal length in pixels, $B$ the stereo
baseline in metres, and $d$ the disparity (the horizontal displacement of the
pixel between the left and right images). The baseline itself is recovered from
the two cameras' projection matrices, $B = (t_x^{L} - t_x^{R}) / f$, where
$t_x^{L}$ and $t_x^{R}$ are the horizontal translation terms of the left and right
projection matrices.

This relationship is central to the entire methodology and reappears as the
dominant source of error analysed in later chapters. Differentiating it with
respect to disparity gives the depth error induced by a disparity error:

$$ \left| \Delta Z \right| \;\approx\; \frac{Z^{2}}{f \cdot B}\, \left| \Delta d \right| $$

Because the depth error scales with the *square* of the distance, a fixed
disparity error of a fraction of a pixel translates into a small error nearby but
a large one far away. The system is therefore expected, by construction, to be
most accurate on nearby vehicles — a property that informs the design of every
subsequent stage and that is quantified empirically in Chapter 5.

**Design.** The stage is built around two interchangeable depth estimators that
represent two distinct engineering trade-offs, selectable without altering any
downstream stage:

- A **classical semi-global block-matching estimator** [CITATION: semi-global
  matching (SGBM)], which produces a *sparse* disparity map: it reports depth only
  where a confident left–right correspondence exists and leaves textureless or
  ambiguous regions unfilled. Its sparsity is, in effect, a built-in reliability
  filter — the pixels it does report tend to lie on well-defined surfaces.
- A **learned deep-stereo estimator** [CITATION: learned deep stereo matching],
  which produces a *dense* disparity map with full coverage, at greater
  computational cost. It is run directly as an in-process inference step.

Offering both is a deliberate architectural choice. It allows the methodology to
separate questions of *depth accuracy* from questions of *downstream robustness*:
a sparse, conservative map and a dense, complete map stress the later stages
differently, and carrying both through the pipeline lets the project reason about
that trade-off rather than presuppose a single answer.

**Interface.** The stage emits a per-pixel disparity map in which invalid pixels
are explicitly marked, so that downstream stages can distinguish "no measurement"
from "a measurement of zero." This explicit invalid-pixel contract is what allows
the sparse and dense estimators to share one interface.

## 3.7 Stage 2 — 2D Object Detection

**Purpose.** Stage 2 locates the surrounding vehicles in the (left) camera image
as two-dimensional regions. It answers *where in the image* the objects of
interest are, deferring the question of *where in space* to Stage 3.

**Design.** The stage uses a pretrained transformer-based object detector
[CITATION: RT-DETR / DETR-family 2D detection] applied to the left image of the
stereo pair. Detection operates on the single image
rather than on the depth map because appearance-based detection is mature and
robust, whereas the depth map is better used for *localizing* an already-detected
object than for *finding* it. The detector is pretrained on a general-purpose
object vocabulary; the stage maps the relevant vehicle categories of that
vocabulary onto the project's target class and discards the rest, applying a
confidence threshold to suppress weak detections.

**Class scope.** The pipeline is deliberately scoped to a single object class —
the car. This scoping is a methodological decision rather than a limitation of
convenience: the controlled cooperative ground truth consists of cars, and stereo
detection and lifting of smaller, more deformable classes such as pedestrians
proved unreliable in a way that would have confounded the cooperative measurement.
Restricting the class keeps the evaluation interpretable and aligned with the
honesty principle (NFR4). The architecture does not preclude additional classes;
re-enabling them is a configuration concern localized to the detection and
matching stages.

**Interface.** The stage emits, per image, a set of labelled 2D regions with
associated confidences, which become the input to the lifting stage alongside the
Stage 1 depth map.

## 3.8 Stage 3 — Lifting to 3D Position

**Purpose.** Stage 3 is the junction of the two preceding stages: it combines each
2D detection (Stage 2) with the depth map (Stage 1) and the camera calibration to
place the detected vehicle at a three-dimensional position in the observing
vehicle's camera frame. This is the stage at which the system crosses from image
space into the metric 3D world in which cooperation takes place.

**The position-only design decision.** The single most important methodological
decision in the single-vehicle chain is that Stage 3 outputs *3D position only* —
a point `(x, y, z)` per vehicle, carried together with the source 2D region — and
deliberately does **not** output a full 3D bounding box with size and heading.
This decision is a direct application of the honesty principle (NFR4) and follows
from the range-dependent stereo uncertainty at the heart of the problem statement
(Chapter 1).

The reasoning is geometric. Recovering an object's *position* requires
establishing how far away it is, which stereo supports. Recovering an object's
*size* and *heading*, by contrast, requires resolving its three-dimensional extent
and orientation from a single viewpoint at range — and the same disparity-to-depth
sensitivity that limits position accuracy at distance makes extent and orientation
far less observable still. An object's heading, in particular, cannot be reliably
recovered from stereo at range: the cues that might fix it are either degenerate
(they assume the object faces the camera) or dominated by depth noise. Rather than
emit size and heading values that the sensor cannot support — and thereby invite
false confidence downstream — the design restricts the output to the quantity
stereo can defensibly provide. The empirical basis for this decision is presented
in the results (Chapter 5); here it is stated as a design principle.

**Design.** For each 2D region, the stage samples a representative depth from the
disparity map within the region, converts it to a metric distance via the stereo
relationship of Section 3.6, and unprojects the region's centre to a 3D point
using the camera calibration. All 3D quantities are expressed in the KITTI camera
convention ($X$ right, $Y$ down, $Z$ forward into the scene; Figure 3.3). Given a
region
centre at pixel $(u, v)$ and its sampled depth $Z$, the inverse of the pinhole
projection yields the 3D point

$$ X = \frac{(u - c_x)\, Z}{f_x}, \qquad Y = \frac{(v - c_y)\, Z}{f_y}, \qquad Z = Z $$

where $(f_x, f_y)$ are the focal lengths and $(c_x, c_y)$ the principal point of
the left camera, both read from its calibration. The point $(X, Y, Z)$ is the
vehicle's position in the observing vehicle's camera frame. Two considerations
shape this:

- **Robust depth sampling within a region.** A detection's region contains pixels
  belonging to the vehicle but also, typically, road surface and background. The
  stage therefore does not take a naive average; it samples depth in a way
  designed to favour the object surface over contaminating background, with the
  sampling strategy tuned per depth estimator because the sparse and dense maps of
  Stage 1 have very different pixel distributions inside a region. The specific
  tuning is treated as an experimental matter and reported in Chapter 5; the
  *architecture* simply exposes it as a per-method parameter so the same lifting
  logic serves both estimators.
- **Confidence propagation.** The 3D position inherits a confidence that scales
  the 2D detection confidence by the proportion of the region for which valid
  depth was available, $c_{3D} = c_{2D} \cdot \rho$, where the coverage ratio
  $\rho \in [0, 1]$ is the fraction of region pixels carrying a valid depth
  measurement. A position lifted from a well-covered region is thus trusted more
  than one lifted from a sparsely measured region, and this confidence later
  participates directly in the fusion merge (Section 3.9).

**Interface.** The stage emits, per detected vehicle, a labelled 3D position with
its propagated confidence and the source 2D region. This compact, position-only
representation is intentionally schema-compatible with the fusion core, which is
designed to operate whether or not size and heading are present.

## 3.9 Stage 4 — Cooperative V2V Fusion

Stage 4 is the cooperative layer and the realization of the thesis goal. It takes
the independent 3D observations of two vehicles and combines them into a single
shared scene in which objects visible to either vehicle become known to both.
Combining detections after each vehicle has perceived independently is the *late
fusion* strategy for cooperative perception [CITATION: cooperative perception /
late fusion, e.g. OPV2V], chosen here because it keeps the inter-vehicle interface
compact (3D positions rather than raw imagery) and the single-vehicle chain
unaware of cooperation. Its
design separates a *source-agnostic fusion core* from the *data plumbing* that
feeds it, so that the same fusion logic operates on either real or simulated
inputs and on either position-only or fully-specified boxes.

The fusion proceeds in three conceptual steps — **register, match, merge** —
preceded by the construction of the common frame.

**Common frame and registration.** Each vehicle perceives in its own camera frame;
the two frames are unrelated until the vehicles' relative pose is known.
Cooperation therefore begins by choosing one vehicle's frame (Vehicle A's) as the
reference and transforming the other vehicle's observations into it by a single
rigid 4×4 homogeneous transform derived from the two vehicles' poses,

$$ T_{B \rightarrow A} = M \; T_{W,A}^{-1} \; T_{W,B} \; M^{-1} $$

where $T_{W,A}$ and $T_{W,B}$ are the world-from-camera transforms of the two
vehicles (each built from the vehicle's world pose composed with its fixed camera
mount), and $M$ is the constant axis map from the vehicle frame to the KITTI
camera convention, $(x, y, z)_{\text{cam}} = (y, -z, x)_{\text{vehicle}}$
(Figure 3.3). Any 3D
point observed by Vehicle B is then carried into A's frame by
$\tilde{p}_A = T_{B \rightarrow A}\, \tilde{p}_B$ (in homogeneous coordinates),
and a box's heading is offset by the transform's rotation about the camera
vertical axis, $\theta_{\text{offset}} = \operatorname{atan2}(R_{13}, R_{33})$,
extracted from the rotation block $R$ of $T_{B \rightarrow A}$. Because the
cooperative source
records simultaneous observations, this is a purely spatial alignment — there is
no temporal extrapolation, and a static object seen by both vehicles should, after
registration, coincide. The design exploits exactly this property as a built-in
correctness check: an object seen by both agents must register to near-zero
displacement, and a large displacement is treated as a symptom of a coordinate or
pose error rather than of a moving object. This sanity check is part of the
methodology, not an afterthought — it is the mechanism by which the most dangerous
class of bug in cooperative perception, a silent frame misalignment, is caught.

**Matching.** Registered observations from the two vehicles are associated greedily
by bird's-eye-view (ground-plane) centre distance, within object class, accepting a
pair as the same physical object only when their separation falls below a
per-class threshold $\tau$. The bird's-eye-view distance discards the vertical
axis and measures separation in the ground plane,

$$ d_{\text{BEV}}(a, b) = \sqrt{(x_a - x_b)^2 + (z_a - z_b)^2}, \qquad \text{match if } d_{\text{BEV}} \le \tau $$

candidate pairs being considered in ascending order of distance so that the
closest compatible pair is committed first. Matching on ground-plane centre
distance — rather than on 3D
box overlap — is the natural choice given the position-only output of Stage 3, and
it is exactly what makes the fusion core able to consume both position-only and
fully-specified inputs through one code path.

**Merge.** A corroborated pair — one object that both vehicles saw — is merged into
a single estimate. Writing the two observations' confidences as $c_A$ and $c_B$,
each merged position coordinate is the confidence-weighted average of the two
observations, and the fused confidence is combined by a noisy-OR rule:

$$ \hat{x} = \frac{c_A\, x_A + c_B\, x_B}{c_A + c_B} \quad (\text{and likewise for } y, z), \qquad \hat{c} = 1 - (1 - c_A)(1 - c_B) $$

The noisy-OR rule reflects that two independent detections of the same object are
more trustworthy than either alone (the fused confidence exceeds both inputs).
Where the inputs happen to carry size and heading, those are merged too —
dimensions by the same confidence-weighted average, and heading by a
confidence-weighted *circular* mean,
$\hat{\theta} = \operatorname{atan2}\!\big(\sum_i c_i \sin\theta_i,\; \sum_i c_i \cos\theta_i\big)$,
which averages angles correctly across the $\pm\pi$ wrap-around. The merge is
nonetheless designed to degrade gracefully to position-only when size and heading
are absent. An object
seen by only one vehicle is carried into the shared scene unmerged, tagged with
its source — this is precisely the blind-spot recovery that motivates the whole
system. A matched pair whose post-registration displacement is implausibly large
is *not* merged but flagged, on the principle that for simultaneous observations a
large displacement indicates a faulty match or pose error rather than genuine
motion.

```
   A's observations          B's observations (registered into A's frame)
        a1  a2  a3                  b1     b2        b3
         │   │   │                   │      │         │
         └───┼───┴──────match by BEV centre distance──┘
             │           (within class, thresholded)
     ┌───────┼───────────────┬────────────────────┐
     ▼       ▼               ▼                     ▼
  a1 ⊕ b? matched pair   a2 only-A (B's blind   b3 only-B (A's blind
  (corroborated:        spot → recovered for B) spot → recovered for A)
   merge position,
   noisy-OR confidence)
     └───────────────┬────────────────────────────┘
                     ▼
            SHARED FUSED SCENE (A's frame)
   contains: corroborated objects + objects either agent saw alone
```
*Figure 3.4 — The register–match–merge logic of the fusion core. Objects seen by
both agents are corroborated and merged; objects seen by only one agent are
carried into the shared scene, which is the blind-spot recovery that benefits the
other agent. The output is a single scene in the reference frame, from which each
agent's gain can be scored.*

**Symmetric benefit by design.** The fused output is a single *shared* scene, and
this is what allows the cooperative benefit to be measured symmetrically (FR5).
Evaluation is framed against a *cooperative ground truth* — the union of every
object visible to either vehicle, de-duplicated by object identity — and the
shared scene is scored from both vehicles' standpoints: how many objects each
vehicle recovered that it could not have seen alone, and whether its localization
of objects it could see improved. The architecture treats the two vehicles
identically in this accounting, so neither agent's gain is privileged. Because the
two ego vehicles are themselves part of the simulated scene yet are not perception
*targets* (their poses are shared directly over the cooperative link), the
evaluation design treats an agent's detection of the *other* ego as neither a hit
nor a false alarm — an ignore region — so that the cooperation metric reflects
perceived traffic rather than the agents perceiving each other.

## 3.10 Cross-Cutting Engineering Concerns

Beyond the per-stage algorithms, several engineering concerns span the whole
pipeline and are what make the non-functional requirements of Section 3.2
achievable. They are presented here as architecture, not as code.

- **Configuration-driven design (NFR1).** Every stage is parameterized by its own
  configuration file, with shared settings centralized so they are defined once
  and read everywhere. No operational value — paths, thresholds, model choices,
  tuned parameters — is embedded in the logic. This is what lets a stage be
  re-run, re-tuned, or pointed at a different data source without editing code,
  and it is the mechanism by which empirically-tuned parameters are kept as a
  single source of truth rather than copied between the pipeline and its
  validators.
- **Experiment tracking and reproducibility (NFR2).** Every run records its
  parameters and resulting metrics to a persistent experiment-tracking store, so
  that no outcome exists only as transient console output. Combined with fixed
  random seeds established at every entry point, this makes runs deterministic and
  auditable: any reported result can be regenerated from its recorded
  configuration.
- **Standalone validation per stage (NFR3).** Each stage is paired with a
  dedicated validation procedure that scores it against ground truth without
  invoking later stages. This is the realization of the V-model's
  stage-validation level (Section 3.3): it ensures that when a later stage
  misbehaves, the question "is the input to this stage already wrong?" can always
  be answered independently.
- **Automated unit testing (NFR3).** The detailed design of each stage is pinned by
  an automated test module — one per stage — exercising its components in
  isolation. These tests are the bottom of the V and the first line of defence
  against regression; notably, the fusion core is tested against *both* the
  position-only and the fully-specified box schemas, so that its source-agnostic
  contract is verified rather than assumed.

Together these concerns implement the right arm of the V-model as living
infrastructure: unit tests, stage validators, and the end-to-end run are not
one-off activities but standing artifacts that any change to the system must
satisfy.

## 3.11 Validation Strategy Summary

The validation strategy mirrors the V-model exactly and is summarized in Table
3.3. Each design artifact has a corresponding verification activity, and the
artifacts are arranged so that confidence accumulates from the bottom up: a result
at any level is only trusted once the level beneath it has been confirmed.

| Design level (left arm) | Verification level (right arm) | What it confirms |
|-------------------------|--------------------------------|------------------|
| Detailed design / implementation | Per-stage unit tests | Each component's logic is correct in isolation |
| Stage design (Sec. 3.6–3.9) | Standalone stage validation vs. ground truth | Each stage performs its function on real inputs |
| System architecture (Sec. 3.4) | End-to-end pipeline run on an agent pair | The stage interfaces compose correctly |
| Objectives (Ch. 1) | Characterisation of cooperative localisation dynamics | The thesis question — measurable V2V benefit (coverage / localisation) and its operating limits |

*Table 3.3 — The verification strategy as a direct image of the V-model. Each row
is one horizontal correspondence in Figure 3.1.*

The crucial structural property is **independence between levels**: because every
stage can be validated against ground truth without trusting its successors
(NFR3), the cooperative result at the top of the V rests on a chain of separately
confirmed links rather than on a single end-to-end number. This is what allows
Chapter 5 to present the cooperative characterisation not as an isolated figure
but as the top of a verified stack — depth confirmed against ground-truth disparity,
detection against labelled regions, lifting against 3D positions, and fusion
against cooperative ground truth — each established on the data source best able
to test it.

## 3.12 Summary

This chapter has presented the engineering architecture of the proposed system and
the V-model process under which it was developed. The system is a four-stage
pipeline — depth, detection, lifting, and fusion — in which an identical
single-vehicle perception chain runs on each of two cooperating vehicles and a
source-agnostic fusion core combines their 3D observations into one shared scene.
A dual-dataset strategy assigns real-image stereo to the single-vehicle chain and
a simultaneous multi-agent simulation to the cooperative layer, so that each part
of the system is exercised on the data able to test it.

Two design commitments give the framework its character. The first is *honesty
about sensing limits*: because stereo cannot reliably recover object size or
heading at range, the system outputs 3D position only, and structures both its
representation and its evaluation around what cameras can actually provide. The
second is *verification by construction*: the V-model pairs every level of design
with the level of testing that confirms it, so that the headline cooperative
result rests on independently validated stages rather than on end-to-end
behaviour alone. With the architecture and process established, Chapter 4 details
the experimental setup and simulation environment, and Chapter 5 reports the
stagewise validation and the characterisation of the cooperative localisation
dynamics that this framework was built to measure.

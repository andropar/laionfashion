# Fashion Embedding Explorer: Living Project Spec

## One-line vision

Build a visually polished, foundation-model-powered atlas of fashion/style space from large-scale natural images.

The goal is to show that I can take a huge, messy visual dataset, extract meaningful structure with modern ML, and turn it into an interpretable, beautiful, communicative data product.

## Core portfolio message

> I can work with messy large-scale visual data, use foundation models pragmatically, build useful embeddings/search/visualizations, and communicate the result in a way that feels polished and understandable.

This is primarily a **showcase project**, not a startup or research paper.

It should demonstrate:

- large-scale data curation
- foundation-model exploitation
- embedding analysis
- vector search
- visual communication
- interactive data-product design
- pragmatic ML evaluation
- taste, judgment, and technical maturity

---

# 1. Project framing

## Working title options

- Fashion Embedding Explorer
- Mapping Style Space
- Style Space Atlas
- Fashion-in-the-Wild Explorer
- Context-Aware Fashion Embeddings
- Foundation Model Style Explorer

## Preferred framing

> A context-aware fashion embedding explorer that uses foundation models to map, search, and visualize clothing styles in natural images.

## Stronger case-study title

> Mapping Style Space: a foundation-model explorer for fashion embeddings in natural images

## Short public description

> I built an embedding explorer for fashion in the wild. Starting from LAION-derived natural images, the project filters for people wearing visible clothing, extracts visual/style embeddings using foundation models, derives interpretable style axes, and presents the result as an interactive atlas of fashion/style space.

## What makes this interesting

Most fashion search demos focus on item similarity:

> “Show me similar sweaters.”

This project should focus more on contextual style structure:

> “Where does this outfit live in style space?”  
> “What visual/style axes do foundation models already encode?”  
> “Can we expose those axes in an intuitive interface?”  
> “Can we distinguish photo aesthetics, outfit coherence, and garment similarity?”

The central idea is **not** to train a better fashion model from scratch.

The central idea is:

> Use foundation models as latent structure extractors, then make that structure explorable.

---

# 2. Goals and non-goals

## MVP goal

Create a working explorer that can:

1. Filter a subset of LAION-Natural / LAION-2B for images containing visible clothing/outfits.
2. Compute or load embeddings from existing foundation models.
3. Derive prompt-based style axes.
4. Build a vector index for image/outfit retrieval.
5. Visualize outfits/items in a 2D embedding map.
6. Let users click an outfit and see similar/style-neighbor examples.
7. Show interpretable style scores such as minimal, formal, streetwear, colorful, coherent.
8. Present the result in a polished interface suitable for screenshots and a demo video.

## Beyond-MVP goal

Produce a complete public-facing showcase package:

- interactive explorer
- case study page
- technical blog/report
- GitHub repo
- visual style-space atlas
- demo video
- small evaluation benchmark
- limitations/failure-case section
- dataset/model card where possible

## Non-goals

Do **not** initially aim to:

- train a large new fashion model
- create a production shopping recommender
- solve subjective taste
- claim universal aesthetic judgment
- build a complex e-commerce product
- redistribute copyrighted images carelessly
- make strong claims about “good style”
- spend months on segmentation before the embedding explorer works

---

# 3. Target audience

## Primary audience

Hiring managers / technical interviewers for:

- ML Engineer
- Applied Scientist
- Research Engineer
- Data Scientist
- Computer Vision Engineer
- AI Product / Data Product roles

## Secondary audience

- collaborators
- ML/data researchers
- visual AI people
- people interested in foundation-model interpretability
- personal website visitors

## What viewers should understand in 30 seconds

The project should immediately communicate:

- this is a serious data/ML project
- it works with large-scale image data
- it uses modern foundation models
- it is visually polished
- it exposes complex embeddings clearly
- it was built with strong technical and aesthetic judgment

---

# 4. Conceptual model

Do not collapse everything into one score.

## 4.1 Image aesthetic quality

Question:

> Is the photo/image visually pleasing?

Possible confounds:

- lighting
- composition
- resolution
- camera quality
- background
- professional photography
- pose

Important distinction:

> A beautiful photo can contain a bad outfit.  
> A poor mirror selfie can contain a good outfit.

## 4.2 Garment/item similarity

Question:

> Are these two items visually similar?

Examples:

- two beige knit sweaters
- two black leather boots
- two straight-leg blue jeans
- two oversized coats

This is relatively easy with CLIP/FashionCLIP-like embeddings.

## 4.3 Outfit compatibility/coherence

Question:

> Do the items work well together in context?

This is the most interesting target, but also the hardest.

It may involve:

- color harmony
- silhouette balance
- formality consistency
- occasion fit
- style identity
- cultural/contextual taste
- seasonal coherence

## 4.4 Central thesis

Foundation models likely already encode many weak style and aesthetic directions.

The project asks:

> Can we extract, inspect, visualize, and lightly calibrate those directions without training a large model?

---

# 5. Data sources

## Primary source

LAION-Natural / LAION-2B assets available on the server.

Expected resources:

- image files or image URLs
- metadata
- CLIP embeddings, if already computed
- naturalness scores, if available
- dataset-level filters
- server-side storage and compute

## Useful derived subset

Create a subset of images likely containing:

- one visible person
- clothing clearly visible
- upper body or full body
- reasonable image quality
- non-crowded scene
- non-explicit content
- ideally adult subjects
- enough visible garments to infer style

## Initial scale targets

Start small enough to iterate.

| Stage | Scale | Purpose |
|---|---:|---|
| Smoke test | 1k–5k images | Debug paths, filtering, thumbnails |
| First MVP | 5k–50k images | UI and retrieval prototype |
| Strong demo | 50k–200k images | meaningful embedding map |
| Large-scale version | 1M+ images | only after pipeline/UI are stable |

---

# 6. Filtering strategy

## 6.1 Stage 1: prompt-based candidate retrieval

Use CLIP/SigLIP/FashionCLIP text-image similarity with positive and negative prompts.

### Positive prompts

```text
a full body photo of a person wearing clothes
a person showing their outfit
street fashion photo
casual outfit
fashion outfit
person wearing a stylish outfit
a photo of clothing worn by a person
a portrait showing visible clothes
a person wearing a sweater and pants
a person wearing a jacket and trousers
a person wearing an outfit outdoors
a person wearing everyday clothing
Negative prompts
a close-up face portrait
a crowd of people
a person in swimwear
a person in underwear
a child
a blurry low quality image
a product photo on white background
a mannequin
a cartoon illustration
a drawing
text or poster
uniform
sports team uniform
6.2 Stage 2: person detection / quality filtering

Estimate:

number of persons
visible body extent
bounding box size
face/person confidence
image quality
NSFW probability
duplicate likelihood

Prefer:

one main person
visible torso and legs if possible
high enough resolution
clothing visible
non-extreme crop
non-crowded setting
6.3 Stage 3: optional garment segmentation/parsing

Potential tools:

SAM/SAM2/SAM3 if available
human parsing model
fashion parsing model
grounding model + segmentation refinement

Important note:

SAM-style segmentation alone gives regions, not garment labels.

For useful garment-level data, combine:

person detection
garment/human parsing
mask refinement
category assignment
6.4 Initial garment classes

Keep categories coarse:

top
bottom
outerwear
dress
shoes
bag
hat
accessory

Avoid fine-grained taxonomy at first.

7. Representation strategy
7.1 Foundation models to try
General image embeddings
CLIP ViT-L/14 or ViT-H/14
OpenCLIP variants
SigLIP
DINOv2
Fashion-specific embeddings
FashionCLIP
any available fashion retrieval model
Aesthetic predictors
LAION aesthetic predictor on CLIP embeddings
other open aesthetic scorers if easy
Vision-language model weak labels

Use a VLM to generate:

style tags
outfit coherence rating
occasion tags
color harmony notes
possible failure descriptions

Treat these as weak labels, not ground truth.

8. Prompt-axis strategy
8.1 Basic idea

Create interpretable directions in embedding space from contrastive text prompts.

axis = embedding("positive prompt") - embedding("negative prompt")
score(image) = cosine(image_embedding, axis)

This gives cheap, interpretable style scores.

8.2 Candidate axes
Aesthetic / quality
a beautiful stylish outfit
vs
an ugly badly styled outfit
a well-composed fashion photograph
vs
a blurry low quality outfit photo
Outfit coherence
a coherent well-matched outfit
vs
an awkward mismatched outfit
clothes that work well together
vs
clothes that clash
Formality
a formal elegant outfit
vs
a casual relaxed outfit
business formal clothing
vs
everyday casual clothing
Minimalism
a minimalist clean outfit
vs
a maximalist busy outfit
simple monochrome style
vs
colorful patterned style
Streetwear
streetwear outfit
vs
classic formal outfit
Sportiness
sporty athletic outfit
vs
elegant dressy outfit
Colorfulness
a colorful vibrant outfit
vs
a neutral muted outfit
Vintage
a vintage retro outfit
vs
a modern contemporary outfit
Outdoors / urban
outdoor functional clothing
vs
urban city fashion
Expensive / polished
an expensive refined outfit
vs
a cheap poorly styled outfit

Use this axis carefully. It is socioculturally loaded and likely confounded with photography quality, class markers, and brand aesthetics.

8.3 Axis validation

For each axis:

inspect top examples
inspect bottom examples
inspect random middle examples
compare prompt variants
compare CLIP vs FashionCLIP vs SigLIP
look for confounds
optionally collect pairwise human labels
9. Minimal human labeling
Purpose

Do not label a huge dataset.

Use small labels to calibrate and evaluate foundation-model-derived axes.

9.1 Pairwise outfit coherence

Question:

Which outfit looks more coherent?

Target data:

300 to 1,000 pairs

Possible model:

logistic regression on embedding differences
Bradley-Terry ranking model
tiny MLP only if needed
9.2 Attribute comparison

Questions:

Which outfit is more formal?
Which outfit is more minimalist?
Which outfit is more colorful?
Which outfit looks more streetwear?
Which outfit looks more coherent?

Target data:

100 to 300 pairs per axis
9.3 Outfit completion

Question:

Given this top, which bottom fits better?

Target data:

200 to 500 choices
9.4 Output

Create a small benchmark containing:

examples
labels
metadata
model predictions
accuracy
agreement
failure cases

This adds methodological credibility and makes the project more than a pretty dashboard.

10. Modeling hierarchy

Start with the least custom approach. Stop once the explorer is good enough.

Level 0: existing embeddings only
CLIP / FashionCLIP image embeddings
nearest-neighbor retrieval
UMAP map
prompt-axis scores
Level 1: existing aesthetic predictor
add LAION aesthetic score
use as a feature
do not equate with outfit quality
Level 2: VLM weak labels
style tags
outfit coherence ratings
explanations
use as metadata and for filtering
Level 3: tiny human-calibrated probe
logistic regression or linear probe
trained on 300–1,000 labels
predicts selected subjective axes
Level 4: small compatibility head
item embedding + outfit context embedding
outfit completion objective
corrupted-outfit negatives
optional only if MVP is already strong
Avoid initially
training large models from scratch
graph neural networks
full recommender-system stack
personalization
complicated segmentation-first pipeline
11. MVP explorer UI
Overall design goal

A polished, communicative interface that feels like an embedding microscope for fashion/style.

11.1 Query item / outfit panel

Capabilities:

select image from dataset
optional upload
show selected person/outfit
show extracted garment crops if available
show attributes:
garment category
dominant colors
style tags
aesthetic score
prompt-axis scores
11.2 Similar outfits/items

Show nearest neighbors by:

CLIP
FashionCLIP
DINO/SigLIP
selected axis-conditioned similarity

Distinguish:

visually similar outfits
same-category item similarity
style-neighborhood similarity
11.3 Compatible / related items

Optional for MVP, but nice if garment extraction works.

Given a top, show:

compatible bottoms
compatible shoes
compatible outerwear
compatible bags

Initial version can use co-occurrence / neighborhood heuristics rather than trained compatibility.

11.4 Style / embedding map

2D UMAP or similar projection.

Features:

scatter or tile map
cluster labels
selected point highlighted
zoom/pan
filters by prompt-axis sliders
click point to update query

Candidate cluster labels:

Minimal / Clean
Streetwear
Sporty
Smart Casual
Vintage
Boho / Natural
Formal
Outdoor
Workwear
Monochrome
Colorful / Patterned

Labels can be generated by inspecting cluster exemplars or using VLM summaries.

11.5 Axis explorer

Sliders or toggles:

minimal ↔ maximal
formal ↔ casual
neutral ↔ colorful
sporty ↔ elegant
streetwear ↔ classic
coherent ↔ clashing
modern ↔ vintage

For each axis, show:

top positive examples
top negative examples
local neighborhoods
failure cases
11.6 Model comparison

Optional but valuable.

Compare:

CLIP
FashionCLIP
DINO/SigLIP
aesthetic predictor
human-calibrated probe

Core question:

Do different models organize style space differently?

11.7 Failure cases

Important for credibility.

Show examples where:

photo quality dominates fashion quality
background affects scores
body/identity confounds appear
cultural style is misread
segmentation fails
model assigns shallow stereotypes
prompt axis behaves unexpectedly

This makes the project look thoughtful rather than naive.

12. Suggested visual aesthetic
UI style
image-heavy
polished dashboard
dark or clean light theme
high contrast
minimal chrome
no noisy decoration
clear typography
smooth interactions
beautiful tiles/cards
Homepage/case-study style

For jroth.space, likely use:

clean white case-study page
large hero screenshot
right-side fact box
technical sections
architecture diagram
outcome / what-it-demonstrates section
clear links to app, repo, report, video
13. Suggested architecture
Backend

Use the simplest stack that gets to a polished result.

Possible components:

FastAPI
Python CLI scripts
DuckDB / SQLite / Postgres
FAISS / Annoy for vector search
Parquet for metadata
local files / object storage for thumbnails
Frontend

Options:

React / Next.js for polished UI
Streamlit for fastest internal MVP
Gradio/HF Space if public deployment is desired quickly

Recommended:

internal MVP: Streamlit or simple React
public showcase: React/Next + FastAPI or static data bundle
Data processing
Python
PyTorch
OpenCLIP / transformers
PIL/OpenCV
pandas/polars
UMAP / sklearn
FAISS/Annoy
optional segmentation models
Data artifacts
data/
  raw/
  interim/
    candidate_images.parquet
    person_filtered.parquet
    embeddings.parquet
    axis_scores.parquet
    thumbnails/
  processed/
    explorer_index.faiss
    explorer_metadata.parquet
    umap_2d.parquet
    cluster_labels.parquet
Repo structure
fashion-embedding-explorer/
  README.md
  docs/
    project_spec.md
    data_card.md
    model_card.md
    case_study.md
  configs/
    filtering.yaml
    models.yaml
    axes.yaml
  scripts/
    filter_candidates.py
    compute_embeddings.py
    compute_axis_scores.py
    build_index.py
    build_umap.py
    generate_thumbnails.py
  app/
    backend/
    frontend/
  notebooks/
    01_filtering_exploration.ipynb
    02_axis_validation.ipynb
    03_embedding_map.ipynb
  assets/
    screenshots/
    diagrams/
14. Data schema
Image-level metadata
image_id
source_dataset
source_url_or_reference
local_path
thumbnail_path
width
height
naturalness_score
nsfw_score
aesthetic_score
person_count
main_person_bbox
body_visibility_score
quality_score
embedding_clip_path_or_id
embedding_fashionclip_path_or_id
umap_x
umap_y
cluster_id
style_tags
axis_minimal
axis_formal
axis_streetwear
axis_colorful
axis_coherent
axis_vintage
axis_sporty
Person/outfit-level metadata
outfit_id
image_id
person_bbox
person_crop_path
visible_body_score
garment_ids
outfit_embedding_id
style_tags
occasion_tags
aesthetic_score
coherence_score
axis_scores
Garment-level metadata
garment_id
outfit_id
image_id
category
mask_path
crop_path
bbox
dominant_colors
embedding_id
axis_scores
quality_score
15. Evaluation ideas
15.1 Retrieval sanity checks

For selected queries:

are nearest neighbors visually similar?
do they preserve style?
do they preserve garment category?
does FashionCLIP beat generic CLIP for item retrieval?
do prompt-axis-conditioned neighbors look meaningfully different?
15.2 Axis validation

For each axis:

inspect top 50 and bottom 50 examples
compute stability across models
compare prompt variants
run small human-label correlation
document common failure modes
15.3 Compatibility evaluation

If garment extraction exists:

outfit completion task
corrupted outfit detection
compare random vs co-occurrence vs embedding-based retrieval
compare VLM-based weak labels vs prompt-axis scores
15.4 Human evaluation

Minimal benchmark:

500 pairwise comparisons
5 axes
100 pairs per axis
report agreement and model accuracy

Possible axes:

more formal
more minimal
more colorful
more coherent
more streetwear
15.5 Failure taxonomy

Track:

segmentation failures
prompt-axis confounds
cultural bias
aesthetic vs outfit quality confusion
background leakage
body/identity leakage
duplicate clusters
low-quality images
over-reliance on professional photography
over-reliance on Western fashion norms
16. Ethics, safety, and legal notes

This project uses web-scale image data and foundation models. Be careful about:

copyright / redistribution
personal images
minors
NSFW content
body and gender stereotypes
cultural bias in fashion judgments
class bias in “expensive-looking” / “good style” axes
using images of real people in a public demo
Recommended public-demo policy
Avoid exposing sensitive or questionable images.
Prefer thumbnails only if legally and ethically acceptable.
Consider using image IDs/metadata only for public dataset artifacts.
Avoid faces where possible, or blur/crop to clothing if feasible.
Avoid children.
Avoid NSFW.
Label model judgments as exploratory, not objective.
Include a limitations section.
Suggested disclaimer

This project explores style-related structure in foundation-model embeddings. Scores and labels are model-derived and should not be interpreted as objective judgments of taste, attractiveness, social status, or personal value. The system is intended as a visualization and data-product prototype, not as a normative fashion-ranking tool.

17. Showcase artifact roadmap
Artifact 1: Interactive explorer

Purpose:

Main proof of capability.

Completion criteria:

polished UI
query image/outfit
nearest-neighbor retrieval
embedding map
style-axis sliders
model/pipeline overview
limitations/failure cases
public demo or clean screen-recordable local demo
Artifact 2: Case study page

Purpose:

Explain the work to hiring managers.

Suggested sections:

Hero
Problem
Data
Pipeline
Explorer
Technical challenges
Evaluation / findings
Limitations
What this demonstrates
Links
Artifact 3: Technical blog/report

Purpose:

Demonstrate thinking and rigor.

Suggested title:

What do vision-language models know about style?

Sections:

motivation
dataset filtering
embeddings
prompt axes
aesthetic vs coherence
evaluation
failure cases
limitations
Artifact 4: Demo video

Purpose:

Fast communication.

Length:

60 to 90 seconds

Storyboard:

Start with huge LAION image pool.
Filter for people/outfits.
Compute embeddings and scores.
Show style map.
Click an outfit.
Show nearest neighbors and axes.
Show failure cases.
End with project message.
Artifact 5: GitHub repo

Purpose:

Technical credibility.

Must include:

clean README
screenshots
local setup
sample data or synthetic demo
architecture diagram
scripts
caveats
license note
Artifact 6: Visual style-space atlas

Purpose:

Visual wow factor for homepage.

Could be:

large poster figure
clusters and representative tiles
prompt-axis arrows
annotation of failure cases
model comparison panels
Artifact 7: Small benchmark

Purpose:

Show validation mindset.

Could include:

pairwise labels
model comparison
axis accuracy
human agreement
failure taxonomy
Artifact 8: Dataset/model card

Purpose:

Data responsibility and professionalism.

Include:

source
filtering
intended use
limitations
known biases
redistribution policy
metadata schema
18. Project phases
Phase 0: Server inventory and planning

Goal:

Understand available data, embeddings, paths, compute, and constraints.

Tasks:

locate LAION-Natural images / metadata
locate existing embeddings
locate naturalness scores
check available GPUs
check storage limits
check thumbnail generation feasibility
inspect current repo/setup
decide initial subset size

Deliverable:

docs/server_inventory.md
Phase 1: Candidate filtering

Goal:

Create first outfit/person candidate set.

Tasks:

implement prompt-based filtering
select 5k to 50k candidates
generate thumbnails
manually inspect random samples
iterate filters

Deliverables:

candidate_outfits.parquet
thumbnail grid report
filtering notes

Completion criteria:

at least 70% of inspected images contain visible relevant clothing
enough variety for visual map
obvious failure categories documented
Phase 2: Embeddings and scores

Goal:

Compute foundation-model features.

Tasks:

compute/load CLIP embeddings
compute/load FashionCLIP embeddings if feasible
compute aesthetic score
compute prompt-axis scores
store all scores in metadata table

Deliverables:

embeddings
axis_scores.parquet
axis inspection grids

Completion criteria:

nearest-neighbor search works
top/bottom axis examples are interpretable
at least 5 useful axes are retained after inspection
Phase 3: Embedding map and clusters

Goal:

Create communicative style-space map.

Tasks:

sample embeddings
compute UMAP/t-SNE/PCA variants
cluster points
generate cluster exemplars
optionally use VLM to label clusters
create static atlas figure

Deliverables:

umap_2d.parquet
cluster labels
atlas draft

Completion criteria:

clusters are visually interpretable
map supports UI interaction
atlas figure is good enough to show to another person
Phase 4: Explorer MVP

Goal:

Build working interactive demo.

Tasks:

build backend or static API
build vector index
build image tile viewer
add query selection
add nearest neighbors
add style map
add axis filters
add dataset/model overview

Deliverables:

local web app
screenshots
rough demo video

Completion criteria:

someone can use it without explanation
interface looks portfolio-worthy enough for iteration
explorer has at least one “wow” moment
Phase 5: Optional garment segmentation

Goal:

Add item-level structure.

Tasks:

run person detection
run garment/human parsing on subset
generate crops/masks
compute garment embeddings
add item-level retrieval
add outfit context panel

Deliverables:

garment metadata
crop/mask thumbnails
item retrieval UI

Completion criteria:

segmentation quality improves the demo
if segmentation is noisy, keep it as experimental appendix rather than central feature
Phase 6: Human-label calibration

Goal:

Add small rigorous evaluation.

Tasks:

create annotation interface
label 300–1,000 pairs
train linear probes
compare to zero-shot prompt axes
document results

Deliverables:

benchmark dataset
evaluation report
updated explorer scores if useful

Completion criteria:

model comparison table
clear story about what labels improve
at least one axis shows measurable improvement over zero-shot prompt scoring
Phase 7: Public artifact polish

Goal:

Turn project into portfolio package.

Tasks:

clean repo
write case study
create demo video
create visual atlas
write limitations
add links from homepage

Deliverables:

public case study
GitHub repo
demo video
screenshots
visual atlas

Completion criteria:

project is linkable in job applications
homepage visitor understands it in 30 seconds
technical reader can inspect the repo and trust the implementation
visual reader sees a polished, memorable artifact
19. Ambitious completion criteria

The project is fully complete when it produces a coherent portfolio package, not just a working app.

19.1 Technical completion

The technical project should include:

curated fashion/person subset from LAION-derived data
reproducible filtering scripts
thumbnail/contact-sheet inspection workflow
CLIP or OpenCLIP embeddings
at least one fashion-specific or alternative embedding model tested
prompt-axis scoring
vector search index
2D embedding projection
cluster exemplars
interactive explorer
documented failure cases
clean config-driven pipeline
clear repo structure
minimal reproducibility instructions
19.2 Data-product completion

The explorer should support:

selecting an outfit/image
viewing similar/style-neighbor images
seeing where it lies in style space
using at least 5 interpretable style axes
comparing at least 2 embedding/scoring approaches
showing model limitations/failure cases
generating screenshots that look professional
19.3 Communication completion

The public-facing project should include:

polished case study page on jroth.space
one strong hero screenshot
visual pipeline diagram
style-space atlas figure
60–90 second demo video
GitHub README with screenshots
concise technical explanation
limitations / ethics section
short “what this demonstrates” section
19.4 Evaluation completion

At least one lightweight evaluation should exist:

pairwise human labels for one or more axes
comparison of prompt-axis scoring vs human-calibrated probe
retrieval sanity checks
model comparison table
documented failure taxonomy

Ambitious version:

500+ pairwise labels
5 evaluated axes
CLIP vs FashionCLIP vs aesthetic predictor vs calibrated probe
small written report: “What do vision-language models know about style?”
19.5 Portfolio completion

The project should be usable in job applications as:

one homepage case study
one GitHub repo
one demo video
one image-heavy artifact
one technical writeup

A hiring manager should be able to conclude:

This person can handle large-scale visual data, use modern ML pragmatically, build data products, communicate clearly, and make technically complex things visually understandable.

19.6 “Wow” completion

The project has at least one moment where a viewer intuitively gets it.

Possible wow moments:

clicking an outfit and seeing a coherent neighborhood appear
moving a “minimal ↔ maximal” axis slider and seeing style change smoothly
seeing a huge style map with meaningful clusters
comparing CLIP vs FashionCLIP neighborhoods
seeing a model failure case explained clearly
watching a messy LAION image pool become an elegant style atlas

The project is not fully complete without at least one of these moments.

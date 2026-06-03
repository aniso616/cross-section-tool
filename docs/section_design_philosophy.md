# The Visual Display of Geological Understanding
## A Design Philosophy for Section

---

## The Problem with Scientific Software

Scientific software has a dirty secret: it treats visualization as a byproduct. The data model is the real work. The computation is the real work. The visualization is what happens when you press a button at the end.

This is backwards.

For a geologist constructing a cross-section, the visualization *is* the scientific instrument. The image on screen is not a report of thinking that happened elsewhere — it is the medium in which thinking happens. When you trace a horizon, you are not describing a geological interpretation you already hold in your head. You are discovering it. The act of drawing, of making something visible, is the act of understanding.

Software that treats this as a mere display problem misunderstands the science.

---

## The Tufte Problem

Edward Tufte demonstrated that the quality of information design and the quality of thinking are not separable. In his analysis of the Challenger disaster, the engineers had the data to make the right decision. They failed because the data was presented in a way that obscured the pattern. The visualization was not a neutral container for facts — it was a cognitive filter that determined what could be seen and therefore what could be known.

This principle applies with full force to geoscience interpretation. The way a cross-section is drawn encodes assumptions. The way a well log is displayed against stratigraphy either reveals or conceals correlation. The way uncertainty is rendered — or more commonly, not rendered at all — determines whether a geologist is reasoning about a model or mistaking a model for reality.

Most geoscience software fails the Tufte test. It produces pictures that are busy with chrome and sparse with information. Toolbars, docked panels, status bars, and dialog boxes crowd the frame. The geology itself — the thing the geologist came to understand — is relegated to a window within a window, surrounded by controls for things that could be handled elsewhere.

Tufte called this "chartjunk." In geoscience software, it is everywhere.

---

## The Bret Victor Problem

Bret Victor argued that software that presents information should behave like a dynamic document, not a control panel. The distinction matters. A control panel is something you operate to produce an output somewhere else. A document is something you read, and in reading, understand.

Most geoscience software is a control panel. You set parameters in dialogs, run processes, and receive outputs that you then look at separately. The interpretation and the visualization are decoupled. This is not just an aesthetic failure — it is a cognitive one. The lag between action and consequence breaks the feedback loop that makes thinking possible.

Victor's principle of direct manipulation is the corrective. When you can grab a horizon and drag it and watch the structure update in real time, you are not operating a control panel — you are thinking with the data. The software becomes an extension of geological reasoning rather than a machine that geological reasoning operates.

---

## The Norman Problem

Don Norman established that good interface design requires that the system image — what the software shows — must match the user's mental model of the problem.

A geologist's mental model of a cross-section is not a list of objects in a database. It is a spatial, temporal, and kinematic whole. Formations have thickness and attitude and age. Faults have geometry and displacement history. The section exists in the context of a map, and both exist in the context of geologic time.

Software that fragments this mental model into separate, poorly connected modules — a section editor here, a map viewer there, a stratigraphic column somewhere else — is not just inconvenient. It is cognitively hostile. Every context switch between modules is a moment when the geologist must reconstruct their mental model from scratch. Every disconnection between views is an opportunity for inconsistency to creep in unnoticed.

The Norman imperative for Section is this: the interface should feel like the geology, not like the software.

---

## First Principles

From these three critiques — Tufte's on information density, Victor's on direct manipulation, Norman's on mental models — we can derive first principles for Section.

### I. The Section Is Primary

The cross-section is the primary epistemic object. Everything else in the application — the map view, the 3D view, the stratigraphic column, the thermal model — exists in service of understanding the section. This hierarchy must be felt in the interface. The section dominates the screen. Other views are contextual, always present, never competing.

This is not a layout decision. It is a statement about the nature of the science.

### II. Seeing Is Thinking

The visualization is not a report of interpretation. It is the medium in which interpretation happens. This means the visualization must be fast enough to think with — changes rendered in milliseconds, not seconds. It means parameters must be manipulable directly, in place, without dialogs. It means the consequence of every action must be immediately visible, not just in the section but in every view that the section affects.

If the geologist has to wait, they have been kicked out of their own thinking.

### III. Every Pixel Must Work

Tufte's data-ink ratio is not an aesthetic preference. It is a statement about cognitive load. Every element of the display that does not carry information is using up the limited bandwidth of human attention. Toolbars can be hidden. Panels can be collapsed. Labels can be elided until needed. The geology should dominate the frame.

This principle governs not just layout but visual encoding. Color should encode information — lithology, age, confidence, temperature — not decorate. Line weight should encode importance or certainty, not style. Typography should be invisible in the sense that it never calls attention to itself, only to what it labels.

### IV. Uncertainty Is Data

Most geoscience software displays interpretations as though they were measurements. Horizons are drawn as crisp lines. Formation boundaries are sharp. The model looks authoritative even when it is speculative.

This is a form of epistemic dishonesty, and it is dangerous. Geologists make decisions based on what they see. If what they see looks certain, they will reason as though it is certain.

Section must represent the epistemic status of every element. An interpreted horizon looks different from a modeled one. A constrained fault looks different from a projected one. Uncertainty in thermal history is visible in the display of thermal history. The visual grammar of certainty is part of the scientific content of the software.

### V. Context Always Present

Don Norman's principle: never make the user remember where they are. A geologist working on a section at 3km depth should always know, without looking away, where that section is in map view, what stratigraphic interval it is in, and how it relates to the 3D model.

This is not achieved by providing tabs or toggles that reveal this information on demand. It is achieved by showing it always, at the periphery of attention, so that context is maintained continuously rather than reconstructed from scratch every time the geologist needs it.

Overview and detail simultaneously. Not sequentially.

### VI. Consistent Visual Language

A lithology that is tan in the section is tan in the 3D view and tan in the map view. An age that is rendered in a particular shade of blue in the section is that same shade in the stratigraphic column. The color is not a style choice — it is a code, and codes must be consistent to be useful.

This seems obvious. Almost no geoscience software does it.

### VII. The Construction Is the Model

Most geoscience software treats interpretation as recovery: there is a true subsurface, recorded in seismic and sampled by wells, and the geologist traces what is already there. The pick measures a signal; the model is what you extract.

This assumes the data to recover from. Often it isn't there. The structure is built from a few wells, an outcrop, a regional analog — and the geologist is not recovering a known subsurface but proposing one, constrained by what little is known. A fault drawn across a section no seismic line ever crossed is not a measurement. It is a geometry that expresses a hypothesis.

So in Section the source of truth is the construction, not a recovered world geometry that the lines approximate. What was drawn, how it was drawn — the dip imposed, the bed held parallel, the assertion that two traces are one structure — and the epistemic status of each: that is the model. A feature's position in three dimensions is derived from its construction, only as far as the construction constrains it, and left explicitly underdetermined where it does not.

This is what separates Section from seismic interpretation software, and it is a difference in kind rather than degree. Seismic software treats the world geometry as truth and the picks as its recovery; given one section, it still returns a confident surface, because its model cannot represent "underdetermined." Section treats the construction as truth and the geometry as a consequence; given one section, it returns the trace, the assertion, and nothing more — because nothing more has been constrained.

Preserving the construction rather than baking it away is what makes the model interrogable. You can ask why a surface sits where it does, change a dip and see the consequence, rename a bed without breaking the structures built on it, and carry the construction into restoration, where the rules that built the geometry are inverted to unbuild it. The construction is not provenance kept for tidiness; it is the content of the model, and it must survive every save, reload, and rename. Lose the construction and you have lost the model, even with its baked geometry intact.

This is the structural corollary to *Uncertainty Is Data*. There, uncertainty is what the display must show; here, it is what the model must hold.

---

## The Method, Not the Model

There is a quiet assumption in most subsurface software: that the goal is a model — a single, consistent, three-dimensional object that represents the geology. The geologist's job is to build it; the software's job is to hold it. Everything is in service of arriving at the solid.

For the geology this tool is built for, that assumption is misplaced. When data is sparse — a few wells, an outcrop, a section or two, a regional analog — there is no single consistent solid to arrive at. There is a *space* of geologies consistent with what little is known, and the honest object is not one model but the family of possibilities and the constraints that bound it. A tool that insists on the solid will fabricate one, and the geologist, looking at it, will mistake a guess for a result.

The working geologist already knows this, and already has a method for it. It is older than the software and it is not a limitation — it is how structural thinking is done when data is thin. You map contacts and measure attitudes in plan. You draw cross-sections through the map. You sketch the structure at depth, an axial trace here, a fault's map pattern there. You hold the three-dimensional structure not as an instantiated solid but across a family of two-dimensional representations that are mutually consistent — each constraining the others, the third dimension implied by how they relate, never committed to beyond what the observations support. Pen and paper, done well, is not a poor approximation of 3D modelling. It is a different and often more honest method.

Section is the digitisation of that method, not the modelling that replaced it.

The cross-section and the map are not different tools here; they are the same kind of thing — a slice through the earth, carrying a coordinate system and the interpretations drawn on it. A vertical slice is a section. A horizontal slice is a plan at a depth. The surface map is the horizontal slice at the top. Each is at once a window — showing where every other slice's interpretations cross it — and a workspace you draw on. Drawing on any one constrains all the others, because they share the geological entities and a single coordinate frame. The consistency the geologist used to hold in their head is enforced by the structure instead.

The third dimension is not avoided. It is *earned*. A structure observed on several slices is constrained in three dimensions by the family of its traces — the dots, connected only within what is geologically feasible, and rendered carrying the breadth of what remains unconstrained. Where observations are dense, the implied geometry is tight; where they are sparse, it is honestly broad. The solid, when shown at all, is a consequence of the two-dimensional interpretations and their feasibility, never an assertion that outruns them.

This is what the principles above are in service of. *The Section Is Primary* is the first slice. *Seeing Is Thinking* is why drawing must be direct. *Uncertainty Is Data* and *The Construction Is the Model* are how the earned third dimension stays honest about how much of itself it has earned. The aim is not to build the geologist a model. It is to give the geologist back their method, with the bookkeeping done and the georeferencing kept.

---

## What This Is Not

This philosophy does not argue for Section to be beautiful in the decorative sense. Decoration is the enemy of information. A chart with a gradient background, drop shadows, and a colored border is not more beautiful than a clean one — it is more cluttered, and clutter is the failure mode of all information design.

Nor does it argue for Section to be simple in the sense of having fewer features. The geology is complex. The software must be capable of representing that complexity. The goal is not simplicity but clarity — the condition in which complexity is visible without being confusing.

The goal is that a geologist, looking at a Section display, sees the geology. Not the software. Not the interface. The geology.

---

## A Test

Here is a practical test for every design decision in Section:

> *Does this make the geology more visible, or does it make itself more visible?*

A toolbar button that does something rarely used makes itself visible. Move it to a menu. A color scheme that uses five shades of gray to represent five different lithologies makes itself visible. Replace it with a perceptually distinct palette. A dialog box that requires three steps to change a horizon's interpretation makes itself visible. Replace it with a direct manipulation.

If the answer to the test is "it makes itself visible," reconsider.

---

## The Aspiration

There is a standard for this kind of work, and it is not set by software. It is set by the best geological atlases, the best structural cross-sections drawn by hand, the best stratigraphic charts produced by the masters of the craft. Those objects are beautiful in the way that Tufte means: the beauty is the clarity, and the clarity is the understanding.

Section should aspire to that standard. Not to replicate the hand-drawn section — the software can do things the hand cannot, and it should — but to take seriously the idea that the display of geological information is a form of scientific communication that has a standard of excellence, and that standard is worth pursuing.

The geological record is one of the most complex and beautiful archives of Earth history that exists. It deserves software that treats it accordingly.

---

*This document is a living argument. As Section develops, the principles stated here will be tested against implementation reality, revised where they prove naive, and extended where the geology demands it.*

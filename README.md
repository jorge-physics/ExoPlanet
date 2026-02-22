# Exoplanet Detection Methods Visualizer

An interactive real-time visualization of the two primary methods used to detect and characterize exoplanets — **transit photometry** and **Doppler spectroscopy** — built with Python and pygame.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![pygame](https://img.shields.io/badge/pygame-2.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

---

## Overview

The simulation renders a quad-panel display showing the physics of an exoplanet system in real time:

| Panel | Description |
|---|---|
| **Top-left** | Edge-on orbital view — watch the planet transit in front of and pass behind the star |
| **Bottom-left** | Top-down orbital view — see the barycentric motion of both bodies |
| **Top-right** | Running transit light curve — flux dips as the planet crosses the stellar disk |
| **Bottom-right** | Hydrogen Balmer absorption spectrum with live Doppler shift + radial velocity history |

---

## Physics

### Coordinate System (right-hand)
- **x** → toward observer
- **y** → right
- **z** → up

The edge-on view looks along **−x** (y/z plane on screen).  
The top-down view is rotated 90° CW around y, so x points down — the observer is at the **bottom** of that panel.

### Orbital Mechanics
Circular orbits about the common barycenter:

```
x_planet =  R_p · cos(θ),   y_planet =  R_p · sin(θ)
x_star   = -R_s · cos(θ),   y_star   = -R_s · sin(θ)
```

### Barycenter Scaling
With constant density, mass ∝ R³. The star orbit radius scales as:

```
R_s = R_p × (ρ_planet / ρ_star) × (R_plan / R_star)³
```

Both planet **radius** and **density** independently perturb the star's orbit.

### Doppler Radial Velocity
Star radial velocity toward the observer:

```
v_r = d(x_s)/dt = R_s · ω · sin(θ)
```

- **θ = 0** → transit midpoint → zero crossing (blueshift → redshift)
- **θ = π/2** → maximum blueshift (star approaching)
- **θ = π** → occultation midpoint → zero crossing
- **θ = 3π/2** → maximum redshift (star receding)

The spectral shift is exaggerated for visual clarity. Balmer lines shown: Hα (656.3 nm), Hβ (486.1 nm), Hγ (434.0 nm), Hδ (410.2 nm).

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/exoplanet-detection-viz.git
cd exoplanet-detection-viz
pip install -r requirements.txt
python exoplanet_detection.py
```

---

## Controls

| Input | Action |
|---|---|
| `↑` | Increase planet radius |
| `↓` | Decrease planet radius |
| `drag slider` | Adjust planet density (0.5× – 5.0× stellar density) |
| `SPACE` | Pause / resume |
| `ESC` | Quit |

---

## Planet Density Colour Scale

The planet colour reflects its bulk composition:

| Density | Colour | Composition analogue |
|---|---|---|
| 0.5× ρ★ | Neptune blue | Gas giant |
| 1.5× ρ★ | Teal | Ice / water world |
| 3.0× ρ★ | Terracotta | Rocky / terrestrial |
| 5.0× ρ★ | Steel grey | Iron / metallic |

---

## Key Parameters

All adjustable at the top of `exoplanet_detection.py`:

```python
PERIOD       = 9.0     # orbital period (animation seconds)
R_STAR_PX    = 27      # star display radius (pixels)
R_PLAN_MAX   = 13      # max planet radius = R_STAR_PX // 2
DOPPLER_SCALE= 14.0    # exaggerated nm shift at max amplitude
F_LO, F_HI   = 0.75, 1.05  # light curve y-axis limits
DENS_MIN     = 0.5     # slider minimum density ratio
DENS_MAX     = 5.0     # slider maximum density ratio
```

---

## Requirements

```
pygame>=2.0
numpy>=1.20
```

---

## Intended Use

Built for science communication and education. Designed to be used as a live demonstration of:
- How transit depth relates to the planet-to-star radius ratio: **ΔF = (R_p / R_★)²**
- How stellar radial velocity amplitude scales with planet mass and orbital radius
- The phase relationship between the light curve and the Doppler signal

---

## License

MIT License — see `LICENSE` for details.

---

## Authors

**Jorge.Physics** — physicist, science communicator, and TikTok educator.  
Built with Python + pygame for educational physics content creation.

**Claude (Anthropic)** — AI collaborator. Worked iteratively with Jorge on architecture, physics implementation, and all the little fixes along the way.

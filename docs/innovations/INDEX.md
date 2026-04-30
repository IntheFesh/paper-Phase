# Innovations Index · Development Log Navigator

The `round-*-summary.md` files under this directory are **historical
development logs**. Each one captures the full context of an innovation at the
time it was proposed, derived, implemented, stress-tested with counterexamples,
and regression-checked. They're written in chronological order and preserve
the names, experimental parameters, and decisions that were in force back then.

**Related documents:**

- Architecture and math derivations: [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
- Academic abstract: [`../PROJECT_ABSTRACT.md`](../PROJECT_ABSTRACT.md)
- Operations manual: [`../OPERATIONS_GUIDE.md`](../OPERATIONS_GUIDE.md)

Suggested reading order: start with `ARCHITECTURE.md` for the current system,
then dip into the per-round log here when you want to trace a particular
design decision.

---

## Algorithmic implementation

| Change | Code entry | Architecture doc | Development log (historical) |
| :-- | :-- | :-- | :-- |
| I1 · Chunk-level InfoNCE (phase identifiability) | `phase_centric/identifiability.py` | [ARCH §2.1, §3.1, §5.2](../ARCHITECTURE.md) | [`round-3-summary.md`](round-3-summary.md) |
| I2 · Bhattacharyya posterior $\beta_t$ | `phase_centric/phase_posterior.py` | [ARCH §2.2, §3.2, §5.3](../ARCHITECTURE.md) | [`round-4-summary.md`](round-4-summary.md) |
| I3 · PACE-A (weighted FM + entropy regulariser) | `phase_centric/pace_a_loss.py` | [ARCH §2.3, §3.3, §5.4](../ARCHITECTURE.md) | [`round-5-summary.md`](round-5-summary.md) |
| I4 · PACE-B (phase-gated MoE) | `phase_centric/pace_b_moe.py` | [ARCH §3.4, §5.5](../ARCHITECTURE.md) | [`round-6-summary.md`](round-6-summary.md) |
| I5 · PACE-C (phase-density curriculum) | `phase_centric/pace_c_curriculum.py` | [ARCH §3.5](../ARCHITECTURE.md) | [`round-6-summary.md`](round-6-summary.md) |
| I6 · PCAR (budget-adaptive replanning) | `phase_centric/pcar_trigger.py` | [ARCH §2.4, §3.6, §5.6](../ARCHITECTURE.md) | [`round-7-summary.md`](round-7-summary.md) |

---

## Architectural changes

| Topic | Architecture doc | Development log |
| :-- | :-- | :-- |
| PhaseQFlow++ four-layer policy skeleton | [ARCH §1](../ARCHITECTURE.md) | [`round-1-summary.md`](round-1-summary.md) |
| FSQ discrete codes / Shortcut flow head / IQL Verifier | [ARCH §1.3](../ARCHITECTURE.md) | [`round-2-summary.md`](round-2-summary.md) |
| Ablation matrix design and paper artifacts | [ARCH §4](../ARCHITECTURE.md) | [`round-8-summary.md`](round-8-summary.md); numbers in [`round-8-paper-stats.md`](round-8-paper-stats.md) |
| Theory verification scripts | [ARCH §5](../ARCHITECTURE.md) | see the verification section of each round |

---

## File list

| File | Scope |
| :-- | :-- |
| [`round-1-summary.md`](round-1-summary.md) | PhaseQFlow++ four-layer skeleton |
| [`round-2-summary.md`](round-2-summary.md) | FSQ + Shortcut + IQL + BID + ACT base components |
| [`round-3-summary.md`](round-3-summary.md) | Chunk-InfoNCE identifiability |
| [`round-4-summary.md`](round-4-summary.md) | Bhattacharyya $\beta_t$ posterior |
| [`round-5-summary.md`](round-5-summary.md) | PACE-A weighted FM |
| [`round-6-summary.md`](round-6-summary.md) | PACE-B MoE + PACE-C curriculum |
| [`round-7-summary.md`](round-7-summary.md) | PCAR budget-triggered replanning |
| [`round-8-summary.md`](round-8-summary.md) | Ablation matrix + paper artifacts |
| [`round-8-paper-stats.md`](round-8-paper-stats.md) | Round 8 placeholder proxy numbers |

> The old script paths, intermediate names, and placeholder numbers that appear
> in the development logs no longer reflect current state. Treat
> `ARCHITECTURE.md` and the source code as authoritative.

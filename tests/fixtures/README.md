# tests/fixtures/

Toy fixtures for the software test suite.

## Purpose

These fixtures contain **synthetic, minimal data invented solely to test
software correctness** — not to represent the case-study municipality or
any real geographic area. They have known expected answers and are used
exclusively in unit and integration tests.

## What does NOT belong here

- Real census data
- Actual flood extent rasters
- Official shelter registries
- Any data derived from the case-study area

Keep this directory clearly separated from `data/` which holds (or will hold)
real case-study data.

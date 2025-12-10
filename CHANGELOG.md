# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.0.3] - 2025-12-10

### Added

- `TileExtractor` builder pattern for fluent XYZ tile extraction API
- `Reprojector` and `StreamingReprojector` for raster reprojection to arbitrary CRS
- `GeoTiffWriter` for writing reprojected rasters to GeoTIFF files
- `PointQuery` trait with `sample_lonlat()` and `sample_crs()` methods
- `CoordTransformer` for reusable CRS-to-CRS coordinate transforms
- `ResamplingMethod` enum with `Nearest`, `Bilinear`, and `Bicubic` options
- Band selection via `TileExtractor::bands(&[0, 1, 2])`
- Concurrent tile extraction with `extract_xyz_tiles_concurrent()`
- `OverviewQualityHint` for pre-computed overview quality control
- `JPEG` and `WebP` compression support
- `LZW` 16-bit sample support
- Floating-point predictor (`predictor=3`) support
- `HttpRangeReader` for serving COGs from URLs
- `Point` struct in geometry module
- GDAL verification test suite
- RGB test COG (Natural Earth) with global coverage
- Criterion benchmarks for tile extraction and point queries

### Changed

- Tile extraction is now async-only (removed sync API duplication)
- `predictor=2` now correctly accumulates per-component for multi-band data
- `tile_cache` API renamed to `get_by_path`, `contains_by_path`, `insert_by_path`
- Improved error handling with `try_lon_lat_to_mercator` and `try_mercator_to_lon_lat`
- `HttpRangeReader` returns error if `Content-Length` header is missing

### Fixed

- Half-pixel shift for `PixelIsPoint` datasets per GDAL RFC 33
- `predictor=2` multi-band handling (was incorrectly accumulating across bands)
- Silent projection failures now return proper errors

### Removed

- Sync tile extraction functions
- Unused `TileKind` enum from `tile_cache`

## [0.0.2] - 2025-12-05

### Added

- `MemoryRangeReader` for in-memory COG parsing
- Global LRU tile cache with overview index support
- GitHub Actions CI workflow

### Changed

- Renamed from `geocog` to `cogrs`
- Use `proj4rs` for all CRS transformations
- Use `ahash` for faster `HashMap` lookups
- Optimized pixel loop with precomputed transforms
- Added fast inline `EPSG:3857`↔`EPSG:4326` transform (2x speedup for `EPSG:4326` COGs)

### Fixed

- `merc_y_to_lat` formula (was using `PI/2` instead of `PI`)

## [0.0.1] - 2025-12-04

### Added

- Initial release extracted from `tileyolo`
- `CogReader` for reading Cloud Optimized GeoTIFFs
- Local and S3 range reader support
- `DEFLATE`, `LZW`, and `ZSTD` compression support
- XYZ tile extraction
- Basic coordinate projection utilities

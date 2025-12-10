# cogrs

Pure Rust COG (Cloud Optimized `GeoTIFF`) reader library.

## Features

- [Local, HTTP, and S3 sources](#sources)
- [Point queries](#point-queries)
- [XYZ tile extraction](#tile-extraction)
- [Coordinate transforms](#coordinate-transforms)
- [Compression: DEFLATE, LZW, ZSTD, JPEG, WebP](#compression)

## Quick Start

```rust,no_run
use cogrs::{CogReader, PointQuery, TileExtractor};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let reader = CogReader::open("path/to/file.tif")?;

    // Point query (sync)
    let result = reader.sample_lonlat(-122.4, 37.8)?;

    // XYZ tile extraction (async)
    let tile = TileExtractor::new(&reader)
        .xyz(10, 163, 395)
        .extract()
        .await?;

    Ok(())
}
```

## Sources

```rust,no_run
use cogrs::CogReader;
# fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
// Local file
let reader = CogReader::open("path/to/file.tif")?;

// HTTP
let reader = CogReader::open("https://example.com/file.tif")?;

// S3 (uses AWS_* environment variables for credentials)
let reader = CogReader::open("s3://bucket/path/to/file.tif")?;
# Ok(())
# }
```

## Point Queries

```rust,no_run
use cogrs::{CogReader, PointQuery};
# fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
let reader = CogReader::open("elevation.tif")?;

// Sample at lon/lat
let result = reader.sample_lonlat(-122.4, 37.8)?;
for (band, value) in &result.values {
    println!("Band {band}: {value}");
}

// Sample in specific CRS (e.g., UTM zone 10N)
let result = reader.sample_crs(32610, 551000.0, 4185000.0)?;
# Ok(())
# }
```

## Tile Extraction

```rust,no_run
use cogrs::{CogReader, TileExtractor, ResamplingMethod};
# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
let reader = CogReader::open("imagery.tif")?;

// Simple extraction (256x256)
let tile = TileExtractor::new(&reader)
    .xyz(10, 163, 395)
    .extract()
    .await?;

// With options
let tile = TileExtractor::new(&reader)
    .xyz(10, 163, 395)
    .output_size(512, 512)
    .resampling(ResamplingMethod::Bilinear)
    .bands(&[0, 1, 2])
    .extract()
    .await?;
# Ok(())
# }
```

## Coordinate Transforms

```rust
use cogrs::{CoordTransformer, project_point};

// One-off transform
let (x, y) = project_point(4326, 3857, -122.4, 37.8)?;

// Reusable transformer
let transformer = CoordTransformer::new(4326, 32610)?;
let (utm_x, utm_y) = transformer.transform(-122.4, 37.8)?;
# Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
```

## Compression

Supported formats are detected automatically:

- DEFLATE
- LZW (8/16-bit, predictors 1-3)
- ZSTD
- JPEG
- WebP

## License

MIT

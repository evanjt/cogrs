//! Benchmarks for cogrs tile extraction and point query performance.
//!
//! Run with: `cargo bench`
//!
//! These benchmarks measure the critical hot paths:
//! - XYZ tile extraction at various zoom levels
//! - Point queries (single and batch)
//! - Coordinate transformation

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::hint::black_box;
use std::sync::Arc;
use tokio::runtime::Runtime;

use cogrs::{
    CogReader, BoundingBox, CoordTransformer, PointQuery,
    extract_xyz_tile, extract_tile_with_extent,
    range_reader::LocalRangeReader,
};

/// Get path to test COG file (if it exists)
fn get_test_cog_path() -> Option<String> {
    let paths = [
        "/home/evan/projects/personal/geo/tileyolo/data/viridis/output_cog.tif",
        "/home/evan/projects/personal/geo/tileyolo/data/grayscale/gray_3857-cog.tif",
    ];

    for path in paths {
        if std::path::Path::new(path).exists() {
            return Some(path.to_string());
        }
    }
    None
}

/// Benchmark XYZ tile extraction at various zoom levels
fn bench_xyz_tile_extraction(c: &mut Criterion) {
    let Some(path) = get_test_cog_path() else {
        eprintln!("Skipping benchmark: no test COG found");
        return;
    };

    let rt = Runtime::new().unwrap();
    let reader = LocalRangeReader::new(&path).unwrap();
    let cog = CogReader::from_reader(Arc::new(reader)).unwrap();

    let mut group = c.benchmark_group("xyz_tile_extraction");

    // Benchmark at different zoom levels
    for zoom in [0, 2, 4, 6, 8] {
        // Use center tile at each zoom level
        let max_tile = 2u32.pow(zoom);
        let x = max_tile / 2;
        let y = max_tile / 2;

        group.bench_with_input(
            BenchmarkId::new("zoom", zoom),
            &(zoom, x, y),
            |b, &(z, x, y)| {
                b.iter(|| {
                    rt.block_on(extract_xyz_tile(black_box(&cog), z, x, y, (256, 256)))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark tile extraction with different output sizes
fn bench_tile_sizes(c: &mut Criterion) {
    let Some(path) = get_test_cog_path() else {
        eprintln!("Skipping benchmark: no test COG found");
        return;
    };

    let rt = Runtime::new().unwrap();
    let reader = LocalRangeReader::new(&path).unwrap();
    let cog = CogReader::from_reader(Arc::new(reader)).unwrap();

    let mut group = c.benchmark_group("tile_sizes");
    let extent = BoundingBox::from_xyz(4, 8, 8);

    for size in [128, 256, 512, 1024] {
        group.bench_with_input(
            BenchmarkId::new("size", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    rt.block_on(extract_tile_with_extent(black_box(&cog), &extent, (size, size)))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark point queries
fn bench_point_query(c: &mut Criterion) {
    let Some(path) = get_test_cog_path() else {
        eprintln!("Skipping benchmark: no test COG found");
        return;
    };

    let reader = LocalRangeReader::new(&path).unwrap();
    let cog = CogReader::from_reader(Arc::new(reader)).unwrap();

    let mut group = c.benchmark_group("point_query");

    // Single point query
    group.bench_function("single_lonlat", |b| {
        b.iter(|| {
            cog.sample_lonlat(black_box(0.0), black_box(0.0))
        });
    });

    // Batch point queries
    let points: Vec<(f64, f64)> = vec![
        (0.0, 0.0),
        (-122.4, 37.8),
        (139.7, 35.7),
        (2.3, 48.9),
        (-43.2, -22.9),
        (10.0, 51.5),
        (-74.0, 40.7),
        (116.4, 39.9),
        (37.6, 55.7),
        (151.2, -33.9),
    ];

    group.bench_function("batch_10_points", |b| {
        b.iter(|| {
            cog.sample_points_lonlat(black_box(&points))
        });
    });

    group.finish();
}

/// Benchmark coordinate transformation
fn bench_coord_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("coord_transform");

    // Test creating transformers
    group.bench_function("create_4326_to_3857", |b| {
        b.iter(|| {
            CoordTransformer::new(black_box(4326), black_box(3857))
        });
    });

    // Test transforming coordinates
    let transformer = CoordTransformer::new(4326, 3857).unwrap();

    group.bench_function("transform_single", |b| {
        b.iter(|| {
            transformer.transform(black_box(-122.4), black_box(37.8))
        });
    });

    // Batch transform
    let points: Vec<(f64, f64)> = (0..100)
        .map(|i| {
            let lon = -180.0 + (i as f64 * 3.6);
            let lat = (i as f64 * 0.9) - 45.0;
            (lon, lat)
        })
        .collect();

    group.bench_function("transform_batch_100", |b| {
        b.iter(|| {
            transformer.transform_batch(black_box(&points))
        });
    });

    group.finish();
}

/// Benchmark BoundingBox operations
fn bench_bounding_box(c: &mut Criterion) {
    let mut group = c.benchmark_group("bounding_box");

    group.bench_function("from_xyz_z0", |b| {
        b.iter(|| {
            BoundingBox::from_xyz(black_box(0), black_box(0), black_box(0))
        });
    });

    group.bench_function("from_xyz_z10", |b| {
        b.iter(|| {
            BoundingBox::from_xyz(black_box(10), black_box(512), black_box(512))
        });
    });

    group.finish();
}

/// Benchmark COG file opening (metadata parsing)
fn bench_cog_open(c: &mut Criterion) {
    let Some(path) = get_test_cog_path() else {
        eprintln!("Skipping benchmark: no test COG found");
        return;
    };

    c.bench_function("cog_open", |b| {
        b.iter(|| {
            let reader = LocalRangeReader::new(black_box(&path)).unwrap();
            CogReader::from_reader(Arc::new(reader))
        });
    });
}

criterion_group!(
    benches,
    bench_xyz_tile_extraction,
    bench_tile_sizes,
    bench_point_query,
    bench_coord_transform,
    bench_bounding_box,
    bench_cog_open,
);

criterion_main!(benches);

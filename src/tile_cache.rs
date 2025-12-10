use lru::LruCache;
use std::cmp::max;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

const CACHE_CAPACITY_BYTES: usize = 512 * 1024 * 1024; // 512 MB upper bound

/// Key for cached decompressed tiles
/// Includes source identifier, tile index, and optional overview index
#[derive(Clone, Eq, PartialEq)]
struct TileKey {
    /// Source identifier (file path or URL)
    source: Arc<str>,
    /// Tile index within the IFD
    tile_index: u32,
    /// Overview index (None = full resolution, Some(n) = overview n)
    overview_idx: Option<u16>,
}

impl Hash for TileKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.source.hash(state);
        self.tile_index.hash(state);
        self.overview_idx.hash(state);
    }
}

impl TileKey {
    fn new(source: &str, tile_index: usize, overview_idx: Option<usize>) -> Self {
        // Casts are safe: tile counts and overview counts are typically small
        #[allow(clippy::cast_possible_truncation)]
        TileKey {
            source: Arc::from(source),
            tile_index: tile_index as u32,
            overview_idx: overview_idx.map(|i| {
                #[allow(clippy::cast_possible_truncation)]
                { i as u16 }
            }),
        }
    }
}

struct CacheEntry {
    data: Arc<Vec<f32>>,
    size_bytes: usize,
}

pub struct TileCache {
    current_bytes: usize,
    capacity_bytes: usize,
    entries: LruCache<TileKey, CacheEntry>,
    hits: u64,
    misses: u64,
}

impl TileCache {
    fn new(capacity_bytes: usize) -> Self {
        TileCache {
            current_bytes: 0,
            capacity_bytes,
            entries: LruCache::unbounded(),
            hits: 0,
            misses: 0,
        }
    }

    fn get(&mut self, key: &TileKey) -> Option<Arc<Vec<f32>>> {
        if let Some(entry) = self.entries.get(key) {
            self.hits += 1;
            Some(Arc::clone(&entry.data))
        } else {
            self.misses += 1;
            None
        }
    }

    fn contains(&mut self, key: &TileKey) -> bool {
        self.entries.contains(key)
    }

    fn insert(&mut self, key: TileKey, data: Arc<Vec<f32>>, size_bytes: usize) {
        if size_bytes > self.capacity_bytes {
            return;
        }

        if let Some(old) = self.entries.pop(&key) {
            self.current_bytes = self.current_bytes.saturating_sub(old.size_bytes);
        }

        while self.current_bytes + size_bytes > self.capacity_bytes {
            if let Some((_key, entry)) = self.entries.pop_lru() {
                self.current_bytes = self.current_bytes.saturating_sub(entry.size_bytes);
            } else {
                break;
            }
        }

        self.current_bytes = self.current_bytes.saturating_add(size_bytes);
        self.entries.put(key, CacheEntry { data, size_bytes });
    }
}

static TILE_CACHE: std::sync::LazyLock<Mutex<TileCache>> = std::sync::LazyLock::new(|| {
    let cap = max(CACHE_CAPACITY_BYTES, 64 * 1024 * 1024); // never below 64MB
    Mutex::new(TileCache::new(cap))
});

fn make_key(source: &str, tile_index: usize, overview_idx: Option<usize>) -> TileKey {
    TileKey::new(source, tile_index, overview_idx)
}

/// Get a cached tile by source identifier, tile index, and optional overview index
/// - `source`: File path or URL identifying the COG
/// - `tile_index`: Tile index within the IFD
/// - `overview_idx`: None for full resolution, Some(n) for overview n
///
/// # Panics
/// Panics if the cache mutex lock is poisoned.
pub fn get(source: &str, tile_index: usize, overview_idx: Option<usize>) -> Option<Arc<Vec<f32>>> {
    let key = make_key(source, tile_index, overview_idx);
    TILE_CACHE.lock().unwrap().get(&key)
}

/// Check if a tile is cached
///
/// # Panics
/// Panics if the cache mutex lock is poisoned.
pub fn contains(source: &str, tile_index: usize, overview_idx: Option<usize>) -> bool {
    let key = make_key(source, tile_index, overview_idx);
    TILE_CACHE.lock().unwrap().contains(&key)
}

/// Insert a decompressed tile into the cache
///
/// # Panics
/// Panics if the cache mutex lock is poisoned.
pub fn insert(source: &str, tile_index: usize, overview_idx: Option<usize>, data: Arc<Vec<f32>>) {
    let size_bytes = data.len() * std::mem::size_of::<f32>();
    let key = make_key(source, tile_index, overview_idx);
    TILE_CACHE.lock().unwrap().insert(key, data, size_bytes);
}

// ============================================================================
// Path-based API for tiff_chunked.rs and lzw_fallback.rs
// ============================================================================

use std::path::Path;

/// Get cached tile by file path (used by tiff_chunked and lzw_fallback modules)
///
/// # Panics
/// Panics if the cache mutex lock is poisoned.
#[must_use]
pub fn get_by_path(path: &Path, index: usize) -> Option<Arc<Vec<f32>>> {
    let source = path.to_string_lossy();
    get(&source, index, None)
}

/// Check if tile is cached by file path
///
/// # Panics
/// Panics if the cache mutex lock is poisoned.
#[must_use]
pub fn contains_by_path(path: &Path, index: usize) -> bool {
    let source = path.to_string_lossy();
    contains(&source, index, None)
}

/// Insert tile into cache by file path
///
/// # Panics
/// Panics if the cache mutex lock is poisoned.
pub fn insert_by_path(path: &Path, index: usize, data: Arc<Vec<f32>>) {
    let source = path.to_string_lossy();
    insert(&source, index, None, data);
}

/// Cache statistics: (`entry_count`, `current_bytes`, `capacity_bytes`, hits, misses)
///
/// # Panics
/// Panics if the cache mutex lock is poisoned.
pub fn stats() -> (usize, usize, usize, u64, u64) {
    let cache = TILE_CACHE.lock().unwrap();
    (cache.entries.len(), cache.current_bytes, cache.capacity_bytes, cache.hits, cache.misses)
}

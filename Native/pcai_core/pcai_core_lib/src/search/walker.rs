use ignore::{WalkBuilder, WalkState};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Configuration for the walker
pub struct WalkerConfig<'a> {
    pub root_path: &'a Path,
    pub include_patterns: Vec<&'a str>,
    pub exclude_patterns: Vec<&'a str>,
    pub git_ignore: bool,
    pub hidden: bool,
}

impl Default for WalkerConfig<'_> {
    fn default() -> Self {
        Self {
            root_path: Path::new("."),
            include_patterns: Vec::new(),
            exclude_patterns: Vec::new(),
            git_ignore: true,
            hidden: false,
        }
    }
}

/// A generic parallel walker that executes a callback for each file.
pub fn run_walker<F>(config: WalkerConfig, callback: F) -> WalkerStats
where
    F: Fn(&ignore::DirEntry) -> WalkState + Send + Sync + Clone + 'static,
{
    let stats = Arc::new(WalkerStats::default());

    let mut builder = WalkBuilder::new(config.root_path);
    builder
        .hidden(!config.hidden)
        .git_ignore(config.git_ignore)
        .git_global(config.git_ignore)
        .git_exclude(config.git_ignore);

    for pat in config.include_patterns {
        builder.add_custom_ignore_filename(pat); // Simplified; likely need overrides builder for includes
    }

    // Note: 'ignore' crate handles includes/excludes via Overrides usually, ensuring we set this up right.
    // For now, standard gitignore behavior is the baseline.

    let walker = builder.build_parallel();

    let stats_clone = stats.clone();
    walker.run(move || {
        let callback = callback.clone();
        let stats = stats_clone.clone();
        Box::new(move |result| {
            match result {
                Ok(entry) => {
                    stats.files_scanned.fetch_add(1, Ordering::Relaxed);
                    if entry.file_type().is_some_and(|ft| ft.is_file()) {
                        return callback(&entry);
                    }
                }
                Err(_) => {
                    stats.errors.fetch_add(1, Ordering::Relaxed);
                }
            }
            WalkState::Continue
        })
    });

    // Extract values from Arc
    WalkerStats {
        files_scanned: stats.files_scanned.load(Ordering::Relaxed).into(),
        errors: stats.errors.load(Ordering::Relaxed).into(),
        ..Default::default()
    }
}

#[derive(Default)]
pub struct WalkerStats {
    pub files_scanned: AtomicU64,
    pub errors: AtomicU64,
}

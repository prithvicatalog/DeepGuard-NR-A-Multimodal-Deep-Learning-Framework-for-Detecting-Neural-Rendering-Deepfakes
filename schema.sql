CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(64) UNIQUE NOT NULL,
    email VARCHAR(128) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    created_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    filename VARCHAR(256) NOT NULL,
    stored_path VARCHAR(512) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'uploaded',
    created_at DATETIME NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER UNIQUE NOT NULL,
    verdict VARCHAR(8) NOT NULL,
    confidence FLOAT NOT NULL,
    artifact_score FLOAT NOT NULL,
    depth_score FLOAT NOT NULL,
    temporal_score FLOAT NOT NULL,
    lighting_score FLOAT NOT NULL,
    heatmap_path VARCHAR(512) NOT NULL,
    module_summary JSON NOT NULL,
    created_at DATETIME NOT NULL,
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

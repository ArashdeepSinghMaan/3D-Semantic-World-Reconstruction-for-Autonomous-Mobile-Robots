"""Minimal ROS 2 SQLite bag access and saved-frame I/O.

The ``.db3`` file is read directly with ``sqlite3``, so neither a ROS
installation nor ``metadata.yaml`` is required. Message deserialization uses
the ``rosbags`` typestore (pip install rosbags).
"""

import json
from pathlib import Path
import sqlite3
import struct

import numpy as np


def find_db3_files(path):
    """Accept a .db3 file or a bag directory; return the .db3 files in order."""
    path = Path(path).expanduser()
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob('*.db3'))
        if files:
            return files
    raise FileNotFoundError(f'No .db3 file found at {path}')


def list_topics(db_files):
    """Return {topic: (msgtype, message_count)} over all bag files."""
    topics = {}
    for db_file in db_files:
        with sqlite3.connect(f'file:{db_file}?mode=ro', uri=True) as conn:
            rows = conn.execute(
                'SELECT t.name, t.type, COUNT(m.id) FROM topics t '
                'LEFT JOIN messages m ON m.topic_id = t.id GROUP BY t.id').fetchall()
        for name, msgtype, count in rows:
            previous = topics.get(name, (msgtype, 0))
            topics[name] = (msgtype, previous[1] + int(count))
    return topics


def iter_messages(db_files, topic_names):
    """Yield (topic, bag_timestamp_ns, raw_cdr_bytes) in recording order, file by file.

    Rows are read in insertion (rowid) order, which is the order rosbag2
    received them. This is a sequential read; ORDER BY timestamp would force
    random access through the index and is much slower on large bags.
    """
    topic_names = list(topic_names)
    for db_file in db_files:
        with sqlite3.connect(f'file:{db_file}?mode=ro', uri=True) as conn:
            ids = dict(conn.execute('SELECT id, name FROM topics').fetchall())
            wanted = {i: name for i, name in ids.items() if name in topic_names}
            if not wanted:
                continue
            marks = ','.join('?' * len(wanted))
            cursor = conn.execute(
                f'SELECT topic_id, timestamp, data FROM messages '
                f'WHERE topic_id IN ({marks}) ORDER BY id', list(wanted))
            for topic_id, timestamp, data in cursor:
                yield wanted[topic_id], int(timestamp), data


def cdr_header_stamp(raw):
    """Read std_msgs/Header.stamp directly from a CDR buffer, in seconds.

    Valid for any message whose first field is a Header (PointCloud2, Image,
    CompressedImage, CameraInfo, Imu, Odometry, ...). Avoids deserializing
    multi-megabyte clouds just to read their timestamp.
    """
    little_endian = raw[1] == 1  # CDR encapsulation: 0x0001 = CDR_LE
    sec, nanosec = struct.unpack_from('<iI' if little_endian else '>iI', raw, 4)
    return sec + nanosec * 1e-9


def make_deserializer():
    """Return deserialize(raw, msgtype) using rosbags' ROS 2 Humble typestore."""
    try:
        from rosbags.typesys import get_typestore, Stores
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise SystemExit(
            'This tool needs the rosbags library: pip install rosbags') from exc
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    return typestore.deserialize_cdr


# ─── Saved frames ────────────────────────────────────────────────────────────

METADATA_FILE = 'frames.json'


def save_frame(out_dir, index, cloud, msg, bag_time_ns):
    """Save one decoded cloud as .npy plus an entry in frames.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = f'frame_{index:05d}.npy'
    np.save(out_dir / file_name, np.ascontiguousarray(cloud), allow_pickle=False)

    meta_path = out_dir / METADATA_FILE
    entries = json.loads(meta_path.read_text()) if meta_path.exists() else []
    entries = [e for e in entries if e['file'] != file_name]
    entries.append({
        'file': file_name,
        'index': int(index),
        'frame_id': str(msg.header.frame_id),
        'stamp': int(msg.header.stamp.sec) + int(msg.header.stamp.nanosec) * 1e-9,
        'bag_time_ns': int(bag_time_ns),
        'height': int(msg.height),
        'width': int(msg.width),
    })
    entries.sort(key=lambda e: e['index'])
    meta_path.write_text(json.dumps(entries, indent=2))


def load_frames(frames_dir, limit=0):
    """Load frames saved by ``inspect_bag --save-frames``.

    Returns a list of dicts with keys cloud, height, width and the metadata.
    Directories of bare .npy files (without frames.json) are also accepted
    and treated as unorganized clouds.
    """
    frames_dir = Path(frames_dir).expanduser()
    meta_path = frames_dir / METADATA_FILE
    if meta_path.exists():
        entries = json.loads(meta_path.read_text())
    else:
        entries = [{'file': p.name} for p in sorted(frames_dir.glob('*.npy'))]
    if limit and limit > 0:
        entries = entries[:limit]
    if not entries:
        raise FileNotFoundError(f'No saved frames in {frames_dir}')

    frames = []
    for entry in entries:
        cloud = np.load(frames_dir / entry['file'], allow_pickle=False).reshape(-1)
        height = int(entry.get('height', 1))
        width = int(entry.get('width', cloud.shape[0]))
        if height * width != cloud.shape[0]:
            height, width = 1, cloud.shape[0]
        frames.append({**entry, 'cloud': cloud, 'height': height, 'width': width})
    return frames

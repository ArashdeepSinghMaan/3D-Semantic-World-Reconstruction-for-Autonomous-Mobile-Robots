"""Inspect the LiDAR topic of a ROS 2 .db3 bag without running ROS.

Answers the open Phase 0 LiDAR / synchronization checklist items:

* PointCloud2 fields, frame_id, organized layout, is_dense
* header-stamp rate, jitter, gaps; header stamp vs. bag receive time
* invalid / zero-point fractions, range distribution, per-point time span
* optional: nearest-stamp offsets from the LiDAR to other topics
  (cameras, odometry) -> camera-LiDAR and LiDAR-pose synchronization

and optionally saves decoded frames for the offline experiments.

Example::

    ros2 run lidar_pointcloud_processing inspect_bag ~/data/grass.db3 \\
        --compare-topics /stereo/frame_left/image_raw/compressed /unitree/body_odom \\
        --save-frames 20 --out-dir ~/data/grass_frames --report-json grass_lidar.json
"""

import argparse
import json
import sys
import time

from ..analysis import (
    aggregate_quality,
    cloud_quality,
    format_table,
    nearest_offsets,
    offset_statistics,
    stamp_statistics,
)
from ..pointcloud_utils import describe_layout, pointcloud2_to_array
from .bag_io import (
    cdr_header_stamp,
    find_db3_files,
    iter_messages,
    list_topics,
    make_deserializer,
    save_frame,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('bag', help='Path to a .db3 file or a rosbag2 directory')
    parser.add_argument('--topic', default='/os_cloud_node/points',
                        help='PointCloud2 topic to inspect')
    parser.add_argument('--compare-topics', nargs='*', default=[],
                        help='Header-stamped topics to measure time offsets against')
    parser.add_argument('--quality-stride', type=int, default=10,
                        help='Fully decode every Nth cloud for quality metrics')
    parser.add_argument('--max-messages', type=int, default=0,
                        help='Stop after this many clouds (0 = all)')
    parser.add_argument('--save-frames', type=int, default=0,
                        help='Number of decoded clouds to save as .npy')
    parser.add_argument('--save-stride', type=int, default=30,
                        help='Save every Nth cloud (30 = every 3 s at 10 Hz)')
    parser.add_argument('--out-dir', default='frames', help='Directory for saved frames')
    parser.add_argument('--skip-topic-list', action='store_true',
                        help='Skip listing all topics (a full table scan on large bags)')
    parser.add_argument('--report-json', default='', help='Write all statistics to JSON')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    db_files = find_db3_files(args.bag)
    deserialize = make_deserializer()
    report = {'bag_files': [str(f) for f in db_files], 'topic': args.topic}

    topics = list_topics(db_files) if not args.skip_topic_list else None
    if topics is not None:
        width = max(len(t) for t in topics) if topics else 10
        print(f'Topics in {", ".join(f.name for f in db_files)}:')
        for name in sorted(topics):
            msgtype, count = topics[name]
            print(f'  {name:<{width}}  {count:>9,}  {msgtype}')
        print()
        report['topics'] = {k: {'type': v[0], 'count': v[1]} for k, v in topics.items()}
        missing = [t for t in [args.topic, *args.compare_topics] if t not in topics]
        if missing:
            print(f'ERROR: topics not in bag: {missing}', file=sys.stderr)
            return 1
        cloud_type = topics[args.topic][0]
    else:
        cloud_type = 'sensor_msgs/msg/PointCloud2'

    header_stamps, bag_stamps = [], []
    compare_stamps = {t: [] for t in args.compare_topics}
    qualities = []
    frame_ids = set()
    n_clouds = saved = 0
    started = time.perf_counter()

    for topic, bag_time_ns, raw in iter_messages(db_files, [args.topic, *args.compare_topics]):
        if topic != args.topic:
            compare_stamps[topic].append(cdr_header_stamp(raw))
            continue
        if args.max_messages and n_clouds >= args.max_messages:
            if not args.compare_topics:
                break
            continue

        header_stamps.append(cdr_header_stamp(raw))
        bag_stamps.append(bag_time_ns * 1e-9)
        index = n_clouds
        n_clouds += 1

        want_quality = index % max(1, args.quality_stride) == 0
        want_save = (saved < args.save_frames and index % max(1, args.save_stride) == 0)
        if index == 0 or want_quality or want_save:
            msg = deserialize(raw, cloud_type)
            frame_ids.add(msg.header.frame_id)
            cloud = pointcloud2_to_array(msg)
            if index == 0:
                print(describe_layout(msg))
                print()
                report['layout'] = {
                    'frame_id': msg.header.frame_id,
                    'height': int(msg.height), 'width': int(msg.width),
                    'point_step': int(msg.point_step), 'row_step': int(msg.row_step),
                    'is_dense': bool(msg.is_dense), 'is_bigendian': bool(msg.is_bigendian),
                    'fields': [{'name': f.name, 'offset': int(f.offset),
                                'datatype': int(f.datatype), 'count': int(f.count)}
                               for f in msg.fields],
                }
            if want_quality:
                qualities.append(cloud_quality(cloud))
            if want_save:
                save_frame(args.out_dir, index, cloud, msg, bag_time_ns)
                saved += 1

        if n_clouds % 500 == 0:
            print(f'  ... {n_clouds} clouds read ({time.perf_counter() - started:.0f} s)',
                  file=sys.stderr)

    if n_clouds == 0:
        print(f'ERROR: no messages on {args.topic}', file=sys.stderr)
        return 1

    sections = []
    report['frame_ids'] = sorted(frame_ids)
    report['header_stamps'] = stamp_statistics(header_stamps)
    sections.append(format_table(f'{args.topic} header stamps', report['header_stamps']))
    report['bag_stamps'] = stamp_statistics(bag_stamps)
    sections.append(format_table(f'{args.topic} bag receive times', report['bag_stamps']))
    report['receive_minus_header'] = offset_statistics(
        [b - h for b, h in zip(bag_stamps, header_stamps)])
    sections.append(format_table('Bag receive time - header stamp',
                                 report['receive_minus_header']))

    if qualities:
        aggregated = aggregate_quality(qualities)
        report['quality'] = {k: dict(zip(('mean', 'min', 'max'), v))
                             for k, v in aggregated.items()}
        sections.append(format_table(
            f'Cloud quality over {len(qualities)} sampled frames', aggregated))

    report['compare'] = {}
    for topic, stamps in compare_stamps.items():
        entry = {
            'stamps': stamp_statistics(stamps),
            'nearest_offset_from_lidar': offset_statistics(
                nearest_offsets(header_stamps, stamps)),
        }
        report['compare'][topic] = entry
        sections.append(format_table(f'{topic} header stamps', entry['stamps']))
        sections.append(format_table(f'{topic} - nearest to each LiDAR stamp',
                                     entry['nearest_offset_from_lidar']))

    print('\n\n'.join(sections))
    print(f'\nframe_ids seen: {sorted(frame_ids)}')
    if saved:
        print(f'Saved {saved} frames to {args.out_dir}')
    if args.report_json:
        with open(args.report_json, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
        print(f'Wrote {args.report_json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

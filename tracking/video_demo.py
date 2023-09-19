import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker


def run_video(tracker_name, tracker_param, videofile, optional_box=None, debug=None, save_results=False, result_path=None):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = Tracker(tracker_name, tracker_param, "video")
    category = videofile.split('-')[0]
    gt_path = f'/ocean/projects/ele220002p/tongshen/dataset/lasot/LaSOTBenchmark/{category}/{videofile}/groundtruth.txt'
    gt_bbox = open(gt_path).readlines()[0].strip().split(',')
    gt_bbox = list(map(int, gt_bbox))
    videopath = f'/ocean/projects/ele220002p/tongshen/code/vl_tracking/latest/VLTracker/work_dirs/example_seqs/{videofile}.avi'
    tracker.run_video(videofilepath=videopath, optional_box=gt_bbox, debug=debug, save_results=save_results, result_path=result_path)


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('videofile', type=str, help='path to a video file.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.add_argument('--result_path', type=str, default=None, help='Save bounding boxes')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    run_video(args.tracker_name, args.tracker_param, args.videofile, args.optional_box, args.debug, args.save_results, args.result_path)


if __name__ == '__main__':
    main()

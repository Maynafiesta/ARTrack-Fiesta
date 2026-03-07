import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker


def run_video(tracker_name, tracker_param, videofile, optional_box=None, debug=None, save_results=False):
    """Run the tracker on a video file.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        videofile: Path to the video file.
        optional_box: Initial bbox (x, y, w, h). If None, user will be prompted to select it.
        debug: Debug level.
        save_results: Whether to save bounding boxes.
    """
    tracker = Tracker(tracker_name, tracker_param, "video")
    tracker.run_video(videofilepath=videofile, optional_box=optional_box, debug=debug, save_results=save_results)


def get_available_models():
    # Sadece %100 sorunsuz çalışan "En Hızlı" ve "En Kaliteli" modeller kalsın kanka
    working_models = ['artrack_seq_large_384_full', 'artrack_seq_256_full']
    return working_models


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your video.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method (e.g. artrack_seq).')
    parser.add_argument('--tracker_param', type=str, default=None, help='Name of parameter file. If None, you will be prompted.')
    parser.add_argument('videofile', type=str, help='Path to a video file.')
    parser.add_argument('--optional_box', type=int, default=None, nargs=4, help='Initial box with format x y w h.')
    parser.add_argument('--debug', type=int, default=1, help='Debug level (default 1 to show visualizer).')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    tracker_param = args.tracker_param
    if tracker_param is None:
        models = get_available_models()
        if not models:
            print("Error: No models found in experiments/artrack_seq")
            sys.exit(1)
        
        print("\n" + "="*50)
        print("Mevcut Modeller (Available Models):")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        print("="*50)
        
        try:
            choice = int(input(f"\nLütfen bir model numarası seçin (1-{len(models)}): "))
            if 1 <= choice <= len(models):
                tracker_param = models[choice-1]
            else:
                print("Geçersiz seçim!")
                sys.exit(1)
        except ValueError:
            print("Lütfen bir sayı girin!")
            sys.exit(1)

    print(f"\nSeçilen Model: {tracker_param}\n")
    run_video(args.tracker_name, tracker_param, args.videofile, args.optional_box, args.debug, args.save_results)


if __name__ == '__main__':
    main()
